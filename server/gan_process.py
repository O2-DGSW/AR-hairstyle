"""HairFastGAN 을 **별도 프로세스**에서 돌리기 위한 클라이언트/워커.

왜 이 파일이 존재하는가
-----------------------
이 서버는 SLO 가 정반대인 두 모델을 한 GPU 에서 돌린다.

  - SegFormer 얼굴파싱 : 프레임당 16ms. **지연**이 전부인 실시간 경로.
  - HairFastGAN        : 적재 ~90초/5.5GB, 추론 ~9초. **처리량**만 중요한 배치 경로.

둘이 같은 프로세스에 있으면 GAN 이 적재/추론하는 동안 파이썬 GIL, CUDA 컨텍스트,
VRAM 을 실시간 경로와 나눠 쓴다. 그래서 지금까지 server.py 는 GAN 이 도는 동안
아예 **검은 프레임을 내보내는 것**으로 때웠다. 아키텍처 문제를 UX 로 덮은 것이다.
게다가 GAN 쪽에서 CUDA OOM 이나 세그폴트가 나면 서버 전체가 같이 죽고,
HairFast 모델은 해제 API 가 없어서 한 번 올린 5.5GB 를 영영 돌려받지 못한다.

프로세스를 나누면 셋 다 해결된다: 실시간 경로는 GAN 적재에 절대 막히지 않고,
GAN 이 죽어도 부모는 재기동만 하면 되고, 자식을 죽이면 VRAM 이 통째로 돌아온다.

왜 multiprocessing.Process 가 아니라 subprocess 인가
----------------------------------------------------
Windows 는 start method 가 spawn 이다. multiprocessing.Process 로 띄우면 자식이
부모의 `__main__`(= server.py)을 다시 import 하고, server.py 의 모듈 레벨 코드
(hair_asset.load_assets(), ThreadPoolExecutor 생성 등)가 자식에서도 실행된다.
그래서 자식은 `python -m gan_process --worker` 로 **깨끗하게** 띄우고, 통신만
multiprocessing.connection(= pickle 왕복)으로 한다.

폴백
----
프로세스 기동에 실패하면 조용히 죽는 대신 기존 in-process GanWorker 로 폴백한다
(backend 프로퍼티가 "thread" 를 반환). 느려도 동작하는 쪽이 낫다.
"""
from __future__ import annotations

import logging
import os
import secrets
import subprocess
import sys
import threading
import time
import traceback
from multiprocessing import connection

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    # 자식은 `python -m gan_process` 로 뜨므로 cwd 가 sys.path 에 들어가지만,
    # 부모가 다른 cwd 에서 띄우거나 -m 이 아닌 경로로 import 될 수도 있어
    # 여기서 명시적으로 보정한다. 아래 `import gan_worker` 가 이것에 의존한다.
    sys.path.insert(0, _HERE)

from config import CONFIG, ROOT          # noqa: E402
import gan_worker                        # noqa: E402

# server.py 가 gan_worker 에서 가져다 쓰는 것들. 백엔드를 바꿔도 import 경로를
# 하나로 볼 수 있게 여기서도 노출한다(같은 객체이므로 동작은 동일).
from gan_worker import CAPTURE_DIR, REF_DIR, list_references   # noqa: E402,F401

logger = logging.getLogger("gan_process")

#: authkey 를 넘기는 환경변수. argv 로 넘기면 같은 머신의 다른 사용자가
#: 프로세스 목록(tasklist / wmic)에서 그대로 읽을 수 있다.
_AUTHKEY_ENV = "HEDDY_GAN_AUTHKEY"

#: 자식이 떠서 접속해 올 때까지 기다리는 상한(초). 자식은 torch 를 import 하지
#: 않고(모델 적재는 첫 swap 때) cv2/numpy 만 올리므로 실측 수 초면 충분하다.
_STARTUP_TIMEOUT_S = 120.0
#: 재기동 폭주 방지. 자식이 뜨자마자 죽는 상황(드라이버 문제 등)에서 초당 수십
#: 번 프로세스를 만드는 것을 막는다.
_MIN_RESTART_INTERVAL_S = 5.0
#: shutdown 요청 후 자식이 스스로 끝나기를 기다리는 시간.
_SHUTDOWN_GRACE_S = 5.0


class _Child:
    """자식 프로세스 한 번의 기동에 딸린 것들 묶음.

    재기동할 때 새 인스턴스를 만든다. 그래야 늦게 끝난 예전 accept 스레드가
    새 세대의 커넥션을 덮어쓰는 사고가 구조적으로 불가능하다.
    """

    __slots__ = ("proc", "listener", "conn", "error", "ready", "threads", "worker_pid",
                 "stopping")

    def __init__(self, proc, listener):
        self.proc = proc
        self.listener = listener
        self.conn = None
        self.error = None
        self.ready = threading.Event()     # conn 또는 error 가 정해지면 set
        # 종료가 시작됐음을 accept 스레드에 알린다.
        # 이게 없으면 teardown 이 listener 를 닫는 순간 accept 가 OSError 를 받는데,
        # 그 시점엔 자식이 아직 살아 있고 마감시한도 안 지나서 **닫힌 소켓에 계속
        # 재시도한다.** 서버를 띄우자마자 Ctrl+C 하면 이 경합이 그대로 드러난다
        # (WinError 10038 이 반복되고 자식은 ConnectionRefusedError 로 죽는다).
        self.stopping = threading.Event()
        self.threads = []
        # Windows venv 의 Scripts\python.exe 는 실제 인터프리터가 아니라 **런처**다.
        # 베이스 인터프리터를 자식으로 다시 띄우므로 Popen.pid 와 워커의 os.getpid()
        # 가 다르다(실측 확인). VRAM 5.5GB 를 들고 있는 쪽은 후자이므로 진단용으로는
        # 워커가 직접 알려준 pid 를 쓴다.
        # (런처는 자식을 job object 로 묶어 두어서 Popen.terminate() 하면 실제
        #  인터프리터도 같이 죽는다. 이것도 실측으로 확인했다 - 고아는 안 남는다.)
        self.worker_pid = None

    def alive(self) -> bool:
        return self.proc.poll() is None


class GanClient:
    """HairFastGAN 을 별도 프로세스에서 돌리는 클라이언트. GanWorker 의 드롭인 대체.

    swap() 은 블로킹이다. 호출부(server.py)가 전용 executor 스레드에서 부른다.
    """

    def __init__(self, cfg=None, log=print):
        self._cfg = cfg if cfg is not None else CONFIG
        self._log = log
        self._backend = "thread" if str(self._cfg.gan_backend).lower() == "thread" else "process"

        # _lock 은 상태 전이(자식 기동/정리/백엔드 전환)만 짧게 감싼다.
        # _io_lock 은 파이프 왕복 전체(최대 ~90초)를 잡는다. 둘을 나눈 이유는
        # close() 가 진행 중인 swap 을 기다리지 않고 커넥션을 끊어 깨울 수
        # 있어야 하기 때문이다. 잠금 순서는 항상 _io_lock -> _lock 이다.
        self._lock = threading.RLock()
        self._io_lock = threading.Lock()

        self._child = None
        self._thread_worker = None      # 폴백용 in-process GanWorker
        self._loaded = False
        self._load_seconds = None
        self._restarts = 0
        self._last_error = None
        self._last_start = 0.0
        self._ever_connected = False
        self._closed = False

    # ---------- 공개 프로퍼티 ----------
    @property
    def loaded(self) -> bool:
        """워커에 모델이 올라가 있는가. 절대 블로킹하지 않는다
        (server.py 가 이벤트 루프 위에서 상태 응답에 쓴다)."""
        if self._backend == "thread":
            w = self._thread_worker
            return bool(w is not None and w.loaded)
        return self._loaded

    @property
    def backend(self) -> str:
        """"process" | "thread". 폴백이 일어나면 런타임에 "thread" 로 바뀐다."""
        return self._backend

    # ---------- 수명 주기 ----------
    def start(self) -> None:
        """워커를 예열한다. 여러 번 불러도 안전하고, 블로킹하지 않는다.

        여기서 접속 완료까지 기다리지 않는 이유: 서버 기동 경로에서 부르는데
        자식 import 에 수 초가 걸리고, 그동안 실시간 경로가 멈추면 안 된다.
        실제 접속은 백그라운드 accept 스레드가 받고, 첫 swap 이 기다린다.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("이미 close() 된 GanClient 입니다")
            if self._backend == "thread":
                self._ensure_thread_worker()
                return
            if self._child is not None and self._child.alive():
                return
            self._start_child()

    def close(self) -> None:
        """자식 프로세스를 내리고 리소스를 놓는다. 여러 번 불러도 안전하다."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            worker, self._thread_worker = self._thread_worker, None
            self._teardown_child(graceful=True)
        if worker is not None:
            worker.close()

    def health(self) -> dict:
        """상태 스냅샷. 블로킹하지 않는다(파이프 왕복 없음)."""
        ch = self._child
        if self._backend == "thread":
            alive = self._thread_worker is not None
            pid = os.getpid()
            load_seconds = getattr(self._thread_worker, "load_seconds", None)
        else:
            alive = bool(ch is not None and ch.alive())
            pid = (ch.worker_pid or ch.proc.pid) if ch is not None else None
            load_seconds = self._load_seconds
        return {
            "backend": self._backend,
            "alive": alive,
            "loaded": self.loaded,
            "load_seconds": load_seconds,
            "pid": pid,
            "restarts": self._restarts,
            "last_error": self._last_error,
        }

    def ping(self, timeout: float = 10.0) -> dict:
        """자식이 살아서 요청을 처리하는지 확인한다. **모델을 적재하지 않는다.**

        health() 와 달리 실제로 파이프를 왕복하므로 블로킹한다. 진단/테스트용이며
        실시간 경로에서 부르지 말 것.
        """
        if self._backend == "thread":
            return {"ok": True, "backend": "thread", "loaded": self.loaded,
                    "pid": os.getpid()}
        with self._io_lock:
            child = self._ensure_child()
            if child is None:                     # 폴백됨
                return {"ok": True, "backend": "thread", "loaded": self.loaded,
                        "pid": os.getpid()}
            rep = self._roundtrip(child, {"op": "ping"}, timeout)
        rep = dict(rep)
        rep["backend"] = "process"
        return rep

    # ---------- 합성 ----------
    def set_rotate(self, ckpt_path, log=print):
        """Rotate 체크포인트만 교체한다 -> 적용된 절대경로.

        GanWorker.set_rotate 와 시그니처가 같다. 모델이 아직 안 올라갔으면
        이 호출이 적재(~90초)를 포함하므로 그때는 긴 타임아웃을 쓴다.
        """
        ckpt_path = os.path.abspath(ckpt_path)
        if self._backend == "thread":
            return self._thread_set_rotate(ckpt_path, log)

        with self._io_lock:
            child = self._ensure_child(log)
            if child is None:                     # 기동 실패 -> thread 로 폴백됨
                return self._thread_set_rotate(ckpt_path, log)
            timeout = (self._cfg.gan_load_timeout_s if not self._loaded
                       else self._cfg.gan_call_timeout_s)
            rep = self._roundtrip(child, {"op": "set_rotate",
                                          "checkpoint": ckpt_path}, timeout)

        if not rep.get("ok"):
            tb = rep.get("traceback")
            if tb:
                logger.error("Rotate 교체 실패:\n%s", tb)
            self._last_error = rep.get("error") or "알 수 없는 Rotate 교체 실패"
            raise RuntimeError(self._last_error)
        self._loaded = True
        self._rotate_checkpoint = rep["checkpoint"]
        return rep["checkpoint"]

    @property
    def rotate_checkpoint(self):
        return getattr(self, "_rotate_checkpoint", None)

    def swap(self, face_bgr, shape_path, color_path=None, log=print):
        """웹캠 BGR 프레임 + 헤어 참고사진 경로 -> (합성된 BGR 이미지, 소요 초).

        GanWorker.swap 과 시그니처/반환값이 같다. 블로킹한다.
        """
        if self._backend == "thread":
            return self._thread_swap(face_bgr, shape_path, color_path, log)

        with self._io_lock:
            child = self._ensure_child(log)
            if child is None:                     # 기동 실패 -> thread 로 폴백됨
                return self._thread_swap(face_bgr, shape_path, color_path, log)

            # 모델이 아직 안 올라갔으면 이 호출이 적재(~90초)를 포함한다.
            first = not self._loaded
            timeout = (self._cfg.gan_load_timeout_s if first
                       else self._cfg.gan_call_timeout_s)
            if first:
                log("GAN 워커에 첫 요청 전송 (모델 적재 포함, 최대 %.0fs 대기)" % timeout)

            req = {
                "op": "swap",
                "face": face_bgr,
                # 자식은 HairFastGAN 디렉터리로 chdir 하므로 상대경로는 깨진다.
                "shape": os.path.abspath(shape_path),
                "color": os.path.abspath(color_path) if color_path else None,
            }
            rep = self._roundtrip(child, req, timeout)

        if not rep.get("ok"):
            # 자식은 살아 있다. 입력 탓(얼굴 미검출 등)이므로 재기동하지 않는다.
            self._loaded = bool(rep.get("loaded", self._loaded))
            tb = rep.get("traceback")
            if tb:
                logger.error("GAN 워커 예외:\n%s", tb)
            self._last_error = rep.get("error") or "알 수 없는 GAN 실패"
            raise RuntimeError(self._last_error)

        self._loaded = True
        if rep.get("load_seconds") is not None:
            self._load_seconds = rep["load_seconds"]
        if rep.get("rotate_checkpoint"):
            self._rotate_checkpoint = rep["rotate_checkpoint"]
        return rep["result"], float(rep["seconds"])

    # ---------- 내부: 파이프 왕복 ----------
    def _roundtrip(self, child: _Child, req: dict, timeout: float) -> dict:
        """요청 하나를 보내고 응답을 받는다. 실패하면 자식을 죽이고 예외를 던진다.

        타임아웃을 반드시 거는 이유: 자식이 CUDA 안에서 멈추면 recv() 는 영원히
        돌아오지 않고, 그러면 GAN executor 스레드가 영구히 잠긴다. 실시간 경로는
        살아 있어도 촬영 기능이 통째로 죽는다.
        """
        conn = child.conn
        try:
            conn.send(req)
            if not conn.poll(timeout):
                raise TimeoutError("GAN 워커가 %.0f초 안에 응답하지 않았습니다" % timeout)
            rep = conn.recv()
            if not isinstance(rep, dict):
                raise RuntimeError("GAN 워커 응답 형식이 이상합니다: %r" % (type(rep),))
            return rep
        except BaseException as e:
            self._on_broken(child, e)
            raise RuntimeError("GAN 워커 통신 실패: %s" % (self._last_error,)) from e

    def _on_broken(self, child: _Child, exc: BaseException) -> None:
        """타임아웃/파손 처리: 자식을 죽이고 다음 호출 때 새로 띄우게 한다."""
        self._last_error = "%s: %s" % (type(exc).__name__, exc)
        logger.warning("GAN 워커가 깨졌습니다 (%s). 자식을 종료하고 다음 호출 때 재기동합니다.",
                       self._last_error)
        with self._lock:
            self._restarts += 1
            self._loaded = False
            if self._child is child:
                self._teardown_child(graceful=False)

    # ---------- 내부: 자식 수명 주기 ----------
    def _ensure_child(self, log=None):
        """쓸 수 있는 _Child 를 돌려준다. thread 로 폴백됐으면 None."""
        with self._lock:
            if self._closed:
                raise RuntimeError("이미 close() 된 GanClient 입니다")
            if self._backend == "thread":
                return None

            child = self._child
            if child is not None and child.ready.is_set() and not child.alive():
                # 우리가 죽인 게 아니라 자식이 스스로 죽었다(CUDA OOM, 세그폴트,
                # 밖에서 taskkill 등). _on_broken 을 안 거쳤으므로 여기서 센다.
                logger.warning("GAN 자식 프로세스가 종료되어 있습니다 (exit=%s). 재기동합니다.",
                               child.proc.returncode)
                self._restarts += 1
                self._last_error = "자식 프로세스가 예기치 않게 종료했습니다 (exit=%s)" % (
                    child.proc.returncode,)
                self._teardown_child(graceful=False)
                child = None

            if child is None:
                wait = _MIN_RESTART_INTERVAL_S - (time.monotonic() - self._last_start)
                if wait > 0 and self._last_start:
                    # 재기동 폭주 방지. 9초짜리 추론에 비하면 무시할 수 있는 대기다.
                    time.sleep(min(wait, _MIN_RESTART_INTERVAL_S))
                child = self._start_child()
                if child is None:
                    return None            # _start_child 안에서 폴백함

        # 접속 대기는 _lock 밖에서 한다(그동안 health()/close() 가 막히면 안 된다).
        try:
            self._await_ready(child)
        except Exception as e:
            with self._lock:
                self._last_error = "%s: %s" % (type(e).__name__, e)
                if self._child is child:
                    self._teardown_child(graceful=False)
                if not self._ever_connected:
                    # 한 번도 붙어본 적이 없다 = 이 환경에서 프로세스 경로가
                    # 원천적으로 안 되는 것이다. 매번 2분씩 헛기다리지 말고 폴백.
                    self._fallback("자식 프로세스가 접속하지 못했습니다: %s" % (self._last_error,))
                    return None
                self._restarts += 1
            raise RuntimeError("GAN 워커 기동 실패: %s" % (self._last_error,)) from e
        return child

    def _await_ready(self, child: _Child) -> None:
        if not child.ready.wait(_STARTUP_TIMEOUT_S + 5.0):
            raise TimeoutError("자식 프로세스 접속 대기 시간 초과")
        if child.conn is None:
            raise RuntimeError(child.error or "자식 프로세스 접속 실패")

    def _start_child(self):
        self._last_start = time.monotonic()
        try:
            child = self._spawn()
        except Exception as e:
            self._fallback("자식 프로세스를 기동하지 못했습니다: %r" % (e,))
            return None
        self._child = child
        return child

    def _spawn(self) -> _Child:
        # 부모가 리스너를 먼저 열고 **커널이 정해준 포트**(0)를 자식에게 알려준다.
        # 고정 포트를 쓰면 서버를 두 번 띄웠을 때 충돌하고, 더 나쁘게는 남의
        # 프로세스에 붙을 수 있다. authkey 로 신원을 한 번 더 막는다.
        authkey = secrets.token_bytes(32)
        listener = connection.Listener(("127.0.0.1", 0), authkey=authkey)
        port = int(listener.address[1])

        env = dict(os.environ)
        env[_AUTHKEY_ENV] = authkey.hex()
        # 자식 stdout 을 파이프로 받는데, 기본 콘솔 코덱(cp949)이면 한글 로그에서
        # UnicodeEncodeError 로 자식이 죽는다.
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        # HEDDY_* 설정은 env 를 그대로 물려주므로 자식에서도 같은 값이 뜬다.

        cmd = [sys.executable, "-u", "-m", "gan_process", "--worker",
               "--host", "127.0.0.1", "--port", str(port)]
        kwargs = {}
        flag = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if flag:
            kwargs["creationflags"] = flag   # 콘솔 창이 뜨지 않게(출력은 파이프로 받는다)
        try:
            proc = subprocess.Popen(
                cmd, cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                **kwargs)
        except Exception:
            listener.close()
            raise

        child = _Child(proc, listener)
        # HairFastGAN 은 진행 상황을 print/tqdm 으로 뱉는다. 파이프를 안 읽으면
        # 버퍼(수십 KB)가 차는 순간 자식이 write 에서 **영원히 멈춘다**.
        # 그래서 반드시 전용 스레드로 계속 비워 주고, 겸사겸사 부모 로그로 옮긴다.
        for name, stream in (("out", proc.stdout), ("err", proc.stderr)):
            t = threading.Thread(target=self._relay, args=(stream, name, proc.pid),
                                 name="gan-%s-%d" % (name, proc.pid), daemon=True)
            t.start()
            child.threads.append(t)

        t = threading.Thread(target=self._accept, args=(child,),
                             name="gan-accept-%d" % proc.pid, daemon=True)
        t.start()
        child.threads.append(t)

        self._log("GAN 워커 프로세스 기동 (pid=%d, port=%d)" % (proc.pid, port))
        return child

    def _relay(self, stream, tag, pid):
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                line = line.rstrip()
                if line:
                    self._log("[gan:%d:%s] %s" % (pid, tag, line))
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _accept(self, child: _Child):
        """자식의 접속을 받는다. 자식이 죽었으면 무한정 기다리지 않고 포기한다."""
        listener = child.listener
        sock = getattr(getattr(listener, "_listener", None), "_socket", None)
        deadline = time.monotonic() + _STARTUP_TIMEOUT_S
        try:
            while True:
                if child.stopping.is_set():
                    child.error = "종료 중이라 자식 접속을 기다리지 않습니다"
                    return
                if sock is not None:
                    # accept 자체에 타임아웃이 없으면 자식이 뜨자마자 죽는 경우
                    # 이 스레드가 영원히 남는다. 0.5초씩 끊어서 자식 생사를 본다.
                    sock.settimeout(0.5)
                try:
                    conn = listener.accept()
                    # 자식은 접속 직후 hello 를 보낸다. 이걸 받아야 "소켓이
                    # 붙었다"가 아니라 "워커 루프가 실제로 돌고 있다"가 확인된다.
                    if not conn.poll(30.0):
                        conn.close()
                        child.error = "자식이 hello 를 보내지 않았습니다"
                        return
                    hello = conn.recv()
                    if not isinstance(hello, dict) or hello.get("op") != "hello":
                        conn.close()
                        child.error = "예상치 못한 첫 메시지: %r" % (hello,)
                        return
                    child.worker_pid = hello.get("pid")
                    child.conn = conn
                    child.error = None
                    self._ever_connected = True
                    return
                except (TimeoutError, OSError) as e:
                    if child.stopping.is_set():
                        # teardown 이 listener 를 닫아서 난 예외다. 정상 종료이므로
                        # 에러로 시끄럽게 남기지 않는다.
                        child.error = "종료 중 accept 중단"
                        return
                    if sock is None:
                        child.error = "accept 실패: %r" % (e,)
                        return
                    if not child.alive():
                        child.error = ("자식 프로세스가 접속 전에 종료했습니다 "
                                       "(exit=%s)" % (child.proc.returncode,))
                        return
                    if time.monotonic() > deadline:
                        child.error = "자식 프로세스 접속 시간 초과(%.0fs)" % _STARTUP_TIMEOUT_S
                        return
                except Exception as e:      # 인증 실패 등
                    child.error = "%s: %s" % (type(e).__name__, e)
                    return
        finally:
            try:
                listener.close()      # 커넥션 하나만 쓴다. 포트를 물고 있을 이유가 없다.
            except Exception:
                pass
            child.ready.set()

    def _teardown_child(self, graceful: bool) -> None:
        """self._lock 을 쥔 채로 부를 것."""
        child, self._child = self._child, None
        if child is None:
            return
        # listener 를 닫기 **전에** 알려야 accept 스레드가 그 예외를 종료 신호로
        # 읽는다. 순서가 바뀌면 예전 경합이 그대로 돌아온다.
        child.stopping.set()
        self._loaded = False
        conn = child.conn
        if graceful and conn is not None and child.alive():
            try:
                conn.send({"op": "shutdown"})
                if conn.poll(_SHUTDOWN_GRACE_S):
                    conn.recv()
            except Exception:
                pass
        for closer in (conn, child.listener):
            try:
                if closer is not None:
                    closer.close()
            except Exception:
                pass
        proc = child.proc
        if graceful and proc.poll() is None:
            # shutdown 을 받아들인 자식은 스스로 끝난다. 여기서 조금 기다려 주지
            # 않으면 아직 정리 중인 프로세스를 terminate 해서 종료코드가 1 로
            # 남고(실측), 자식의 finally(FacePose 해제)도 건너뛴다.
            try:
                proc.wait(_SHUTDOWN_GRACE_S)
            except Exception:
                pass
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(_SHUTDOWN_GRACE_S)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(5)
                except Exception:
                    pass
        # 릴레이 스레드는 파이프 EOF 로 알아서 끝난다. 잠깐만 기다려 준다.
        for t in child.threads:
            if t is not threading.current_thread():
                t.join(2.0)

    # ---------- 폴백 ----------
    def _fallback(self, reason: str) -> None:
        """self._lock 을 쥔 채로 부를 것."""
        if self._backend == "thread":
            return
        logger.warning("=" * 72)
        logger.warning("GAN 프로세스 백엔드를 쓸 수 없어 in-process 스레드로 폴백합니다.")
        logger.warning("이유: %s", reason)
        logger.warning("이 모드에서는 GAN 적재(~90초)/추론(~9초) 동안 실시간 경로가 "
                       "같은 GPU/GIL 을 나눠 쓰므로 프레임이 밀립니다.")
        logger.warning("=" * 72)
        self._last_error = reason
        self._backend = "thread"
        self._teardown_child(graceful=False)
        self._ensure_thread_worker()

    def _ensure_thread_worker(self):
        if self._thread_worker is None:
            self._thread_worker = gan_worker.GanWorker()
        return self._thread_worker

    def _thread_set_rotate(self, ckpt_path, log):
        with self._lock:
            if self._closed:
                raise RuntimeError("이미 close() 된 GanClient 입니다")
            worker = self._ensure_thread_worker()
        path = worker.set_rotate(ckpt_path, log=log)
        self._rotate_checkpoint = path
        return path

    def _thread_swap(self, face_bgr, shape_path, color_path, log):
        with self._lock:
            if self._closed:
                raise RuntimeError("이미 close() 된 GanClient 입니다")
            worker = self._ensure_thread_worker()
        return worker.swap(face_bgr, shape_path, color_path, log=log)


# =====================================================================
# 자식 프로세스 쪽
# =====================================================================

def _worker_main(host: str, port: int) -> int:
    """자식 엔트리포인트. 요청을 **하나씩 순차로** 처리한다.

    큐잉은 부모가 워커 1개짜리 executor 로 이미 해 주므로 여기서 더 할 게 없고,
    HairFast 모델이 스레드 안전하지 않으므로 해서도 안 된다.
    """
    def log(msg):
        # 부모의 릴레이 스레드가 이 줄을 받아 서버 로그로 옮긴다.
        print(msg, flush=True)

    key = os.environ.get(_AUTHKEY_ENV)
    if not key:
        print("%s 환경변수가 없습니다" % _AUTHKEY_ENV, file=sys.stderr, flush=True)
        return 2
    try:
        conn = connection.Client((host, port), authkey=bytes.fromhex(key))
    except Exception as e:
        print("부모에 접속하지 못했습니다: %r" % (e,), file=sys.stderr, flush=True)
        return 3

    worker = gan_worker.GanWorker()
    # 접속 직후 hello 를 보낸다. 부모가 이걸 받아야 "TCP 가 붙었다"가 아니라
    # "워커 루프가 실제로 돌기 시작했다"를 확인할 수 있다. 겸사겸사 진짜 pid 도
    # 알려 준다(Windows venv 런처 때문에 부모의 Popen.pid 와 다르다).
    conn.send({"op": "hello", "pid": os.getpid()})
    log("GAN 워커 프로세스 준비 완료 (pid=%d). 모델은 첫 swap 요청 때 올린다." % os.getpid())
    try:
        while True:
            try:
                req = conn.recv()
            except (EOFError, OSError):
                # 부모가 죽거나 커넥션을 닫으면 여기로 온다. 고아로 남지 않도록
                # 그대로 종료한다(부모를 강제 종료해도 소켓은 닫힌다).
                log("부모와의 연결이 끊겼습니다. 종료합니다.")
                break

            op = req.get("op") if isinstance(req, dict) else None
            if op == "ping":
                # 모델을 절대 건드리지 않는다. 살아있음 확인용이다.
                conn.send({"ok": True, "loaded": worker.loaded, "pid": os.getpid(),
                           "load_seconds": worker.load_seconds})
            elif op == "shutdown":
                try:
                    conn.send({"ok": True})
                except Exception:
                    pass
                log("종료 요청을 받았습니다.")
                break
            elif op == "set_rotate":
                try:
                    path = worker.set_rotate(req["checkpoint"], log=log)
                    conn.send({"ok": True, "checkpoint": path, "loaded": True,
                               "load_seconds": worker.load_seconds})
                except Exception as e:
                    conn.send({"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                               "traceback": traceback.format_exc(),
                               "loaded": worker.loaded})
            elif op == "swap":
                try:
                    out, seconds = worker.swap(req["face"], req["shape"],
                                               req.get("color"), log=log)
                    conn.send({"ok": True, "result": out, "seconds": float(seconds),
                               "loaded": True, "load_seconds": worker.load_seconds,
                               "rotate_checkpoint": worker.rotate_checkpoint})
                except Exception as e:
                    # 예외를 그대로 pickle 하면 사용자 정의 예외 클래스를 부모가
                    # import 못 해 recv 에서 다시 터진다. 문자열로 평탄화한다.
                    conn.send({"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                               "traceback": traceback.format_exc(),
                               "loaded": worker.loaded})
            else:
                conn.send({"ok": False, "error": "알 수 없는 op: %r" % (op,),
                           "traceback": ""})
    finally:
        try:
            conn.close()
        except Exception:
            pass
        worker.close()
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="HairFastGAN 워커 프로세스")
    ap.add_argument("--worker", action="store_true", help="자식 워커로 실행")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()

    if args.worker:
        sys.exit(_worker_main(args.host, args.port))
    ap.error("--worker 없이 직접 실행할 용도가 아닙니다")
