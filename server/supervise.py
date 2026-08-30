"""server.py 를 감시하며 죽으면 다시 띄운다.

왜 필요한가
-----------
지금까지는 서버를 띄운 셸(또는 그 셸을 띄운 세션)이 끝나면 서버도 같이 내려갔다.
그리고 GAN 워커가 CUDA 오류로 죽거나 서버 프로세스 자체가 죽으면 아무도 되살리지
않았다. 실제로 세션 중에 여러 번 내려갔다.

이 파일은 그 둘을 다 막는다:
  - 자식이 어떤 이유로 끝나든 다시 띄운다.
  - 부모(이 프로세스)를 detached 로 띄우면 셸/세션과 수명이 끊긴다.

크래시 루프 방지
----------------
자식이 뜨자마자 죽는 상황(포트 점유, 드라이버 문제, 코드 오류)에서 초당 수십 번
프로세스를 만들면 로그만 채우고 원인도 안 보인다. 그래서 백오프를 둔다. 다만
**오래 살아 있다가 죽은 경우는 백오프를 초기화한다** - 그건 크래시 루프가 아니라
일회성 사고이므로 즉시 되살리는 게 맞다.

사용:
    python supervise.py [-- server.py 에 넘길 인자들]
    python supervise.py --status
    python supervise.py --stop
"""
import argparse
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "server.log")
PIDFILE = os.path.join(ROOT, "supervise.pid")

#: 재기동 간격. 연달아 죽을수록 늘린다.
BACKOFF = (2, 5, 10, 20, 30, 60)
#: 자식이 이 시간보다 오래 살아 있었으면 백오프를 초기화한다.
HEALTHY_S = 120.0


def log(msg):
    line = f"[supervise {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def alive(pid):
    if not pid:
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True, timeout=10).stdout
        return str(pid) in out
    except Exception:
        return False


def read_pidfile():
    try:
        with open(PIDFILE, encoding="utf-8") as f:
            parts = f.read().split()
        return int(parts[0]), (int(parts[1]) if len(parts) > 1 else None)
    except Exception:
        return None, None


def cmd_status():
    sup, child = read_pidfile()
    print(f"감시 프로세스 pid={sup} 살아있음={alive(sup)}")
    print(f"서버   프로세스 pid={child} 살아있음={alive(child)}")
    print(f"로그: {LOG}")
    return 0 if alive(sup) else 1


def cmd_stop():
    sup, child = read_pidfile()
    # 감시자를 먼저 죽여야 한다. 서버부터 죽이면 감시자가 곧바로 되살린다.
    for name, pid in (("감시", sup), ("서버", child)):
        if alive(pid):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True)
            print(f"{name} 프로세스 {pid} 종료")
    try:
        os.remove(PIDFILE)
    except OSError:
        pass
    return 0


def main(extra):
    with open(PIDFILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    cmd = [sys.executable, "-u", os.path.join(ROOT, "server.py")] + extra
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"

    fails = 0
    stopping = False

    def _bye(*_):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _bye)
        except (ValueError, OSError):
            pass

    log(f"감시 시작 pid={os.getpid()}  대상: {' '.join(cmd[1:])}")
    while not stopping:
        started = time.time()
        with open(LOG, "a", encoding="utf-8") as out:
            out.write(f"\n===== 서버 기동 {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            out.flush()
            try:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                        stdout=out, stderr=subprocess.STDOUT)
            except Exception as e:
                log(f"기동 실패: {e}")
                time.sleep(BACKOFF[min(fails, len(BACKOFF) - 1)])
                fails += 1
                continue

            with open(PIDFILE, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()} {proc.pid}")
            log(f"서버 기동 pid={proc.pid}")

            try:
                rc = proc.wait()
            except KeyboardInterrupt:
                stopping = True
                proc.terminate()
                break

        lived = time.time() - started
        if stopping:
            break
        # 오래 버텼으면 일회성 사고다. 즉시 되살린다.
        fails = 0 if lived >= HEALTHY_S else fails + 1
        wait = BACKOFF[min(fails, len(BACKOFF) - 1)] if fails else 1
        log(f"서버 종료 rc={rc}, {lived:.0f}초 생존 -> {wait}초 뒤 재기동 (연속실패 {fails})")
        for _ in range(int(wait * 4)):
            if stopping:
                break
            time.sleep(0.25)

    log("감시 종료")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="server.py 감시/자동 재기동")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true")
    args, extra = ap.parse_known_args()
    if args.status:
        sys.exit(cmd_status())
    if args.stop:
        sys.exit(cmd_stop())
    sys.exit(main([a for a in extra if a != "--"]))
