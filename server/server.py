"""AI 서버: 브라우저 웹캠 -> WebRTC -> GPU 얼굴파싱 -> **합성된 영상**을 WebRTC로 반환.

설계 의도
---------
클라이언트에 레이어를 하나만 준다. 브라우저가 "실시간 비디오 + 늦게 오는
오버레이"를 각자 그리면 둘이 서로 다른 시점을 보여줘서 헤어가 머리에서
떨어져 따로 논다. 서버가 아예 합쳐서 완성된 프레임을 내려보내면 클라이언트는
그냥 비디오 하나를 재생하므로 어긋남이 구조적으로 불가능하다.

트레이드오프(정직하게): 왕복 인코딩/디코딩 때문에 절대 지연은 로컬 처리보다
크다. 어긋남 0을 절대 지연과 맞바꾼 구조다. 로컬 비교군은 /warp.html 에 있다.

세그멘테이션은 GPU(SegFormer face parsing + CUDA 그래프, ~9ms)를 쓴다.
이전 MediaPipe CPU 경로는 98ms로 이 구조에선 쓸 수 없었다.

시그널링: HTTP POST /offer (SDP offer/answer JSON, non-trickle ICE)

상태를 왜 모듈 전역에 두지 않는가
---------------------------------
예전에는 import 시점에 에셋을 읽고 executor 셋을 만들고 GAN 워커까지 만들었다.
두 가지가 터진다.
  - GAN 을 별도 프로세스로 돌리는 구조(gan_process)에서 자식이 이 모듈을 다시
    import 하는 경로가 생기면, 무거운 초기화가 자식에서도 그대로 돌아간다.
    특히 GanClient 를 모듈 레벨에서 만들고 start() 까지 부르면 서버를 두 번
    띄웠을 때 자식도 둘이 되어 VRAM 이 2배가 된다.
  - 테스트나 도구가 `import server` 만 해도 모델 디렉터리를 읽고 스레드를 띄운다.
그래서 가변 상태는 전부 AppState 하나에 모으고 앱 팩토리(create_app)에서
만든다. aiohttp 앱에는 app["state"] 로 붙는다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import time
import urllib.parse
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender
from av import VideoFrame

from config import CONFIG
import hair_asset
# gan_process 는 GanClient(별도 프로세스) + gan_worker 의 CAPTURE_DIR/REF_DIR/
# list_references 를 재노출한다. import 자체는 torch 를 건드리지 않아 가볍다.
import gan_process
import metrics as metrics_mod

ROOT = os.path.dirname(os.path.abspath(__file__))
CLIENT_DIR = os.path.join(os.path.dirname(ROOT), "client")
TRAIN_DIR = os.path.join(ROOT, "train")
REC_ROOT = os.path.join(TRAIN_DIR, "frames")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")


# ---------------------------------------------------------------------------
# 앱 상태
# ---------------------------------------------------------------------------

class AppState:
    """서버 프로세스 하나의 가변 상태 전부.

    executor 를 셋으로 나눈 이유는 각각 다르다:
      gpu  - CUDA 그래프는 캡처한 스트림에 묶여 있어서 반드시 같은 단일
             스레드에서만 재생해야 한다. 그래서 워커 1개짜리 전용 executor.
      gan  - GAN(고화질 촬영)은 장당 ~9초라 실시간 루프와 같은 워커를 쓰면
             프레임이 통째로 밀린다. 모델도 스레드 안전하지 않아 워커 1개 고정.
      pose - 랜드마커 전용. gpu 와 나누어야 세그멘테이션과 겹쳐 돌릴 수 있다.
             MediaPipe VIDEO 모드가 상태를 들고 있으므로 워커는 반드시 1개.
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        self.started_at = time.time()
        self.pcs = set()
        self.sessions = set()          # 활성 PeerState
        self.relay = MediaRelay()

        self.gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        # 맨이마 인페인팅(LaMa, ~45ms) 전용. gpu_executor 에서 돌리면 그 시간만큼 영상 경로가
        # 멈춘다(주기 갱신이면 매초 한 번 끊김). 별도 스레드 + 별도 CUDA 스트림으로 겹친다.
        self.forehead_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forehead")
        self.gan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gan")
        self.pose_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pose")

        self.segmenter = None          # 첫 연결 때 지연 초기화 (모델 로딩이 몇 초)
        self.forehead = None           # forehead.ForeheadInpainter (LaMa). 3D 그룸 첫 선택 때 적재.
        self.seg_error = None          # 적재 실패 사유. /readyz 가 이걸 돌려준다.
        self._seg_lock = asyncio.Lock()

        self.static_assets = {}        # 전 세션 공유. **읽기 전용.**
        self.references = {}
        self.gan = None

        self.metrics = metrics_mod.Metrics()
        #: GPU executor 에 지금 몇 장이 걸려 있는가. 세션별 카운터만으로는
        #: 부족하다 - recv() 는 트랙마다 직렬이라 자기 세션 카운터는 거의 항상
        #: 0 이고, 실제로 대기열을 채우는 건 **다른 세션**이기 때문이다.
        self.gpu_inflight = 0
        self.reaper = None
        self._closed = False

    # ---- 세그멘터 ----
    async def get_segmenter(self):
        async with self._seg_lock:
            if self.segmenter is None:
                logger.info("GPU 세그멘터 로딩 중... (첫 연결 시 수 초 소요)")
                loop = asyncio.get_event_loop()

                def _load():
                    from gpu_segmenter import GpuFaceParser
                    return GpuFaceParser()

                try:
                    seg = await loop.run_in_executor(self.gpu_executor, _load)
                except Exception as e:
                    # 사유를 남겨야 /readyz 가 "왜 안 되는지"를 말할 수 있다.
                    # 여기서 삼키면 매 프레임 같은 예외가 조용히 반복된다.
                    self.seg_error = "%s: %s" % (type(e).__name__, e)
                    raise
                self.segmenter = seg
                self.seg_error = None
                logger.info("GPU 세그멘터 준비 완료 (device=%s, cuda_graph=%s)",
                            seg.device, seg.graph is not None)
                bpath = os.path.join(TRAIN_DIR, "blender.pt")
                ok = await loop.run_in_executor(self.gpu_executor, seg.load_blender, bpath)
                logger.info("증류 블렌더: %s", "로드됨" if ok else "없음 (워핑만 사용)")
        return self.segmenter

    def evict_from_gpu(self, name: str) -> None:
        """세션 레지스트리가 에셋을 버릴 때 부르는 훅.

        파이썬 쪽 참조만 지우면 캐싱 얼로케이터가 VRAM 을 반납하지 않는다.
        세그멘터 캐시에서도 같이 빼야 실제로 돌아온다. 세그멘터가 아직 없는
        시점(첫 연결 전)에도 세션이 만들어질 수 있으므로 None 을 견뎌야 한다.
        """
        seg = self.segmenter
        if seg is not None:
            seg.evict_asset(name)

    def ready(self):
        """(준비됨?, 사유). /readyz 가 쓴다."""
        if self.segmenter is None:
            return False, self.seg_error or "segmenter not loaded"
        seg = self.segmenter
        # CPU 폴백에는 CUDA 그래프가 애초에 없다. 그걸 not-ready 로 보면
        # GPU 없는 머신에서 영원히 503 이 된다.
        if self.cfg.use_cuda_graph and seg.device == "cuda" and seg.graph is None:
            return False, "cuda graph not captured"
        return True, "ok"

    async def close(self):
        if self._closed:
            return
        self._closed = True
        for st in list(self.sessions):
            st.cleanup()
        self.sessions.clear()
        await asyncio.gather(*(pc.close() for pc in list(self.pcs)),
                             return_exceptions=True)
        self.pcs.clear()
        if self.gan is not None:
            # 자식이 남으면 VRAM 5.5GB 가 안 돌아온다. 실측으로 close() 가
            # 7.9GB 를 회수하는 것을 확인했다(9366 -> 1453 MiB).
            self.gan.close()
        if self.segmenter is not None:
            self.segmenter.close()
        self.gpu_executor.shutdown(wait=False)
        self.forehead_executor.shutdown(wait=False)
        self.gan_executor.shutdown(wait=False)
        self.pose_executor.shutdown(wait=False)


class PeerState:
    """연결(세션) 하나가 들고 있는 것 전부.

    세션 스코프 자원(GPU 플레이트 텐서, MediaPipe 네이티브 그래프, 생성 에셋)이
    여기 모여 있어야 cleanup() 한 번으로 확실히 놓을 수 있다. 예전에는 트랙
    객체 안에 흩어져 있어서 연결이 끊겨도 아무것도 해제되지 않았다.
    """

    def __init__(self, app: AppState):
        self.app = app
        self.cfg = app.cfg
        self.sid = uuid.uuid4().hex[:12]
        self.pc = None
        self.channel = None
        self.track = None          # 최초 1회만 설정. cleanup 에서 끊는다.
        self.capturing = False
        # 기본은 원본 패스스루. 프로덕션 플로우는 "헤어 고르기 -> 각도 수집
        # -> 생성 -> tryon" 이라, 에셋이 생기기 전까지 보여줄 것은 원본뿐이다.
        # (seg/plate/remove 는 디버그용으로 남아 있고 클라이언트가 요청하면 된다)
        self.mode = "raw"
        self.asset_name = None      # None이면 첫 번째 에셋
        self.bank = None            # 다각도 뱅크 이름 (설정 시 yaw로 자동 선택)
        self.scale_mul = 1.0
        self.offset_up = 0.0
        self.harmonize = True     # 조명/화이트밸런스 정합
        self.shadow = 0.35        # 헤어라인 그림자 세기
        # 증류 블렌더는 기본 꺼짐. 검증셋 지표(기준선 대비 30%)는 좋았지만
        # 실제 영상에서 체감 이득이 없다고 확인됨. 비교용으로만 남긴다.
        self.blend = 0.0
        self.smooth = 1.0         # 앵커 평활화 세기 (0=끔)
        # 3D 그룸(헤어카드 GLB). 설정되면 tryon 이 2D 에셋/뱅크 대신 이걸 포즈로 렌더한다.
        # groom_obj 는 {"name","path","meta"} - warm 이 끝난 뒤에만 채워진다(첫 프레임 멈춤 방지).
        self.groom = None
        self.groom_obj = None
        self.groom_fwd_cm = 0.0   # 앞뒤 (cm). 상하는 offset_up(px) 을 cm 로 환산해 같이 쓴다.
        self.groom_color = None   # (b,g,r) 염색 미리보기. None 이면 사용자 머리색에 맞춤.
        # 2차 운동 상태(세션별 스프링). dyn_amount 는 UI 슬라이더(0=강체, 1=기본, 1.5=과장).
        import hair_dynamics
        self.hair_dyn = hair_dynamics.HairDynamics(
            stiffness=self.cfg.groom_dyn_stiffness, damping_ratio=self.cfg.groom_dyn_damping,
            gravity=self.cfg.groom_dyn_gravity, max_shift_cm=self.cfg.groom_dyn_max_shift_cm,
            idle_cm=self.cfg.groom_dyn_idle_cm)
        self.dyn_amount = 1.0
        self.dyn_last_t = None
        self.forehead_name = None # 인페인팅으로 만든 맨이마 패치(세션 레지스트리 이름). 그룸 tryon 이 이마에 씀.
        self.forehead_building = False
        self.forehead_built_at = 0.0   # 주기 갱신 기준 시각 (monotonic)
        self.forehead_prev = None      # 교체 직후 크로스페이드용 이전 패치 이름
        self.forehead_fade_at = 0.0
        self.livebank = None      # LiveBank: 세션 중 GAN 으로 각도별 헤어를 생성
        self.recording = False    # 학습 데이터용 원본 프레임 수집
        self.rec_dir = None
        self.rec_count = 0

        # 세션 스코프 에셋. 정적 에셋은 공유(읽기 전용)하고 생성분만 여기 가둔다.
        # 축출 훅이 세그멘터 GPU 캐시까지 비워야 VRAM 이 실제로 돌아온다.
        self.registry = hair_asset.AssetRegistry(
            app.static_assets, on_evict=app.evict_from_gpu)

        # 세션 스코프 자원
        self.plate = None         # SessionPlate (GPU 텐서)
        self.pose = None          # FacePose (MediaPipe 네이티브 그래프)
        self.smoother = None
        self.pose_future = None
        self.last_raw = None      # 촬영용: 오버레이 없는 원본 프레임

        self.created_at = time.monotonic()
        self.last_frame_at = self.created_at
        self.inflight = 0
        self.frames = 0
        self.dropped = 0
        self.errors = 0
        self._closed = False

    # ---- 정리 ----
    def cleanup(self) -> None:
        """세션 자원을 놓는다. 여러 번 불려도 안전해야 한다.

        호출 경로가 넷이다(connectionstatechange failed/closed, 트랙 ended,
        idle 리퍼, 앱 shutdown). 넷 다 서로를 모르므로 idempotent 가 아니면
        같은 텐서를 두 번 놓거나 MediaPipe 를 두 번 닫는다.
        """
        if self._closed:
            return
        self._closed = True

        # 진행 중인 랜드마커 future 부터 정리한다.
        fut, self.pose_future = self.pose_future, None
        pose, self.pose = self.pose, None
        if pose is not None:
            if fut is not None and not fut.done() and not fut.cancel():
                # 아직 워커 스레드에서 돌고 있다. 지금 close() 하면 MediaPipe
                # 네이티브 그래프를 **쓰는 도중에** 해제하는 것이라 프로세스가
                # 통째로 죽는다(파이썬 예외가 아니라 네이티브 크래시라 로그도
                # 안 남는다). 끝난 뒤에 닫도록 미룬다.
                fut.add_done_callback(lambda _f, p=pose: _close_quietly(p))
            else:
                _close_quietly(pose)

        # 플레이트는 GPU 텐서를 들고 있다. 참조를 끊어야 캐싱 얼로케이터가
        # 블록을 재사용할 수 있다.
        self.plate = None
        self.smoother = None
        self.last_raw = None
        # 트랙 <-> 상태 순환참조를 끊는다. 남겨두면 GC 가 늦어져서 다음 세션이
        # 시작될 때까지 이전 세션의 프레임 버퍼가 살아 있다.
        self.track = None
        self.livebank = None
        self.recording = False
        # 이 세션이 만든 에셋을 전부 축출한다(on_evict 가 GPU 캐시도 비운다).
        try:
            self.registry.close()
        except Exception:
            logger.exception("세션 에셋 정리 실패 (%s)", self.sid)


def _close_quietly(obj) -> None:
    try:
        obj.close()
    except Exception:
        logger.exception("close() 실패")


# ---------------------------------------------------------------------------
# 프레임 루프
# ---------------------------------------------------------------------------

class SegmentedVideoTrack(VideoStreamTrack):
    """소스 트랙을 받아 GPU로 합성한 프레임을 내보내는 트랙."""

    def __init__(self, source_track, state: PeerState):
        super().__init__()
        self._source = source_track
        self._state = state
        self._frame_idx = 0
        self._recv_times = deque(maxlen=max(2, state.cfg.log_every))
        self._last_recv_at = None
        self._last_pose = None
        self._yaw_ema = None       # yaw 흔들림이 각도 전환을 튀게 하므로 완만하게
        self._cur_asset = None     # 히스테리시스: 지금 쓰는 뱅크 칸
        self._last_out = None      # 드롭/에러 시 내보낼 직전 합성 결과
        self._frame_size = None    # 수신 해상도 (h, w). 바뀌면 로그

    def _pose_rgb(self, img):
        """랜드마커 워커 스레드에서 실행. 색변환도 여기서 해야 메인 루프가 안 막힌다."""
        return self._state.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                        int(time.monotonic() * 1000))

    @staticmethod
    def _sharpness(img, pose):
        """얼굴 주변 크롭의 라플라시안 분산. 같은 칸의 후보끼리만 비교한다."""
        el, er = pose["eye_l"], pose["eye_r"]
        d = max(20.0, float(np.hypot(er[0] - el[0], er[1] - el[1])))
        cx, cy = (el[0] + er[0]) / 2.0, (el[1] + er[1]) / 2.0
        h, w = img.shape[:2]
        x0, x1 = int(max(0, cx - 1.5 * d)), int(min(w, cx + 1.5 * d))
        y0, y1 = int(max(0, cy - 1.5 * d)), int(min(h, cy + 1.5 * d))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return 0.0
        g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_32F).var())

    def _finalize_bucket(self, lb, target):
        """후보 중 최선을 그 칸의 프레임으로 확정한다."""
        st = self._state
        best, frame, yaw, n = lb.cands.pop(target)
        lb.status[target] = "captured"
        lb.frames[target] = frame
        lb.measured[target] = yaw
        logger.info("라이브 뱅크 %s: yaw %+.0f 확정 (후보 %d장, 선명도 %.0f)",
                    lb.name, target, n, best)
        notify_peer(st, {"type": "livebank", **lb.report(),
                         "status": "captured", "captured_yaw": target})

        # 다 모았으면 생성 단계로. 수집과 생성을 겹치지 않는 이유는 GPU 다.
        # 같은 4070 에서 세그멘테이션과 GAN 이 경합하면 양쪽 다 느려진다.
        if all(v == "captured" for v in lb.status.values()):
            lb.phase = "generate"
            asyncio.create_task(run_bank_generation(st, lb))

    def _collect_for_bank(self, pose):
        """라이브 뱅크가 켜져 있으면, 목표 각도에 들어온 프레임 중 가장 선명한
        것을 잡아 큐에 넣는다.

        움직이는 도중에 잡으면 모션 블러가 그대로 GAN 입력이 되어 결과가
        뭉개진다. 측정값이 EMA 를 얼마나 앞질렀는지로 '지금 움직이는 중인가'
        를 판정하고(정지해 있으면 둘이 붙고, 돌리는 중이면 벌어진다), 그
        판정을 통과한 프레임 중에서도 후보 N 장의 선명도를 비교해 고른다
        (CONFIG.live_pick_frames). yaw 만 보는 정지 판정은 상하 흔들림이나
        초점 흔들림을 못 거르기 때문이다.
        """
        st = self._state
        lb = st.livebank
        if lb is None or lb.phase != "collect":
            return
        moving = (pose is None or self._yaw_ema is None
                  or abs(float(pose["yaw"]) - self._yaw_ema) > st.cfg.live_steady)
        target = None if moving or st.last_raw is None else lb.match(self._yaw_ema)

        # 후보가 쌓여 있는데 그 칸을 벗어났으면(다른 칸이거나 범위 밖) 지금까지
        # 의 최선으로 확정한다 - 사용자가 다시 돌아오길 기다리지 않는다.
        for t in list(lb.cands):
            if t != target and lb.status[t] == "pending":
                self._finalize_bucket(lb, t)
        if target is None:
            return

        score = self._sharpness(st.last_raw, pose)
        c = lb.cands.get(target)
        if c is None or score > c[0]:
            c = [score, st.last_raw.copy(), float(self._yaw_ema), 0]
        c[3] = (lb.cands[target][3] if target in lb.cands else 0) + 1
        lb.cands[target] = c
        if c[3] >= max(1, int(st.cfg.live_pick_frames)):
            self._finalize_bucket(lb, target)

    def _record_frame(self, img):
        """학습 데이터 수집: 원본 프레임을 그대로 떨군다.

        포즈가 다양해야 쓸모가 있으므로 몇 프레임 걸러 저장해 비슷한 연속
        프레임이 쌓이는 걸 줄인다. 쿼터가 없으면 DataChannel 메시지 하나로
        서버 디스크를 끝까지 채울 수 있다.
        """
        st = self._state
        cfg = st.cfg
        if not st.recording or self._frame_idx % cfg.rec_every:
            return
        if st.rec_count >= cfg.record_max_frames:
            st.recording = False
            notify_peer(st, {"type": "record", "on": False, "count": st.rec_count,
                             "dir": st.rec_dir,
                             "message": "프레임 상한(%d장)에 도달해 자동 중지했습니다."
                                        % cfg.record_max_frames})
            logger.warning("프레임 수집 자동 중지: 상한 %d장", cfg.record_max_frames)
            return
        # 디렉터리 용량은 매 프레임 재는 게 아니라(os.walk 가 수백 파일에서
        # 수십 ms 든다) 100장에 한 번만 잰다. 그 사이 최대 100장(수십 MB)만
        # 초과할 수 있어 상한의 의미는 유지된다.
        if st.rec_count and st.rec_count % 100 == 0:
            if _dir_size_mb(REC_ROOT) > cfg.record_dir_max_mb:
                st.recording = False
                notify_peer(st, {"type": "record", "on": False, "count": st.rec_count,
                                 "dir": st.rec_dir,
                                 "message": "디스크 쿼터(%dMB)를 넘어 자동 중지했습니다."
                                            % cfg.record_dir_max_mb})
                logger.warning("프레임 수집 자동 중지: 쿼터 %dMB 초과", cfg.record_dir_max_mb)
                return
        try:
            cv2.imwrite(os.path.join(st.rec_dir, f"{st.rec_count:05d}.png"), img)
            st.rec_count += 1
        except Exception:
            logger.exception("프레임 저장 실패")

    async def recv(self):
        st = self._state
        app = st.app
        cfg = st.cfg

        t_wait0 = time.perf_counter()
        frame = await self._source.recv()
        wait_ms = (time.perf_counter() - t_wait0) * 1000

        now = time.perf_counter()
        if self._last_recv_at is not None:
            self._recv_times.append(now - self._last_recv_at)
        self._last_recv_at = now

        img = frame.to_ndarray(format="bgr24")
        # 수신 해상도. 클라이언트가 720p 를 요청해도 브라우저는 업링크 대역폭에
        # 따라 조용히 낮춰 보낸다. 그러면 GAN 입력부터 흐려지는데 로그에
        # 안 남으면 원인을 찾을 수 없다. 처음과 바뀔 때만 남긴다.
        if img.shape[:2] != self._frame_size:
            self._frame_size = img.shape[:2]
            logger.info("수신 해상도 %dx%d (%s)", img.shape[1], img.shape[0], st.sid)
        # 촬영은 오버레이가 얹히기 전 원본을 써야 한다 (GAN 입력에 마젠타 색칠이
        # 들어가면 안 됨). 매 프레임 최신본만 보관.
        st.last_raw = img
        st.last_frame_at = time.monotonic()
        if st.track is None:
            # 최초 1회만. 매 프레임 재대입하면 순환참조가 계속 새로 맺어져
            # GC 가 늦어진다.
            st.track = self
        self._frame_idx += 1
        st.frames = self._frame_idx
        app.metrics.frames_total += 1

        # 생성 단계에서는 **아무것도 하지 않는다.**
        #
        # GAN 7칸이 같은 4070 을 1분 동안 쓴다. 그 옆에서 세그멘테이션과
        # 랜드마커를 돌리면 양쪽 다 느려지는데, 그동안 사용자는 진행률만 보면
        # 되고(클라이언트가 오버레이를 덮는다) 영상을 볼 이유가 없다.
        # 포즈 계산보다 **앞에** 두어 CPU 도 안 쓴다.
        #
        # 진행률은 이 경로와 무관하게 run_bank_generation 이 DataChannel 로
        # 계속 보낸다. 검은 프레임을 내보내는 이유는 트랙을 살려 두기 위해서다 -
        # 아무것도 안 보내면 연결 상태 판정이 애매해진다.
        #
        # **여기는 recv() 다. VideoFrame 을 돌려줘야 한다.** _compose() 처럼
        # ndarray 를 반환하면 인코더가 pts 를 찾다가 죽고(AttributeError),
        # 송신 태스크가 끝나면서 세션째 정리된다. 그러면 생성 중이던 라이브
        # 뱅크도 "세션 종료로 중단" 되어 첫 칸만 만들어지고 멈춘다.
        if (not cfg.stream_during_gan and st.livebank is not None
                and st.livebank.phase == "generate"):
            return self._wrap(np.zeros_like(img), frame)

        self._record_frame(img)

        out = None
        try:
            out = await self._compose(img, wait_ms)
        except Exception:
            # 프레임 한 장의 실패로 트랙을 죽이면 PeerConnection 이 통째로
            # 날아가고 사용자는 재협상을 해야 한다. 원본 패스스루로 낮춰서
            # 버틴다 - 화면이 잠깐 원본으로 보이는 편이 연결이 끊기는 것보다 낫다.
            st.errors += 1
            app.metrics.frame_errors_total += 1
            # 같은 예외가 30fps 로 쏟아지면 로그가 초당 30줄이 된다.
            # 첫 번째와 그 뒤 log_every 마다만 남긴다.
            if st.errors == 1 or st.errors % cfg.log_every == 0:
                logger.exception("프레임 합성 실패 (누적 %d회) - 원본으로 대체", st.errors)

        if out is None:
            # 직전 합성 결과가 있으면 그걸, 없으면 원본을. 해상도가 바뀌었으면
            # (카메라 재협상) 직전 것은 못 쓴다.
            prev = self._last_out
            out = prev if (prev is not None and prev.shape == img.shape) else img

        return self._wrap(out, frame)

    @staticmethod
    def _wrap(arr, src_frame):
        """내보낼 ndarray -> VideoFrame. recv() 의 **모든** 반환은 이걸 거친다.

        포장을 한 곳에 모아 둔 이유: recv() 안에서 ndarray 를 그대로 반환하면
        인코더가 pts 를 찾다가 AttributeError 로 죽고, 송신 태스크가 끝나면서
        세션째 정리된다. 실제로 그렇게 만들어서 라이브 뱅크가 첫 칸만 생성하고
        멈췄다. _compose() 는 ndarray 를 반환해도 되므로 헷갈리기 쉽다.
        """
        new_frame = VideoFrame.from_ndarray(arr, format="bgr24")
        # pts/time_base 는 반드시 원본 것을 유지한다. 새로 만들면 수신 측
        # 지터 버퍼가 타임라인을 다시 맞추느라 영상이 튄다.
        new_frame.pts = src_frame.pts
        new_frame.time_base = src_frame.time_base
        return new_frame

    async def _compose(self, img, wait_ms):
        """합성 본체. 내보낼 ndarray, 또는 이번 프레임을 버릴 거면 None."""
        st = self._state
        app = st.app
        cfg = st.cfg

        seg = await app.get_segmenter()
        if st.plate is None:
            from gpu_segmenter import SessionPlate
            from face_pose import FacePose
            st.plate = SessionPlate(seg.device)
            st.pose = FacePose()
            st.smoother = hair_asset.AnchorSmoother()

        loop = asyncio.get_event_loop()
        t_exec0 = time.perf_counter()
        asset = st.registry.get(st.asset_name) if st.asset_name else st.registry.default()

        # 랜드마커는 얼굴이 있으면 11.5ms 든다(실측). 입력 해상도를 320x240 까지
        # 낮춰도 9.7ms 라 거의 안 줄어드는데, VIDEO 모드가 이전 얼굴 주변을
        # 크롭해 고정 크기 망을 돌리기 때문이다. 즉 다운스케일로는 못 줄인다.
        #
        # 세그멘테이션(19ms)과 같은 단일 워커 스레드에서 직렬로 돌리면 그대로
        # 더해져 30ms 가 되고 30fps 를 못 지킨다(실측 24fps). 그래서 별도
        # 스레드에서 겹쳐 돌리고, 이번 프레임 앵커로는 '직전 프레임 결과'를 쓴다.
        # 앵커가 한 프레임(33ms) 늦지만 평활화가 이미 그보다 큰 지연을 만들고
        # 있어 체감 차이는 없다. MediaPipe VIDEO 모드는 내부 상태를 들고 있어
        # 워커가 반드시 1개여야 한다.
        # 모드와 무관하게 항상 잰다.
        #
        # 예전에는 tryon/remove/라이브뱅크일 때만 쟀다. 렌더에 앵커가 필요한
        # 경우가 그것뿐이라서였는데, 그 결과 **기본 모드(raw)에서 stats.yaw 가
        # 계속 null 로 나갔다.** 클라이언트가 각도 가이드를 그려야 하는 시점이
        # 정확히 그때(라이브 뱅크 시작 전/직후)라, 가이드를 띄울 방법이 없었다.
        # "원본 그대로 + 각도 가이드"가 수집 단계의 설계인데 구현이 어긋나 있었다.
        #
        # 비용은 프레임당 랜드마커 1회인데 전용 워커에서 돌고 아래 future
        # 백프레셔가 막고 있어 실시간 경로를 늦추지 않는다.
        if st.pose_future is not None and st.pose_future.done():
            try:
                self._last_pose = st.pose_future.result()
            except Exception:
                logger.exception("랜드마커 실패")
            st.pose_future = None
        # 아직 안 끝났으면 새로 던지지 않는다(자연스러운 백프레셔).
        # concurrent Future 로 던진다 - gpu_executor 스레드의 seg.process 가 앵커가 필요한
        # 시점에 .result(timeout) 으로 **이 프레임의** 포즈를 받아 쓰기 위해서다(asyncio Future
        # 는 다른 스레드에서 못 기다린다). 예전엔 항상 직전 프레임 포즈로 그려 한 프레임 뒤처졌다.
        pose_cf = None
        if st.pose_future is None:
            pose_cf = app.pose_executor.submit(self._pose_rgb, img)
            st.pose_future = asyncio.wrap_future(pose_cf, loop=loop)
        pose = self._last_pose

        # 다각도 뱅크: 측정된 yaw 에 가장 가까운 각도의 에셋으로 바꾼다.
        # 닮음변환으로는 만들 수 없는 평면 밖 회전을 '그 각도에서 생성된 헤어'로
        # 대체하는 것이라, 고개를 돌리면 헤어도 그 각도의 모습으로 전환된다.
        # 측정 yaw 는 프레임마다 흔들린다. 그대로 쓰면 혼합 비율이 떨려 전환이
        # 지글거린다. EMA 로 완만하게 만든 값을 뱅크 선택과 라이브 수집이 공유한다.
        if pose is not None:
            y = float(pose["yaw"])
            self._yaw_ema = y if self._yaw_ema is None else self._yaw_ema * 0.75 + y * 0.25
        self._collect_for_bank(pose)

        # 수집 단계에서는 GPU 를 최소로 쓴다.
        #
        # 화면에 원본을 그대로 내보내므로 렌더용 분할이 필요 없다. GPU 를 쓰는
        # 유일한 이유가 배경 플레이트 누적인데, 플레이트는 시간에 걸쳐 쌓는
        # 것이라 매 프레임일 필요가 없다. 여기서 아낀 GPU 는 곧바로 이어질
        # GAN 7칸에 쓰인다.
        #
        # 얼굴 각도는 위에서 이미 갱신됐고(CPU) 칸 포착도 끝났으므로, 각도
        # 가이드와 수집은 이 분기와 무관하게 매 프레임 동작한다.
        #
        # **raw 모드에서만** 건너뛴다. 여기서 내보내는 건 원본 프레임인데,
        # 출력이 원본과 같은 모드는 raw 뿐이다. tryon 상태에서 건너뛰면 6프레임
        # 중 5장이 헤어 없는 원본, 1장만 합성본이 되어 헤어가 초당 5번 깜빡인다
        # (두 번째 라이브 뱅크를 만들 때 실제로 이렇게 됐다 - 서버가 첫 칸에서
        #  tryon 으로 바꿔 두기 때문에 모드가 raw 로 돌아가 있지 않다).
        if (st.livebank is not None and st.livebank.phase == "collect"
                and st.mode == "raw"
                and cfg.collect_seg_every > 1
                and self._frame_idx % cfg.collect_seg_every != 0):
            return img

        # 프레임 드롭. 버퍼는 지연을 줄이지 못한다 - 밀린 프레임을 계속
        # 처리하면 지연만 누적된다. 최신 프레임만 살리고 밀린 건 버린다.
        if st.inflight + app.gpu_inflight >= cfg.max_inflight_frames:
            st.dropped += 1
            app.metrics.frames_dropped_total += 1
            return None

        st.smoother.set_strength(st.smooth)
        # 두 칸을 알파로 섞으면 헤어가 반투명하게 겹쳐 보인다. 그냥 가장 가까운
        # 칸으로 바로 바꾸되, 경계에서 깜빡이지 않도록 이력만 둔다.
        asset2, mix = None, 0.0
        if st.bank and pose is not None and self._yaw_ema is not None:
            a = hair_asset.pick_by_yaw_stable(
                st.registry, st.bank, self._yaw_ema, self._cur_asset)
            if a is not None:
                asset = self._cur_asset = a
        else:
            self._cur_asset = None

        st.inflight += 1
        app.gpu_inflight += 1
        try:
            # 그룸은 상하 슬라이더(px)를 cm 로 환산해 쓴다: 눈 간격 ~50px ≈ 6.3cm 이므로 10px ≈ 1.2cm.
            groom_fit = (st.scale_mul, st.offset_up * 0.12, st.groom_fwd_cm)
            groom_dyn = None
            dyn_dt = None
            if st.groom_obj is not None:
                # 그룸 경로에서 asset 은 이마 패치(.face) 로만 쓰인다 - 2D 헤어/뱅크는 무시.
                asset = st.registry.get(st.forehead_name) if st.forehead_name else None
                asset2, mix = None, 0.0
                now_m = time.monotonic()
                # 2차 운동은 seg.process 안에서 이 프레임의 포즈가 확정된 뒤 굴린다(dyn 인자).
                dyn_dt = 1 / 30 if st.dyn_last_t is None else now_m - st.dyn_last_t
                st.dyn_last_t = now_m
                # 방금 갱신됐으면 이전 패치와 크로스페이드 (둘 다 같은 앵커로 워핑되므로 겹침 없음)
                if st.forehead_prev is not None:
                    f = (now_m - st.forehead_fade_at) / max(cfg.forehead_fade_s, 1e-3)
                    prev = st.registry.get(st.forehead_prev)
                    if f < 1.0 and prev is not None:
                        asset2, mix = prev, 1.0 - f
                    else:
                        st.registry.remove(st.forehead_prev)
                        st.forehead_prev = None
                # 주기 갱신: 정면이고 안정적일 때만, 마지막 생성 후 refresh_s 지났으면 백그라운드로.
                if (cfg.forehead_refresh_s > 0 and st.forehead_name is not None
                        and not st.forehead_building and pose is not None
                        and abs(float(pose["yaw"])) < cfg.forehead_refresh_yaw
                        and self._yaw_ema is not None
                        and abs(float(pose["yaw"]) - self._yaw_ema) < 5.0
                        and now_m - st.forehead_built_at > cfg.forehead_refresh_s):
                    asyncio.ensure_future(build_forehead(st, quiet=True))
            processed, timings = await loop.run_in_executor(
                app.gpu_executor, seg.process, img, st.plate, st.mode,
                asset, st.scale_mul, st.offset_up, pose, st.harmonize, st.shadow,
                st.blend, asset2, mix, st.smoother,
                st.groom_obj, groom_fit, st.groom_color, groom_dyn,
                pose_cf, (st.hair_dyn, st.dyn_amount, dyn_dt) if dyn_dt is not None else None)
        finally:
            st.inflight -= 1
            app.gpu_inflight -= 1
        exec_ms = (time.perf_counter() - t_exec0) * 1000
        if timings.get("pose") is not None:
            pose = timings["pose"]
            self._last_pose = pose
            if st.pose_future is not None and st.pose_future.done():
                st.pose_future = None

        app.metrics.process.observe(exec_ms / 1000.0)
        app.metrics.infer.observe(float(timings["infer_ms"]) / 1000.0)

        fps = (len(self._recv_times) / sum(self._recv_times)
               if self._recv_times and sum(self._recv_times) > 0 else 0.0)

        ch = st.channel
        if ch is not None and ch.readyState == "open":
            try:
                ch.send(json.dumps({
                    "type": "stats",
                    "frame": self._frame_idx,
                    "server_fps": round(fps, 1),
                    "wait_ms": round(wait_ms, 1),
                    "proc_ms": round(exec_ms, 1),
                    "infer_ms": round(timings["infer_ms"], 1),
                    "pre_ms": round(timings["pre_ms"], 1),
                    "post_ms": round(timings["post_ms"], 1),
                    "hair_px": timings["hair_px"],
                    "coverage": timings["coverage"],
                    "plate_frames": timings["plate_frames"],
                    "mode": st.mode,
                    "anchor": timings["anchor"],
                    "harmonized": timings["harmonized"],
                    "blended": timings["blended"],
                    "blender_ready": seg._blender is not None,
                    "rec_count": st.rec_count if st.recording else None,
                    # 세션 레지스트리 기준이다. 다른 세션이 GAN 으로 만든
                    # 에셋(그 사람 얼굴에서 뽑은 것)은 여기 절대 안 보인다.
                    "assets": st.registry.names(),
                    "banks": st.registry.banks(),
                    "bank": st.bank,
                    "asset_used": ("3D " + st.groom) if st.groom_obj else _asset_label(asset, asset2, mix),
                    "groom": st.groom,
                    "groom_ms": round(timings.get("groom_ms", 0.0), 2),
                    "pose_ms": round(timings.get("pose_ms", 0.0), 1),
                    "yaw_ema": round(self._yaw_ema, 1) if self._yaw_ema is not None else None,
                    "references": list(app.references.keys()),
                    "gan_loaded": app.gan.loaded if app.gan else False,
                    "yaw": round(pose["yaw"], 1) if pose else None,
                    "tz": round(pose["tz"], 1) if pose and pose["tz"] else None,
                    "d_measured": round(pose["d_measured"], 1) if pose else None,
                    "d_corrected": round(pose["d_corrected"], 1) if pose else None,
                    "device": seg.device,
                    "cuda_graph": seg.graph is not None,
                    "frame_size": [img.shape[1], img.shape[0]],
                    # --- 아래는 추가된 키다(기존 키는 하나도 안 바꿨다) ---
                    "dropped": st.dropped,
                    "errors": st.errors,
                    "sessions": len(app.sessions),
                    "gan_backend": app.gan.backend if app.gan else None,
                }))
            except Exception:
                logger.exception("datachannel send 실패")

        if self._frame_idx % cfg.log_every == 0:
            logger.info("frame %d: fps=%.1f wait=%.0fms proc=%.1fms "
                        "(infer=%.1f pre=%.1f post=%.1f) drop=%d err=%d",
                        self._frame_idx, fps, wait_ms, exec_ms,
                        timings["infer_ms"], timings["pre_ms"], timings["post_ms"],
                        st.dropped, st.errors)

        self._last_out = processed
        return processed


def _asset_label(asset, asset2, mix):
    """stats 의 asset_used. 에셋 디렉터리가 비면 asset 이 None 일 수 있다."""
    if asset is None:
        return None
    if asset2 is not None and mix > 0.001:
        return f"{asset.name} + {asset2.name} ({mix:.0%})"
    return asset.name


def _dir_size_mb(path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# 에셋 생성
# ---------------------------------------------------------------------------

async def build_asset_from_result(state: PeerState, result_bgr, reference: str,
                                  yaw=None, bank=None, livebank=None):
    """GAN 결과 이미지에서 실시간 워핑용 헤어 에셋을 추출해 **이 세션에** 등록한다.

    yaw/bank 를 주면 다각도 뱅크의 한 칸으로 등록된다. 오프라인
    train/make_asset_bank.py 가 파일로 굽는 것과 같은 물건을 메모리에 만든다.
    """
    app = state.app
    seg = await app.get_segmenter()
    loop = asyncio.get_event_loop()

    # 세그멘테이션은 CUDA 그래프 때문에 반드시 gpu_executor 스레드에서.
    cls = await loop.run_in_executor(app.gpu_executor, seg.class_map, result_bgr)

    from face_pose import FacePose
    # with 문으로 닫는다. 예전 try/finally 와 같은 동작이지만 close() 를
    # 빠뜨릴 여지가 없다 - MediaPipe 는 네이티브 핸들이라 안 닫으면 프로세스가
    # 끝날 때까지 남는다.
    with FacePose() as poser:
        pose = poser.process(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), 0)
    if pose is None:
        logger.warning("GAN 결과에서 얼굴을 찾지 못해 에셋 추출을 건너뜁니다")
        return None

    tag = "" if yaw is None else f"-yaw{int(round(yaw)):+03d}"
    name = f"gan-{reference}{tag}-{int(time.time() * 1000) % 1000000}"
    # 추출 규칙(머리 + 얼굴 패치)은 asset_extract 한 곳에 있다. 오프라인
    # 뱅크 스크립트도 같은 함수를 쓰므로 두 경로가 어긋나지 않는다.
    import asset_extract
    asset, px = asset_extract.extract(result_bgr, cls, pose["eye_l"], pose["eye_r"], name)
    if asset is None:
        logger.warning("GAN 결과에서 머리를 찾지 못했습니다 (%s px)", px)
        return None

    asset.yaw = yaw
    asset.bank = bank

    # 크기 정규화. 이게 없으면 각도 칸이 sec(yaw) 배로 부풀어 보인다.
    #
    # 렌더 배율은 (라이브 눈간격 / D_asset) 인데, D_asset 은 그 칸을 구울 때의
    # **투영된** 눈 간격이라 고개를 돌린 각도일수록 짧다. 그대로 두면 나눗셈이
    # 작아져서 헤어가 커진다. 오프라인 뱅크는 JSON 의 scaleAdjust 로 이걸
    # 잡아뒀는데 라이브 뱅크는 아무도 설정하지 않아 전부 1.0 이었다.
    # (증상: 왼쪽으로 크게 돌렸을 때 끝 칸의 헤어가 유독 커진다)
    #
    # 뱅크는 정면 칸부터 굽는다(run_bank_generation 이 abs 로 정렬). 그래서
    # 첫 칸의 눈 간격을 기준 L 로 잡고 나머지를 그에 맞춘다.
    d_asset = float(np.linalg.norm(np.asarray(asset.eye_r) - np.asarray(asset.eye_l)))
    norm = 1.0
    if livebank is not None and d_asset > 1e-3:
        if livebank.ref_eye_len is None:
            livebank.ref_eye_len = d_asset
        norm = d_asset / livebank.ref_eye_len
    asset.scale_adjust = norm * state.cfg.gan_asset_scale

    state.registry.add(asset)
    app.metrics.assets_generated_total += 1

    # GPU 피라미드를 지금(GAN 생성 직후, 화면이 진행률로 덮여 있는 동안) 미리
    # 굽는다. 이걸 안 하면 나중에 고개를 돌려 이 칸이 처음 화면에 뜰 때
    # _warp_asset 이 즉석으로 만들어 그 프레임이 45~70ms 로 튄다(실측). 반드시
    # gpu_executor(CUDA 단일 스레드)에서. 실패해도 렌더가 알아서 만드므로 무해.
    try:
        await loop.run_in_executor(app.gpu_executor, seg.warm_asset, asset)
    except Exception:
        logger.exception("에셋 피라미드 예열 실패 (렌더가 필요 시 만든다)")

    # 디스크에 남긴다. 목적은 **내구성과 수동 승격**이다: 마음에 든 결과를
    # 사람이 골라 server/assets/ 로 옮기면 그때부터 공유 정적 에셋이 된다.
    # 시작할 때 이 디렉터리를 공유 정적 목록으로 자동 로드하면 안 된다 -
    # 생성 에셋에는 그 사람의 헤어라인/피부톤이 그대로 구워져 있어서, 다음에
    # 접속한 남의 목록에 뜨는 순간 그게 곧 유출이다(hair_asset.load_assets()
    # 가 generated_dir 를 읽지 않는 것도 같은 이유).
    # PNG 인코딩은 수십 ms 지만 여기는 GAN 9초 뒤의 느린 경로라 무시할 수 있다.
    try:
        hair_asset.save_asset(asset, os.path.join(state.cfg.generated_dir, state.sid))
    except Exception:
        logger.exception("생성 에셋 저장 실패 (메모리 등록은 성공)")
    return name


# --- 라이브 뱅크 ------------------------------------------------------------

class LiveBank:
    """세션 중에 각도별 헤어를 GAN 으로 생성해 채우는 뱅크.

    왜 미리 굽지 않는가
    -------------------
    미리 구운 뱅크는 '그때 그 사람, 그때 그 조명' 에 맞춰진 물건이다. 다른
    사람이 앉으면 두상도 피부톤도 달라 다시 구워야 한다. 세션에서 만들면
    언제나 지금 앉은 사람 기준이 된다.

    왜 점진적으로 채우는가
    ----------------------
    한 칸에 GAN 이 9초쯤 걸려 세 칸이면 30초다. 다 찰 때까지 아무것도 안
    보여주면 그 30초가 통째로 대기 시간이 된다. pick_pair_by_yaw 는 뱅크가
    덜 찼어도 양 끝을 클램프하므로(hair_asset.py), 정면 한 칸만 들어와도 바로
    입혀놓고 옆 칸은 도착하는 대로 얹으면 체감 대기가 9초로 줄어든다.

    즉시 전환(크로스페이드 없음)이라 칸이 촘촘할수록 전환이 눈에 덜 띈다.
    기본 각도/허용오차는 CONFIG.live_targets / live_tol 이다(검출기 dlib CNN
    한계가 ±40도라 그 안쪽에서 12도 간격).
    """

    def __init__(self, reference: str, targets=None, tol=None):
        self.reference = reference
        self.name = f"live-{reference}-{int(time.time()) % 100000}"
        self.targets = [float(t) for t in (targets or CONFIG.live_targets)]
        self.tol = float(CONFIG.live_tol if tol is None else tol)
        self.status = {t: "pending" for t in self.targets}
        # collect: 각도별 원본을 모으는 중 (영상 계속 보여줌 - 각도를 맞춰야 하니까)
        # generate: 모은 걸로 GAN 을 도는 중
        self.phase = "collect"
        self.frames = {}          # target -> 캡처된 원본 프레임
        # target -> **실제로 측정된** yaw. 라벨(목표 각도)과 다를 수 있고,
        # 런타임 칸 선택은 이 값을 써야 한다. 목표 각도를 적어 두면 허용오차
        # 만큼 어긋난 채로 기록돼 엉뚱한 각도에서 그 칸이 선택된다.
        self.measured = {}
        # target -> [최고 선명도, 그 프레임, 그때 yaw, 본 후보 수]. 확정 전 후보.
        self.cands = {}
        # 첫 칸(정면)의 눈 간격. 나머지 칸의 크기를 여기에 맞춰 정규화한다.
        # build_asset_from_result 가 채운다.
        self.ref_eye_len = None

    def next_target(self):
        """남은 칸 중 정면에 가까운 것부터. 정면이 가장 자주 보이는 각도다."""
        left = [t for t in self.targets if self.status[t] == "pending"]
        return min(left, key=abs) if left else None

    def match(self, yaw: float):
        """현재 각도가 아직 안 채운 칸의 허용 범위에 들어왔으면 그 칸을 준다."""
        best, best_d = None, self.tol
        for t in self.targets:
            if self.status[t] != "pending":
                continue
            d = abs(yaw - t)
            if d <= best_d:
                best, best_d = t, d
        return best

    def report(self):
        return {
            "phase": self.phase,
            "bank": self.name,
            "reference": self.reference,
            "buckets": [{"yaw": t, "status": self.status[t]} for t in self.targets],
            "next": self.next_target(),
            "done": sum(1 for v in self.status.values() if v == "done"),
            "total": len(self.targets),
        }


def notify_peer(state, payload: dict):
    ch = state.channel
    if ch is not None and ch.readyState == "open":
        try:
            ch.send(json.dumps(payload))
        except Exception:
            logger.exception("알림 전송 실패")


async def prepare_gan_input(state: PeerState, frame):
    """GAN 에 넣을 프레임 전처리. 지금은 앞머리 사전 제거(gan_input.py) 하나다.

    파싱은 gpu_executor(CUDA 그래프 단일 스레드), 눈 위치는 MediaPipe IMAGE
    모드. GAN 자식 프로세스에는 파서가 없으므로 여기(부모)서 끝내고 보낸다.
    실패하면 원본을 그대로 쓴다 - 전처리는 개선이지 전제 조건이 아니다.
    """
    app = state.app
    if not state.cfg.gan_prefill_forehead:
        return frame
    try:
        seg = await app.get_segmenter()
        loop = asyncio.get_event_loop()
        from face_pose import landmarks_image
        import gan_input
        # 랜드마커(~10ms)와 채움(수십 ms)은 CPU 작업이라 루프 밖에서 돌린다.
        # 촬영은 스트리밍 중에 오므로 루프를 막으면 모든 세션이 그만큼 멈춘다.
        lm = await loop.run_in_executor(
            None, landmarks_image, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if lm is None:
            return frame
        cls = await loop.run_in_executor(app.gpu_executor, seg.class_map, frame)
        out, npx = await loop.run_in_executor(
            None, gan_input.prefill_forehead, frame, cls, lm["eye_l"], lm["eye_r"])
        if npx:
            logger.info("GAN 입력: 앞머리 %d px 사전 제거", npx)
        return out
    except Exception:
        logger.exception("GAN 입력 전처리 실패 - 원본 프레임 사용")
        return frame


async def run_bank_bucket(state: PeerState, lb: LiveBank, target: float, frame):
    """뱅크 한 칸을 GAN 으로 만들어 등록한다.

    gan_executor 는 워커가 1개라 여러 칸이 동시에 들어와도 자동으로 줄을 선다.
    """
    app = state.app

    def report(**kw):
        notify_peer(state, {"type": "livebank", **lb.report(), **kw})

    lb.status[target] = "running"
    report(status="running", message=f"{target:+.0f}° 생성 중... (약 9초)")

    ref_path = app.references.get(lb.reference)
    if ref_path is None:
        lb.status[target] = "failed"
        report(status="error", message=f"참고 사진 없음: {lb.reference}")
        return

    try:
        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()
        frame = await prepare_gan_input(state, frame)
        result, gan_ms = await loop.run_in_executor(
            app.gan_executor, app.gan.swap, frame, ref_path, ref_path, logger.info)
        app.metrics.gan_swaps_total += 1
        app.metrics.gan.observe(float(gan_ms))
        name = await build_asset_from_result(
            state, result, lb.reference,
            yaw=float(lb.measured.get(target, target)), bank=lb.name,
            livebank=lb)
        if not name:
            lb.status[target] = "failed"
            report(status="error", message=f"{target:+.0f}° 에서 머리를 찾지 못했습니다")
            return

        lb.status[target] = "done"
        # 첫 칸이 들어온 순간부터 곧바로 실시간에 물린다. 나머지 칸은 도착하는
        # 대로 같은 뱅크에 쌓이므로 별도 전환 처리가 필요 없다.
        if state.bank != lb.name:
            state.bank = lb.name
            state.asset_name = None
            state.mode = "tryon"
        logger.info("라이브 뱅크 %s: yaw %+.0f 완료 (GAN %.1fs, 전체 %.1fs)",
                    lb.name, target, gan_ms, time.perf_counter() - t0)
        report(status="filled", filled_yaw=target, gan_seconds=round(gan_ms, 1),
               banks=state.registry.banks())
    except Exception as e:
        # 자식 프로세스 쪽 예외(얼굴 미검출 등)는 입력 탓이므로 자식은 살아 있고
        # 재기동도 하지 않는다. 이 칸만 실패로 두고 다음 칸으로 넘어간다.
        logger.exception("라이브 뱅크 칸 생성 실패 (yaw %+.0f)", target)
        app.metrics.gan_errors_total += 1
        lb.status[target] = "failed"
        report(status="error", message=str(e))


async def run_bank_generation(state: PeerState, lb: LiveBank):
    """수집이 끝난 각도들을 한 번에 생성한다.

    정면(|yaw| 가 작은 것)부터 돌린다. 중간에 실패하거나 사용자가 끊어도
    가장 자주 보이는 각도는 건지기 위해서다.
    """
    order = sorted(lb.targets, key=abs)
    t0 = time.perf_counter()
    for i, target in enumerate(order):
        if state._closed:
            # 세션이 이미 정리됐다. 남은 칸을 계속 돌리면 GPU 만 태우고
            # 결과는 아무도 안 본다(칸당 9초 x 7칸 = 1분이 통째로 낭비된다).
            logger.info("라이브 뱅크 %s: 세션 종료로 중단", lb.name)
            return
        frame = lb.frames.get(target)
        if frame is None:
            lb.status[target] = "failed"
            continue
        notify_peer(state, {"type": "livebank", **lb.report(),
                            "status": "generating", "index": i + 1,
                            "total": len(order), "current_yaw": target})
        await run_bank_bucket(state, lb, target, frame)

    lb.phase = "done"
    lb.frames.clear()
    # 스트리밍 재개. livebank 를 비워야 recv() 의 로딩 프레임 분기에서 빠져나온다.
    state.livebank = None
    done = sum(1 for v in lb.status.values() if v == "done")
    logger.info("라이브 뱅크 %s 완료: %d/%d칸, %.1fs",
                lb.name, done, len(order), time.perf_counter() - t0)
    notify_peer(state, {"type": "livebank", **lb.report(), "status": "complete",
                        "seconds": round(time.perf_counter() - t0, 1),
                        "banks": state.registry.banks()})


async def compose_still(state: PeerState, frame, asset_name: str):
    """프레임 한 장에 에셋을 실시간과 같은 경로로 합성한다. -> BGR 또는 None.

    세션의 플레이트(기존 머리 지울 배경)와 세그멘터를 그대로 쓴다. 포즈는 이
    프레임에서 새로 잰다 - 세션 포즈 추적기는 워커 스레드가 쓰고 있고, 어차피
    한 장이라 거리 보정이 필요 없다.
    """
    app = state.app
    seg = await app.get_segmenter()
    asset = state.registry.get(asset_name)
    if asset is None or state.plate is None:
        return None
    # 세션 추적기의 마지막 포즈를 쓴다. 거리 보정(d_corrected = K/tz)은 정면
    # 프레임에서 캘리브레이션된 K 가 있어야 고개를 돌려도 크기가 유지되는데,
    # 새 FacePose 로 한 장만 재면 K 가 없어 yaw 만큼 헤어가 작아진다
    # (36° 면 ~19%). 촬영 프레임은 last_raw 라 추적기 포즈와 한 프레임 차이다.
    pose = getattr(state.track, "_last_pose", None) if state.track is not None else None
    if pose is None:
        from face_pose import FacePose
        with FacePose() as poser:
            pose = poser.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0)
    if pose is None:
        return None
    loop = asyncio.get_event_loop()
    # 평활화기는 넘기지 않는다(한 장). 블렌더도 끈다.
    out, _ = await loop.run_in_executor(
        app.gpu_executor, seg.process, frame, state.plate, "tryon",
        asset, state.scale_mul, state.offset_up, pose, state.harmonize, state.shadow,
        0.0, None, 0.0, None)
    return out


async def run_capture(state: PeerState, reference: str):
    """현재 프레임을 GAN으로 고화질 합성한다. 진행 상황은 DataChannel로 보고."""
    app = state.app

    def notify(**kw):
        notify_peer(state, {"type": "capture", **kw})

    if state.last_raw is None:
        notify(status="error", message="아직 영상 프레임이 없습니다.")
        state.capturing = False
        return

    ref_path = app.references.get(reference)
    if ref_path is None:
        notify(status="error", message=f"참고 사진을 찾을 수 없습니다: {reference}")
        state.capturing = False
        return

    frame = state.last_raw.copy()
    loop = asyncio.get_event_loop()

    try:
        if not app.gan.loaded:
            notify(status="loading",
                   message="GAN 모델을 처음 올리는 중입니다 (약 90초, 최초 1회)")

        notify(status="running", message="합성 중...")
        t0 = time.perf_counter()
        gan_in = await prepare_gan_input(state, frame)
        result, gan_ms = await loop.run_in_executor(
            app.gan_executor, app.gan.swap, gan_in, ref_path, ref_path, logger.info)
        total = time.perf_counter() - t0
        app.metrics.gan_swaps_total += 1
        app.metrics.gan.observe(float(gan_ms))

        os.makedirs(gan_process.CAPTURE_DIR, exist_ok=True)
        name = f"capture_{int(time.time() * 1000)}.png"
        out_path = os.path.join(gan_process.CAPTURE_DIR, name)
        before_name = name.replace("capture_", "before_")
        gan_name = name.replace("capture_", "gan_")
        # GAN 원본(1024, 얼굴 전체가 재생성된 것)은 gan_*.png 로 남긴다.
        # capture_*.png 는 아래에서 **본인 얼굴 위에 합성한** 것으로 바뀐다.
        cv2.imwrite(os.path.join(gan_process.CAPTURE_DIR, gan_name), result)
        cv2.imwrite(out_path, result)
        cv2.imwrite(os.path.join(gan_process.CAPTURE_DIR, before_name), frame)

        # 촬영본은 한 장에 1~2MB 다. 상한을 안 두면 DataChannel 명령 하나로
        # 디스크를 채울 수 있다. 오래된 것부터 지운다 - 방금 찍은 것이 가장
        # 쓸모 있고, 그 전 것들은 사용자가 이미 받아갔을 것이다.
        try:
            removed = hair_asset.prune_dir(gan_process.CAPTURE_DIR,
                                           state.cfg.capture_dir_max_mb)
            if removed:
                logger.info("captures 정리: %d장 삭제 (상한 %dMB)",
                            removed, state.cfg.capture_dir_max_mb)
        except Exception:
            logger.exception("captures 정리 실패")

        logger.info("촬영 완료: %s (GAN %.1fs, 전체 %.1fs)", name, gan_ms, total)

        # --- GAN 결과에서 실시간용 에셋을 뽑는다 ---
        # 이게 핵심이다. GAN은 이미 **이 사람의 두상/피부톤/조명에 맞춰** 헤어를
        # 다시 그려놨으므로, 거기서 오려낸 헤어는 남의 참고사진에서 오려온 것보다
        # 훨씬 자연스럽게 붙는다. 한 번 생성해서 계속 워핑하는 구조.
        asset_name = None
        try:
            notify(status="running", message="실시간용 헤어 에셋 추출 중...")
            asset_name = await build_asset_from_result(state, result, reference)
            if asset_name:
                state.asset_name = asset_name
                state.mode = "tryon"
                logger.info("실시간 에셋 등록: %s", asset_name)
        except Exception:
            logger.exception("에셋 추출 실패 (촬영 결과는 정상)")

        # --- 촬영 결과 = 본인 얼굴 + 합성 헤어 ---
        # GAN 출력을 그대로 주면 얼굴까지 StyleGAN 이 다시 그린 "남의 얼굴"이
        # 된다(입력이 웹캠이라 인버전이 정체성을 다 못 살린다). 실시간과 같은
        # 합성 경로(기존 머리 제거 + GAN 헤어/이마 패치 + 조명/그레인 정합)를
        # 원본 프레임에 한 번 돌려 그걸 촬영본으로 준다. 얼굴 픽셀은 100%
        # 본인 것이고 헤어만 GAN 것이다. 실패하면 GAN 원본이 그대로 남는다.
        if asset_name:
            try:
                composed = await compose_still(state, frame, asset_name)
                if composed is not None:
                    cv2.imwrite(out_path, composed)
            except Exception:
                logger.exception("촬영 합성 실패 - GAN 원본을 그대로 둡니다")

        notify(status="done", url=f"/captures/{name}", before=f"/captures/{before_name}",
               gan=f"/captures/{gan_name}",
               gan_seconds=round(gan_ms, 1), total_seconds=round(total, 1),
               asset=asset_name, assets=state.registry.names())
    except Exception as e:
        logger.exception("촬영 실패")
        app.metrics.gan_errors_total += 1
        msg = str(e)
        if "face" in msg.lower() or "detect" in msg.lower():
            msg += " — 정면을 보고 머리 위/양옆에 여백이 있도록 조금 물러나 보세요."
        notify(status="error", message=msg)
    finally:
        state.capturing = False


# ---------------------------------------------------------------------------
# HTTP 핸들러
# ---------------------------------------------------------------------------

def rotate_variants():
    """고를 수 있는 Rotate 체크포인트 -> {이름: 절대경로}.

    **미리 정의한 것만 노출한다.** 클라이언트가 임의 경로를 주게 두면 디스크의
    아무 파일이나 로드를 시도하게 되는데, 비교 UI 하나 만들자고 열어 줄 문이
    아니다. 클라이언트는 이 이름 중에서만 고른다.
    """
    out = {"base": os.path.join(gan_process.gan_worker.HAIRFAST_DIR,
                                "pretrained_models", "Rotate", "rotate_best.pth")}

    def add(name, raw):
        if not raw:
            return
        p = raw if os.path.isabs(raw) else os.path.abspath(
            os.path.join(os.path.dirname(ROOT), raw))
        if os.path.isfile(p):
            out.setdefault(name, p)

    add("finetuned", CONFIG.gan_rotate_finetuned)
    # 기동 시 지정한 것이 위 둘과 다르면 "startup" 이라는 이름으로 함께 노출한다.
    # 안 그러면 그 체크포인트로 띄웠을 때 되돌아갈 방법이 없다.
    ck = CONFIG.gan_rotate_checkpoint
    if ck:
        p = ck if os.path.isabs(ck) else os.path.abspath(
            os.path.join(os.path.dirname(ROOT), ck))
        if os.path.isfile(p) and p not in out.values():
            out["startup"] = p
    return out


def _variant_of(path):
    if not path:
        return None
    for name, p in rotate_variants().items():
        if os.path.normcase(os.path.abspath(p)) == os.path.normcase(os.path.abspath(path)):
            return name
    return "custom"


async def model_get(request):
    """현재 Rotate 체크포인트와 고를 수 있는 목록."""
    app = request.app["state"]
    cur = app.gan.rotate_checkpoint if app.gan else None
    return web.json_response({
        "variant": _variant_of(cur),
        "checkpoint": cur,
        "available": sorted(rotate_variants().keys()),
        "loaded": bool(app.gan and app.gan.loaded),
    })


async def model_set(request):
    """Rotate 체크포인트 교체. body: {"variant": "base"|"finetuned"}

    원본과 파인튜닝본을 번갈아 보려면 지금까지 서버를 재기동해야 했고 매번
    모델 적재에 20~90초가 들었다. Rotate 는 25MB 라 이것만 갈아끼우면 된다.
    """
    app = request.app["state"]
    if app.gan is None:
        return web.json_response(
            {"error": "no_gan", "message": "GAN 워커가 없습니다"}, status=503)
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": "bad_json", "message": "JSON 본문이 필요합니다"}, status=400)

    variants = rotate_variants()
    name = str(body.get("variant") or "")
    if name not in variants:
        return web.json_response(
            {"error": "bad_variant",
             "message": "variant 는 %s 중 하나여야 합니다" % ", ".join(sorted(variants))},
            status=400)

    loop = asyncio.get_event_loop()
    try:
        # 모델이 아직 안 올라갔으면 여기서 적재까지 일어난다(최대 ~90초).
        # 실시간 경로를 막지 않도록 GAN 전용 executor 에서 돌린다.
        path = await loop.run_in_executor(
            app.gan_executor, app.gan.set_rotate, variants[name], logger.info)
    except Exception as e:
        logger.exception("Rotate 교체 실패")
        return web.json_response(
            {"error": "swap_failed",
             "message": "%s: %s" % (type(e).__name__, e)}, status=500)

    logger.info("Rotate 체크포인트 -> %s (%s)", name, path)
    return web.json_response({"variant": name, "checkpoint": path})


async def references_list(request):
    """참고 헤어스타일 목록. 연결 전에도 UI를 채울 수 있도록 HTTP로 노출한다.
    (DataChannel 통계에만 실어 보내면 [연결 시작] 전에는 목록이 비어 보인다)

    **여기에는 정적 에셋만 넣는다.** 이 엔드포인트에는 세션이 없으므로 세션에서
    생성된 에셋을 섞으면 그게 곧 다른 사람에게 목록이 새는 경로가 된다.
    세션 에셋은 그 세션의 DataChannel stats 로만 나간다.
    """
    app = request.app["state"]
    return web.json_response({
        # styles 가 클라이언트가 쓸 것이다. id/표시명/썸네일/원본 URL 이 들어 있다.
        "styles": _style_entries(app, request),
        # references 는 이름만 있는 예전 형식이다. 웹 클라이언트와 이미 배포한
        # 연동 문서가 이걸 쓰고 있어서 남겨 둔다. styles 로 옮기고 나면 뺀다.
        "references": list(app.references.keys()),
        "assets": list(app.static_assets.keys()),
        "banks": hair_asset.list_banks(app.static_assets),
        "gan_loaded": app.gan.loaded if app.gan else False,
    })


_REF_NAME_OK = re.compile(r"^[A-Za-z0-9가-힣._-]{1,60}$")


def _inspect_reference(img_bgr):
    """참고사진의 품질을 잰다. (눈 간격 px, 경고 문구 또는 None)

    GAN 은 참고사진을 FFHQ 1024(눈 간격 ~256px)로 정렬해서 쓴다. 눈 간격이
    작으면 그만큼 업스케일되어 **헤어스타일 디테일이 뭉개진 채로** 들어간다.
    기존 korean-frontal.png 이 50px 이라 5배 확대되고 있었고, 그게 결과가
    흐릿했던 원인 중 하나다. 그래서 올릴 때 바로 알려준다.
    """
    from face_pose import FacePose
    with FacePose() as poser:
        pose = poser.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 0)
    if pose is None:
        return None, None, "얼굴을 찾지 못했습니다."
    return float(pose["d_measured"]), float(pose["yaw"]), None


async def references_upload(request):
    """참고 헤어스타일 사진 업로드. multipart/form-data 의 'file' 필드.

    선택 인자: 'name' (없으면 파일명에서 딴다)
    """
    app = request.app["state"]
    cfg = app.cfg
    limit = int(cfg.reference_max_mb * 1024 * 1024)
    if request.content_length and request.content_length > limit:
        return web.json_response(
            {"error": "too_large",
             "message": f"파일이 너무 큽니다 (상한 {cfg.reference_max_mb:.0f}MB)"},
            status=413)

    reader = await request.multipart()
    data, name, force = None, None, False
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "file":
            name = name or os.path.splitext(os.path.basename(part.filename or ""))[0]
            data = await part.read(decode=False)
            if len(data) > limit:
                return web.json_response(
                    {"error": "too_large", "message": "파일이 너무 큽니다"}, status=413)
        elif part.name == "name":
            name = (await part.text()).strip() or name
        elif part.name == "force":
            force = (await part.text()).strip().lower() in ("1", "true", "on", "yes")

    if not data:
        return web.json_response(
            {"error": "no_file", "message": "file 필드가 없습니다"}, status=400)

    name = (name or "ref").strip().replace(" ", "_")
    if not _REF_NAME_OK.match(name):
        return web.json_response(
            {"error": "bad_name",
             "message": "이름에 쓸 수 없는 문자가 있습니다 (영문/숫자/한글/._- 만)"},
            status=400)
    if name in app.references:
        # 덮어쓰기를 막는다. GAN 워커가 참고사진을 **경로로** 캐시하기 때문에
        # (정규화 + FFHQ 정렬 결과), 같은 경로의 내용이 바뀌면 낡은 캐시가
        # 계속 쓰인다. 다른 이름으로 올리게 하는 편이 안전하다.
        return web.json_response(
            {"error": "exists", "message": f"같은 이름이 이미 있습니다: {name}"},
            status=409)

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return web.json_response(
            {"error": "bad_image", "message": "이미지로 읽을 수 없습니다"}, status=400)

    loop = asyncio.get_event_loop()
    eye_px, yaw, err = await loop.run_in_executor(
        app.pose_executor, _inspect_reference, img)
    if err:
        return web.json_response({"error": "no_face", "message": err}, status=400)

    # 거부는 최후의 수단이다. 이 가드는 "알려주기" 용도인데 예전 임계값(120px)이
    # 상반신/전신 헤어 사진을 통째로 막았다. 게다가 눈 간격은 **투영값**이라
    # 고개를 돌린 사진에서는 cos(yaw) 만큼 짧아져, 얼굴이 충분히 커도 걸린다.
    # 그래서 (1) 각도를 감안해 정면 등가로 환산하고 (2) 그래도 걸리면 사유를
    # 알려주되 force 로 넘어갈 수 있게 한다.
    eff_px = eye_px / max(0.35, math.cos(math.radians(min(abs(yaw or 0.0), 70.0))))
    if eff_px < cfg.reference_min_eye_px and not force:
        hint = ""
        if abs(yaw or 0.0) >= 20:
            hint = (f" 고개가 {abs(yaw):.0f}도 돌아가 있어 눈 간격이 실제보다 "
                    f"짧게 잡혔습니다 (정면 등가 {eff_px:.0f}px).")
        return web.json_response({
            "error": "too_small",
            "message": (f"얼굴이 작습니다 (눈 간격 {eye_px:.0f}px, 최소 "
                        f"{cfg.reference_min_eye_px}px).{hint} GAN 이 1024 로 "
                        f"확대해 쓰기 때문에 헤어 디테일이 뭉개집니다. "
                        f"그래도 쓰려면 '작아도 등록'을 체크하세요."),
            "eye_px": round(eye_px, 1), "yaw": round(yaw or 0.0, 1),
            "effective_eye_px": round(eff_px, 1), "can_force": True,
        }, status=400)

    os.makedirs(cfg.reference_upload_dir, exist_ok=True)
    path = os.path.join(cfg.reference_upload_dir, name + ".png")
    if not cv2.imwrite(path, img):
        return web.json_response(
            {"error": "write_failed", "message": "저장에 실패했습니다"}, status=500)

    app.references = gan_process.list_references()
    warn = None
    if eye_px < cfg.reference_warn_eye_px:
        warn = (f"눈 간격 {eye_px:.0f}px 은 권장({cfg.reference_warn_eye_px}px)보다 "
                f"작습니다. 1024 로 {256/max(eye_px,1.0):.1f}배 확대되어 디테일이 "
                f"줄어듭니다.")
        if abs(yaw or 0.0) >= 20:
            warn += f" (고개 {abs(yaw):.0f}도 - 정면이면 더 크게 잡힙니다)"
    logger.info("참고사진 등록: %s (%dx%d, 눈 간격 %.0fpx)",
                name, img.shape[1], img.shape[0], eye_px)
    return web.json_response({
        "name": name, "width": img.shape[1], "height": img.shape[0],
        "eye_px": round(eye_px, 1), "upscale": round(256.0 / max(eye_px, 1.0), 2),
        "yaw": round(yaw or 0.0, 1),
        # 방금 올린 것의 썸네일 URL 까지 같이 준다. 안 그러면 클라이언트가
        # 목록을 다시 받아야 새 스타일을 캐러셀에 그릴 수 있다.
        "style": next((s for s in _style_entries(app, request) if s["id"] == name), None),
        "styles": _style_entries(app, request),
        "warning": warn, "references": list(app.references.keys()),
    })


THUMB_DIR = os.path.join(ROOT, "references", "thumbs")


def _meta_path(img_path):
    return os.path.splitext(img_path)[0] + ".meta.json"


def _read_ref_meta(name, img_path):
    """<이름>.meta.json 이 있으면 읽는다. 없으면 파일명에서 표시명을 만든다.

    별도 DB 를 두지 않는 이유: 참고사진은 파일 하나로 추가/삭제되는데 메타만
    다른 곳에 있으면 곧 어긋난다. 사진 옆에 두면 같이 움직인다.
    """
    meta = {}
    p = _meta_path(img_path)
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                meta = raw
        except (OSError, ValueError):
            logger.warning("참고사진 메타를 읽지 못했습니다: %s", p)
    # 표시명이 없으면 id 를 사람이 읽을 만하게 다듬는다. 어디까지나 임시값이라
    # 앱에 그대로 내보낼 이름은 meta.json 에 적어 주는 게 맞다.
    if not meta.get("name"):
        pretty = name.replace("-", " ").replace("_", " ").strip()
        meta["name"] = pretty.title() if pretty.isascii() else pretty
    return meta


def _make_thumb(img_bgr, size, half_eyes):
    """머리 전체가 들어가는 정사각 썸네일 (BGR).

    원형 캐러셀에 쓰이므로 얼굴이 가운데 오고 머리가 잘리지 않아야 한다.
    그냥 가운데를 자르면 상반신 사진에서 얼굴이 위쪽에 치우쳐 잘린다.
    그래서 눈 위치를 재서 머리를 중심에 놓는다. 얼굴을 못 찾으면 가운데를 자른다.
    """
    h, w = img_bgr.shape[:2]
    eye_l = eye_r = None
    try:
        from face_pose import FacePose
        with FacePose() as poser:
            pose = poser.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 0)
        if pose is not None:
            eye_l, eye_r = pose["eye_l"], pose["eye_r"]
    except Exception:
        logger.exception("썸네일용 얼굴 검출 실패 - 가운데 크롭으로 대체")

    if eye_l is not None:
        cx = float(eye_l[0] + eye_r[0]) / 2.0
        cy = float(eye_l[1] + eye_r[1]) / 2.0
        d = max(1.0, float(np.hypot(eye_r[0] - eye_l[0], eye_r[1] - eye_l[1])))
        half = d * float(half_eyes)
        # 눈은 머리 한가운데가 아니라 아래쪽에 있다. 헤어까지 담으려면 위로 민다.
        cy -= d * 0.30
    else:
        cx, cy = w / 2.0, h / 2.0
        half = min(w, h) / 2.0

    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    # 사진 밖으로 나가면 가장자리를 늘려 채운다. 검은 여백보다 낫다.
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - w), max(0, y1 - h)
    if pl or pt or pr or pb:
        img_bgr = cv2.copyMakeBorder(img_bgr, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
        x0 += pl; x1 += pl; y0 += pt; y1 += pt
    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        crop = img_bgr
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _thumb_path(name, img_path):
    """썸네일 경로. 원본보다 오래됐으면 다시 만든다."""
    os.makedirs(THUMB_DIR, exist_ok=True)
    out = os.path.join(THUMB_DIR, name + ".png")
    try:
        if os.path.isfile(out) and os.path.getmtime(out) >= os.path.getmtime(img_path):
            return out
    except OSError:
        pass
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    thumb = _make_thumb(img, CONFIG.reference_thumb_px,
                        CONFIG.reference_thumb_half_eyes)
    if not cv2.imwrite(out, thumb):
        return None
    logger.info("참고사진 썸네일 생성: %s", name)
    return out


def _base_url(request):
    """이미지 URL 앞부분. 네이티브 앱은 상대경로를 못 푸니 절대 URL 로 준다."""
    if CONFIG.public_base_url:
        return CONFIG.public_base_url.rstrip("/")
    # 프록시 뒤라면 원래 스킴/호스트가 여기 실려 온다.
    fwd_host = request.headers.get("X-Forwarded-Host")
    fwd_proto = request.headers.get("X-Forwarded-Proto")
    host = fwd_host or request.host
    scheme = fwd_proto or request.scheme
    return f"{scheme}://{host}"


def _style_entries(app, request):
    """클라이언트가 캐러셀을 그리는 데 필요한 것 전부."""
    base = _base_url(request)
    out = []
    for name, path in sorted(app.references.items()):
        meta = _read_ref_meta(name, path)
        q = urllib.parse.quote(name, safe="")
        entry = {
            "id": name,
            "name": meta.get("name") or name,
            "thumbnailUrl": f"{base}/references/{q}/thumbnail",
            "referenceImageUrl": f"{base}/references/{q}/image",
            # 절대 URL 과 **상대경로를 함께** 준다.
            #
            # 절대 URL 은 요청 Host 로 만드는데, 클라이언트가 프록시를 거쳐
            # 들어오면(웹 dev 의 /ar-server 같은) 그 URL 이 프록시를 우회한다.
            # 앱이 HTTPS 인데 서버가 HTTP 면 혼합 콘텐츠로 차단되기도 한다.
            # 상대경로가 있으면 클라이언트가 자기 base 에 붙여 쓸 수 있다.
            #   웹(프록시)  : `${base}${thumbnailPath}`
            #   네이티브     : thumbnailUrl 그대로
            "thumbnailPath": f"/references/{q}/thumbnail",
            "referenceImagePath": f"/references/{q}/image",
        }
        for k in ("description", "category", "gender"):
            if meta.get(k):
                entry[k] = meta[k]
        out.append(entry)
    return out


async def reference_image(request):
    """참고사진 원본. 인증 없이 열려 있다(서버 전체가 그렇다)."""
    app = request.app["state"]
    name = request.match_info["name"]
    if not _REF_NAME_OK.match(name):
        raise web.HTTPNotFound()
    path = app.references.get(name)
    if not path or not os.path.isfile(path):
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={"Cache-Control": "public, max-age=300"})


async def reference_thumbnail(request):
    """원형 캐러셀용 정사각 썸네일. 처음 요청될 때 만들어 두고 재사용한다."""
    app = request.app["state"]
    name = request.match_info["name"]
    if not _REF_NAME_OK.match(name):
        raise web.HTTPNotFound()
    path = app.references.get(name)
    if not path or not os.path.isfile(path):
        raise web.HTTPNotFound()
    loop = asyncio.get_event_loop()
    # 얼굴 검출이 들어가므로 이벤트 루프에서 돌리면 안 된다.
    out = await loop.run_in_executor(app.pose_executor, _thumb_path, name, path)
    if not out:
        raise web.HTTPInternalServerError(reason="썸네일 생성 실패")
    return web.FileResponse(out, headers={"Cache-Control": "public, max-age=300"})


async def references_delete(request):
    """업로드된 참고사진 삭제. 손으로 넣어 둔 것은 지우지 않는다."""
    app = request.app["state"]
    name = request.match_info["name"]
    if not _REF_NAME_OK.match(name):
        raise web.HTTPNotFound()
    path = app.references.get(name)
    updir = os.path.abspath(app.cfg.reference_upload_dir)
    if not path or os.path.commonpath([os.path.abspath(path), updir]) != updir:
        return web.json_response(
            {"error": "not_uploaded",
             "message": "업로드된 참고사진만 지울 수 있습니다"}, status=403)
    try:
        os.remove(path)
    except OSError as e:
        return web.json_response({"error": "delete_failed", "message": str(e)},
                                 status=500)
    app.references = gan_process.list_references()
    logger.info("참고사진 삭제: %s", name)
    return web.json_response({"deleted": name,
                              "references": list(app.references.keys())})


async def captures_file(request):
    name = request.match_info["name"]
    if "/" in name or "\\" in name or ".." in name:
        raise web.HTTPNotFound()
    path = os.path.join(gan_process.CAPTURE_DIR, name)
    if not os.path.isfile(path):
        raise web.HTTPNotFound()
    return web.FileResponse(path)


#: 3D 헤어 그룸(헤어카드 GLB + 정합 메타 json). Im2Haircut 스트랜드를
#: _hair3d_work/strands_to_cards.py 로 변환한 결과를 여기 둔다. 파일명이 곧 스타일 이름.
GROOM_DIR = os.path.join(ROOT, "grooms")


def list_grooms():
    """{name: meta}. meta 는 옆의 .json (없으면 {}). 정렬은 이름순."""
    out = {}
    if not os.path.isdir(GROOM_DIR):
        return out
    for fn in sorted(os.listdir(GROOM_DIR)):
        if not fn.endswith(".glb"):
            continue
        name = fn[:-4]
        meta = {}
        jp = os.path.join(GROOM_DIR, name + ".json")
        if os.path.isfile(jp):
            try:
                with open(jp, encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError) as e:
                logger.warning("그룸 메타 읽기 실패 %s: %s", jp, e)
        out[name] = meta
    return out


async def grooms_list(request):
    """hair3d.html 드롭다운용. 클라이언트에 필요한 것만 추린다(source_ply 같은 로컬 경로는 뺌)."""
    items = []
    for name, meta in list_grooms().items():
        items.append({
            "name": name,
            "url": f"/grooms/{name}.glb",
            "n_strands": meta.get("n_strands"),
            "user_fit": meta.get("user_fit", {}),
        })
    return web.json_response({"grooms": items})


def _parse_bgr(v):
    """'#rrggbb' -> (b,g,r) 0~255. 빈 값/이상한 값은 None(=사용자 머리색 맞춤)."""
    if not v or not isinstance(v, str):
        return None
    h = v.lstrip("#")
    if len(h) != 6:
        return None
    try:
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None
    return (float(b), float(g), float(r))


async def _set_groom(state: "SessionState", name):
    """세션 그룸 교체. GLB 적재(~1초)는 gpu_executor 에서 미리 하고, 끝난 뒤에 세션에 건다 -
    실시간 경로에서 처음 렌더할 때 적재하면 그 프레임이 1초 멈춘다."""
    app = state.app
    if not name:
        state.groom = state.groom_obj = None
        if state.mode == "tryon" and state.asset_name is None and not state.bank:
            state.mode = "raw"
        return
    grooms = list_grooms()
    if name not in grooms:
        notify_peer(state, {"type": "groom", "status": "error",
                            "message": f"그룸을 찾을 수 없습니다: {name}"})
        return
    obj = {"name": name, "path": os.path.join(GROOM_DIR, name + ".glb"), "meta": grooms[name]}
    seg = await app.get_segmenter()
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(app.gpu_executor, seg.warm_groom, obj)
    if not ok:
        notify_peer(state, {"type": "groom", "status": "error", "message": f"GLB 로드 실패: {name}"})
        return
    state.groom = name
    state.groom_obj = obj
    state.bank = None
    if state.mode == "raw":
        state.mode = "tryon"
    logger.info("그룸 -> %s (%s)", name, state.sid)
    notify_peer(state, {"type": "groom", "status": "ok", "groom": name})
    # 맨이마 패치가 아직 없으면 지금 프레임으로 만든다 (LaMa 인페인팅, ~50ms + 첫 적재 수 초).
    if state.forehead_name is None:
        asyncio.ensure_future(build_forehead(state))


async def build_forehead(state: "SessionState", force: bool = False, quiet: bool = False):
    """현재 프레임에서 앞머리를 인페인팅으로 걷어낸 얼굴 패치를 만들어 세션에 등록한다.

    GAN 촬영을 쓰지 않는 이유는 forehead.py 머리 주석 참고. 파싱은 gpu_executor(CUDA 그래프
    스레드), 인페인트는 forehead_executor(별도 스트림), 눈 위치는 MediaPipe IMAGE 모드.

    quiet: 주기 갱신. 성공 알림을 보내지 않고 이전 패치와 크로스페이드로 바꿔 끼운다 -
    매초 새 인페인트로 툭 바뀌면 이마가 깜빡인다.
    """
    app = state.app
    if state.forehead_building or state.last_raw is None:
        return
    state.forehead_building = True
    try:
        seg = await app.get_segmenter()
        loop = asyncio.get_event_loop()
        # 실시간 경로를 건드리지 않는다: 파싱은 프레임 루프가 방금 계산한 클래스맵을 재사용하고
        # (gpu_executor 에 6ms 작업을 끼워 넣지 않음), 랜드마크는 forehead_executor 에서 IMAGE
        # 모드로 - pose_executor 를 쓰면 그 사이 프레임의 포즈 대기가 25ms 까지 늘어난다.
        from face_pose import landmarks_image

        def _parse_and_landmarks(frame, cls_t):
            frame = frame.copy()
            cls = cls_t.cpu().numpy() if cls_t is not None else seg.class_map(frame)
            return frame, cls, landmarks_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 세그멘터는 세션 공용이라 마지막 클래스맵이 **이 세션의** 프레임인지 확인한다(객체 동일성).
        # 다른 세션 프레임이면 파싱을 다시 한다(gpu_executor - 드문 경우라 비용 무시).
        frame = state.last_raw
        cls_t = seg.last_cls if seg.last_cls_frame is frame else None
        if cls_t is None:
            cls_np = await loop.run_in_executor(app.gpu_executor, seg.class_map, frame)
            frame, cls, lm = await loop.run_in_executor(
                app.forehead_executor, lambda: (frame.copy(), cls_np,
                                                landmarks_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        else:
            frame, cls, lm = await loop.run_in_executor(app.forehead_executor, _parse_and_landmarks, frame, cls_t)
        if lm is None:
            notify_peer(state, {"type": "forehead", "status": "error", "message": "얼굴을 찾지 못했습니다"})
            return
        import forehead
        if app.forehead is None:
            app.forehead = forehead.ForeheadInpainter(seg.device)
            notify_peer(state, {"type": "forehead", "status": "loading", "message": "이마 인페인터 적재 중 (최초 1회)"})
        name = f"forehead-{int(time.time() * 1000) % 1000000}"
        asset, bangs_px, ms = await loop.run_in_executor(
            app.forehead_executor, forehead.build_forehead_asset,
            app.forehead, frame, cls, lm["eye_l"], lm["eye_r"], name, 7, lm.get("matrix"), lm.get("brows"))
        if asset is None:
            if not quiet:
                notify_peer(state, {"type": "forehead", "status": "error", "message": "얼굴 패치를 만들지 못했습니다"})
            return
        # 이전 패치는 바로 지우지 않는다 - 프레임 루프가 fade 동안 둘을 섞는다. 그 전 것만 정리.
        stale, state.forehead_prev = state.forehead_prev, state.forehead_name
        state.registry.add(asset)
        await loop.run_in_executor(app.gpu_executor, seg.warm_asset, asset)
        state.forehead_name = name
        state.forehead_built_at = time.monotonic()
        state.forehead_fade_at = state.forehead_built_at
        if stale:
            state.registry.remove(stale)
        if quiet:
            logger.debug("맨이마 패치 갱신 %s: 앞머리 %dpx, %.0fms", name, bangs_px, ms)
        else:
            logger.info("맨이마 패치 %s: 앞머리 %dpx, 인페인트 %.0fms (%s)", name, bangs_px, ms, state.sid)
            notify_peer(state, {"type": "forehead", "status": "ok", "name": name,
                                "bangs_px": bangs_px, "inpaint_ms": round(ms)})
    except Exception as e:
        logger.exception("맨이마 패치 생성 실패")
        if not quiet:
            notify_peer(state, {"type": "forehead", "status": "error", "message": str(e)})
        state.forehead_built_at = time.monotonic()      # 실패 직후 매 프레임 재시도하지 않게
    finally:
        state.forehead_building = False


async def groom_file(request):
    name = request.match_info["name"]
    if "/" in name or "\\" in name or ".." in name or not name.endswith(".glb"):
        raise web.HTTPNotFound()
    path = os.path.join(GROOM_DIR, name)
    if not os.path.isfile(path):
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={"Content-Type": "model/gltf-binary"})


# 개발 중에는 브라우저 캐시가 계속 발목을 잡는다. 코드를 고쳐도 예전 client.js가
# 캐시에서 나오면 "왜 안 바뀌지"로 시간을 버린다. 클라이언트 파일은 캐시 금지.
_NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate"}


async def index(request):
    return web.FileResponse(os.path.join(CLIENT_DIR, "index.html"), headers=_NO_CACHE)


async def client_file(request):
    """client/ 안의 파일을 그대로 서빙 (warp.html, warp.js 등)."""
    name = request.match_info["name"]
    if "/" in name or "\\" in name or ".." in name:
        raise web.HTTPNotFound()
    path = os.path.join(CLIENT_DIR, name)
    if not os.path.isfile(path):
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers=_NO_CACHE)


async def healthz(request):
    """프로세스가 살아 있는가. 의존성을 확인하지 않는다.

    여기서 모델 상태까지 보면, 모델 적재가 느린 순간에 오케스트레이터가
    '죽었다'고 판단해 프로세스를 재시작한다. 그러면 적재를 처음부터 다시
    하므로 영원히 안 뜬다. liveness 와 readiness 를 나누는 이유가 이것이다.
    """
    app = request.app["state"]
    return web.json_response({"status": "ok",
                              "uptime_s": round(time.time() - app.started_at, 1)})


async def readyz(request):
    """트래픽을 받을 준비가 됐는가.

    --preload 없이 뜨면 첫 연결이 모델 적재를 기다린다(수 초 ~ 수십 초).
    로드밸런서 뒤에서 이 신호가 없으면 그 콜드 인스턴스로 트래픽이 그대로
    들어가서 첫 사용자만 손해를 본다.
    """
    app = request.app["state"]
    ok, reason = app.ready()
    seg = app.segmenter
    body = {
        "ready": ok,
        "reason": reason,
        "device": seg.device if seg is not None else None,
        "cuda_graph": (seg.graph is not None) if seg is not None else None,
        "sessions": len(app.sessions),
    }
    return web.json_response(body, status=200 if ok else 503)


async def metrics_handler(request):
    """Prometheus 텍스트 노출.

    계측이 DataChannel 로만 나가면 화면을 보고 있는 사람만 서버 상태를 안다.
    사람이 안 보고 있을 때가 정확히 문제가 나는 때다.
    """
    app = request.app["state"]
    cfg = app.cfg
    if not cfg.metrics_enabled:
        raise web.HTTPNotFound()

    m = app.metrics
    e = metrics_mod.Exposition()

    e.gauge("heddy_up", "서버 프로세스가 응답 중이면 1.", 1)
    e.gauge("heddy_uptime_seconds", "프로세스 기동 후 경과 시간(초).",
            round(time.time() - app.started_at, 1))

    ok, reason = app.ready()
    # 사유를 라벨로 붙이면 '왜 안 준비됐는가'를 그래프에서 바로 본다.
    # 라벨 값에 예외 메시지가 들어오므로 이스케이프는 Exposition 이 한다.
    e.gauge("heddy_ready", "세그멘터가 적재되고 CUDA 그래프까지 준비됐으면 1.",
            1 if ok else 0, {"reason": reason})

    # --- 다운링크 품질 (서버 -> 클라이언트) ---
    #
    # "영상이 끊긴다" 의 원인이 GPU 인지 회선인지 구분하려면 이게 필요하다.
    # 지금까지는 클라이언트에 getStats() 를 물어봐야만 알 수 있었는데, 앱이든
    # 웹이든 붙어 있기만 하면 서버가 직접 볼 수 있다 - 수신 측이 RTCP Receiver
    # Report 로 유실률을 계속 보내주고 aiortc 가 그걸 remote-inbound-rtp 로
    # 노출한다.
    #
    # fraction_lost 는 직전 보고 구간의 유실 비율(0~1)이다. 0.02 를 넘기면
    # 비트레이트가 회선을 넘어선 것으로 보면 된다.
    for st_ in list(app.sessions):
        pc = getattr(st_, "pc", None)
        if pc is None:
            continue
        try:
            report = await pc.getStats()
        except Exception:
            continue
        for stat in report.values():
            if getattr(stat, "type", None) != "remote-inbound-rtp":
                continue
            lbl = {"sid": str(getattr(st_, "sid", "?"))}
            e.gauge("heddy_downlink_fraction_lost",
                    "수신 측이 보고한 직전 구간 유실 비율(0~1).",
                    round(float(getattr(stat, "fractionLost", 0.0) or 0.0), 4), lbl)
            e.gauge("heddy_downlink_packets_lost",
                    "수신 측이 보고한 누적 유실 패킷 수.",
                    int(getattr(stat, "packetsLost", 0) or 0), lbl)
            e.gauge("heddy_downlink_jitter_seconds",
                    "수신 측이 보고한 지터(초).",
                    round(float(getattr(stat, "jitter", 0.0) or 0.0), 5), lbl)
            rtt = getattr(stat, "roundTripTime", None)
            if rtt is not None:
                e.gauge("heddy_downlink_rtt_seconds", "RTCP 로 잰 왕복 시간(초).",
                        round(float(rtt), 4), lbl)

    e.gauge("heddy_sessions_active", "지금 붙어 있는 피어 수.", len(app.sessions))
    e.gauge("heddy_sessions_max", "동시 접속 상한(CONFIG.max_sessions).", cfg.max_sessions)
    e.counter("heddy_sessions_total", "수락된 세션 누적 수.", m.sessions_total)
    e.counter("heddy_sessions_rejected_total",
              "동시 접속 상한 때문에 503 으로 거절한 세션 누적 수.",
              m.sessions_rejected_total)

    e.counter("heddy_frames_total", "수신해 처리를 시도한 프레임 누적 수.", m.frames_total)
    e.counter("heddy_frames_dropped_total",
              "GPU 대기열이 밀려 합성을 건너뛴 프레임 누적 수.", m.frames_dropped_total)
    e.counter("heddy_frame_errors_total",
              "합성이 실패해 원본으로 패스스루한 프레임 누적 수.", m.frame_errors_total)

    e.histogram("heddy_infer_seconds", "세그멘테이션 추론 시간(초).", m.infer)
    e.histogram("heddy_frame_process_seconds",
                "프레임 한 장의 서버측 처리 시간(초). 워커 대기 포함.", m.process)
    # 히스토그램만으로는 '지금 값'을 못 본다. 최근값은 gauge 로 따로 낸다
    # (gauge 를 histogram 이라고 선언하지 않는다).
    # 이름을 heddy_infer_seconds_last 로 하지 않는 이유: 히스토그램 계열은
    # <name>_bucket/_sum/_count 를 쓰는데, 같은 접두어로 시작하는 다른 계열이
    # 있으면 엄격한 파서가 한 계열로 묶으려다 거부한다.
    e.gauge("heddy_last_infer_seconds", "가장 최근 추론 시간(초).", m.infer.last)
    e.gauge("heddy_last_frame_process_seconds", "가장 최근 프레임 처리 시간(초).",
            m.process.last)

    # 세션 스코프 상태
    plate_frames = 0
    for st in list(app.sessions):
        plate = st.plate
        if plate is not None:
            plate_frames += int(plate.frames)
    e.gauge("heddy_plate_frames",
            "배경 플레이트에 누적된 프레임 수(전 세션 합).", plate_frames)

    seg = app.segmenter
    if seg is not None:
        try:
            cache = seg.cache_stats()
            e.gauge("heddy_asset_cache_assets", "GPU 에 올라가 있는 헤어 에셋 수.",
                    cache.get("assets", 0))
            e.gauge("heddy_asset_cache_bytes", "헤어 에셋 GPU 캐시 크기(바이트).",
                    cache.get("bytes", 0))
        except Exception:
            # 관측이 서버를 죽이면 안 된다. 캐시가 다른 스레드에서 갱신되는
            # 도중이면 순회가 던질 수 있다.
            logger.debug("cache_stats 실패", exc_info=True)
        try:
            gpu = seg.gpu_stats()
        except Exception:
            gpu = {}
        if gpu:
            lb = {"device": gpu.get("device", "?")}
            e.gauge("heddy_gpu_memory_allocated_bytes",
                    "torch 가 실제로 쓰고 있는 VRAM(바이트).", gpu.get("allocated", 0), lb)
            e.gauge("heddy_gpu_memory_reserved_bytes",
                    "torch 캐싱 얼로케이터가 잡고 있는 VRAM(바이트).",
                    gpu.get("reserved", 0), lb)

    if app.gan is not None:
        h = app.gan.health()
        lb = {"backend": h.get("backend", "?")}
        e.gauge("heddy_gan_alive", "GAN 워커(자식 프로세스 또는 스레드)가 살아 있으면 1.",
                1 if h.get("alive") else 0, lb)
        e.gauge("heddy_gan_loaded", "HairFastGAN 모델이 적재돼 있으면 1.",
                1 if h.get("loaded") else 0, lb)
        e.counter("heddy_gan_restarts_total", "GAN 워커 재기동 누적 수.",
                  int(h.get("restarts") or 0), lb)
        if h.get("load_seconds"):
            e.gauge("heddy_gan_load_seconds", "HairFastGAN 모델 적재에 걸린 시간(초).",
                    float(h["load_seconds"]), lb)
        e.counter("heddy_gan_swaps_total", "성공한 GAN 합성 누적 수.", m.gan_swaps_total)
        e.counter("heddy_gan_errors_total", "실패한 GAN 합성 누적 수.", m.gan_errors_total)
        e.histogram("heddy_gan_swap_seconds", "GAN 합성 1회 소요 시간(초).", m.gan)

    e.counter("heddy_assets_generated_total",
              "GAN 결과에서 뽑아 세션에 등록한 헤어 에셋 누적 수.",
              m.assets_generated_total)
    e.gauge("heddy_static_assets", "전 세션이 공유하는 정적 헤어 에셋 수.",
            len(app.static_assets))
    e.gauge("heddy_references", "GAN 참고 사진 수.", len(app.references))

    return web.Response(text=e.text(),
                        content_type="text/plain",
                        charset="utf-8",
                        headers={"Cache-Control": "no-store"})


def _apply_rtp_packet_size(cfg) -> None:
    """내보내는 RTP 패킷을 경로 MTU 안쪽으로 줄인다.

    aiortc 는 페이로드 상한을 코덱 모듈의 전역 상수 PACKET_MAX 로 들고 있고
    패킷을 자를 때마다 그 전역을 읽는다. 그래서 여기서 바꿔 두면 이후 만들어지는
    모든 패킷에 적용된다(설정 API 가 따로 없다).

    왜 필요한지는 config.rtp_packet_max 주석 참고 - 기본 1300 은 회선상 약
    1350 바이트가 되어, MTU 가 그보다 낮은 경로에서 **서버가 보내는 것만**
    통째로 버려진다. 받는 쪽(Chrome, 1200)은 멀쩡하니 원인이 잘 안 보인다.
    """
    if not cfg.rtp_packet_max:
        return
    from aiortc.codecs import h264 as _h264, vpx as _vpx
    for mod in (_vpx, _h264):
        mod.PACKET_MAX = int(cfg.rtp_packet_max)
    logger.info("RTP 페이로드 상한: %d 바이트 (aiortc 기본 1300)", cfg.rtp_packet_max)


def _apply_video_bitrate(cfg) -> None:
    """서버가 내보내는 영상의 비트레이트 상한을 올린다.

    aiortc 는 설정 API 없이 코덱 모듈의 전역 상수로 들고 있다. 세터가 매번
    그 전역을 읽으므로(vpx.py: `max(MIN_BITRATE, min(bitrate, MAX_BITRATE))`)
    여기서 바꿔 두면 이후 만들어지는 인코더 전부에 적용된다. PACKET_MAX 와
    같은 방식이다.

    DEFAULT_BITRATE 는 인코더 생성 시점에 한 번 읽히므로 시작값이 되고,
    MAX_BITRATE 는 REMB 가 올라올 때마다 상한으로 쓰인다.

    **다운링크에만 적용된다.** 올라오는 영상은 클라이언트 인코더와 aiortc 가
    수신자로서 보내는 REMB 가 정하므로 이 상수와 무관하다.
    """
    if not cfg.video_max_bitrate:
        return
    from aiortc.codecs import vpx as _vpx
    lo = int(cfg.video_start_bitrate or _vpx.DEFAULT_BITRATE)
    hi = int(cfg.video_max_bitrate)
    # 시작값이 상한보다 크면 세터가 상한으로 깎아 버려 의도가 뒤집힌다.
    lo = min(lo, hi)
    _vpx.MAX_BITRATE = hi
    _vpx.DEFAULT_BITRATE = lo
    logger.info("영상 비트레이트: 시작 %.1f Mbps / 상한 %.1f Mbps "
                "(aiortc 기본 0.5 / 1.5)", lo / 1e6, hi / 1e6)


def _prefer_codec(pc, want: str) -> None:
    """보낼 비디오 코덱을 고정한다. setRemoteDescription 뒤, createAnswer 앞에서.

    코덱은 원래 **브라우저 offer 의 순서**대로 정해진다. aiortc 는 VP8 과 H.264
    만 지원하므로 Chrome 이 H.264 를 앞에 두면 그쪽으로 붙는데, 그러면 패킷은
    도착하는데 framesDecoded 가 0 인 채 검은 화면이 된다(실측: bytesReceived 는
    오르는데 디코딩된 프레임이 0). 브라우저가 고르게 두면 안 되는 이유다.
    """
    if want == "auto":
        return
    mime = "video/" + ("VP8" if want == "vp8" else "H264")
    caps = RTCRtpSender.getCapabilities("video")
    # rtx(재전송)는 남겨둔다. 빼면 패킷 손실 복구가 사라져 화면이 잘 깨진다.
    prefs = [c for c in caps.codecs
             if c.mimeType == mime or c.mimeType == "video/rtx"]
    if not prefs:
        logger.warning("코덱 %s 를 쓸 수 없어 브라우저 선택에 맡깁니다", want)
        return
    for t in pc.getTransceivers():
        if t.kind == "video":
            try:
                t.setCodecPreferences(prefs)
            except Exception:
                logger.exception("코덱 고정 실패 - 브라우저 선택으로 진행합니다")


def _log_negotiated_codec(sdp: str, sid: str) -> None:
    """실제로 합의된 코덱을 남긴다.

    이게 로그에 없어서 "검은 화면"의 원인이 전송인지 디코딩인지 코덱인지
    구분하는 데 한참 걸렸다. 한 줄이면 다음엔 바로 보인다.
    """
    in_video = False
    names = []
    for line in sdp.splitlines():
        if line.startswith("m="):
            in_video = line.startswith("m=video")
        elif in_video and line.startswith("a=rtpmap:"):
            name = line.split(" ", 1)[1].split("/")[0]
            if name.lower() != "rtx":
                names.append(name)
    logger.info("세션 %s 비디오 코덱: %s", sid, ", ".join(names) or "(없음)")


async def offer(request):
    app = request.app["state"]
    cfg = app.cfg

    # GPU 워커가 1개다. 상한을 넘겨 받으면 아무도 못 막아 다 같이 느려지고
    # 30fps 가 무너진다. 조용히 전부 느려지는 것보다 명시적으로 거절하는 편이
    # 낫다 - 거절당한 쪽은 이유를 알고 나중에 다시 오면 된다.
    if len(app.sessions) >= cfg.max_sessions:
        app.metrics.sessions_rejected_total += 1
        logger.warning("세션 거절: 활성 %d / 상한 %d", len(app.sessions), cfg.max_sessions)
        return web.json_response({
            "error": "server_busy",
            "message": ("동시 접속 상한(%d)에 도달했습니다. "
                        "GPU 워커가 1개라 더 받으면 모두 느려집니다."
                        % cfg.max_sessions),
            "active": len(app.sessions),
            "max_sessions": cfg.max_sessions,
        }, status=503)

    # 본문이 비었거나 sdp/type 이 없으면 **클라이언트 오류**다. 그냥 request.json()
    # 을 부르면 JSONDecodeError 가 그대로 올라가 500 이 되는데, 그러면 받는 쪽은
    # "서버가 answer 생성 중 터졌다" 로 읽는다. 실제로 그렇게 몇 시간을 잃었다.
    #
    # 실제 사례: 네이티브 앱이 RTCSessionDescription 객체를 그대로 실어 보냈다.
    # sdp/type 이 프로토타입 게터라 브리지가 직렬화하면 빈 객체가 되고 본문이
    # 비어서 도착한다. 브라우저는 JSON.stringify 가 toJSON() 을 타서 멀쩡했다.
    raw = await request.text()
    try:
        params = json.loads(raw) if raw.strip() else None
    except ValueError:
        params = None
    if not isinstance(params, dict) or not params.get("sdp") or not params.get("type"):
        logger.warning("잘못된 /offer 본문 (%d바이트, UA=%s): %.200r",
                       len(raw), request.headers.get("User-Agent", "?"), raw)
        return web.json_response({
            "error": "bad_offer",
            "message": ("본문에 sdp 와 type 이 있어야 합니다. 받은 본문 %d바이트. "
                        "RTCSessionDescription 을 그대로 보내면 sdp/type 이 "
                        "프로토타입 게터라 직렬화에서 사라집니다 - "
                        "{sdp, type} 평범한 객체로 만들어 보내세요."
                        % len(raw)),
            "received_bytes": len(raw),
        }, status=400)

    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    app.pcs.add(pc)
    state = PeerState(app)
    state.pc = pc
    app.sessions.add(state)
    app.metrics.sessions_total += 1
    logger.info("세션 시작 %s (활성 %d/%d)", state.sid, len(app.sessions), cfg.max_sessions)

    def _teardown():
        """세션 자원 해제 + 집합에서 제거. 어느 경로로 들어와도 한 번만 돈다."""
        state.cleanup()
        app.sessions.discard(state)
        app.pcs.discard(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("connection state: %s (%s)", pc.connectionState, state.sid)
        if pc.connectionState in ("failed", "closed"):
            _teardown()
            await pc.close()

    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info("datachannel opened: %s", channel.label)
        state.channel = channel

        @channel.on("message")
        def on_message(message):
            try:
                data = json.loads(message)
            except (TypeError, ValueError):
                return
            if data.get("type") == "ping":
                channel.send(json.dumps({"type": "pong", "t_client": data.get("t_client")}))
            elif data.get("type") == "mode":
                m = data.get("mode")
                if m in ("raw", "seg", "remove", "plate", "tryon"):
                    state.mode = m
                    logger.info("mode -> %s", m)
            elif data.get("type") in ("livebank", "capture") and app.gan is None:
                notify_peer(state, {"type": data["type"], "status": "error",
                                    "message": "GAN 경로는 꺼져 있습니다 (3D 스타일을 쓰세요. HEDDY_GAN_ENABLED=1 로 복구)"})
                if data.get("type") == "capture":
                    state.capturing = False
            elif data.get("type") == "livebank":
                if data.get("on"):
                    ref = data.get("reference") or (app.references and
                                                    next(iter(app.references)))
                    if ref not in app.references:
                        channel.send(json.dumps({
                            "type": "livebank", "status": "error",
                            "message": f"참고 사진을 찾을 수 없습니다: {ref}"}))
                    else:
                        state.livebank = LiveBank(ref, data.get("targets"))
                        # 수집 중에는 원본을 그대로 보여준다.
                        # 각도를 맞춰야 하는 단계라 예전 헤어를 씌워 둘 이유가
                        # 없고, 무엇보다 raw 여야 세그멘테이션을 건너뛸 수 있다
                        # (collect_seg_every). 두 번째 뱅크를 만들 때는 이미
                        # tryon 상태라 서버가 여기서 되돌려 주지 않으면 GPU 를
                        # 계속 쓰게 된다.
                        state.mode = "raw"
                        logger.info("라이브 뱅크 시작: %s (각도 %s)",
                                    state.livebank.name, list(state.livebank.targets))
                        notify_peer(state, {"type": "livebank",
                                            **state.livebank.report(),
                                            "status": "started", "mode": "raw"})
                else:
                    lb = state.livebank
                    state.livebank = None
                    logger.info("라이브 뱅크 중지")
                    notify_peer(state, {"type": "livebank", "status": "stopped",
                                        **(lb.report() if lb else {})})
            elif data.get("type") == "record":
                _handle_record(state, channel, data)
            elif data.get("type") == "capture":
                if state.capturing:
                    return
                state.capturing = True
                asyncio.ensure_future(
                    run_capture(state, data.get("reference") or
                                (next(iter(app.references)) if app.references else "")))
            elif data.get("type") == "fit":
                if "groom" in data:
                    asyncio.ensure_future(_set_groom(state, data["groom"] or None))
                if data.get("forehead") == "refresh":
                    asyncio.ensure_future(build_forehead(state, force=True))
                if "dyn" in data:
                    state.dyn_amount = max(0.0, min(2.0, float(data["dyn"])))
                if "groom_fwd" in data:
                    state.groom_fwd_cm = max(-15.0, min(15.0, float(data["groom_fwd"])))
                if "groom_color" in data:
                    state.groom_color = _parse_bgr(data["groom_color"])
                if "asset" in data and data["asset"] in state.registry:
                    state.asset_name = data["asset"]
                    state.bank = None          # 개별 에셋을 고르면 뱅크는 해제
                if "bank" in data:
                    state.bank = data["bank"] or None
                if "scale" in data:
                    state.scale_mul = max(0.5, min(2.0, float(data["scale"])))
                if "offset" in data:
                    state.offset_up = max(-150.0, min(150.0, float(data["offset"])))
                if "harmonize" in data:
                    state.harmonize = bool(data["harmonize"])
                if "shadow" in data:
                    state.shadow = max(0.0, min(1.0, float(data["shadow"])))
                if "blend" in data:
                    state.blend = max(0.0, min(1.0, float(data["blend"])))
                if "smooth" in data:
                    state.smooth = max(0.0, min(2.0, float(data["smooth"])))
            else:
                # 여기 걸리면 클라이언트/서버 프로토콜이 어긋난 것이다.
                # 지금까지는 조용히 버려져서 "눌러도 아무 일이 없다" 로만 보였다.
                logger.warning("알 수 없는 DataChannel 커맨드: %r", data.get("type"))

    @pc.on("track")
    def on_track(track):
        logger.info("track received: %s", track.kind)
        if track.kind == "video":
            # buffered=False: 릴레이가 최신 프레임 하나만 들고 있고 밀린 프레임은 버린다.
            # 기본(True)은 무제한 asyncio.Queue 라, 처리 속도가 도착 속도보다 조금만 느려도
            # (고개 돌릴 때 포즈 재검출, 3초마다 이마 갱신 스파이크) 밀린 프레임이 쌓이고 그걸
            # 순서대로 다 처리하느라 **지연이 계속 자랐다** - "처음엔 실시간이다가 점점 늦어짐".
            # recv() 의 inflight 드롭은 루프가 직렬이라 이 경우를 못 잡는다(항상 0).
            pc.addTrack(SegmentedVideoTrack(app.relay.subscribe(track, buffered=False), state))

        @track.on("ended")
        async def on_ended():
            logger.info("source track ended: %s (%s)", track.kind, state.sid)
            # 트랙이 끝나도 pc 는 한동안 살아 있을 수 있다. 여기서 안 놓으면
            # MediaPipe 그래프와 플레이트 텐서가 ICE 타임아웃까지 남는다.
            _teardown()

    await pc.setRemoteDescription(offer_desc)
    _prefer_codec(pc, cfg.video_codec)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    _log_negotiated_codec(pc.localDescription.sdp, state.sid)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })


def _handle_record(state: PeerState, channel, data):
    """학습용 프레임 수집 켜기/끄기.

    얼굴 프레임은 생체정보다. 한번 디스크에 떨어지면 서버를 굴리는 쪽이 그
    사람의 얼굴 수천 장을 갖게 된다. 그래서 기본은 꺼짐이고
    (CONFIG.allow_frame_recording), 켜는 것은 서버 운영자의 명시적 결정이어야
    한다 - 브라우저 버튼 한 번이 그 결정이 되면 안 된다. 예전에는 이 메시지
    하나로 디스크가 찰 때까지 얼굴이 쌓였다.
    """
    cfg = state.cfg
    if data.get("on"):
        if not cfg.allow_frame_recording:
            logger.warning("프레임 수집 요청 거절: allow_frame_recording=False")
            channel.send(json.dumps({
                "type": "record", "on": False, "count": state.rec_count,
                "dir": None,
                "message": ("서버에서 프레임 수집이 꺼져 있습니다. 얼굴 프레임은 "
                            "생체정보라 기본값이 꺼짐입니다 "
                            "(HEDDY_ALLOW_FRAME_RECORDING=1 로 켤 수 있습니다).")}))
            return
        import datetime
        sid = datetime.datetime.now().strftime("%m%d_%H%M%S")
        state.rec_dir = os.path.join(REC_ROOT, sid)
        os.makedirs(state.rec_dir, exist_ok=True)
        state.rec_count = 0
        state.recording = True
        logger.info("프레임 수집 시작 -> %s (상한 %d장 / %dMB)",
                    state.rec_dir, cfg.record_max_frames, cfg.record_dir_max_mb)
    else:
        state.recording = False
        logger.info("프레임 수집 종료: %d장", state.rec_count)
    channel.send(json.dumps({"type": "record", "on": state.recording,
                             "count": state.rec_count,
                             "dir": state.rec_dir}))


# ---------------------------------------------------------------------------
# 앱 팩토리 / 라이프사이클
# ---------------------------------------------------------------------------

async def _reaper(app_state: AppState):
    """프레임이 안 들어오는 세션을 정리한다.

    브라우저 탭을 그냥 닫으면 ICE 가 failed 로 갈 때까지 (구현에 따라 수십 초
    ~ 사실상 영원히) 아무 이벤트도 안 온다. 그동안 MediaPipe 그래프와 GPU
    플레이트가 그대로 살아 있어서, 몇 번 반복하면 max_sessions 가 유령 세션으로
    다 차고 정상 사용자가 503 을 받는다.
    """
    cfg = app_state.cfg
    while True:
        await asyncio.sleep(cfg.session_reaper_interval_s)
        now = time.monotonic()
        for st in list(app_state.sessions):
            idle = now - st.last_frame_at
            if idle < cfg.session_idle_timeout_s:
                continue
            logger.warning("idle 세션 정리 %s (%.0fs 동안 프레임 없음)", st.sid, idle)
            st.cleanup()
            app_state.sessions.discard(st)
            pc = st.pc
            if pc is not None:
                app_state.pcs.discard(pc)
                try:
                    await pc.close()
                except Exception:
                    logger.exception("pc.close 실패")


def create_app(cfg=CONFIG, preload=False, preload_gan=False) -> web.Application:
    """aiohttp 앱을 만든다. 무거운 초기화는 전부 on_startup 에서."""
    app = web.Application()
    state = AppState(cfg)
    app["state"] = state

    async def _startup(_app):
        # 첫 피어가 붙기 전에 해둬야 한다. 패킷을 자를 때 읽히는 전역이라
        # 세션이 이미 돌고 있으면 그 세션에는 안 먹는다.
        _apply_rtp_packet_size(cfg)
        _apply_video_bitrate(cfg)

        # 정적 에셋은 여기서 **한 번만** 읽어 전 세션이 공유한다(읽기 전용).
        # 끄면 아예 안 읽는다 - 목록에서 감추는 게 아니라 존재하지 않게 된다.
        # 그래야 /references 와 stats.assets 양쪽에서 동시에 사라진다.
        if cfg.serve_static_assets:
            state.static_assets = hair_asset.load_assets()
        else:
            state.static_assets = {}
            logger.info("정적 에셋 비활성 (serve_static_assets=False) - 라이브 뱅크만 씁니다")
        state.references = gan_process.list_references()
        logger.info("정적 에셋 %d개, 참고 사진 %d개",
                    len(state.static_assets), len(state.references))

        # 생성 에셋 저장소를 상한 이하로 줄인다. 지난 실행들이 남긴 것이
        # 계속 쌓이면 디스크가 조용히 찬다.
        try:
            removed = hair_asset.prune_dir(cfg.generated_dir, cfg.generated_dir_max_mb)
            if removed:
                logger.info("생성 에셋 정리: %d개 삭제 (상한 %dMB)",
                            removed, cfg.generated_dir_max_mb)
        except Exception:
            logger.exception("생성 에셋 정리 실패")

        # GanClient 는 **여기서** 만든다. 모듈 레벨에서 만들고 start() 까지
        # 부르면 서버를 두 번 띄웠을 때 자식도 둘이 되어 VRAM 이 2배가 된다.
        # log=logger.info 는 자식 stdout/stderr 중계에 쓰이며, 부모의 데몬
        # 스레드에서 호출되므로 스레드 안전한 콜러블이어야 한다.
        if cfg.gan_enabled:
            state.gan = gan_process.GanClient(cfg=cfg, log=logger.info)
            try:
                state.gan.start()          # 논블로킹(실측 0.008s). 모델은 첫 swap 때.
            except Exception:
                logger.exception("GAN 워커 기동 실패 (첫 촬영 때 다시 시도한다)")
        else:
            logger.info("GAN 경로 비활성 (gan_enabled=False) - 3D 그룸만 사용")

        if preload:
            await state.get_segmenter()
        if preload_gan and state.gan is not None:
            await _warm_gan(state)

        state.reaper = asyncio.create_task(_reaper(state))

    async def _shutdown(_app):
        # 리퍼를 먼저 확실히 죽인다. 살아 있으면 close() 가 비운 집합을
        # 계속 훑고, 이벤트 루프가 안 닫힌다.
        task, state.reaper = state.reaper, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("리퍼 종료 실패")
        await state.close()

    app.on_startup.append(_startup)
    app.on_shutdown.append(_shutdown)

    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_get("/model", model_get)
    app.router.add_post("/model", model_set)
    app.router.add_get("/references", references_list)
    app.router.add_post("/references", references_upload)
    app.router.add_get("/references/{name}/image", reference_image)
    app.router.add_get("/references/{name}/thumbnail", reference_thumbnail)
    app.router.add_delete("/references/{name}", references_delete)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/readyz", readyz)
    app.router.add_get("/metrics", metrics_handler)
    app.router.add_get("/captures/{name}", captures_file)
    app.router.add_get("/grooms", grooms_list)
    app.router.add_get("/grooms/{name}", groom_file)
    # 반드시 마지막 (catch-all). 위에 있으면 /healthz 같은 새 라우트를 전부
    # 삼켜서 404 가 된다.
    app.router.add_get("/{name}", client_file)
    return app


async def _warm_gan(state: AppState):
    """GAN 을 실제로 예열한다(모델 적재 ~90초).

    워커 프로토콜에는 '모델만 올려라' op 가 없다 - 적재는 첫 swap 이 유발한다.
    그래서 참고 사진 한 장을 얼굴 겸 헤어로 넣어 한 번 돌리고 결과는 버린다.
    참고 사진에서 dlib 이 얼굴을 못 찾으면 실패하는데, 그건 예열이 안 됐다는
    뜻일 뿐이라 경고만 남기고 서버는 그대로 뜬다(첫 촬영 때 다시 적재한다).
    """
    if not state.references:
        logger.warning("--preload-gan: 참고 사진이 없어 예열을 건너뜁니다")
        return
    name, path = next(iter(state.references.items()))
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("--preload-gan: 참고 사진을 읽을 수 없습니다: %s", path)
        return
    logger.info("--preload-gan: %s 로 예열 시작 (모델 적재 ~90초)", name)
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(state.gan_executor, state.gan.swap,
                                   img, path, path, logger.info)
    except Exception as e:
        logger.warning("--preload-gan 실패 (%s). 첫 촬영 때 다시 적재한다.", e)
        return
    logger.info("--preload-gan 완료: %.1fs (health=%s)",
                time.perf_counter() - t0, state.gan.health())


def main():
    # argparse 기본값을 CONFIG 에서 가져온다. CONFIG 는 이미 HEDDY_* 환경변수로
    # 덮어써져 있으므로 우선순위가 자동으로 명령행 > 환경변수 > 기본값이 된다.
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=CONFIG.host)
    parser.add_argument("--port", type=int, default=CONFIG.port)
    # 기본 켬(CONFIG.preload). 끄면 서버를 띄운 뒤 첫 연결이 모델 적재(수 초)를 기다려서, 그동안
    # 화면이 검게 멈춰 있었다 - 재시작할 때마다 "연결이 너무 느리다" 로 보였다(사용자 피드백).
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=CONFIG.preload,
                        help="시작할 때 세그멘터를 미리 올린다 (--no-preload 로 끔)")
    parser.add_argument("--preload-gan", action="store_true",
                        help="HairFastGAN 도 미리 올린다 (~90초. 기본으로 켜면 안 된다)")
    args = parser.parse_args()

    app = create_app(CONFIG, preload=args.preload, preload_gan=args.preload_gan)

    # 인증서가 있으면 HTTPS 도 같이 연다 (run_app 은 사이트 하나만 열어서 직접 구성).
    # 자체서명(IP 접속용)과 Tailscale 정식 인증서(앱용)는 SNI 로 갈라 쓴다.
    import ssl

    def _tls_ctx(cert, key):
        if not (os.path.isfile(cert) and os.path.isfile(key)):
            return None
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.load_cert_chain(cert, key)
        return ctx

    self_ctx = _tls_ctx(CONFIG.tls_cert, CONFIG.tls_key)
    ts_ctx = _tls_ctx(CONFIG.tls_ts_cert, CONFIG.tls_ts_key)
    ssl_ctx = self_ctx or ts_ctx
    if self_ctx is not None and ts_ctx is not None:
        def _sni(sock, server_name, _ctx):
            if server_name and server_name.lower().endswith(".ts.net"):
                sock.context = ts_ctx
        self_ctx.sni_callback = _sni
    if ssl_ctx is not None:
        logger.info("starting on http://%s:%s + https://%s:%s (자체서명=%s, tailscale=%s)",
                    args.host, args.port, args.host, CONFIG.tls_port,
                    self_ctx is not None, ts_ctx is not None)
    else:
        logger.info("starting on http://%s:%s (TLS 없음: %s 없음)", args.host, args.port, CONFIG.tls_cert)

    async def _serve():
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, args.host, args.port).start()
        if ssl_ctx is not None:
            await web.TCPSite(runner, args.host, CONFIG.tls_port, ssl_context=ssl_ctx).start()
        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            await runner.cleanup()

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
