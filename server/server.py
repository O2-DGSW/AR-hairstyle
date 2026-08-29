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
import os
import time
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
        self.gan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gan")
        self.pose_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pose")

        self.segmenter = None          # 첫 연결 때 지연 초기화 (모델 로딩이 몇 초)
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
        self.mode = "seg"
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

    def _pose_rgb(self, img):
        """랜드마커 워커 스레드에서 실행. 색변환도 여기서 해야 메인 루프가 안 막힌다."""
        return self._state.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                        int(time.monotonic() * 1000))

    def _collect_for_bank(self, pose):
        """라이브 뱅크가 켜져 있으면, 목표 각도에 들어온 순간의 원본을 잡아 큐에 넣는다.

        움직이는 도중에 잡으면 모션 블러가 그대로 GAN 입력이 되어 결과가
        뭉개진다. 측정값이 EMA 를 얼마나 앞질렀는지로 '지금 움직이는 중인가'
        를 판정한다 - 정지해 있으면 둘이 붙고, 돌리는 중이면 벌어진다.
        """
        st = self._state
        lb = st.livebank
        if lb is None or lb.phase != "collect" or pose is None or self._yaw_ema is None:
            return
        if abs(float(pose["yaw"]) - self._yaw_ema) > st.cfg.live_steady:
            return
        target = lb.match(self._yaw_ema)
        if target is None or st.last_raw is None:
            return

        lb.status[target] = "captured"
        lb.frames[target] = st.last_raw.copy()
        notify_peer(st, {"type": "livebank", **lb.report(),
                         "status": "captured", "captured_yaw": target})

        # 다 모았으면 생성 단계로. 수집과 생성을 겹치지 않는 이유는 GPU 다.
        # 같은 4070 에서 세그멘테이션과 GAN 이 경합하면 양쪽 다 느려진다.
        if all(v == "captured" for v in lb.status.values()):
            lb.phase = "generate"
            asyncio.create_task(run_bank_generation(st, lb))

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

        new_frame = VideoFrame.from_ndarray(out, format="bgr24")
        # pts/time_base 는 반드시 원본 것을 유지한다. 새로 만들면 수신 측
        # 지터 버퍼가 타임라인을 다시 맞추느라 영상이 튄다.
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
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
        pose = None
        if st.mode in ("tryon", "remove") or st.livebank is not None:
            if st.pose_future is not None and st.pose_future.done():
                try:
                    self._last_pose = st.pose_future.result()
                except Exception:
                    logger.exception("랜드마커 실패")
                st.pose_future = None
            # 아직 안 끝났으면 새로 던지지 않는다(자연스러운 백프레셔).
            if st.pose_future is None:
                st.pose_future = loop.run_in_executor(app.pose_executor,
                                                      self._pose_rgb, img)
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

        # 예전에는 생성 중에 검은 프레임을 흘렸다. GAN 이 같은 프로세스에 있어서
        # 이벤트 루프와 GPU 를 통째로 잡아먹었기 때문인데, gan_process 로 자식
        # 프로세스에 분리한 지금은 실시간 경로가 GIL 로 막히는 일이 구조적으로
        # 없다. 같은 4070 을 공유하므로 fps 는 떨어지지만 검은 화면보다 낫다.
        # 경합이 심한 환경에서는 CONFIG.stream_during_gan=False 로 되돌린다.
        # (진행률은 어느 쪽이든 DataChannel 로 계속 나간다)
        if (not cfg.stream_during_gan and st.livebank is not None
                and st.livebank.phase == "generate"):
            return np.zeros_like(img)

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
            processed, timings = await loop.run_in_executor(
                app.gpu_executor, seg.process, img, st.plate, st.mode,
                asset, st.scale_mul, st.offset_up, pose, st.harmonize, st.shadow,
                st.blend, asset2, mix, st.smoother)
        finally:
            st.inflight -= 1
            app.gpu_inflight -= 1
        exec_ms = (time.perf_counter() - t_exec0) * 1000

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
                    "asset_used": _asset_label(asset, asset2, mix),
                    "yaw_ema": round(self._yaw_ema, 1) if self._yaw_ema is not None else None,
                    "references": list(app.references.keys()),
                    "gan_loaded": app.gan.loaded if app.gan else False,
                    "yaw": round(pose["yaw"], 1) if pose else None,
                    "tz": round(pose["tz"], 1) if pose and pose["tz"] else None,
                    "d_measured": round(pose["d_measured"], 1) if pose else None,
                    "d_corrected": round(pose["d_corrected"], 1) if pose else None,
                    "device": seg.device,
                    "cuda_graph": seg.graph is not None,
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
                                  yaw=None, bank=None):
    """GAN 결과 이미지에서 실시간 워핑용 헤어 에셋을 추출해 **이 세션에** 등록한다.

    yaw/bank 를 주면 다각도 뱅크의 한 칸으로 등록된다. 오프라인
    train/make_asset_bank.py 가 파일로 굽는 것과 같은 물건을 메모리에 만든다.
    """
    app = state.app
    seg = await app.get_segmenter()
    loop = asyncio.get_event_loop()

    # 세그멘테이션은 CUDA 그래프 때문에 반드시 gpu_executor 스레드에서.
    cls = await loop.run_in_executor(app.gpu_executor, seg.class_map, result_bgr)

    from gpu_segmenter import CLS_HAIR, CLS_SKIN
    hair = (cls == CLS_HAIR).astype("uint8")
    # 이 결과의 피부색을 함께 기록해두면, 나중에 조명이 달라져도 그 비율로
    # 헤어 색을 보정할 수 있다.
    ref_skin = hair_asset.skin_mean(result_bgr, (cls == CLS_SKIN))

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
    asset, _, px = hair_asset.build_from_photo(
        result_bgr, hair, pose["eye_l"], pose["eye_r"], name, ref_skin=ref_skin)
    if asset is None or px < 500:
        logger.warning("GAN 결과에서 머리를 찾지 못했습니다 (%s px)", px)
        return None

    asset.yaw = yaw
    asset.bank = bank
    state.registry.add(asset)
    app.metrics.assets_generated_total += 1

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
        result, gan_ms = await loop.run_in_executor(
            app.gan_executor, app.gan.swap, frame, ref_path, ref_path, logger.info)
        app.metrics.gan_swaps_total += 1
        app.metrics.gan.observe(float(gan_ms))
        name = await build_asset_from_result(
            state, result, lb.reference, yaw=float(target), bank=lb.name)
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
        result, gan_ms = await loop.run_in_executor(
            app.gan_executor, app.gan.swap, frame, ref_path, ref_path, logger.info)
        total = time.perf_counter() - t0
        app.metrics.gan_swaps_total += 1
        app.metrics.gan.observe(float(gan_ms))

        os.makedirs(gan_process.CAPTURE_DIR, exist_ok=True)
        name = f"capture_{int(time.time() * 1000)}.png"
        out_path = os.path.join(gan_process.CAPTURE_DIR, name)
        before_name = name.replace("capture_", "before_")
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

        notify(status="done", url=f"/captures/{name}", before=f"/captures/{before_name}",
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

async def references_list(request):
    """참고 헤어스타일 목록. 연결 전에도 UI를 채울 수 있도록 HTTP로 노출한다.
    (DataChannel 통계에만 실어 보내면 [연결 시작] 전에는 목록이 비어 보인다)

    **여기에는 정적 에셋만 넣는다.** 이 엔드포인트에는 세션이 없으므로 세션에서
    생성된 에셋을 섞으면 그게 곧 다른 사람에게 목록이 새는 경로가 된다.
    세션 에셋은 그 세션의 DataChannel stats 로만 나간다.
    """
    app = request.app["state"]
    return web.json_response({
        "references": list(app.references.keys()),
        "assets": list(app.static_assets.keys()),
        "banks": hair_asset.list_banks(app.static_assets),
        "gan_loaded": app.gan.loaded if app.gan else False,
    })


async def captures_file(request):
    name = request.match_info["name"]
    if "/" in name or "\\" in name or ".." in name:
        raise web.HTTPNotFound()
    path = os.path.join(gan_process.CAPTURE_DIR, name)
    if not os.path.isfile(path):
        raise web.HTTPNotFound()
    return web.FileResponse(path)


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

    params = await request.json()
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
                if m in ("seg", "remove", "plate", "tryon"):
                    state.mode = m
                    logger.info("mode -> %s", m)
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
                        logger.info("라이브 뱅크 시작: %s (각도 %s)",
                                    state.livebank.name, list(state.livebank.targets))
                        notify_peer(state, {"type": "livebank",
                                            **state.livebank.report(),
                                            "status": "started"})
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
            pc.addTrack(SegmentedVideoTrack(app.relay.subscribe(track), state))

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

        # 정적 에셋은 여기서 **한 번만** 읽어 전 세션이 공유한다(읽기 전용).
        state.static_assets = hair_asset.load_assets()
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
        state.gan = gan_process.GanClient(cfg=cfg, log=logger.info)
        try:
            state.gan.start()          # 논블로킹(실측 0.008s). 모델은 첫 swap 때.
        except Exception:
            logger.exception("GAN 워커 기동 실패 (첫 촬영 때 다시 시도한다)")

        if preload:
            await state.get_segmenter()
        if preload_gan:
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
    app.router.add_get("/references", references_list)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/readyz", readyz)
    app.router.add_get("/metrics", metrics_handler)
    app.router.add_get("/captures/{name}", captures_file)
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
    parser.add_argument("--preload", action="store_true",
                        help="첫 연결을 기다리지 않고 시작할 때 세그멘터를 미리 올린다")
    parser.add_argument("--preload-gan", action="store_true",
                        help="HairFastGAN 도 미리 올린다 (~90초. 기본으로 켜면 안 된다)")
    args = parser.parse_args()

    app = create_app(CONFIG, preload=args.preload, preload_gan=args.preload_gan)
    logger.info("starting on http://%s:%s", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
