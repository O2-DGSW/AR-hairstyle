"""HairFastGAN 래퍼 — "사진 찍기" 모드의 고화질 합성 담당.

실시간 워핑(16ms/프레임)이 미리보기를, 이쪽이 최종 결과 한 장을 만든다.
GAN은 장당 ~9초라 프레임당 실행은 원천적으로 불가능하므로, 사용자가 각도를
맞춘 뒤 결정적 순간에만 호출하는 구조다.

주의사항 (실측/문서 기반)
------------------------
- 모델 적재에 ~88초, GPU 5.5GB. 첫 촬영 때 한 번만 로드하고 계속 물고 있는다.
- dlib 정면 검출기를 쓰므로 반측면 얼굴은 실패한다. 업샘플링으로 완화.
- 머리 위/양옆에 여백이 없는 클로즈업이면 블렌딩 경계가 배경을 침범해
  전체가 뿌옇게 뭉개진다. 웹캠 프레임은 대개 여백이 부족하므로 패딩을 덧댄다.
- 얼굴 사진에 원래 머리카락이 있어야 정렬 기준이 잡힌다(대머리 입력은 결과 불량).

이 클래스를 별도 프로세스에서 돌리는 래퍼는 gan_process.GanClient 에 있다.
같은 프로세스에서 쓰면(=thread 백엔드) 아래 chdir 문제와 VRAM 경합이 남는다.
"""
import os
import sys
import threading
import time

import cv2
import numpy as np

from config import CONFIG, ROOT

HAIRFAST_DIR = os.path.join(os.path.dirname(ROOT), "external", "HairFastGAN")
REF_DIR = os.path.join(ROOT, "references")
CAPTURE_DIR = os.path.join(ROOT, "captures")

# HairFastGAN 저장소는 pretrained_models 를 **상대경로**로 찾기 때문에 적재/추론
# 구간에서 os.chdir 로 작업 디렉터리를 옮겨야 한다. 그런데 chdir 은 스레드가
# 아니라 **프로세스 전역** 상태다. 그 사이에 다른 스레드가 상대경로로 파일을
# 열면(에셋 저장, 캡처 쓰기 등) 엉뚱한 곳에 쓰거나 FileNotFoundError 가 난다.
# 지금은 GAN 워커가 1개라 우연히 겹치지 않을 뿐이므로, chdir 구간 전체를 이
# 모듈 전역 락으로 감싸고 try/finally 로 반드시 되돌린다.
# (gan_process 로 별도 프로세스에 분리하면 chdir 이 그 프로세스 안에서만
#  일어나므로 이 위험 자체가 사라진다. thread 폴백 경로를 위해 남겨 둔다.)
_CHDIR_LOCK = threading.RLock()


class GanWorker:
    """스레드 하나에서만 호출할 것 (모델이 스레드 안전하지 않음)."""

    def __init__(self):
        self._hf = None
        self._poser = None
        self._ref_cache = {}     # 정규화된 참고사진 캐시 (경로 -> RGB 배열)
        self._aligned_ref_cache = {}   # FFHQ 정렬까지 끝낸 참고사진 (경로 -> 텐서)
        self._shape_predictor = None   # dlib 68점 예측기. 한 번만 만든다.
        self._lock = threading.Lock()
        self.load_seconds = None
        #: 현재 적재된 Rotate 체크포인트 절대경로. 적재 전에는 None.
        self.rotate_checkpoint = None

    # ---------- 모델 ----------
    def ensure_loaded(self, log=print):
        with self._lock:
            if self._hf is not None:
                return self._hf
            log("HairFastGAN 적재 시작 (최초 1회, ~90초 소요)")
            t = time.perf_counter()

            if HAIRFAST_DIR not in sys.path:
                sys.path.insert(0, HAIRFAST_DIR)
            with _CHDIR_LOCK:
                cwd = os.getcwd()
                os.chdir(HAIRFAST_DIR)
                try:
                    from hair_swap import HairFast, get_parser
                    opts = get_parser().parse_args([])
                    ck = CONFIG.gan_rotate_checkpoint
                    if ck:
                        # 여기서 cwd 는 HAIRFAST_DIR 이다. 사용자가 준 상대경로는
                        # 레포 루트 기준이므로 절대경로로 바꿔서 넘긴다.
                        if not os.path.isabs(ck):
                            ck = os.path.abspath(os.path.join(os.path.dirname(ROOT), ck))
                        if not os.path.isfile(ck):
                            raise FileNotFoundError(
                                f"gan_rotate_checkpoint 를 찾을 수 없다: {ck}")
                        opts.rotate_checkpoint = ck
                        log(f"파인튜닝 Rotate 체크포인트 사용: {ck}")
                    self._hf = HairFast(opts)
                    # opts.rotate_checkpoint 는 기본값일 때 HAIRFAST_DIR 상대경로다.
                    # 지금 cwd 가 거기라 abspath 가 올바르게 풀린다.
                    self.rotate_checkpoint = os.path.abspath(opts.rotate_checkpoint)
                finally:
                    os.chdir(cwd)

            self.load_seconds = time.perf_counter() - t
            log(f"HairFastGAN 적재 완료: {self.load_seconds:.1f}s")
            return self._hf

    @property
    def loaded(self):
        return self._hf is not None

    def set_rotate(self, ckpt_path: str, log=print) -> str:
        """Rotate 모듈 가중치만 갈아끼운다. -> 적용된 절대경로.

        원본과 파인튜닝본을 번갈아 보려고 서버를 다시 띄우면 매번 모델 적재에
        20~90초가 들고, 그 사이 조건도 흔들려 비교가 어렵다. Rotate 는 6.6M
        (25MB)뿐이고 StyleGAN(30.4M)/e4e 와 독립이라 이것만 바꾸면 즉시 끝난다.

        캐시는 건드리지 않아도 된다. _aligned_ref_cache 는 FFHQ 정렬 결과라
        Rotate 와 무관하다. 이미 만들어 둔 에셋은 예전 모델의 산출물로 그대로
        남으므로, 비교하려면 새로 촬영해야 한다.
        """
        import torch
        hf = self.ensure_loaded(log)
        ckpt_path = os.path.abspath(ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Rotate 체크포인트가 없다: {ckpt_path}")
        with self._lock:
            dev = hf.align.opts.device
            sd = torch.load(ckpt_path, map_location=dev)["model_state_dict"]
            hf.align.rotate_model.load_state_dict(sd)
            hf.align.rotate_model.to(dev).eval()
            self.rotate_checkpoint = ckpt_path
        log(f"Rotate 체크포인트 교체: {ckpt_path}")
        return ckpt_path

    def close(self):
        """들고 있는 보조 리소스를 놓는다. 여러 번 불러도 안전하다.

        FacePose(MediaPipe)는 네이티브 핸들을 쥐고 있어서 명시적으로 닫지
        않으면 프로세스가 끝날 때까지 남는다. GAN 을 별도 프로세스로 돌릴 때
        자식이 종료 직전에 이걸 부른다.
        (HairFast 모델 자체는 해제 API 가 없다. 프로세스가 끝나야 VRAM 5.5GB 가
         돌아온다 - 그래서 별도 프로세스로 빼는 것이 회수 수단이기도 하다.)
        """
        poser, self._poser = self._poser, None
        if poser is not None:
            try:
                poser.close()
            except Exception:
                pass
        self._ref_cache.clear()
        self._aligned_ref_cache.clear()
        self._shape_predictor = None

    # ---------- 입력 준비 ----------
    @staticmethod
    def _pad(img_rgb: np.ndarray) -> np.ndarray:
        """머리 주변 여백을 확보한다. 가장자리 색으로 늘려서 이어붙이면
        단색 띠보다 정렬/블렌딩이 덜 흔들린다."""
        h, w = img_rgb.shape[:2]
        py, px = int(h * CONFIG.pad_ratio), int(w * CONFIG.pad_ratio)
        return cv2.copyMakeBorder(img_rgb, py, py, px, px, cv2.BORDER_REPLICATE)

    def _prepare(self, frame_bgr: np.ndarray) -> np.ndarray:
        """dlib이 찾을 수 있고 블렌딩 여백도 있는 RGB 이미지를 만든다.

        웹캠 프레임을 그냥 패딩해서 넣으면 두 가지가 동시에 나빠진다:
        얼굴이 프레임 대비 작으면 dlib 정면 검출기가 놓치고(No faces detected),
        패딩은 그 비율을 **더** 작게 만든다.

        그래서 순서를 뒤집는다. MediaPipe로 (dlib보다 훨씬 안정적으로) 얼굴을
        찾아 머리 주변을 넉넉히 잘라낸 뒤, 눈 간격이 적당해지도록 키운다.
        결과적으로 여백은 확보하면서 얼굴은 커진다.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._poser is None:
            from face_pose import FacePose
            self._poser = FacePose()
        pose = self._poser.process(rgb, 0)
        if pose is None:
            return self._pad(rgb)          # 얼굴을 못 찾으면 기존 방식으로 폴백

        d = float(pose["d_measured"])
        if d < 1.0:
            return self._pad(rgb)

        cx = float(pose["eye_l"][0] + pose["eye_r"][0]) / 2.0
        cy = float(pose["eye_l"][1] + pose["eye_r"][1]) / 2.0
        half = d * CONFIG.crop_half_eyes
        x0, x1 = int(cx - half), int(cx + half)
        y0, y1 = int(cy - half * 1.15), int(cy + half * 1.35)   # 머리 위/턱 아래

        # 프레임 밖으로 나가는 만큼만 가장자리 색으로 늘린 뒤 자른다
        h, w = rgb.shape[:2]
        pl, pt = max(0, -x0), max(0, -y0)
        pr, pb = max(0, x1 - w), max(0, y1 - h)
        if pl or pt or pr or pb:
            rgb = cv2.copyMakeBorder(rgb, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
            x0 += pl; x1 += pl; y0 += pt; y1 += pt
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return self._pad(rgb)

        # dlib이 편하게 찾을 수 있는 눈 간격이 되도록 스케일 조정
        s = CONFIG.target_eye_px / d
        if s > 1.02:
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        return crop

    # ---------- FFHQ 정렬 ----------
    # 여기가 실제 병목이었다. 실측(warm, 3장 기준):
    #   align_face  2,770ms  /  equal_replacer  50ms  /  실제 GAN  770ms
    # 즉 시간의 77% 가 GAN 이 아니라 dlib 얼굴검출+정렬이다. 인터넷에 도는
    # "장당 0.5초" 는 **이미 FFHQ 정렬된 1024 이미지**를 넣었을 때의 GAN
    # 시간이라 이 단계가 아예 빠져 있다.
    #
    # 낭비가 두 군데 있었다:
    #   1) align_face 는 predictor=None 이면 dlib .dat 를 매 호출 디스크에서
    #      새로 읽는다 (실측 -644ms)
    #   2) shape/color 로 넘기는 **참고사진은 바뀌지 않는데** 매번 다시 정렬된다.
    #      3장 -> 1장으로 줄이면 -1,713ms
    # 그래서 예측기를 한 번만 만들고, 정렬된 참고사진을 캐시하고, hf.swap 에는
    # align=False 로 이미 정렬된 텐서를 넘긴다.
    def _predictor(self):
        if self._shape_predictor is None:
            import dlib
            path = os.path.join(HAIRFAST_DIR, "pretrained_models", "ShapeAdaptor",
                                "shape_predictor_68_face_landmarks.dat")
            self._shape_predictor = dlib.shape_predictor(path)
        return self._shape_predictor

    def _align(self, rgb: np.ndarray):
        """RGB 배열 -> FFHQ 정렬된 (3,1024,1024) 텐서. 얼굴을 못 찾으면 예외."""
        import torchvision.transforms.functional as TF
        from utils.shape_predictor import align_face
        # align_face 는 리스트를 받아 리스트를 준다. 한 장만 넘긴다.
        return align_face([TF.to_tensor(rgb)], predictor=self._predictor())[0]

    def _aligned_ref(self, path):
        """참고사진의 정렬 결과. 파일이 안 바뀌므로 프로세스 수명 내내 캐시한다."""
        cached = self._aligned_ref_cache.get(path)
        if cached is None:
            cached = self._align(self._prepare_ref(path))
            self._aligned_ref_cache[path] = cached
        return cached

    def _prepare_ref(self, path):
        """참고사진을 얼굴 사진과 같은 기준(눈간격 TARGET_EYE_PX)으로 맞춘다.
        같은 파일이 반복 사용되므로 캐시한다."""
        cached = self._ref_cache.get(path)
        if cached is not None:
            return cached
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"참고사진을 읽을 수 없습니다: {path}")
        out = self._prepare(img)
        self._ref_cache[path] = out
        return out

    # ---------- 합성 ----------
    def swap(self, face_bgr: np.ndarray, shape_path: str, color_path: str = None,
             log=print):
        """웹캠 BGR 프레임 + 헤어 참고사진 경로 -> 합성된 BGR 이미지."""
        hf = self.ensure_loaded(log=log)
        color_path = color_path or shape_path
        # 아래에서 작업 디렉터리를 HairFastGAN 쪽으로 옮기므로 상대 경로는 깨진다.
        shape_path = os.path.abspath(shape_path)
        color_path = os.path.abspath(color_path)

        face_rgb = self._prepare(face_bgr)
        # 참고사진도 **같은 기준으로** 정규화한다.
        # 얼굴 사진만 정규화하고 참고사진을 원본 그대로 넣으면 두 이미지의
        # 얼굴-프레임 비율이 크게 어긋나고, HairFastGAN 의 pose alignment 가
        # 그 차이를 스케일로 흡수하지 못해 **헤어가 머리보다 크게 생성된다.**
        # (문서에 기록된 "참고사진과 대상사진의 프레임 내 얼굴 비율을 맞추라"는
        #  완화책을 사람이 손으로 하는 대신 여기서 자동으로 한다)
        # 추론 중에도 저장소가 상대경로로 가중치/캐시를 건드리므로 chdir 이 필요하다.
        # 정렬도 안에서 한다 - align_face 가 상대경로로 모델을 찾을 수 있고,
        # 참고사진 정렬은 첫 호출에만 도니 구간이 길어지지도 않는다.
        # _CHDIR_LOCK 설명은 모듈 상단 참고.
        with _CHDIR_LOCK:
            cwd = os.getcwd()
            os.chdir(HAIRFAST_DIR)
            try:
                t = time.perf_counter()
                # 참고사진은 캐시된 정렬 결과를 그대로 쓴다(두 번째 호출부터 0ms).
                shape_t = self._aligned_ref(shape_path)
                color_t = (shape_t if color_path == shape_path
                           else self._aligned_ref(color_path))
                # 얼굴은 매번 달라지므로 이것만 정렬한다.
                face_t = self._align(face_rgb)
                # 이미 정렬했으므로 align=False. True 로 두면 안에서 3장을
                # 다시 정렬해 2.7초가 그대로 붙는다.
                out = hf.swap(face_t, shape_t, color_t, align=False)
                took = time.perf_counter() - t
            finally:
                os.chdir(cwd)

        # align=False 면 최종 이미지만 온다(align=True 였을 때는 튜플이었다)
        final = out[0] if isinstance(out, tuple) else out

        arr = final.detach().float().clamp(0, 1).cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)      # CHW RGB -> HWC
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), took


REF_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def list_references():
    """참고사진 목록 {이름: 경로}. GAN 은 오려낸 에셋이 아니라 원본 사진이 필요하다.

    두 곳을 본다:
      server/references/          손으로 넣어 둔 것 (git 추적)
      server/references/uploads/  API 로 올라온 것 (git 제외 - 남의 얼굴 사진)
    이름이 겹치면 업로드가 이긴다.
    """
    out = {}
    for d in (REF_DIR, CONFIG.reference_upload_dir):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(REF_EXTS):
                out[os.path.splitext(fn)[0]] = os.path.join(d, fn)
    return out
