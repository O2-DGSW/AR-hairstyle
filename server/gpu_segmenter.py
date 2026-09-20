"""GPU 얼굴 파싱 세그멘테이션 (SegFormer / CelebAMask-HQ 19클래스).

MediaPipe(윈도우 Python에서 CPU 전용, 98ms)를 대체한다. 이 모델을 고른 이유:
  - PyTorch/CUDA 네이티브 -> 4070을 실제로 쓴다
  - 19클래스라 MediaPipe 6클래스보다 헤어라인 디테일이 훨씬 좋다
  - HairFastGAN 계열이 내부적으로 쓰는 face parsing과 같은 계보라
    나중에 GAN을 붙일 때 마스크 규격이 그대로 맞는다
  - 문서상 파인튜닝 1순위가 바로 이 네트워크다

성능 메모: 순수 eager 실행은 해상도를 512->224로 낮춰도 50ms에서 안 내려간다.
연산이 아니라 커널 실행 오버헤드에 묶여 있기 때문(작은 레이어가 아주 많은 구조).
CUDA 그래프로 실행 계획을 통째로 캡처하면 9ms대로 떨어진다 -> 5배 이상.
그래서 입력 크기를 고정(CONFIG.input_size)하고 그래프를 캡처해 재생하는 구조로 짰다.

주의: CUDA 그래프는 캡처한 스트림에 묶이므로 반드시 단일 스레드에서만
호출해야 한다. server.py가 전용 단일 워커 executor로 호출한다.
"""
import logging
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

from config import CONFIG

logger = logging.getLogger("segmenter")

# 아래 CLS_* 는 튜닝 값이 아니라 CelebAMask-HQ 19클래스의 **모델 규격**이다.
# 체크포인트가 바뀌지 않는 한 바뀔 수 없으므로 설정으로 뺄 이유가 없다.
CLS_BG = 0
CLS_SKIN = 1
CLS_NOSE = 2
CLS_EYE_G = 3             # 안경
CLS_EYE_L = 4
CLS_EYE_R = 5
CLS_BROW_L = 6
CLS_BROW_R = 7
CLS_HAIR = 13

# ImageNet 정규화 (SegformerImageProcessor 기본값과 동일)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# 오버레이 색 (BGR)
_COLOR_HAIR = (255, 0, 200)
_COLOR_SKIN = (0, 220, 255)
_ALPHA = 0.45


class SessionPlate:
    """연결(세션)마다 하나. "머리가 아닌 것"을 시간에 걸쳐 누적한다.

    Phase 1(가림 문제)의 핵심. 정지 사진에서는 머리 뒤에 뭐가 있었는지
    정보가 아예 없어서 인페인팅이 환각할 수밖에 없었다(LaMa/TELEA가 실패한
    이유). 영상에서는 사람이 조금만 움직여도 가려졌던 픽셀이 실제로 드러나므로,
    그 순간을 기록해두면 나중에 **실제로 관측된 픽셀**로 채울 수 있다.

    "배경"이 아니라 "머리가 아닌 것" 전부를 모으는 게 중요하다. 그래야 머리
    뒤가 배경이든 옷이든 목이든 귀든 똑같이 처리된다.
    """

    def __init__(self, device):
        self.device = device
        self.plate = None      # (H, W, 3) float32 - 마지막으로 본 "머리 아닌" 픽셀
        self.seen = None       # (H, W) bool     - 한 번이라도 본 적 있는가
        self.frames = 0

    def _ensure(self, h, w):
        if self.plate is None or self.plate.shape[0] != h or self.plate.shape[1] != w:
            self.plate = torch.zeros(h, w, 3, device=self.device, dtype=torch.float32)
            self.seen = torch.zeros(h, w, device=self.device, dtype=torch.bool)
            self._alpha = torch.tensor(0.15, device=self.device)
            self._one = torch.tensor(1.0, device=self.device)
            self.frames = 0

    def update(self, frame_f: torch.Tensor, non_hair: torch.Tensor, alpha: float = 0.15):
        """frame_f: (H,W,3) float32 BGR, non_hair: (H,W) bool

        갱신률을 픽셀별 맵 하나로 표현해서 in-place 두 번으로 끝낸다:
          머리인 곳            -> a=0    (건드리지 않음)
          처음 보는 비-머리     -> a=1    (그대로 기록)
          이미 본 비-머리       -> a=알파 (천천히 갱신, 조명 변화 추종)
        torch.where로 중간 텐서를 여러 개 만들면 프레임당 수 ms가 그냥 샌다.
        """
        h, w = frame_f.shape[:2]
        self._ensure(h, w)
        a = torch.where(self.seen, self._alpha, self._one)
        a = (a * non_hair).unsqueeze(-1)          # 머리인 곳은 0
        self.plate.mul_(1 - a).add_(frame_f * a)
        self.seen |= non_hair
        self.frames += 1

    def coverage_tensor(self, mask: torch.Tensor):
        """(채울 수 있는 픽셀 수, 전체 마스크 픽셀 수) - GPU 텐서로 반환.
        .item()은 GPU 동기화를 강제하므로 호출부에서 모아서 한 번만 한다."""
        return (mask & self.seen).sum(), mask.sum()


class GpuFaceParser:
    def __init__(self, use_cuda_graph: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = self.device == "cuda"

        # 리비전을 고정한다. 핀이 없으면 업스트림이 가중치나 config 를 갱신한 날
        # **코드를 한 줄도 안 고쳤는데 마스크가 달라진다**. 증류 블렌더는 이
        # 파서의 출력으로 학습됐으므로 그 순간 조용히 품질이 어긋난다.
        # 이 해시는 로컬 HF 캐시에 이미 받아둔 스냅샷이라 오프라인에서도 뜬다.
        model = SegformerForSemanticSegmentation.from_pretrained(
            CONFIG.model_id, revision=CONFIG.model_revision)
        model = model.to(self.device).eval()
        if self.half:
            model = model.half()
        self.model = model

        dt = torch.half if self.half else torch.float32
        self.mean = _MEAN.to(self.device, dt)
        self.std = _STD.to(self.device, dt)

        self._color = torch.zeros(19, 3, device=self.device, dtype=torch.float32)
        self._color[CLS_HAIR] = torch.tensor(_COLOR_HAIR, dtype=torch.float32)
        self._color[CLS_SKIN] = torch.tensor(_COLOR_SKIN, dtype=torch.float32)
        self._alpha_cls = torch.zeros(19, device=self.device, dtype=torch.float32)
        self._alpha_cls[CLS_HAIR] = _ALPHA
        self._alpha_cls[CLS_SKIN] = _ALPHA

        self._last_hair_px = 0
        self._stat_cache = None    # (dark, sigma) 캐시. 몇 프레임에 한 번만 갱신.
        self._stat_frame = -999
        self._last_coverage = None
        self._grid = None      # (ys, xs) 좌표 그리드 캐시 - 무게중심/워핑 공용
        # LRU. 512^2 RGBA float32 = 4MB/개이고 라이브 뱅크 한 번이 7개를 만든다.
        # 상한 없이 두면 세션을 몇 번 돌리는 것만으로 VRAM 이 수백 MB 샌다.
        self._asset_cache = OrderedDict()   # 에셋 이름 -> {"levels", "dark"} (_asset_entry)
        self._lum_w = torch.tensor([0.114, 0.587, 0.299], device=self.device)   # BGR 휘도
        self._blender = None    # 증류 블렌더 (load_blender 로 주입)
        self._cgrid = None      # 크롭 좌표 그리드 캐시
        self._groom = None      # GroomRenderer (3D 헤어카드). 처음 쓸 때 gpu_executor 스레드에서 생성.
        self._hair_color = None # 사용자 머리 평균색 (BGR 텐서, EMA). 그룸 색 맞춤용.
        self._hair_std = None   # 사용자 머리 휘도 표준편차 (EMA). 대비 맞춤용.

        # 추론 시간 측정용 CUDA 이벤트. 쌍을 두 조 두고 번갈아 쓴다 - 자세한
        # 이유는 _begin_infer() 주석 참고.
        self._ev_pairs = None
        self._ev_idx = 0
        self._ev_prev = None
        self._last_infer_ms = 0.0

        self.graph = None
        self._static_in = torch.zeros(
            1, 3, CONFIG.input_size, CONFIG.input_size, device=self.device, dtype=dt)
        if use_cuda_graph and self.device == "cuda":
            self._capture_graph()

    def _capture_graph(self):
        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    self.model(pixel_values=self._static_in)
            torch.cuda.current_stream().wait_stream(s)

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._static_out = self.model(pixel_values=self._static_in).logits

    @torch.no_grad()
    def class_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR 프레임 -> (H, W) 클래스 인덱스 numpy 배열. 에셋 추출 등 오프라인용."""
        h, w = frame_bgr.shape[:2]
        frame_t = torch.from_numpy(frame_bgr).to(self.device)
        rgb = frame_t[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = torch.nn.functional.interpolate(
            rgb, size=(CONFIG.input_size, CONFIG.input_size),
            mode="bilinear", align_corners=False)
        if self.half:
            x = x.half()
        x = (x - self.mean) / self.std

        if self.graph is not None:
            self._static_in.copy_(x)
            self.graph.replay()
            logits = self._static_out
        else:
            logits = self.model(pixel_values=x).logits

        cls = logits.argmax(dim=1)
        cls = torch.nn.functional.interpolate(
            cls.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).squeeze(0)
        return cls.to(torch.uint8).cpu().numpy()

    # ---------- 증류 블렌더 ----------
    def load_blender(self, path):
        """학습된 블렌더를 올린다. 없으면 조용히 비활성.

        이게 '실시간 GAN'의 실체다. HairFastGAN 자체는 장당 수 초라 프레임당
        실행이 불가능하므로, 그 출력을 교사로 삼아 증류한 소형 네트워크(0.47M,
        4ms)를 대신 돌린다.
        """
        if not os.path.isfile(path):
            return False
        import sys
        tdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")
        if tdir not in sys.path:
            sys.path.insert(0, tdir)
        from blender_net import BlenderUNet

        ck = torch.load(path, map_location=self.device)
        net = BlenderUNet().to(self.device).eval()
        net.load_state_dict(ck["model"])
        if self.half:
            net = net.half()
        self._blender = net
        return True

    def _blend_grids(self, eye_l, eye_r, h, w):
        """(프레임->크롭 샘플링 그리드, 크롭->프레임 샘플링 그리드)."""
        import hair_asset
        M = hair_asset.blend_crop_matrix(eye_l, eye_r)      # 프레임 -> 크롭
        if M is None:
            return None, None
        Minv = hair_asset.invert_affine(M)                  # 크롭 -> 프레임
        c = hair_asset.BLEND_CROP

        # 크롭 픽셀 -> 프레임 좌표 (크롭을 만들 때 쓰는 그리드)
        cy, cx = self._crop_grid(c)
        fx = Minv[0, 0] * cx + Minv[0, 1] * cy + Minv[0, 2]
        fy = Minv[1, 0] * cx + Minv[1, 1] * cy + Minv[1, 2]
        g_to_crop = torch.stack([(2 * fx + 1) / w - 1, (2 * fy + 1) / h - 1], -1).unsqueeze(0)

        # 프레임 픽셀 -> 크롭 좌표 (되돌릴 때 쓰는 그리드)
        ys, xs = self._ensure_grid(h, w)
        ux = M[0, 0] * xs + M[0, 1] * ys + M[0, 2]
        uy = M[1, 0] * xs + M[1, 1] * ys + M[1, 2]
        g_to_frame = torch.stack([(2 * ux + 1) / c - 1, (2 * uy + 1) / c - 1], -1).unsqueeze(0)
        return g_to_crop, g_to_frame

    def _crop_grid(self, c):
        if self._cgrid is None or self._cgrid[0].shape != (c, c):
            ys, xs = torch.meshgrid(
                torch.arange(c, device=self.device, dtype=torch.float32),
                torch.arange(c, device=self.device, dtype=torch.float32),
                indexing="ij")
            self._cgrid = (ys, xs)
        return self._cgrid

    def _apply_blender(self, out, eye_l, eye_r, h, w, strength=1.0):
        """합성 결과의 얼굴 영역만 잘라 블렌더에 통과시키고 되돌린다."""
        F = torch.nn.functional
        g_c, g_f = self._blend_grids(eye_l, eye_r, h, w)
        if g_c is None:
            return out

        src = (out.permute(2, 0, 1).unsqueeze(0) / 255.0).clamp(0, 1)
        crop = F.grid_sample(src, g_c, mode="bilinear", padding_mode="border",
                             align_corners=False)
        if self.half:
            crop = crop.half()
        with torch.no_grad():
            ref = self._blender(crop)
        ref = ref.float()

        back = F.grid_sample(ref, g_f, mode="bilinear", padding_mode="zeros",
                             align_corners=False)
        back = back.squeeze(0).permute(1, 2, 0) * 255.0

        # 이음매 처리.
        # 크롭의 사각 테두리에서 마스크가 0으로 떨어지면 아무리 흐려도 네모가
        # 그대로 보인다. 그래서 마스크를 **크롭 좌표계의 타원형 감쇠**로 만들어
        # 크롭 경계에 닿기 한참 전에 0이 되게 한다. 이러면 되돌린 뒤에도
        # 직선 경계가 생길 수 없다.
        m_crop = self._falloff(crop.shape[-1])
        m = F.grid_sample(m_crop, g_f, mode="bilinear",
                          padding_mode="zeros", align_corners=False)
        m = m.squeeze(0).permute(1, 2, 0).clamp(0, 1) * strength
        return out * (1 - m) + back * m

    def _falloff(self, c):
        """크롭 중심에서 1, 가장자리로 갈수록 0이 되는 부드러운 타원 마스크."""
        if getattr(self, "_fmask", None) is not None and self._fmask.shape[-1] == c:
            return self._fmask
        ys, xs = self._crop_grid(c)
        nx = (xs - c / 2) / (c / 2)
        ny = (ys - c / 2) / (c / 2)
        r = torch.sqrt(nx * nx + ny * ny)
        t = ((1.0 - r) / 0.45).clamp(0, 1)
        t = t * t * (3 - 2 * t)                 # smoothstep
        self._fmask = t.unsqueeze(0).unsqueeze(0)
        return self._fmask

    def _ensure_grid(self, h, w):
        if self._grid is None or self._grid[0].shape != (h, w):
            ys, xs = torch.meshgrid(
                torch.arange(h, device=self.device, dtype=torch.float32),
                torch.arange(w, device=self.device, dtype=torch.float32),
                indexing="ij")
            self._grid = (ys, xs)
        return self._grid

    #: 밉맵 단계 수 상한. 1024 스케일 에셋을 1/16 까지 (눈 간격 14px 상당).
    _PYRAMID_LEVELS = 5

    def _asset_entry(self, asset):
        """에셋 RGBA -> GPU 캐시 항목 {"levels": [텐서...], "dark": float}.

        levels[i] 는 원본을 1/2^i 로 **안티에일리어스** 축소한 프리멀티플라이드
        BGRA 텐서(1,4,H,W). 두 가지 문제를 여기서 한 번에 잡는다:

        1) 에일리어싱. 에셋은 1024 정렬 스케일(눈 간격 ~220px)인데 프레임은
           50~100px 라 워핑이 3~5배 축소다. grid_sample bilinear 는 축소에
           안티에일리어싱이 없어서 머리카락 고주파가 반짝이고 지글거린다.
           목표 배율에 가장 가까운 레벨에서 샘플링하면 남는 축소가 2배 미만이라
           bilinear 로 충분하다. 축소는 캐시 시점에 한 번만 한다.
        2) 경계 프린지. build_from_photo 는 스트레이트 알파(RGB 가 마스크 밖에서
           GAN 배경/피부색)를 만든다. 그대로 bilinear 하면 페더 경계에서 배경색이
           머리에 섞여 밝은 테두리가 생긴다(실측: 헤어 외곽에 주황/흰 띠).
           RGB 에 알파를 미리 곱해 두면 보간이 색을 끌어오지 않는다.

        "dark" 는 알파가 있는 픽셀의 어두운 분위수 휘도. 장면 블랙레벨 정합에 쓴다.
        """
        e = self._asset_cache.get(asset.name)
        if e is not None:
            self._asset_cache.move_to_end(asset.name)        # 최근 사용 표시
            return e

        F = torch.nn.functional
        arr = asset.rgba.astype(np.float32)                  # (Ha, Wa, 4) BGRA
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        a = t[:, 3:4] / 255.0
        pm = torch.cat([t[:, :3] * a, t[:, 3:4]], dim=1)
        # 다운샘플은 bicubic + antialias. box 평균(avg_pool)보다 통과대역이
        # 살아 있어 머리카락이 덜 뭉갠다(실측 헤어 HF box 1.29 vs bicubic 1.31,
        # 예전 에일리어스 1.41). 피라미드는 등록 시점(warm_asset)에 미리 구우므로
        # 실시간 경로 밖이라 이 비용(에셋당 ~12ms)은 체감되지 않는다. bicubic 은
        # 오버슈트로 음수가 나올 수 있어 클램프한다(프리멀티라 음수는 무의미).
        levels = [pm]
        while (len(levels) < self._PYRAMID_LEVELS
               and min(levels[-1].shape[-2:]) >= 32):
            nxt = F.interpolate(levels[-1], scale_factor=0.5, mode="bicubic",
                                align_corners=False, antialias=True)
            levels.append(nxt.clamp(min=0.0))

        # 어두운 분위수. 불투명 픽셀만, CPU numpy 로 한 번.
        m = arr[..., 3] > 128
        dark = 0.0
        if m.sum() >= 100:
            lum = arr[..., :3][m] @ np.array([0.114, 0.587, 0.299], np.float32)
            dark = float(np.quantile(lum, CONFIG.black_pct))

        e = {"levels": levels, "dark": dark}
        self._asset_cache[asset.name] = e
        # 지금 워핑 중인 에셋이 축출되면 안 되므로 넣고 나서 자른다.
        while len(self._asset_cache) > CONFIG.asset_cache_max:
            self._asset_cache.popitem(last=False)
        return e

    def _asset_tensor(self, asset):
        """호환용: 원본 해상도 프리멀티플라이드 텐서."""
        return self._asset_entry(asset)["levels"][0]

    def warm_asset(self, asset) -> None:
        """에셋(과 얼굴 패치)의 GPU 피라미드를 **미리** 만들어 캐시한다.

        피라미드 빌드는 에셋당 수 ms 인데, 이걸 _warp_asset 이 처음 부를 때
        (=고개를 돌려 그 각도 칸이 처음 화면에 뜰 때) 즉석으로 하면 그 프레임이
        45~70ms 로 튄다(실측). 실시간 경로에서 스터터로 보인다. 뱅크는 GAN
        생성(칸당 ~9초, 그동안 화면은 진행률로 덮임) 직후에 등록되므로, 거기서
        미리 구워 두면 실제로 워핑할 때는 항상 warm 이다.
        """
        self._asset_entry(asset)
        if getattr(asset, "face", None) is not None:
            self._asset_entry(asset.face)

    # ---------- 에셋 GPU 캐시 ----------
    def evict_asset(self, name: str) -> bool:
        """캐시된 에셋 텐서를 버린다. 있었으면 True.

        세션이 끝나면 그 세션이 GAN 으로 만든 에셋은 다시 쓰이지 않는다.
        레지스트리 쪽 축출 훅에서 이걸 불러야 VRAM 이 실제로 돌아온다
        (파이썬 쪽 참조가 남아 있으면 캐싱 얼로케이터가 반납하지 않는다).
        """
        hit = self._asset_cache.pop(name, None) is not None
        # 얼굴 패치는 <name>#face 키로 따로 캐시된다(asset_extract / _read_asset
        # 규칙). 본체와 같이 지워야 VRAM 이 실제로 돌아온다.
        self._asset_cache.pop(name + "#face", None)
        return hit

    def evict_assets(self, names) -> int:
        return sum(1 for n in names if self.evict_asset(n))

    def cache_stats(self) -> dict:
        n = 0
        for e in self._asset_cache.values():
            for t in e["levels"]:
                n += t.numel() * t.element_size()
        return {"assets": len(self._asset_cache), "bytes": int(n)}

    def gpu_stats(self) -> dict:
        """VRAM 사용량(바이트). CPU 실행이면 빈 dict.

        /metrics 가 부르는 경로라 여기서 예외가 나면 관측이 통째로 죽는다.
        드라이버 상태에 따라 torch 쪽이 던질 수 있으므로 전부 삼킨다.
        """
        if self.device != "cuda":
            return {}
        try:
            return {
                "device": torch.cuda.get_device_name(0),
                "allocated": int(torch.cuda.memory_allocated()),
                "reserved": int(torch.cuda.memory_reserved()),
            }
        except Exception:
            return {}

    def _warp_asset(self, asset, eye_l, eye_r, scale_mul, offset_up, h, w):
        """헤어 에셋을 GPU에서 워핑해 (rgb, alpha) 를 돌려준다.

        합성까지 하지 않고 분리해서 반환하는 이유: **새 헤어의 알파를 기존 머리
        제거 단계에서 먼저 써야** 하기 때문이다. 새 헤어가 덮을 자리는 지울 필요가
        없다(그리고 지우면 오히려 사고가 난다 - 아래 process() 주석 참고).

        CPU(cv2.warpAffine + numpy 블렌딩)로 하면 640x480 기준 11ms가 넘는다.
        grid_sample 한 번이면 1ms 수준.
        """
        import hair_asset
        M = hair_asset.similarity_matrix(asset.eye_l, asset.eye_r, eye_l, eye_r,
                                         scale_mul, offset_up)
        if M is None:
            return None, None
        Minv = hair_asset.invert_affine(M)

        # 에셋 px -> 프레임 px 배율. 이보다 크지 않은 밉맵 레벨을 고르면 남는
        # 축소가 2배 미만이라 bilinear 로 에일리어싱이 안 생긴다.
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        levels = self._asset_entry(asset)["levels"]
        lvl = 0
        while lvl + 1 < len(levels) and scale <= 0.5 ** (lvl + 1):
            lvl += 1
        at = levels[lvl]
        f = 0.5 ** lvl
        ha, wa = at.shape[2], at.shape[3]
        ys, xs = self._ensure_grid(h, w)

        # 출력 픽셀 좌표 -> 에셋 좌표 -> 레벨 좌표 -> grid_sample 정규화 좌표([-1,1])
        # 축소 레벨의 픽셀 중심은 (x + 0.5) * f - 0.5 에 놓인다(align_corners=False).
        xa = (Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2] + 0.5) * f - 0.5
        ya = (Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2] + 0.5) * f - 0.5
        gx = (2.0 * xa + 1.0) / wa - 1.0
        gy = (2.0 * ya + 1.0) / ha - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

        warped = torch.nn.functional.grid_sample(
            at, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if CONFIG.edge_feather > 0:
            # 프레임 스케일 페더. 에셋의 3px 페더는 4배 축소되면 1px 미만이라
            # 경계가 오려붙인 것처럼 딱딱해진다. 눈 간격에 비례한 폭으로 흐린다.
            # **프리멀티플라이드 4채널을 함께** 흐려야 한다. 알파만 흐리면 알파가
            # 새로 생긴 바깥 픽셀의 RGB 가 0(검정)이라 어두운 테두리가 생긴다.
            d = float(np.hypot(eye_r[0] - eye_l[0], eye_r[1] - eye_l[1]))
            k = int(round(d * CONFIG.edge_feather)) | 1
            if k >= 3:
                warped = torch.nn.functional.avg_pool2d(warped, k, stride=1, padding=k // 2)
        a = warped[:, 3:4] / 255.0                          # (1,1,h,w)
        # 프리멀티플라이드를 되돌린다. 알파 0 인 곳은 RGB 도 0 이고 합성에서
        # 알파가 곱해져 사라지므로 나눗셈 하한만 두면 된다.
        rgb = warped[:, :3] / a.clamp(min=1e-3)
        return rgb[0].permute(1, 2, 0), a[0].permute(1, 2, 0)

    def _face_zone(self, eye_l, eye_r, h, w):
        """눈 2점 기준으로 '얼굴/이마' 영역의 부드러운 마스크 (h, w) 0~1.

        배경 플레이트를 여기에 쓰면 안 되기 때문에 필요하다. 플레이트는 **화면
        좌표**로 누적되는데, 머리가 움직이면 같은 화면 위치가 어떤 순간엔 이마였다가
        어떤 순간엔 벽이 된다. 이마를 덮던 머리를 지울 때 그 자리에 기록된 '벽'을
        가져오면 이마에 벽이 뚫려 보인다. 이 영역 안에서는 플레이트 대신 피부톤을 쓴다.
        """
        c = ((eye_l[0] + eye_r[0]) * 0.5, (eye_l[1] + eye_r[1]) * 0.5)
        d = max(1.0, float(np.hypot(eye_r[0] - eye_l[0], eye_r[1] - eye_l[1])))
        ys, xs = self._ensure_grid(h, w)
        # 눈 위쪽으로 길게 뻗은 타원 (이마 + 정수리 방향)
        nx = (xs - c[0]) / (1.45 * d)
        ny = (ys - (c[1] - 0.75 * d)) / (1.60 * d)
        r = nx * nx + ny * ny
        return torch.clamp(1.6 - 1.6 * r, 0.0, 1.0)

    def _eye_protect_geo(self, eye_l, eye_r, h, w):
        """랜드마크 눈 2점 기준 보호 마스크 (h, w) 0~1. 눈마다 타원 하나.

        파싱이 눈을 '머리'로 오분류하는 프레임(앞머리+안경+뿌연 조명)에서도
        눈이 지워지지 않게 한다. 세부 근거는 CONFIG.protect_eye_rx 주석.
        """
        rx, ry = CONFIG.protect_eye_rx, CONFIG.protect_eye_ry
        if rx <= 0 or ry <= 0:
            return None
        d = max(1.0, float(np.hypot(eye_r[0] - eye_l[0], eye_r[1] - eye_l[1])))
        ys, xs = self._ensure_grid(h, w)
        # 눈 축 방향/수직 방향으로 분해해 고개가 기울어도 타원이 따라간다.
        ux, uy = (eye_r[0] - eye_l[0]) / d, (eye_r[1] - eye_l[1]) / d
        out = torch.zeros_like(xs)
        for ex, ey in (eye_l, eye_r):
            dx, dy = xs - float(ex), ys - float(ey)
            u = (dx * ux + dy * uy) / (rx * d)
            v = (-dx * uy + dy * ux) / (ry * d)
            r = u * u + v * v
            # r<1 안쪽 1, 1~1.35 에서 부드럽게 0 으로 (반지름 기준 1.16배까지).
            # 더 넓히면 눈썹 아래 앞머리 끝이 보호돼 지워지지 않고 남는다.
            t = ((1.35 - r) / 0.35).clamp(0.0, 1.0)
            out = torch.maximum(out, t * t * (3 - 2 * t))
        return out

    @staticmethod
    def _box(x, k):
        """(1,C,h,w) 분리형 박스 블러. 큰 커널도 O(k) 라 저주파 추출에 쓴다."""
        F = torch.nn.functional
        k = int(k) | 1
        if k < 3:
            return x
        x = F.avg_pool2d(x, (k, 1), stride=1, padding=(k // 2, 0))
        return F.avg_pool2d(x, (1, k), stride=1, padding=(0, k // 2))

    def _frame_stats(self, frame_f, cls, eyes=None):
        """(장면 블랙레벨 휘도, 피부 노이즈 σ) - 둘 다 GPU 스칼라 텐서.

        블랙레벨: **머리 주변** 휘도의 어두운 분위수. 플레어/역광으로 뿌연
        장면에서는 아무것도 진짜 검지 않아서 이 값이 40~80 까지 올라간다. GAN 이
        그린 새까만 머리를 그대로 얹으면 그것만 튄다. 프레임 전체를 쓰지 않는
        이유: 흰 벽/밝은 옷뿐인 또렷한 장면도 "검은 게 없다" 는 이유로 lift 가
        걸린다. 머리 주변(본인 머리, 눈썹, 콧구멍)은 항상 어두운 게 있어서
        플레어가 없으면 분위수가 낮게 나온다.
        노이즈: 피부 픽셀의 (원본 - 3x3 평균) 표준편차. 피부는 평탄해서
        고주파가 거의 전부 센서/압축 노이즈다.
        .item() 없이 텐서로만 돌려준다.
        """
        F = torch.nn.functional
        h, w = frame_f.shape[:2]
        lum = (frame_f @ self._lum_w).unsqueeze(0).unsqueeze(0)          # (1,1,h,w)
        d = 50.0
        region = lum
        if eyes is not None:
            el, er = eyes
            d = max(20.0, float(np.hypot(er[0] - el[0], er[1] - el[1])))
            cx, cy = float((el[0] + er[0]) / 2), float((el[1] + er[1]) / 2)
            x0, x1 = int(max(0, cx - 1.8 * d)), int(min(w, cx + 1.8 * d))
            y0, y1 = int(max(0, cy - 2.2 * d)), int(min(h, cy + 1.5 * d))
            if x1 - x0 >= 16 and y1 - y0 >= 16:
                region = lum[:, :, y0:y1, x0:x1]
        small = F.avg_pool2d(region, 4, stride=4)
        dark = torch.quantile(small.reshape(-1), CONFIG.black_pct)

        # 피부 마스크를 안쪽으로 깎는다. 경계(안경/눈/머리와 맞닿는 곳)와 피부
        # 안의 에지(코 주름, 입꼬리)의 잔차는 노이즈가 아니라서 그대로 넣으면
        # σ 가 1.5~2배 부푼다(실측: 침식 7px 1.5~1.7 vs 강건 추정 1.0). 침식
        # 폭을 얼굴 크기에 비례(0.3D)시키고, 프레임 밖은 '피부 아님' 으로 패딩해
        # 가장자리 픽셀이 침식을 피하지 못하게 한다. 잔차도 상한으로 자른다.
        ke = max(7, int(round(0.3 * d))) | 1
        skin = (cls == CLS_SKIN).float().unsqueeze(0).unsqueeze(0)
        not_skin = F.pad(1.0 - skin, (ke // 2,) * 4, value=1.0)
        skin = 1.0 - F.max_pool2d(not_skin, ke, stride=1)
        hp = lum - F.avg_pool2d(lum, 3, stride=1, padding=1, count_include_pad=False)
        n = skin.sum().clamp(min=1.0)
        var = (torch.clamp(hp * hp, max=100.0) * skin).sum() / n
        # 3x3 평균을 뺀 잔차의 분산은 원 노이즈 분산의 8/9 이다.
        sigma = torch.sqrt(var * 9.0 / 8.0)
        ok = (skin.sum() > 200).float()
        return dark, sigma * ok

    @staticmethod
    def _skin_tone(frame_f, cls):
        """보이는 얼굴 피부의 대표 색 (3,). GPU 동기화 없이 계산한다."""
        m = (cls == CLS_SKIN).unsqueeze(-1).float()
        n = m.sum()
        tone = (frame_f * m).sum(dim=(0, 1)) / n.clamp(min=1.0)
        # 피부가 거의 안 보이면 화면 평균으로 대체 (.item() 없이 GPU에서 분기)
        ok = (n > 200).float()
        return tone * ok + frame_f.mean(dim=(0, 1)) * (1.0 - ok)

    def _skin_plane(self, frame_f, cls, h, w):
        """보이는 피부의 **휘도 평면** L(x,y)=a+b·x+c·y 를 피팅해 (h,w,3) 톤
        필드로 돌려준다. 색도(chroma)는 평균 살색으로 고정하고 휘도 기울기만
        살린다.

        왜: 이마를 평균 살색 **한 색**으로 덮으면 스티커처럼 보인다. 실제 이마는
        중앙이 밝고 관자놀이·헤어라인으로 어두워지는 그라디언트가 있는데 상수는
        그걸 죽인다. 경계를 아무리 부드럽게 해도 안 없어지는 아티팩트다.

        전부 reduction + 3x3 해라 .item() 없이 GPU 에서 끝난다(_skin_tone 의
        무동기화 제약 유지). 기울기 (b,c) 는 사실상 조명 방향 단서라 나중에
        헤어 셰이딩에 재활용할 수 있다.
        """
        m = (cls == CLS_SKIN).float()
        n = m.sum()
        mean_color = self._skin_tone(frame_f, cls)                  # (3,)
        ys, xs = self._ensure_grid(h, w)
        # 조건수를 위해 좌표를 [-1,1] 로 정규화한다(x² 이 수십만이면 행렬이 상함).
        xn = (xs - w * 0.5) / (w * 0.5)
        yn = (ys - h * 0.5) / (h * 0.5)
        lum = frame_f @ self._lum_w                                 # (h,w)
        Sx = (m * xn).sum(); Sy = (m * yn).sum()
        Sxx = (m * xn * xn).sum(); Sxy = (m * xn * yn).sum(); Syy = (m * yn * yn).sum()
        SL = (m * lum).sum(); SxL = (m * xn * lum).sum(); SyL = (m * yn * lum).sum()
        A = torch.stack([torch.stack([n, Sx, Sy]),
                         torch.stack([Sx, Sxx, Sxy]),
                         torch.stack([Sy, Sxy, Syy])])
        A = A + torch.eye(3, device=self.device) * 1e-3            # 정칙화
        coef = torch.linalg.solve(A, torch.stack([SL, SxL, SyL]))  # [a,b,c]
        Lfield = coef[0] + coef[1] * xn + coef[2] * yn             # (h,w)
        meanL = SL / n.clamp(min=1.0)
        # 이마는 피팅 영역(볼·코) 위쪽이라 외삽이다. 평면을 멀리 밀면 폭주하니
        # 휘도 비율을 상한으로 자른다.
        ratio = (Lfield / meanL.clamp(min=1.0)).clamp(0.65, 1.35)
        ok = (n > 200).float()                                     # 피부 적으면 상수로
        ratio = ratio * ok + (1.0 - ok)
        return mean_color.view(1, 1, 3) * ratio.unsqueeze(-1)      # (h,w,3)

    def _centroids(self, cls, h, w):
        """눈/코/눈썹의 무게중심을 한 번의 전송으로 가져온다.

        별도 랜드마크 모델을 붙일 필요가 없다 - 19클래스 파싱에 이미 l_eye,
        r_eye, nose, brow가 들어있으므로 그 무게중심이 곧 앵커다. GPU에 이미
        있는 데이터라 추가 비용이 거의 없고, 모델을 하나 덜 돌린다.
        """
        ys, xs = self._ensure_grid(h, w)

        want = [CLS_EYE_L, CLS_EYE_R, CLS_NOSE, CLS_BROW_L, CLS_BROW_R, CLS_EYE_G]
        rows = []
        for c in want:
            m = (cls == c).float()
            n = m.sum()
            rows.append(torch.stack([(m * xs).sum(), (m * ys).sum(), n]))
        packed = torch.stack(rows).cpu().numpy()   # (6, 3) - 전송 1회

        out = {}
        for c, (sx, sy, n) in zip(want, packed):
            out[c] = (float(sx / n), float(sy / n), int(n)) if n >= CONFIG.min_anchor_px else None
        return out

    @staticmethod
    def eye_anchors(cent):
        """눈 2점을 (화면 왼쪽, 화면 오른쪽) 순으로 반환.

        클래스 라벨(l_eye/r_eye)은 인물 기준인 데다 모델이 가끔 뒤바꿔 붙인다.
        그대로 쓰면 그 프레임만 헤어가 180° 뒤집히므로 x좌표로 정렬한다.
        안경/감은 눈으로 눈이 안 잡히면 눈썹으로 대체.
        """
        import hair_asset
        l, r = cent.get(CLS_EYE_L), cent.get(CLS_EYE_R)
        if l and r:
            a, b = hair_asset.order_by_x((l[0], l[1]), (r[0], r[1]))
            return a, b, "eye"
        bl, br = cent.get(CLS_BROW_L), cent.get(CLS_BROW_R)
        if bl and br:
            # 눈썹은 눈보다 위에 있으므로 눈 위치로 조금 내려 보정
            dy = 0.22 * abs(br[0] - bl[0])
            a, b = hair_asset.order_by_x((bl[0], bl[1] + dy), (br[0], br[1] + dy))
            return a, b, "brow"
        return None, None, None

    # ---------- 추론 시간 측정 ----------
    def _begin_infer(self):
        """추론 구간의 시작 이벤트를 기록하고 이번 프레임의 이벤트 쌍을 돌려준다.

        예전엔 여기서 torch.cuda.synchronize() 를 불렀는데, 그건 **디바이스
        전체**를 멈춘다. GAN 워커가 같은 GPU 를 쓰는 이 서버에서는 통계 숫자
        하나 때문에 남의 스트림까지 같이 기다리게 되는 셈이다.
        이벤트는 자기 스트림에만 마커를 꽂으므로 그런 부작용이 없다.

        쌍을 두 조 두고 번갈아 쓰는 이유: 이번 프레임이 A 에 기록하는 동안
        직전 프레임의 B 를 읽는다. 한 조만 쓰면 아직 기록 중인 이벤트를
        읽으려 들어 값이 뒤섞인다.
        """
        if self.device != "cuda" or CONFIG.profile_blocking_sync:
            return None
        if self._ev_pairs is None:
            self._ev_pairs = [(torch.cuda.Event(enable_timing=True),
                               torch.cuda.Event(enable_timing=True)) for _ in range(2)]
        ev = self._ev_pairs[self._ev_idx]
        self._ev_idx ^= 1
        ev[0].record()
        return ev

    def _end_infer(self, ev, t_inf):
        """추론 구간 시간(ms). 이벤트 경로에서는 **직전 프레임 값**을 돌려준다.

        elapsed_time() 자체는 이벤트가 완료될 때까지 기다린다. 그래서 방금
        기록한 쌍을 바로 읽으면 synchronize 를 없앤 의미가 없다. 통계용
        숫자라 한 프레임 늦어도 상관없으므로, 이미 끝나 있는 직전 프레임
        것만 읽어 파이프라인을 전혀 막지 않는다(query() 는 논블로킹).
        """
        if ev is None:
            if self.device == "cuda":
                # 정밀 프로파일링 모드: 커널이 끝날 때까지 기다린 실제 벽시계 시간.
                torch.cuda.synchronize()
            return (time.perf_counter() - t_inf) * 1000
        ev[1].record()
        prev = self._ev_prev
        if prev is not None and prev[1].query():
            self._last_infer_ms = prev[0].elapsed_time(prev[1])
        self._ev_prev = ev
        return self._last_infer_ms

    @torch.no_grad()
    def process(self, frame_bgr: np.ndarray, plate: "SessionPlate" = None, mode: str = "seg",
                asset=None, scale_mul: float = 1.0, offset_up: float = 0.0, pose=None,
                harmonize: bool = True, shadow: float = 0.35,
                blend: float = 1.0, asset2=None, mix: float = 0.0,
                smoother=None, groom=None, groom_fit=None, groom_color=None, groom_dyn=None):
        """BGR 프레임 -> (합성된 BGR 프레임, 타이밍/상태 dict).

        groom: {"name","path","meta"} 3D 헤어카드 GLB. 주어지면 tryon 에서 2D 에셋
               워핑 대신 포즈 행렬로 렌더한다(_render_groom). 뒤 합성은 동일.
        groom_fit: (scale_mul, up_cm, fwd_cm) 세션 보정. json 의 user_fit 위에 곱/더한다.
        groom_color: (b,g,r) 0~255 또는 None. None 이면 사용자 머리색에 맞춘다.
        groom_dyn: hair_dynamics.HairDynamics.step() 결과(2차 운동 유니폼). None 이면 강체.

        mode:
          raw    - 원본 그대로 (기본). 플레이트는 계속 쌓는다.
          tryon  - 기존 머리 제거 + 새 헤어 씌우기 (프로덕션 화면)
          seg    - 세그멘테이션 색칠 (머리=마젠타, 얼굴피부=시안)  [디버그]
          remove - 기존 머리를 누적된 플레이트로 지움              [디버그]
          plate  - 누적된 플레이트 자체를 표시                     [디버그]

        어느 모드든 plate.update() 는 돈다. 플레이트는 시간에 걸쳐 쌓이는
        것이라, 나중에 tryon 으로 바꿨을 때 바로 쓰려면 그 전부터 채워둬야 한다.
        """
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]

        # --- 전처리 (GPU) ---
        t_pre = time.perf_counter()
        frame_t = torch.from_numpy(frame_bgr).to(self.device, non_blocking=True)
        rgb = frame_t[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = torch.nn.functional.interpolate(
            rgb, size=(CONFIG.input_size, CONFIG.input_size),
            mode="bilinear", align_corners=False
        )
        if self.half:
            x = x.half()
        x = (x - self.mean) / self.std
        pre_ms = (time.perf_counter() - t_pre) * 1000

        # --- 추론 ---
        t_inf = time.perf_counter()
        ev = self._begin_infer()
        if self.graph is not None:
            self._static_in.copy_(x)
            self.graph.replay()
            logits = self._static_out
        else:
            logits = self.model(pixel_values=x).logits
        inf_ms = self._end_infer(ev, t_inf)

        # --- 마스크 -> 합성 (전부 GPU에서) ---
        t_post = time.perf_counter()
        F = torch.nn.functional

        # 로짓은 입력의 1/4 해상도(=128x128)로 나온다. 여기서 바로 argmax를 하고
        # nearest로 확대하면 한 칸이 출력 4~5픽셀이 되어 계단이 그대로 보인다.
        # **먼저 로짓을 bilinear로 확대한 뒤 argmax** 하면 경계가 로짓의 연속적인
        # 변화를 따라가서 훨씬 촘촘해진다. 비용은 0.1ms 수준으로 사실상 공짜.
        lo = logits.float()
        cls = F.interpolate(lo, size=(h, w), mode="bilinear",
                            align_corners=False).argmax(dim=1).squeeze(0)   # (h, w)

        # 머리 경계용 소프트 알파: (머리 로짓 - 나머지 최대) 를 1채널로 확대해
        # sigmoid. 이진 마스크로 자르면 아무리 해상도를 올려도 경계가 딱딱하다.
        hair_l = lo[:, CLS_HAIR:CLS_HAIR + 1]
        other_l = lo.clone()
        other_l[:, CLS_HAIR] = -1e4
        margin = F.interpolate(hair_l - other_l.max(dim=1, keepdim=True).values,
                               size=(h, w), mode="bilinear", align_corners=False)
        hair_a = torch.sigmoid(margin.squeeze(0).squeeze(0) * CONFIG.soft_k)        # (h, w) 0~1

        frame_f = frame_t.float()
        hair = hair_a > 0.5

        if plate is not None:
            # 플레이트에는 **배경만** 기록한다.
            # 처음엔 "머리가 아닌 것 전부"를 모았는데, 그러면 본인의 얼굴과 몸도
            # 같이 들어간다. 고개를 움직이면 조금 전 얼굴이 있던 자리의 픽셀이
            # 플레이트에 남아 있다가 그대로 칠해져서 '자기 얼굴의 잔상'이 된다.
            # 배경만 모으면 사람이 절대 기록되지 않으므로 그 잔상이 사라진다.
            # 대신 머리가 옷/목 위에 걸친 부분은 채울 데이터가 없어지는데,
            # 근거 없이 칠하느니 원본을 남기는 쪽이 낫다.
            plate.update(frame_f, (cls == CLS_BG) & (hair_a < 0.2))

        # .item()은 그때마다 GPU 동기화를 강제한다. 매 프레임 세 번씩 하면
        # 프레임당 십수 ms가 그냥 샌다 -> 몇 프레임에 한 번만 재고 나머지는 캐시.
        want_stats = plate is None or plate.frames % CONFIG.stats_every == 0
        if want_stats:
            hair_sum = hair.sum()
            if plate is not None:
                cov_ok, cov_all = plate.coverage_tensor(hair)
                packed = torch.stack([hair_sum, cov_ok, cov_all]).cpu()
                self._last_hair_px = int(packed[0])
                self._last_coverage = (float(packed[1]) / float(packed[2])
                                       if float(packed[2]) > 0 else 1.0)
            else:
                self._last_hair_px = int(hair_sum.item())
                self._last_coverage = None

        # 눈 앵커. remove/tryon 둘 다 필요하다 - tryon 은 헤어 정합에, remove 는
        # 얼굴 영역을 알아야 거기에 플레이트를 안 쓸 수 있기 때문.
        # (파서는 세션 공용 싱글턴이므로 인스턴스에 담지 않고 지역 변수로 다룬다)
        anchor_src = None
        harmonized = False
        blended = False
        eyes = None
        new_rgb = new_a = None
        face_rgb = face_a = None   # GAN 이 그려 준 이마/눈썹 패치
        sigma_f = None             # 프레임 노이즈 σ (그레인 정합용, tryon 에서만)

        if mode in ("remove", "tryon"):
            # 앵커는 랜드마크를 우선한다.
            #
            # 세그멘테이션 눈 무게중심도 매 프레임 사실상 공짜로 얻을 수 있어
            # 처음엔 그걸 썼는데, 눈 마스크가 수백 픽셀짜리 작은 덩어리라
            # 경계가 프레임마다 흔들리고 깜빡이면 통째로 사라진다. 작은 덩어리의
            # 무게중심은 랜드마크보다 훨씬 심하게 떨려서, 하류에서 아무리
            # 평활화해도 랜드마크만큼 안정되지 않는다(로컬 워핑 테스트가
            # 서버 경로보다 매끈했던 이유가 이것이다).
            # 좌표계는 둘 다 이미지 픽셀이라 그대로 바꿔 끼울 수 있다.
            eye_l = eye_r = None
            if pose is not None:
                # 거리 보정을 **앵커 자체에** 반영한다. 눈 중점과 축 방향은
                # 관측값 그대로 두고 두 점 사이 거리만 d_corrected 로 바꾼다.
                #
                # 예전에는 관측 눈 2점을 그대로 앵커로 쓰고, 크기만 나중에
                # gain = d_corrected/d_measured 로 곱했다. 그런데 앵커는 아래에서
                # **평활화**되고 gain 은 평활화되지 않은 순간값이라, 최종 배율이
                #     (d_smoothed / D_asset) x (d_corrected / d_measured)
                # 즉 d_smoothed / d_measured 에 비례했다. 정지 상태에서는 둘이
                # 같아 상쇄되지만 고개를 빠르게 돌리면 d_measured 는 즉시 줄고
                # (50->25) 평활화된 값은 뒤따라가느라 45 쯤에 머문다. 그 순간
                # 배율이 1.8배로 튀어 헤어가 갑자기 커졌다(녹화로 확인).
                # 완전 측면에서 d_measured -> 0 이면 아예 발산한다.
                #
                # 보정을 앵커에 먼저 넣고 그 결과를 평활화하면 값의 출처가
                # 하나가 되어 이 불일치가 원천적으로 사라진다.
                from face_pose import eyes_scaled
                eye_l, eye_r = eyes_scaled(pose)
                anchor_src = "landmark +거리보정"
            if eye_l is None:
                # 얼굴을 놓친 프레임: 세그멘테이션 쪽으로 버틴다.
                cent = self._centroids(cls, h, w)
                eye_l, eye_r, anchor_src = self.eye_anchors(cent)
                if anchor_src:
                    anchor_src += "(폴백)"
            if eye_l is not None:
                # 시간축 평활화. 세그멘테이션 무게중심은 매 프레임 떨리고,
                # 거리 보정 배율은 몇 프레임에 한 번만 갱신돼 계단식으로 튄다.
                # 최종 앵커에서 잡아야 두 원인이 한 번에 흡수된다.
                if smoother is not None:
                    eye_l, eye_r = smoother.update(eye_l, eye_r)
                eyes = (eye_l, eye_r)

            # 거리 보정은 위에서 앵커에 이미 들어갔다(eyes_scaled). 여기서 또
            # 곱하면 이중 적용이다. 세그멘테이션 폴백 경로는 pose 가 없어서
            # 보정할 근거 자체가 없으므로 그대로 1.0 이다.
            gain = 1.0

            groom_ms = 0.0
            if mode == "tryon" and groom is not None and pose is not None                     and pose.get("matrix") is not None:
                # 3D 그룸: 그 프레임의 포즈 행렬로 직접 래스터라이즈. 평면 밖 회전이
                # 그대로 나오므로 각도 뱅크(asset2/mix)가 필요 없다. 조명 정합(ratio/lift)
                # 대신 아래 색 맞춤이 사용자 머리색(=그 조명 아래의 색)에 맞춘다.
                new_rgb, new_a = self._render_groom(groom, pose["matrix"], groom_fit, h, w, groom_dyn)
                groom_ms = self._groom.last_ms if self._groom is not None else 0.0
                # 이마 패치: 앞머리 있는 사람이 앞머리 없는 스타일을 입으면 지운 자리가
                # 살색 평면으로 남아 이질적이다. 촬영(GAN, 입력에서 앞머리를 미리 걷어냄)이
                # 재구성한 맨이마(asset.face)를 2D 경로와 똑같이 눈 앵커로 워핑해 쓴다.
                # 헤어는 3D 지만 이마는 거의 평면이라 닮음변환으로 충분하다.
                if asset is not None and getattr(asset, "face", None) is not None and eyes is not None:
                    def _face_patch(a_):
                        r_, al_ = self._warp_asset(a_.face, eye_l, eye_r,
                                                   scale_mul * a_.scale_adjust, offset_up, h, w)
                        if r_ is not None and harmonize and a_.ref_skin is not None:
                            cur = self._skin_tone(frame_f, cls)
                            ref = torch.as_tensor(a_.ref_skin, device=self.device, dtype=torch.float32)
                            fr = (cur / ref.clamp(min=1.0)).clamp(CONFIG.harmonize_min, CONFIG.harmonize_max)
                            r_ = (r_ * fr.view(1, 1, 3)).clamp(0, 255)
                        return r_, al_
                    face_rgb, face_a = _face_patch(asset)
                    harmonized = face_rgb is not None and harmonize and asset.ref_skin is not None
                    # 주기 갱신 직후: 이전 패치(asset2)와 크로스페이드. 같은 앵커로 워핑되므로
                    # 겹침 없이 섞인다. 프리멀티플라이드로 섞어야 경계가 어두워지지 않는다.
                    if (face_rgb is not None and asset2 is not None and mix > 0.001
                            and getattr(asset2, "face", None) is not None):
                        rgb2, a2 = _face_patch(asset2)
                        if rgb2 is not None:
                            pa = face_a * (1.0 - mix)
                            pb = a2 * mix
                            tot = pa + pb
                            face_rgb = (face_rgb * pa + rgb2 * pb) / tot.clamp(min=1e-4)
                            face_a = tot
                if new_rgb is not None:
                    new_rgb = self._match_hair_color(new_rgb, new_a, frame_f, hair_a,
                                                     want_stats, groom_color, pose=pose, eyes=eyes)
                    if CONFIG.groom_feather > 0 and eyes is not None:
                        # 리본 가장자리 페더. MSAA 만으로는 원래 머리 경계(소프트 알파)보다
                        # 딱딱해 오려붙인 티가 난다. 눈 간격 비례 폭으로 프리멀티플라이드 블러.
                        d = float(np.hypot(eye_r[0] - eye_l[0], eye_r[1] - eye_l[1]))
                        k = int(round(d * CONFIG.groom_feather)) | 1
                        if k >= 3:
                            pm = torch.cat([new_rgb * new_a, new_a], -1).permute(2, 0, 1).unsqueeze(0)
                            pm = F.avg_pool2d(pm, k, stride=1, padding=k // 2)[0].permute(1, 2, 0)
                            new_a = pm[:, :, 3:4]
                            new_rgb = pm[:, :, :3] / new_a.clamp(min=1e-3)
            elif mode == "tryon" and asset is not None and eyes is not None:
                # 에셋에 구워진 크기 보정까지 함께 적용한다.
                new_rgb, new_a = self._warp_asset(
                    asset, eye_l, eye_r,
                    scale_mul * gain * asset.scale_adjust, offset_up, h, w)

                # 다각도 뱅크: 인접한 두 각도를 섞어 전환을 연속적으로 만든다.
                # 가장 가까운 하나만 쓰면 10도 간격에서 하드 전환이 일어나
                # 뚝뚝 끊긴다.
                if asset2 is not None and mix > 0.001 and new_rgb is not None:
                    rgb2, a2 = self._warp_asset(
                        asset2, eye_l, eye_r,
                        scale_mul * gain * asset2.scale_adjust, offset_up, h, w)
                    if rgb2 is not None:
                        # 알파로 가중한 뒤 나눠야(프리멀티플라이드) 경계에서
                        # 어두운 테두리가 생기지 않는다.
                        pa = new_a * (1.0 - mix)
                        pb = a2 * mix
                        tot = pa + pb
                        new_rgb = (new_rgb * pa + rgb2 * pb) / tot.clamp(min=1e-4)
                        new_a = tot

                # --- 조명 정합 ---
                # (1) 곱셈 배율: 피부 평균색 비율. 화이트밸런스/노출 차이를 잡는다.
                # (2) 블랙레벨 lift: 곱셈은 검은 것을 밝힐 수 없다. GAN 은 노출을
                #     '정상'으로 되돌려 그리므로(실측: 플레어로 뿌연 프레임 ->
                #     GAN 결과는 또렷한 어두운 머리) 뿌연 장면에 얹으면 머리만
                #     새까맣게 튄다. 프레임의 어두운 분위수와 에셋 머리의 어두운
                #     분위수 차이만큼 그림자를 들어올린다. out = x + lift*(1-x/255)
                #     - 밝은 곳은 덜, 어두운 곳은 더 (플레어/안개 모델).
                ratio = None
                lift = None
                dark_f = sigma_f = None
                if new_rgb is not None and (CONFIG.black_match > 0 or CONFIG.grain_match > 0):
                    # 블랙레벨/노이즈 σ 는 조명·카메라 특성이라 프레임마다 거의
                    # 안 변한다. 매 프레임 quantile+블러를 돌리면 비싸고(스파이크
                    # 유발) 이득이 없어, 히스테리시스처럼 stats_every 프레임에 한
                    # 번만 재고 캐시한다. 첫 프레임은 무조건 잰다.
                    fidx = plate.frames if plate is not None else self._stat_frame + CONFIG.stats_every
                    if self._stat_cache is None or fidx - self._stat_frame >= CONFIG.stats_every:
                        self._stat_cache = self._frame_stats(frame_f, cls, eyes)
                        self._stat_frame = fidx
                    dark_f, sigma_f = self._stat_cache
                if new_rgb is not None and harmonize and asset.ref_skin is not None:
                    cur = self._skin_tone(frame_f, cls)
                    ref = torch.as_tensor(asset.ref_skin, device=self.device,
                                          dtype=torch.float32)
                    ratio = (cur / ref.clamp(min=1.0)).clamp(CONFIG.harmonize_min, CONFIG.harmonize_max)
                    new_rgb = (new_rgb * ratio.view(1, 1, 3)).clamp(0, 255)
                    harmonized = True
                if new_rgb is not None and CONFIG.black_match > 0:
                    dark_a = self._asset_entry(asset)["dark"]
                    if ratio is not None:
                        dark_a = dark_a * ratio.mean()
                    lift = ((dark_f - dark_a) * CONFIG.black_match).clamp(0.0, CONFIG.black_lift_max)
                    new_rgb = new_rgb + lift * (1.0 - new_rgb / 255.0)

                # 얼굴 패치. 기존 머리를 지운 자리를 평균 살색 대신 이걸로 덮는다.
                # 머리와 **같은 변환**을 써야 눈/눈썹 위치가 어긋나지 않는다.
                if getattr(asset, "face", None) is not None:
                    face_rgb, face_a = self._warp_asset(
                        asset.face, eye_l, eye_r,
                        scale_mul * gain * asset.scale_adjust, offset_up, h, w)
                    if face_rgb is not None and ratio is not None:
                        # 같은 사진에서 나왔으니 헤어와 같은 조명 보정을 받는다.
                        face_rgb = (face_rgb * ratio.view(1, 1, 3)).clamp(0, 255)
                    if face_rgb is not None and lift is not None:
                        face_rgb = face_rgb + lift * (1.0 - face_rgb / 255.0)

        # --- 눈/눈썹 보호 ---
        # 새 헤어 알파에서 이 영역을 깎아 둔다. 안 그러면 에셋에 앞머리가 있을 때
        # 눈까지 통째로 덮어서 마네킹처럼 보인다. 파싱이 실제로 '눈'으로 분류한
        # 픽셀에만 적용되므로, 원래 앞머리에 가려 안 보이던 눈썹은 여기에 안 잡힌다
        # (애초에 관측된 적이 없어 되살릴 수 없다).
        protect = None
        if mode == "tryon" and (CONFIG.protect_eyes > 0 or CONFIG.protect_brows > 0):
            p = torch.zeros_like(hair_a)
            if CONFIG.protect_eyes > 0:
                eye_m = ((cls == CLS_EYE_L) | (cls == CLS_EYE_R) |
                         (cls == CLS_EYE_G)).float()
                p = torch.maximum(p, eye_m * float(CONFIG.protect_eyes))
            if CONFIG.protect_brows > 0:
                brow_m = ((cls == CLS_BROW_L) | (cls == CLS_BROW_R)).float()
                p = torch.maximum(p, brow_m * float(CONFIG.protect_brows))
            k = max(1, int(CONFIG.protect_blur_k) | 1)          # 홀수로
            p = F.avg_pool2d(p.unsqueeze(0).unsqueeze(0), k, stride=1,
                             padding=k // 2).squeeze(0).squeeze(0)
            if eyes is not None and CONFIG.protect_eyes > 0:
                # 파싱과 무관한 기하학적 눈 보호. 파서가 눈을 '머리'로 오분류하는
                # 프레임에서 눈이 지워지고 GAN 얼굴로 대체되는 사고를 막는다.
                geo = self._eye_protect_geo(eyes[0], eyes[1], h, w)
                if geo is not None:
                    p = torch.maximum(p, geo * float(CONFIG.protect_eyes))
            protect = p.clamp(0.0, 1.0)
            if new_a is not None:
                new_a = new_a * (1.0 - protect).unsqueeze(-1)

        if mode in ("remove", "tryon") and plate is not None and plate.seen is not None:
            # 기존 머리를 "그 자리에서 실제로 관측된 적 있는" 픽셀로 채운다.
            # 두 가지 안전장치가 없으면 사고가 난다:
            #
            # (1) 새 헤어가 덮을 자리는 지우지 않는다. 어차피 가려지므로 지울 이유가
            #     없고, 지우면 새 헤어의 빈틈으로 엉뚱한 픽셀이 비친다.
            # (2) 얼굴/이마 영역에는 플레이트를 쓰지 않는다. 플레이트는 화면 좌표로
            #     누적되므로, 머리가 움직이면 이마 자리에 '벽'이 기록돼 있을 수 있다.
            #     그대로 채우면 이마에 벽이 뚫린 것처럼 보인다. 여기서는 피부톤을 쓴다.
            erase = hair_a
            if new_a is not None:
                erase = erase * (1.0 - new_a.squeeze(-1))

            fill = plate.plate
            src_ok = plate.seen.float()

            if protect is not None:
                # 보이는 눈/눈썹 위에 피부톤을 칠하지 않는다.
                erase = erase * (1.0 - protect)

            if eyes is not None:
                zone = self._face_zone(eyes[0], eyes[1], h, w)          # (h,w) 0~1
                if new_a is not None:
                    # **새 헤어보다 위쪽은 얼굴이 아니라 배경이다.**
                    # _face_zone 은 눈 위 2.35d 까지 뻗는데(불투명 구간만 1.73d),
                    # 실측한 정수리 높이가 1.84d 라 타원이 머리를 넘어선다. 거기까지
                    # 피부톤을 쓰면 원래 머리가 새 헤어보다 클 때 그 차이만큼
                    # **살색 덩어리**가 헤어 위에 뜬다.
                    #
                    # 열마다 "이 픽셀 위쪽에서 새 헤어를 만난 적이 있는가"를
                    # 누적최대로 구해서 그 아래로만 피부톤을 허용한다. 위쪽은
                    # 아래 src_ok 가 플레이트 관측 여부를 그대로 따르므로
                    # 배경으로 채워진다. (.item() 없이 GPU 에서 끝난다)
                    below = torch.cummax(new_a.squeeze(-1), dim=0).values
                    zone = zone * below
                # 얼굴 zone 안에서는 erase 를 단단하게 만든다. 원래 앞머리 경계
                # 픽셀은 소프트 알파가 0.3~0.7 라 절반만 지워져서 이마에 어두운
                # 테두리가 남았다(실측: 지운 이마가 얼룩덜룩). 이 구간을 0~1 로
                # 다시 펴고 1px 팽창해 확실히 지운다. zone 밖(배경/옷 위)은
                # 플레이트 관측이 필요한 곳이라 그대로 둔다.
                lo, hi = CONFIG.erase_hard_lo, CONFIG.erase_hard_hi
                if hi > lo:
                    hard = ((hair_a - lo) / (hi - lo)).clamp(0.0, 1.0)
                    hard = F.max_pool2d(hard.unsqueeze(0).unsqueeze(0), 3, stride=1,
                                        padding=1).squeeze(0).squeeze(0)
                    if new_a is not None:
                        hard = hard * (1.0 - new_a.squeeze(-1))
                    if protect is not None:
                        hard = hard * (1.0 - protect)
                    erase = erase * (1.0 - zone) + hard * zone
                # 상수 살색 대신 휘도 평면 필드(스티커 방지). 패치가 없거나
                # 패치 알파가 낮은 자리의 폴백이 이걸로 그라디언트를 받는다.
                tone = (self._skin_plane(frame_f, cls, h, w) if CONFIG.skin_plane
                        else self._skin_tone(frame_f, cls).view(1, 1, 3))
                # 얼굴 패치가 있으면 그걸 먼저 쓴다. 평균 살색은 패치가 없거나
                # 패치가 안 닿는 자리에만 남는 폴백이다.
                #
                # 평균색이 나빴던 이유: 볼/턱/목이 섞여 이마 색과 안 맞고, 평평해서
                # 음영이 없어 스티커처럼 보였다. 무엇보다 앞머리에 가려져 있던
                # 눈썹 자리까지 살색이 됐다. 패치는 GAN 이 그 사람 피부톤과 조명으로
                # 그린 이마라 셋 다 해결된다.
                face_fill = tone
                if face_rgb is not None:
                    fa = face_a                                    # (h,w,1) 0~1
                    if CONFIG.face_lf_match > 0:
                        # 패치의 저주파(색/큰 음영)를 **실제 피부**의 것으로 바꾼다.
                        # GAN 이마는 색이 조금 다르고(뿌연 장면이면 더 어둡다)
                        # 앞머리 자리에 얼룩이 남기도 하는데 둘 다 저주파다.
                        # 눈썹/잔 음영 같은 세부는 패치 것을 남긴다.
                        # 실제 피부 색 필드는 '지워지지 않는 피부' 픽셀만 넣은
                        # 정규화 컨볼루션으로 이마까지 외삽한다.
                        d = float(np.hypot(eyes[1][0] - eyes[0][0], eyes[1][1] - eyes[0][1]))
                        k = int(round(d * CONFIG.face_lf_k)) | 1
                        skin_m = (((cls == CLS_SKIN) | (cls == CLS_NOSE)).float()
                                  * (1.0 - hair_a)).unsqueeze(0).unsqueeze(0)
                        fr = frame_f.permute(2, 0, 1).unsqueeze(0)
                        num = self._box(fr * skin_m, k)
                        den = self._box(skin_m, k)
                        skin_lf = num / den.clamp(min=1e-3)
                        # 근거(창 안의 실제 피부 비율)가 적을수록 보정을 서서히
                        # 줄인다. 하드 임계면 그 경계에서 수십 레벨이 한 픽셀에
                        # 끊겨 이마에 가로 띠가 생긴다.
                        have = ((den - 0.02) / 0.10).clamp(0.0, 1.0)
                        pf = face_rgb.permute(2, 0, 1).unsqueeze(0)
                        wa = face_a.permute(2, 0, 1).unsqueeze(0)
                        face_lf = self._box(pf * wa, k) / self._box(wa, k).clamp(min=1e-3)
                        corr = (skin_lf - face_lf) * have * CONFIG.face_lf_match
                        face_rgb = (pf + corr).squeeze(0).permute(1, 2, 0).clamp(0, 255)
                    face_fill = face_rgb * fa + tone * (1.0 - fa)
                if sigma_f is not None and CONFIG.grain_match > 0:
                    # 그레인은 패치 유무와 무관하게 얹는다. 채운 자리가 노이즈 0
                    # 이면 실제 카메라 노이즈와 대비돼 튀는 건 평균색 폴백도 같다.
                    face_fill = face_fill + torch.randn(h, w, 1, device=self.device) * (
                        sigma_f * CONFIG.grain_match)
                fill = fill * (1.0 - zone).unsqueeze(-1) + face_fill * zone.unsqueeze(-1)
                # 얼굴 영역은 플레이트 관측 여부와 무관하게 채울 수 있다
                src_ok = torch.clamp(src_ok + zone, 0.0, 1.0)

            a = (erase * src_ok).unsqueeze(-1)
            out = frame_f * (1 - a) + fill * a
        elif mode == "plate" and plate is not None and plate.seen is not None:
            out = plate.plate * plate.seen.unsqueeze(-1)
        elif mode == "raw":
            # 원본을 그대로 내보낸다. 라이브 뱅크가 각도를 모으는 동안 쓰는
            # 화면이다 - 아직 씌울 헤어가 없으니 보여줄 것도 없다.
            #
            # 그런데 세그멘테이션은 **건너뛰지 않는다.** 위에서 plate.update()
            # 가 이미 돌았고, 그게 기존 머리를 지울 때 쓰는 배경 플레이트를
            # 채운다. 여기서 GPU 를 아끼면 tryon 으로 넘어가는 순간 플레이트가
            # 비어 있어 원래 머리가 안 지워진다. 게다가 수집 단계는 사용자가
            # 고개를 크게 돌리는 구간이라 플레이트를 채우기에 가장 좋은 시간이다.
            out = frame_f
        else:
            color = self._color[cls]                       # (h, w, 3) BGR
            a = self._alpha_cls[cls].unsqueeze(-1)         # (h, w, 1)
            out = frame_f * (1 - a) + color * a

        # --- 새 헤어 합성 (앞에서 워핑해 둔 것을 여기서 얹는다) ---
        if new_a is not None:
            if shadow > 0:
                # 헤어라인 그림자.
                # 실제 머리카락은 이마에 그늘을 드리운다. 그게 없으면 아무리 경계를
                # 부드럽게 해도 "떠 있는" 느낌이 남는다. 알파를 흐린 것에서 원래
                # 알파를 빼면 헤어 바깥에 딱 붙은 띠가 나오는데, 그 자리를 살짝
                # 어둡게 해서 접지감을 만든다.
                a2 = new_a.permute(2, 0, 1).unsqueeze(0)          # (1,1,h,w)
                # 띠 폭은 얼굴 크기에 비례해야 한다. 픽셀 고정이면 720p 에서는
                # 480p 의 절반 폭이 되어 그림자가 사라진다. shadow_k 는 눈 간격
                # 50px 기준값이다.
                sk = CONFIG.shadow_k
                if eyes is not None:
                    d = float(np.hypot(eyes[1][0] - eyes[0][0], eyes[1][1] - eyes[0][1]))
                    sk = max(3, int(round(CONFIG.shadow_k * d / 50.0))) | 1
                blurred = F.avg_pool2d(a2, sk, stride=1, padding=sk // 2)
                band = (blurred - a2).clamp(min=0.0).squeeze(0).permute(1, 2, 0)
                out = out * (1.0 - shadow * band)

            if CONFIG.hair_sharpen > 0:
                # 언샤프 마스크. 두 곳에서 디테일이 깎인다: (1) 웹캠 입력이 눈
                # 간격 ~50px 라 GAN 입력이 4배 업스케일된다 (2) 에셋을 프레임에
                # 얹을 때 안티에일리어스 축소가 통과대역을 조금 누른다. 잃은
                # 고주파를 되살린다. 헤어 rgb 에만(얼굴은 실제 픽셀이라 불필요),
                # 축소 배율에 맞춘 커널로. 과하면 예의 'AI 텍스처'가 도로 생기니
                # 기본은 온건하게 두고 HEDDY_HAIR_SHARPEN 으로 조절한다.
                hr = new_rgb.permute(2, 0, 1).unsqueeze(0)
                ksz = 3 if eyes is None else max(3, int(round(
                    0.06 * float(np.hypot(eyes[1][0] - eyes[0][0],
                                          eyes[1][1] - eyes[0][1]))))) | 1
                lo = F.avg_pool2d(hr, ksz, stride=1, padding=ksz // 2)
                hr = (hr + CONFIG.hair_sharpen * (hr - lo)).clamp(0, 255)
                new_rgb = hr.squeeze(0).permute(1, 2, 0)
            if sigma_f is not None and CONFIG.grain_match > 0:
                # 그레인 정합. 프레임의 센서/압축 노이즈 σ 를 피부에서 재서 같은
                # 세기의 노이즈를 헤어에 얹는다. 합성물이 프레임보다 '깨끗' 하면
                # 그 자체가 오려붙인 티다. 매 프레임 새 노이즈 - 실제 노이즈도
                # 그렇다. 휘도 노이즈(3채널 동일)로 충분하다. 샤픈 뒤에 얹어야
                # 노이즈까지 증폭되지 않는다.
                g = torch.randn(h, w, 1, device=self.device) * (sigma_f * CONFIG.grain_match)
                new_rgb = new_rgb + g
            out = out * (1.0 - new_a) + new_rgb * new_a

            # 증류 블렌더: 워핑 합성본을 GAN 품질에 가깝게 정제한다.
            if self._blender is not None and blend > 0 and eyes is not None:
                out = self._apply_blender(out, eyes[0], eyes[1], h, w, blend)
                blended = True

        out_u8 = out.clamp(0, 255).to(torch.uint8).cpu().numpy()
        post_ms = (time.perf_counter() - t_post) * 1000

        return out_u8, {
            "pre_ms": pre_ms,
            "infer_ms": inf_ms,
            "post_ms": post_ms,
            "total_ms": (time.perf_counter() - t0) * 1000,
            "hair_px": self._last_hair_px,
            "coverage": self._last_coverage if mode in ("remove", "tryon", "plate") else None,
            "plate_frames": plate.frames if plate is not None else 0,
            "anchor": anchor_src,
            "harmonized": harmonized,
            "blended": blended,
            "groom_ms": groom_ms if mode in ("remove", "tryon") else 0.0,
        }

    # ---- 3D 그룸 ----
    def warm_groom(self, groom: dict) -> bool:
        """GLB 를 GPU 에 올려 둔다(트라이메시 파싱 + VBO, ~1초). 실시간 경로 밖에서 부를 것 -
        _render_groom 이 처음 부를 때 즉석에서 하면 그 프레임이 1초 멈춘다."""
        try:
            self._groom_renderer().load(groom["name"], groom["path"], groom.get("meta"))
            return True
        except Exception:
            logger.exception("그룸 로드 실패: %s", groom.get("path"))
            return False

    def evict_groom(self, name: str) -> None:
        if self._groom is not None:
            self._groom.evict(name)

    def _groom_renderer(self):
        if self._groom is None:
            from groom_renderer import GroomRenderer
            self._groom = GroomRenderer(self.device)
        return self._groom

    def _render_groom(self, groom: dict, matrix, fit, h: int, w: int, dyn=None):
        r = self._groom_renderer()
        g = r.load(groom["name"], groom["path"], groom.get("meta"))
        sm, up, fwd = fit if fit is not None else (1.0, 0.0, 0.0)
        try:
            # 흰 베이스: 색은 _match_hair_color 가 입힌다 (GLB 의 갈색은 무시)
            return r.render(g, matrix, w, h, g.scale_mul * float(sm),
                            g.offset_up_cm + float(up), g.offset_fwd_cm + float(fwd),
                            base_rgb=(1.0, 1.0, 1.0), dyn=dyn)
        except Exception:
            logger.exception("그룸 렌더 실패")
            return None, None

    def _match_hair_color(self, new_rgb, new_a, frame_f, hair_a, refresh: bool, color=None,
                          pose=None, eyes=None):
        """렌더된 헤어(흰 베이스: 음영·텍스처만)에 색을 입힌다 - 알파가중 평균이 목표색이 되도록.

        목표색은 사용자 머리(파싱 마스크)의 평균색. **그 조명 아래에서 관측된** 색이라
        색상뿐 아니라 노출/화이트밸런스/플레어(뿌연 프레임이면 밝은 회색)까지 한 번에
        들어 있다. 2D 경로의 피부 비율(harmonize)과 블랙레벨 lift 가 하던 일을 이 배율
        하나가 대신한다. 텍스처의 명암 대비는 배율이라 그대로 남는다.

        렌더가 흰 베이스여야 하는 이유: 진한 갈색 베이스에 배율을 곱하면 목표가 밝은
        회색일 때 4배 넘게 필요하고, 클램프에 걸리면 갈색 색상이 남아 **주황**이 된다
        (실측). 흰 베이스면 채널별 배율 = 목표/휘도 라 색상이 정확히 목표를 따른다.
        color 가 주어지면(염색 미리보기) 사용자 색 대신 그 색.
        """
        if color is not None:
            target = torch.as_tensor(color, device=self.device, dtype=torch.float32)
        else:
            # 샘플이 오염되는 두 경우를 막는다: (1) 고개를 크게 젖히거나 돌리면 파서가 피부를
            # '머리'로 찍어 평균이 살색으로 밀린다(실측: 분홍 헤어) (2) 긴 머리가 얼굴 옆으로
            # 내려온 자리는 그림자/피부가 섞인다. 정면·안정 포즈에서 눈 위쪽 픽셀만 잰다.
            steady = pose is None or (abs(float(pose.get("yaw", 0.0))) < CONFIG.groom_color_max_yaw
                                      and abs(float(pose.get("pitch", 0.0))) < CONFIG.groom_color_max_pitch)
            above = None
            if eyes is not None:
                ey = float(min(eyes[0][1], eyes[1][1]))
                above = (torch.arange(hair_a.shape[0], device=self.device, dtype=torch.float32)
                         < ey).unsqueeze(-1).unsqueeze(-1)          # (h,1,1) 눈 위쪽 행만

            def measure():
                # (평균 BGR, 휘도 표준편차) - 마스크 안(눈 위쪽) 픽셀
                m = (hair_a > 0.5).float().unsqueeze(-1)
                if above is not None:
                    m = m * above
                cnt = m.sum().clamp(min=1.0)
                mean = (frame_f * m).sum((0, 1)) / cnt
                lum = (frame_f * self._lum_w).sum(-1, keepdim=True)
                lmean = (lum * m).sum() / cnt
                std = (((lum - lmean) ** 2 * m).sum() / cnt).sqrt()
                return mean, std

            if self._hair_color is None:
                # 첫 프레임: 통계 주기를 기다리면 그동안 흰 머리가 나간다. 한 번만 동기화해서 잰다.
                if steady and int((hair_a > 0.5).sum().item()) > CONFIG.groom_color_min_px:
                    self._hair_color, self._hair_std = measure()
            elif refresh and steady and self._last_hair_px > CONFIG.groom_color_min_px:
                # 이후는 want_stats 프레임(이미 동기화됨)에만 EMA 갱신. .item() 없음.
                mean, std = measure()
                a = CONFIG.groom_color_alpha
                self._hair_color = self._hair_color * (1.0 - a) + mean * a
                self._hair_std = self._hair_std * (1.0 - a) + std * a
            if self._hair_color is None:
                # 사용자 머리가 안 보인다(모자/삭발): 기본 색
                target = torch.as_tensor(CONFIG.groom_default_bgr, device=self.device,
                                         dtype=torch.float32)
            else:
                target = self._hair_color
        # 1) 평균색 맞춤 (채널별 배율)
        asum = new_a.sum().clamp(min=1.0)
        rmean = (new_rgb * new_a).sum((0, 1)) / asum
        gain = (target / rmean.clamp(min=1.0)).clamp(0.02, 8.0)
        out = new_rgb * gain.view(1, 1, 3)
        # 2) 휘도 대비 맞춤. 실제 머리는 어두운 뿌리 + 밝은 윤기라 표준편차가 렌더(텍스처
        #    0.55~1.0 배 음영)보다 크다. 평균은 그대로 두고 편차만 배율로 늘리거나 줄인다.
        if color is None and self._hair_color is not None and CONFIG.groom_contrast_match > 0:
            lum = (out * self._lum_w).sum(-1, keepdim=True)
            lmean = (lum * new_a).sum() / asum
            rstd = (((lum - lmean) ** 2 * new_a).sum() / asum).sqrt()
            k = (self._hair_std / rstd.clamp(min=1.0)).clamp(0.5, 3.0)
            k = 1.0 + (k - 1.0) * CONFIG.groom_contrast_match
            out = target.view(1, 1, 3) + (out - target.view(1, 1, 3)) * k
        return out.clamp(0.0, 255.0)

    def close(self):
        self.graph = None
        self.model = None
        # 캐시를 안 비우면 empty_cache() 를 불러도 VRAM 이 그대로 잡혀 있다.
        # 캐싱 얼로케이터는 파이썬 참조가 살아 있는 블록을 반납하지 못한다.
        self._asset_cache.clear()
        self._blender = None
        if self._groom is not None:
            self._groom.close()
            self._groom = None
        self._ev_pairs = None
        self._ev_prev = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
