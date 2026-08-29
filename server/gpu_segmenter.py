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
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

from config import CONFIG

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
        self._last_coverage = None
        self._grid = None      # (ys, xs) 좌표 그리드 캐시 - 무게중심/워핑 공용
        # LRU. 512^2 RGBA float32 = 4MB/개이고 라이브 뱅크 한 번이 7개를 만든다.
        # 상한 없이 두면 세션을 몇 번 돌리는 것만으로 VRAM 이 수백 MB 샌다.
        self._asset_cache = OrderedDict()   # 에셋 이름 -> GPU 텐서
        self._blender = None    # 증류 블렌더 (load_blender 로 주입)
        self._cgrid = None      # 크롭 좌표 그리드 캐시

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

    def _asset_tensor(self, asset):
        """에셋 RGBA를 GPU에 올려 캐시. 매 프레임 업로드하면 그것만으로 수 ms 샌다."""
        t = self._asset_cache.get(asset.name)
        if t is not None:
            self._asset_cache.move_to_end(asset.name)        # 최근 사용 표시
            return t

        arr = asset.rgba.astype(np.float32)                  # (Ha, Wa, 4) BGRA
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self._asset_cache[asset.name] = t
        # 지금 워핑 중인 에셋이 축출되면 안 되므로 넣고 나서 자른다.
        while len(self._asset_cache) > CONFIG.asset_cache_max:
            self._asset_cache.popitem(last=False)
        return t

    # ---------- 에셋 GPU 캐시 ----------
    def evict_asset(self, name: str) -> bool:
        """캐시된 에셋 텐서를 버린다. 있었으면 True.

        세션이 끝나면 그 세션이 GAN 으로 만든 에셋은 다시 쓰이지 않는다.
        레지스트리 쪽 축출 훅에서 이걸 불러야 VRAM 이 실제로 돌아온다
        (파이썬 쪽 참조가 남아 있으면 캐싱 얼로케이터가 반납하지 않는다).
        """
        return self._asset_cache.pop(name, None) is not None

    def evict_assets(self, names) -> int:
        return sum(1 for n in names if self.evict_asset(n))

    def cache_stats(self) -> dict:
        n = 0
        for t in self._asset_cache.values():
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

        at = self._asset_tensor(asset)
        ha, wa = at.shape[2], at.shape[3]
        ys, xs = self._ensure_grid(h, w)

        # 출력 픽셀 좌표 -> 에셋 좌표 -> grid_sample 정규화 좌표([-1,1])
        xa = Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2]
        ya = Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2]
        gx = (2.0 * xa + 1.0) / wa - 1.0
        gy = (2.0 * ya + 1.0) / ha - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

        warped = torch.nn.functional.grid_sample(
            at, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        a = warped[0, 3:4].permute(1, 2, 0) / 255.0     # (h, w, 1)
        rgb = warped[0, :3].permute(1, 2, 0)            # (h, w, 3)
        return rgb, a

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

    @staticmethod
    def _skin_tone(frame_f, cls):
        """보이는 얼굴 피부의 대표 색 (3,). GPU 동기화 없이 계산한다."""
        m = (cls == CLS_SKIN).unsqueeze(-1).float()
        n = m.sum()
        tone = (frame_f * m).sum(dim=(0, 1)) / n.clamp(min=1.0)
        # 피부가 거의 안 보이면 화면 평균으로 대체 (.item() 없이 GPU에서 분기)
        ok = (n > 200).float()
        return tone * ok + frame_f.mean(dim=(0, 1)) * (1.0 - ok)

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
                smoother=None):
        """BGR 프레임 -> (합성된 BGR 프레임, 타이밍/상태 dict).

        mode:
          seg    - 세그멘테이션 색칠 (머리=마젠타, 얼굴피부=시안)
          remove - 기존 머리를 누적된 플레이트로 지움 (Phase 1 검증용)
          plate  - 누적된 플레이트 자체를 표시 (디버깅)
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

            if mode == "tryon" and asset is not None and eyes is not None:
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

                if new_rgb is not None and harmonize and asset.ref_skin is not None:
                    # 조명/화이트밸런스 정합.
                    # 에셋은 만들어질 때의 조명에 고정돼 있는데 지금 프레임의 조명은
                    # 다르다. 그 차이를 피부색 비율로 추정해 헤어에 그대로 곱한다.
                    # (피부는 어느 장면에나 있고 조명을 그대로 받으므로 조도의
                    #  대리 지표로 쓸 수 있다)
                    cur = self._skin_tone(frame_f, cls)
                    ref = torch.as_tensor(asset.ref_skin, device=self.device,
                                          dtype=torch.float32)
                    ratio = (cur / ref.clamp(min=1.0)).clamp(CONFIG.harmonize_min, CONFIG.harmonize_max)
                    new_rgb = (new_rgb * ratio.view(1, 1, 3)).clamp(0, 255)
                    harmonized = True

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

            if eyes is not None:
                zone = self._face_zone(eyes[0], eyes[1], h, w)          # (h,w) 0~1
                tone = self._skin_tone(frame_f, cls).view(1, 1, 3)
                fill = fill * (1.0 - zone).unsqueeze(-1) + tone * zone.unsqueeze(-1)
                # 얼굴 영역은 플레이트 관측 여부와 무관하게 피부톤으로 채울 수 있다
                src_ok = torch.clamp(src_ok + zone, 0.0, 1.0)

            a = (erase * src_ok).unsqueeze(-1)
            out = frame_f * (1 - a) + fill * a
        elif mode == "plate" and plate is not None and plate.seen is not None:
            out = plate.plate * plate.seen.unsqueeze(-1)
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
                sk = CONFIG.shadow_k
                blurred = F.avg_pool2d(a2, sk, stride=1, padding=sk // 2)
                band = (blurred - a2).clamp(min=0.0).squeeze(0).permute(1, 2, 0)
                out = out * (1.0 - shadow * band)

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
        }

    def close(self):
        self.graph = None
        self.model = None
        # 캐시를 안 비우면 empty_cache() 를 불러도 VRAM 이 그대로 잡혀 있다.
        # 캐싱 얼로케이터는 파이썬 참조가 살아 있는 블록을 반납하지 못한다.
        self._asset_cache.clear()
        self._blender = None
        self._ev_pairs = None
        self._ev_prev = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
