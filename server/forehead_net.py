"""학습된 이마 인페인터 (앞머리를 걷어낸 맨이마 생성).

규칙 기반(forehead.synthesize_forehead: 피부 확산 + 눈썹선 그리기)은 밀랍처럼 매끈하고 눈썹이 그림처럼
보였고, 범용 LaMa 는 앞머리를 다시 그리거나 배경색을 끌어왔다(2026-09-30 실측). 이마만 학습한 작은
네트워크로 바꾼다: FFHQ 맨이마 얼굴에 '다른 사람의 앞머리 모양' 마스크를 씌워 원래 이마를 복원하도록
자기지도 학습(정답이 원본에 있다). 머리카락을 그리는 법은 배우지 않는다 - 정답이 항상 맨이마다.

크롭 정의(crop_box)는 학습 데이터(train/prep_forehead_data.py)와 런타임이 공유한다: 눈 중심 기준,
한 변 CROP_D·d, 눈이 위에서 EYE_Y 비율에 오게.

v4 부터 5번째 입력 = 눈썹 안내(brow_guide): MediaPipe 메시가 추정한 눈썹 윗선/아랫선을 칠한 마스크.
앞머리가 눈썹을 다 덮으면 문맥만으로는 눈썹 위치/모양을 알 수 없어 흐린 얼룩이나 맨살이 됐다.
v5 부터 6번째 입력 = 눈 안내(eye_guide, 눈 윤곽). 눈을 찌르는 앞머리도 지운다 - 그 전엔 눈 둘레 타원을
구멍에서 빼서 눈 위 머리카락이 그대로 남았다. 가중치의 첫 conv 입력 채널 수로 4/5/6ch 를 자동 판별한다.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NET_PX = 256
#: 크롭 한 변 (눈 간격 배수)과 눈 높이(위에서 비율)
CROP_D = 4.4
EYE_Y = 0.55
#: 앞머리 띠 (눈 간격 배수, 눈 중심 기준): 위로 UP, 아래로 DOWN, 좌우 SIDE
BAND_UP, BAND_DOWN, BAND_SIDE = 1.4, 0.35, 1.5
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "forehead_net.pt")


def crop_box(eye_l, eye_r):
    """-> (x0, y0, S) 원본 이미지 좌표의 정사각 크롭."""
    eye_l = np.asarray(eye_l, np.float32); eye_r = np.asarray(eye_r, np.float32)
    c = (eye_l + eye_r) / 2.0
    d = float(np.linalg.norm(eye_r - eye_l))
    S = CROP_D * d
    return float(c[0] - S / 2), float(c[1] - EYE_Y * S), float(S)


def band_mask(px: int = NET_PX) -> np.ndarray:
    """크롭 좌표의 앞머리 띠 (bool, px x px)."""
    d = px / CROP_D
    cy, cx = EYE_Y * px, px / 2
    ys, xs = np.mgrid[0:px, 0:px]
    return (ys > cy - BAND_UP * d) & (ys < cy + BAND_DOWN * d) & (np.abs(xs - cx) < BAND_SIDE * d)


def brow_guide(brows, px: int = NET_PX, M=None) -> np.ndarray:
    """눈썹선 [(윗선 5점, 아랫선 5점), ...] -> 칠한 마스크 (float32 px x px, 0~1, 가장자리 살짝 흐림).
    M: 원본 -> 크롭 2x3 (없으면 이미 크롭 좌표)."""
    import cv2
    g = np.zeros((px, px), np.float32)
    if brows is None:
        return g
    for up, lo in brows:
        poly = np.concatenate([np.asarray(up, np.float32), np.asarray(lo, np.float32)[::-1]], 0)
        if M is not None:
            poly = poly @ np.asarray(M, np.float32)[:, :2].T + np.asarray(M, np.float32)[:, 2]
        cv2.fillPoly(g, [np.round(poly * 4).astype(np.int32)], 1.0, lineType=cv2.LINE_AA, shift=2)
    return cv2.GaussianBlur(g, (0, 0), px / 256.0)


def eye_guide(eyes, px: int = NET_PX, M=None) -> np.ndarray:
    """눈 윤곽 [고리 16점, 고리 16점] -> 칠한 마스크 (float32 px x px)."""
    import cv2
    g = np.zeros((px, px), np.float32)
    if eyes is None:
        return g
    for ring in eyes:
        poly = np.asarray(ring, np.float32)
        if M is not None:
            poly = poly @ np.asarray(M, np.float32)[:, :2].T + np.asarray(M, np.float32)[:, 2]
        cv2.fillPoly(g, [np.round(poly * 4).astype(np.int32)], 1.0, lineType=cv2.LINE_AA, shift=2)
    return cv2.GaussianBlur(g, (0, 0), px / 256.0)


def guides(n: int, brows=None, eyes=None, px: int = NET_PX, M=None) -> np.ndarray:
    """안내 채널 n 개 (0/1/2) -> (n, px, px). 순서: 눈썹, 눈."""
    out = [brow_guide(brows, px, M), eye_guide(eyes, px, M)][:n]
    return np.stack(out) if out else np.zeros((0, px, px), np.float32)


class _Res(nn.Module):
    def __init__(self, c, dil=1):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=dil, dilation=dil), nn.GroupNorm(8, c), nn.SiLU(),
                               nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c))

    def forward(self, x):
        return F.silu(x + self.b(x))


class ForeheadNet(nn.Module):
    """U-Net: 입력 (구멍을 0 으로 지운 RGB [-1,1], 마스크) 4ch -> RGB. 병목은 팽창 잔차로 넓은 문맥."""

    def __init__(self, c=48, in_ch=4):
        super().__init__()
        ch = [c, c * 2, c * 4, c * 8]
        self.in_ch = in_ch
        self.inc = nn.Sequential(nn.Conv2d(in_ch, ch[0], 5, padding=2), nn.SiLU(), _Res(ch[0]))
        self.downs = nn.ModuleList()
        for i in range(3):
            self.downs.append(nn.Sequential(nn.Conv2d(ch[i], ch[i + 1], 4, stride=2, padding=1), nn.SiLU(),
                                            _Res(ch[i + 1])))
        self.mid = nn.Sequential(*[_Res(ch[3], d) for d in (1, 2, 4, 8, 1)])
        self.ups = nn.ModuleList()
        for i in (2, 1, 0):
            self.ups.append(nn.ModuleDict({
                "up": nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                    nn.Conv2d(ch[i + 1], ch[i], 3, padding=1), nn.SiLU()),
                "res": nn.Sequential(nn.Conv2d(ch[i] * 2, ch[i], 3, padding=1), nn.SiLU(), _Res(ch[i]))}))
        self.out = nn.Conv2d(ch[0], 3, 3, padding=1)

    def forward(self, img, mask, guide=None):
        x = [img * (1 - mask), mask]
        if self.in_ch > 4:
            x.append(guide if guide is not None else torch.zeros_like(mask).repeat(1, self.in_ch - 4, 1, 1))
        x = torch.cat(x, 1)
        skips = [self.inc(x)]
        h = skips[0]
        for d in self.downs:
            h = d(h); skips.append(h)
        h = self.mid(skips.pop())
        for u in self.ups:
            h = u["up"](h)
            h = u["res"](torch.cat([h, skips.pop()], 1))
        out = torch.tanh(self.out(h))
        return img * (1 - mask) + out * mask, out


class LearnedForehead:
    """런타임 래퍼: 프레임(BGR) + 눈 + 구멍 마스크 -> 구멍을 채운 프레임. 크롭 -> 256 -> 되돌려 붙임."""

    def __init__(self, device="cuda", path=WEIGHTS):
        self.device = device
        sd = torch.load(path, map_location=device)
        sd = sd.get("ema", sd.get("net", sd))
        self.net = ForeheadNet(in_ch=sd["inc.0.weight"].shape[1]).to(device).eval()
        self.net.load_state_dict(sd)

    @property
    def fills_eyes(self) -> bool:
        """눈 안내(6ch)가 있는 가중치 = 눈 위 머리카락까지 지우도록 학습됨."""
        return self.net.in_ch >= 6

    @torch.inference_mode()
    def fill(self, frame_bgr: np.ndarray, hole: np.ndarray, eye_l, eye_r, brows=None, eyes=None) -> np.ndarray:
        import cv2
        h, w = frame_bgr.shape[:2]
        x0, y0, S = crop_box(eye_l, eye_r)
        M = np.float32([[NET_PX / S, 0, -x0 * NET_PX / S], [0, NET_PX / S, -y0 * NET_PX / S]])
        ci = cv2.warpAffine(frame_bgr, M, (NET_PX, NET_PX), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
        cm = cv2.warpAffine(hole.astype(np.uint8) * 255, M, (NET_PX, NET_PX), flags=cv2.INTER_LINEAR) > 64
        t = torch.from_numpy(cv2.cvtColor(ci, cv2.COLOR_BGR2RGB)).to(self.device).permute(2, 0, 1)[None].float()
        t = t / 127.5 - 1
        m = torch.from_numpy(cm.astype(np.float32)).to(self.device)[None, None]
        g = None
        if self.net.in_ch > 4:
            g = torch.from_numpy(guides(self.net.in_ch - 4, brows, eyes, NET_PX, M)).to(self.device)[None]
        comp, _ = self.net(t, m, g)
        out = ((comp[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        Minv = cv2.invertAffineTransform(M)
        back = cv2.warpAffine(out, Minv, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        res = frame_bgr.copy()
        res[hole] = back[hole]
        return res
