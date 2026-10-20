"""긴 머리 밑(목/어깨/옷/배경) 인페인터 - 크롭 정의 (학습 데이터와 런타임이 공유).

어깨 아래로 내려오는 긴 머리를 지운 자리를 채운다. LaMa/MI-GAN 은 주변 머리결을 이어 그려 유령 가닥이
남았고(2026-10-19 실측), 색 확산은 어깨선을 몰라 덩어리가 됐다. 이마 인페인터(forehead_net)와 같은
자기지도: 머리가 어깨 위로 안 내려온 상반신 사진(정답)에 '다른 사람의 긴 머리 모양' 구멍을 씌워 원래
목/어깨/옷/배경을 복원한다. 정답에 머리가 없으니 머리를 그리는 법을 배우지 않는다.

데이터: EasyPortrait v2 (웹캠/폰 상반신 4만 장, CC BY-SA 4.0 변형 라이선스 - 상업 이용 가능, 저작자 표시,
데이터를 고쳐 배포하면 같은 라이선스). train/prep_torso_data.py 가 이 크롭으로 잘라 저장한다.

크롭: 눈 중심 기준 정사각, 한 변 CROP_D·d (d = 눈 간격), 눈이 위에서 EYE_Y 비율. 폰 세로 화면(800 폭,
d≈130)이 대략 6d 폭이라 좌우 끝과 가슴께까지 들어간다. 크롭이 원본 밖으로 나간 자리는 무효(학습 손실 제외).
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

#: 저장 해상도 (학습은 이걸 줄여 쓴다)
STORE_PX = 512
#: 크롭 한 변 (눈 간격 배수)과 눈 높이(위에서 비율): 눈 위 1d ~ 눈 아래 5.4d
CROP_D = 6.4
EYE_Y = 1.0 / CROP_D
#: 저장 파싱 PNG 의 무효(원본 밖) 값
INVALID = 255
#: 저장하는 포즈 랜드마크 (MediaPipe Pose 번호): 코, 귀, 어깨, 팔꿈치, 엉덩이
POSE_IDS = (0, 7, 8, 11, 12, 13, 14, 23, 24)


#: 학습/런타임 해상도
NET_PX = 384
#: 안내 채널: 포즈 뼈대, 구멍 밖에 보이는 셀피 분류(배경, 옷, 피부)
GUIDES = ("pose", "bg", "cloth", "skin")
#: 이전 프레임 기억 채널 (v2): 구멍 안에서 머리가 비켜 실제로 보였던 픽셀 RGB [-1,1] + 신선도 [0,1].
#: 구멍 밖과 본 적 없는 자리는 0. 안내 채널 뒤에 붙인다 (guide = [GUIDES..., mem_rgb, mem_conf]).
MEM_CH = 4
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "torso_net.pt")


class TorsoNet(nn.Module):
    """ForeheadNet 과 같은 U-Net 이되 한 단 더 깊다(down 4번): 구멍이 이마보다 훨씬 커서(어깨 폭 2d 넘게)
    384 에서 병목 24x24 + 팽창 잔차로 크롭 전체를 본다. 입력 = 구멍을 0 으로 지운 RGB [-1,1], 마스크, 안내."""

    def __init__(self, c=48, in_ch=4 + len(GUIDES), depth=4):
        super().__init__()
        from forehead_net import _Res
        ch = [c * min(2 ** i, 8) for i in range(depth + 1)]
        self.in_ch = in_ch
        self.inc = nn.Sequential(nn.Conv2d(in_ch, ch[0], 5, padding=2), nn.SiLU(), _Res(ch[0]))
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(nn.Sequential(nn.Conv2d(ch[i], ch[i + 1], 4, stride=2, padding=1), nn.SiLU(),
                                            _Res(ch[i + 1])))
        self.mid = nn.Sequential(*[_Res(ch[depth], d) for d in (1, 2, 4, 8, 1)])
        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
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


def crop_box(eye_l, eye_r):
    """-> (x0, y0, S) 원본 이미지 좌표의 정사각 크롭."""
    eye_l = np.asarray(eye_l, np.float32); eye_r = np.asarray(eye_r, np.float32)
    c = (eye_l + eye_r) / 2.0
    d = float(np.linalg.norm(eye_r - eye_l))
    S = CROP_D * d
    return float(c[0] - S / 2), float(c[1] - EYE_Y * S), float(S)


def crop_matrix(box, px: int = STORE_PX) -> np.ndarray:
    """crop_box -> 원본 -> 크롭(px) 2x3 아핀."""
    x0, y0, S = box
    k = px / S
    return np.float32([[k, 0, -x0 * k], [0, k, -y0 * k]])
