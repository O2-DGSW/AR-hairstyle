"""FFHQ 정렬을 GPU 에서 한다. HairFastGAN 의 align_face 대체.

왜 다시 만드는가
----------------
HairFastGAN 은 입력을 FFHQ 규격(1024, 눈 간격 ~256px)으로 맞춰서 쓴다. 그
전처리가 실측 **1,068ms** 로 파이프라인 최대 병목이었다. GAN 네트워크 자체보다
오래 걸린다.

    dlib 68점 랜드마크      356 ms
    PIL 크롭/패딩/워프      712 ms

둘 다 없앨 수 있다.

1) 68점 중 실제로 쓰는 건 4개뿐이다. shape_predictor.py 의 사각형 계산을 보면
   눈 2점(컨투어 평균)과 입꼬리 2점만 들어간다. 눈썹/코/윤곽은 계산만 하고
   버린다. 그 4점은 MediaPipe 가 훨씬 싸게 준다(~10ms). 게다가 dlib 의 HOG
   검출기는 yaw ±25도를 넘으면 놓쳐서 CNN 으로 떨어지는데, 라이브 뱅크는
   ±36도까지 가므로 극단 칸마다 그 느린 경로를 탄다.

2) 사각형이 **닮음변환**이다. x 와 y 가 수직이고 길이가 같게 만들어지므로
   (y = flipud(x)*[-1,1]) 회전+등배율+평행이동뿐이다. PIL 의 크롭→반사패딩→
   블러→QUAD→LANCZOS 체인 전체가 grid_sample 한 번과 같다.

원본과 다른 점
--------------
PIL 경로는 사각형이 이미지 밖으로 나갈 때 반사 패딩을 한 뒤 경계를 블러와
중앙값으로 섞어 티를 지운다. 여기서는 grid_sample 의 reflection 패딩만 쓴다.
얼굴이 화면 가장자리에 붙어 사각형이 크게 넘칠 때 가장자리 질감이 달라질 수
있다. 실측 비교는 tools/compare_align.py 참고.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

#: HairFastGAN 쪽 패치와 같은 값을 쓴다. 거기서는 os.environ 을 직접 읽는다
#: (utils/shape_predictor.py:140). 두 경로가 어긋나면 정렬이 달라지므로
#: 여기서도 같은 환경변수를 본다.
import os

QUAD_SCALE = float(os.environ.get("HEDDY_QUAD_SCALE", "1.15"))


def ffhq_quad(eye_l, eye_r, mouth_l, mouth_r, quad_scale=None):
    """FFHQ 정렬 사각형 (4,2). shape_predictor.align_face 와 같은 식.

    반환 순서는 PIL 의 Image.QUAD 와 같다: 좌상, 좌하, 우하, 우상.
    """
    eye_l = np.asarray(eye_l, dtype=np.float64)
    eye_r = np.asarray(eye_r, dtype=np.float64)
    eye_avg = (eye_l + eye_r) * 0.5
    eye_to_eye = eye_r - eye_l
    mouth_avg = (np.asarray(mouth_l, dtype=np.float64)
                 + np.asarray(mouth_r, dtype=np.float64)) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    n = np.hypot(*x)
    if n < 1e-6:
        raise ValueError("정렬 사각형을 만들 수 없습니다 (랜드마크가 퇴화했습니다)")
    x /= n
    scale = QUAD_SCALE if quad_scale is None else float(quad_scale)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8) * scale
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return np.stack([c - x - y, c - x + y, c + x + y, c + x - y]), float(np.hypot(*x) * 2)


#: 격자를 출력의 몇 배로 찍고 줄일지. 원본 PIL 경로는 항상 4096 으로 변환한 뒤
#: LANCZOS 로 1024 로 줄인다 - 사실상 4배 슈퍼샘플링이다. 1024 에서 bilinear 로
#: 한 번만 찍으면 머리카락 같은 고주파에서 에일리어싱이 생겨 원본과 벌어진다
#: (같은 사각형인데도 PSNR 23~28dB 였다). 2배만 해도 대부분 회복되고, GPU 에서는
#: 픽셀이 4배라도 비용이 밀리초 단위다.
SUPERSAMPLE = 2


def align_to_quad(img, quad, out_size=1024, device=None):
    """이미지를 사각형 기준으로 잘라 out_size 정사각으로 편다.

    img   : (H,W,3) uint8 RGB 또는 (3,H,W) float 텐서 0~1
    quad  : ffhq_quad 가 준 (4,2). 좌상/좌하/우하/우상.
    반환  : (3,out,out) float 텐서 0~1

    사각형이 출력보다 훨씬 크면 먼저 줄인다. 안 그러면 grid_sample 이 원본을
    듬성듬성 찍어 계단이 생긴다. 원본 PIL 경로의 shrink + LANCZOS 가 하던 일이다.
    """
    if isinstance(img, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    else:
        t = img.float()
    if device is not None:
        t = t.to(device)
    t = t.unsqueeze(0)                                   # (1,3,H,W)
    quad = np.asarray(quad, dtype=np.float64).copy()

    ss = max(1, int(SUPERSAMPLE))
    grid_size = out_size * ss

    qsize = float(np.hypot(*(quad[3] - quad[0])))        # 좌상->우상 = 한 변
    shrink = qsize / float(grid_size)
    if shrink > 2.0:
        # 격자보다도 훨씬 큰 원본은 먼저 줄인다. 슈퍼샘플링만으로는 못 덮는
        # 배율이고, 큰 텐서를 그대로 들고 있을 이유도 없다.
        f = 1.0 / shrink
        t = F.interpolate(t, scale_factor=f, mode="bilinear",
                          align_corners=False, antialias=True)
        quad *= f

    _, _, h, w = t.shape
    ul, ll, _lr, ur = quad
    # 출력 픽셀 중심 격자. align_corners=False 규약에 맞춰 (i+0.5)/out 을 쓴다.
    lin = (torch.arange(grid_size, dtype=torch.float64, device=t.device) + 0.5) / grid_size
    vv, uu = torch.meshgrid(lin, lin, indexing="ij")     # v=세로, u=가로

    ul_t = torch.tensor(ul, dtype=torch.float64, device=t.device)
    du = torch.tensor(ur - ul, dtype=torch.float64, device=t.device)
    dv = torch.tensor(ll - ul, dtype=torch.float64, device=t.device)
    # 닮음변환이라 이 쌍선형 보간이 곧 정확한 매핑이다.
    src = ul_t.view(1, 1, 2) + uu.unsqueeze(-1) * du.view(1, 1, 2) \
        + vv.unsqueeze(-1) * dv.view(1, 1, 2)            # (out,out,2) 원본 픽셀좌표

    # grid_sample 은 [-1,1] 정규화 좌표를 받는다. align_corners=False 에서
    # 픽셀 중심 i 는 (2i+1)/size - 1 에 대응한다.
    gx = (src[..., 0] * 2.0 + 1.0) / w - 1.0
    gy = (src[..., 1] * 2.0 + 1.0) / h - 1.0
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).to(t.dtype)

    out = F.grid_sample(t, grid, mode="bilinear", padding_mode="reflection",
                        align_corners=False)

    # 사각형이 원본 밖으로 나간 부분을 부드럽게 덮는다.
    #
    # reflection 패딩만 쓰면 배경이 거울처럼 반복돼 **구조가 있는 가짜 무늬**가
    # 생긴다(실측: 꽃무늬가 좌우로 접혀 보였다). e4e 인버전에 그런 패턴을
    # 먹이는 건 좋지 않다. 원본 PIL 경로도 같은 이유로 반사 패딩 뒤에 블러와
    # 중앙값을 섞어 티를 지운다. 여기서는 그걸 한 번의 혼합으로 흉내 낸다.
    # **평평한 색으로 덮으면 안 된다.** 중앙값으로 칠해 봤더니 사방에 액자 같은
    # 테두리가 생겼다. 원본도 평평한 색이 아니라 **흐린 이미지 내용**으로 잇는다
    # (gaussian_filter 결과를 섞고, 맨 바깥에서만 중앙값으로 간다).
    oob = ((grid[..., 0].abs() > 1.0) | (grid[..., 1].abs() > 1.0)).to(out.dtype)
    if float(oob.sum()) > 0:
        oob = oob.unsqueeze(1)                            # (1,1,H,W)
        k = max(3, (grid_size // 64) | 1)
        soft = F.avg_pool2d(oob, k, stride=1, padding=k // 2).clamp(0, 1)
        # 큰 커널 평균 = 값싼 블러. 반사된 내용이 뭉개져 무늬가 안 보인다.
        kb = max(3, (grid_size // 24) | 1)
        blurred = F.avg_pool2d(out, kb, stride=1, padding=kb // 2)
        out = out * (1 - soft) + blurred * soft

    if ss > 1:
        # 원본의 LANCZOS 축소에 대응한다. antialias 를 켜야 의미가 있다.
        out = F.interpolate(out, size=(out_size, out_size), mode="bilinear",
                            align_corners=False, antialias=True)
    return out.squeeze(0).clamp(0, 1)


def align_face_landmarks(img, eye_l, eye_r, mouth_l, mouth_r,
                         out_size=1024, device=None, quad_scale=None):
    """랜드마크 4점으로 바로 FFHQ 정렬. -> (3,out,out) 0~1 텐서."""
    quad, _ = ffhq_quad(eye_l, eye_r, mouth_l, mouth_r, quad_scale)
    return align_to_quad(img, quad, out_size=out_size, device=device)
