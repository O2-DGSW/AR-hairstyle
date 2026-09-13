"""GAN 입력 프레임 준비 - 앞머리를 미리 지운다.

왜 필요한가
-----------
앞머리로 이마를 덮은 사용자에게 이마를 드러내는 스타일을 입히면 HairFastGAN
결과의 이마에 **어두운 얼룩**이 남는다(실측: 4개 각도 전부). 구조적인 이유다 -
원본에서 머리였다가 목표에서 피부가 되는 영역은 Alignment 단계가 SEAN 으로
인페인팅하고 그걸 e4e(256) 로 다시 인코딩해 F 공간에 섞는데, 그 경로가 앞머리
그림자를 뭉갠 채로 이마에 남긴다.

그 영역을 GAN 에 넣기 **전에** 피부색으로 채우면 원본의 머리 마스크가 이마를
포함하지 않게 되어 그 자리의 F 가 SEAN 인페인팅이 아니라 원본(=평평한 피부)의
FS 인코딩에서 온다. 실측: 정면/±12°/+25°/+32° 모두 이마가 깨끗해지고 헤어라인은
GAN 이 참고 스타일대로 새로 그린다. 정수리 머리는 남기므로("대머리 입력은 결과
불량") 정렬/회전은 영향받지 않는다.

채움은 **눈썹 바로 위의 보이는 피부** 색으로 한다. 얼굴 전체 평균은 볼/턱이
섞여 이마와 안 맞았고, 인페인팅(Telea/NS)은 줄무늬가 GAN 결과까지 살아남았다.
아래 경계는 눈썹선 근처에서 램프로 흐려 단차를 없앤다.

GPU/모델을 직접 잡지 않는다 - 파싱 결과(cls)와 눈 2점은 호출부가 넘긴다.
"""
from __future__ import annotations

import cv2
import numpy as np

from gpu_segmenter import CLS_HAIR, CLS_SKIN

#: 채울 영역(눈 간격 D 배수). 눈 위 top*D 까지의 머리만 - 정수리는 남긴다.
TOP_D = 1.3
#: 아래 한계(눈 중심 위 D 배수). 이 아래는 절대 건드리지 않는다 - 파서가
#: 안경/눈을 머리로 볼 때가 있다. 실측 비교(0.45/0.32/0.25): 눈썹선(0.45)에서
#: 끊으면 그 아래 남은 앞머리 끝을 GAN 이 짙은 띠로 그린다. 0.25 까지 내리면
#: 띠가 거의 사라지고 안경은 GAN 이 그대로 그린다. 눈 위 여유를 조금 남긴 값.
BROW_D = 0.28
#: 램프 폭. 아래 한계에서 위로 이만큼에 걸쳐 채움 알파가 0 -> 1.
RAMP_D = 0.18
#: 좌우 반폭.
HALF_W_D = 1.45
#: 페더 σ (D 배수).
FEATHER_D = 0.10


def prefill_forehead(frame_bgr: np.ndarray, cls: np.ndarray, eye_l, eye_r):
    """앞머리를 피부색으로 지운 프레임과 (채운 픽셀 수). 채울 게 없으면 원본 그대로.

    frame_bgr : (H,W,3) uint8
    cls       : (H,W) uint8 SegFormer 클래스맵 (같은 크기)
    eye_l/r   : 화면 좌표 눈 2점
    """
    h, w = frame_bgr.shape[:2]
    el = np.asarray(eye_l, dtype=np.float32)
    er = np.asarray(eye_r, dtype=np.float32)
    d = float(np.linalg.norm(er - el))
    if d < 8.0:
        return frame_bgr, 0
    cx, cy = float((el[0] + er[0]) / 2), float((el[1] + er[1]) / 2)
    # 머리 주변 크롭 안에서만 계산한다. 프레임 전체 mgrid/float32 연산은 720p 에서
    # ~60ms 라 촬영 경로에서 눈에 띈다. 크롭은 회전(고개 기울임)까지 덮게 넉넉히.
    r = 2.0 * d
    x0, x1 = int(max(0, cx - r)), int(min(w, cx + r))
    y0, y1 = int(max(0, cy - r)), int(min(h, cy + 0.5 * d))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return frame_bgr, 0
    crop = frame_bgr[y0:y1, x0:x1]
    ccls = cls[y0:y1, x0:x1]
    # 눈 축 기준 좌표계. 고개가 기울어도 '위'가 얼굴의 위다.
    ux, uy = (er - el) / d
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    u = (dx * ux + dy * uy) / d              # 좌우 (D 단위)
    v = (-dx * uy + dy * ux) / d             # 아래가 +, 위가 -

    region = (np.abs(u) < HALF_W_D) & (v > -TOP_D) & (v < -BROW_D)
    hair = ccls == CLS_HAIR
    m = (region & hair).astype(np.uint8)
    n = int(m.sum())
    if n < 30:
        return frame_bgr, 0
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # 눈썹선 위 램프: v=-BROW_D 에서 0, v=-(BROW_D+RAMP_D) 에서 1
    ramp = np.clip((-v - BROW_D) / RAMP_D, 0.0, 1.0)
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, d * FEATHER_D)) * ramp
    a = np.clip(a, 0.0, 1.0)[..., None]

    # 채움색: 눈썹 바로 위 ~ 채움 영역 사이의 보이는 피부. 없으면 얼굴 피부 전체.
    band = (ccls == CLS_SKIN) & (v < -0.2) & (v > -TOP_D) & (np.abs(u) < HALF_W_D)
    if band.sum() < 50:
        band = ccls == CLS_SKIN
    if band.sum() < 50:
        band = cls == CLS_SKIN
        src = frame_bgr
    else:
        src = crop
    if band.sum() < 50:
        return frame_bgr, 0
    tone = src[band].reshape(-1, 3).mean(axis=0)

    out = frame_bgr.copy()
    out[y0:y1, x0:x1] = (crop.astype(np.float32) * (1.0 - a)
                         + tone.reshape(1, 1, 3) * a).clip(0, 255).astype(np.uint8)
    return out, n
