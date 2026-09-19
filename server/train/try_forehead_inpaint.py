"""이마(앞머리 자리) 인페인팅 모델 비교 실험.

    python train/try_forehead_inpaint.py train/frames/0816_015751/00000.png --out <dir>

마스크 = 파싱 머리 ∩ 얼굴 zone(눈 위 2.35d 타원, gpu_segmenter._face_zone 과 같은 기하) 을 조금
팽창한 것. 이 마스크로 (1) 기존 사전채움(살색 평면) (2) LaMa 를 돌려 나란히 저장한다.
크롭은 눈 중점 기준 정사각(눈 간격의 ~7배)으로 잘라 512 로 돌린다 - 얼굴 전체가 문맥으로 들어가야
피부 색/음영이 맞고, 원본 해상도(눈 간격 50px) 그대로면 모델 입력이 너무 작다.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu_segmenter import GpuFaceParser, CLS_HAIR   # noqa: E402
from face_pose import FacePose                       # noqa: E402


def face_zone_mask(eye_l, eye_r, h, w, up=2.35, down=1.6, side=1.5):
    """gpu_segmenter._face_zone 과 같은 타원 (눈 중점 기준, 눈 간격 배수)."""
    c = (np.asarray(eye_l) + np.asarray(eye_r)) / 2.0
    d = float(np.linalg.norm(np.asarray(eye_r) - np.asarray(eye_l)))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xs - c[0]) / (side * d)
    dy = np.where(ys < c[1], (ys - c[1]) / (up * d), (ys - c[1]) / (down * d))
    return (dx * dx + dy * dy) <= 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dilate", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    img = cv2.imread(a.frame)
    h, w = img.shape[:2]
    seg = GpuFaceParser()
    cls = seg.class_map(img)
    poser = FacePose()
    pose = poser.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
    eye_l, eye_r = pose["eye_l"], pose["eye_r"]
    d = float(np.linalg.norm(eye_r - eye_l))
    c = (eye_l + eye_r) / 2.0

    zone = face_zone_mask(eye_l, eye_r, h, w)
    bangs = ((cls == CLS_HAIR) & zone).astype(np.uint8) * 255
    k = np.ones((a.dilate, a.dilate), np.uint8)
    bangs = cv2.dilate(bangs, k)
    print(f"눈 간격 {d:.1f}px, 앞머리 마스크 {int((bangs > 0).sum())}px")

    # 얼굴 크롭 (정사각, 눈 간격 x7, 눈 중점이 위에서 40%)
    S = int(round(d * 7))
    x0 = int(round(c[0] - S / 2)); y0 = int(round(c[1] - S * 0.4))
    x0 = max(0, min(w - S, x0)); y0 = max(0, min(h - S, y0))
    crop = img[y0:y0 + S, x0:x0 + S]
    mcrop = bangs[y0:y0 + S, x0:x0 + S]
    crop512 = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)
    m512 = cv2.resize(mcrop, (512, 512), interpolation=cv2.INTER_NEAREST)

    # (1) 기준: 살색 평면 (gpu_segmenter 의 폴백과 비슷하게 마스크 주변 피부 평균)
    skin = (cls == 1)[y0:y0 + S, x0:x0 + S]
    tone = crop[skin].mean(0) if skin.any() else np.array([160, 150, 140.0])
    flat = crop.copy(); flat[mcrop > 0] = tone

    # (2) LaMa
    from simple_lama_inpainting import SimpleLama
    from PIL import Image
    t0 = time.perf_counter()
    lama = SimpleLama()
    print(f"LaMa 적재 {(time.perf_counter() - t0) * 1000:.0f}ms")
    rgb = Image.fromarray(cv2.cvtColor(crop512, cv2.COLOR_BGR2RGB))
    for _ in range(2):
        t0 = time.perf_counter()
        out = lama(rgb, Image.fromarray(m512))
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
    print(f"LaMa 512 추론 {ms:.0f}ms, VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f}MB")
    out = cv2.cvtColor(np.asarray(out)[:512, :512], cv2.COLOR_RGB2BGR)
    lama_full = cv2.resize(out, (S, S), interpolation=cv2.INTER_AREA)
    res = crop.copy()
    mm = (mcrop > 0)
    res[mm] = lama_full[mm]

    # 마스크 표시
    shown = crop.copy(); shown[mm] = (shown[mm] * 0.4 + np.array([255, 0, 255]) * 0.6).astype(np.uint8)
    sheet = np.hstack([shown, flat, res])
    sheet = cv2.resize(sheet, (sheet.shape[1] * 2, sheet.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    p = os.path.join(a.out, "forehead_compare.png")
    cv2.imwrite(p, sheet)
    cv2.imwrite(os.path.join(a.out, "lama_crop512.png"), out)
    print("->", p, "(마스크 | 살색 평면 | LaMa)")


if __name__ == "__main__":
    main()
