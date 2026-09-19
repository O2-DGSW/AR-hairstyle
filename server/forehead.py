"""이마 생성: 앞머리를 인페인팅으로 걷어낸 '맨이마' 패치 (3D 그룸 경로용).

왜 필요한가
-----------
앞머리 있는 사람이 앞머리 없는 스타일을 입으면, 원래 머리를 지운 이마 자리를 뭔가로
채워야 한다. 살색 휘도 평면은 스티커처럼 보이고, HairFastGAN 촬영은 이마 한 조각을 얻자고
266M 파라미터 모델(VRAM 5.5GB, 적재 15초, 2초/장)을 돌리는 데다 실측으로는 입력의
사전채움(평면)을 그대로 흐릿하게 재현할 뿐이었다(2026-09-19 비교). 이마는 "작은 영역
인페인팅" 문제라 범용 인페인터가 더 싸고 낫다: LaMa(big-lama, ~50M, VRAM 0.6GB,
512 크롭 41ms) 가 눈썹을 보존한 채 피부를 자연스럽게 이어 준다.

쓰는 법
-------
스타일을 고를 때 1회 만들고(재생성 명령 있음), 결과를 HairAsset(얼굴 패치)으로 만들어
매 프레임 기존 _warp_asset(눈 앵커 닮음변환)으로 얹는다. 헤어는 3D 지만 이마는 거의
평면이라 이걸로 충분하고, 저주파 색은 런타임의 face_lf_match 가 실제 피부에 맞춘다.

마스크
------
파싱 머리 ∩ 얼굴 zone(눈 위 2.35d 타원 - gpu_segmenter._face_zone 과 같은 기하) 을 조금
팽창한 것. 그룸이 어디까지 덮을지 미리 모르므로 zone 안의 머리를 전부 피부로 채워
"머리 없는 사용자"를 만든다 - 3D 헤어가 덮는 곳은 어차피 안 보인다.
"""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np
import torch

import hair_asset
from gpu_segmenter import CLS_HAIR

logger = logging.getLogger("forehead")

#: 얼굴 zone 타원 (눈 간격 배수). gpu_segmenter._face_zone 과 맞춘다.
ZONE_UP, ZONE_DOWN, ZONE_SIDE = 2.35, 1.6, 1.5
#: 인페인터 입력 크롭: 눈 간격의 몇 배를 한 변으로 (얼굴 전체가 문맥으로 들어가야 색/음영이 맞는다).
CROP_EYES = 7.0
CROP_PX = 512
#: 얼굴 패치로 쓰는 파싱 클래스 범위 (asset_extract 와 동일: 피부~아랫입술).
FACE_CLS_MIN, FACE_CLS_MAX = 1, 12


def zone_mask(eye_l, eye_r, h, w):
    c = (np.asarray(eye_l) + np.asarray(eye_r)) / 2.0
    d = float(np.linalg.norm(np.asarray(eye_r) - np.asarray(eye_l)))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xs - c[0]) / (ZONE_SIDE * d)
    dy = np.where(ys < c[1], (ys - c[1]) / (ZONE_UP * d), (ys - c[1]) / (ZONE_DOWN * d))
    return (dx * dx + dy * dy) <= 1.0


class ForeheadInpainter:
    """LaMa 래퍼. 적재는 처음 부를 때(3초 + JIT 워밍업). 스레드 안전(락)."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._lock = threading.Lock()
        # 영상 경로(기본 스트림)와 겹치도록 별도 스트림. 같은 스트림이면 프레임 루프의 동기화가
        # 여기 큐잉된 커널까지 기다린다.
        self._stream = torch.cuda.Stream() if device == "cuda" else None

    def _ensure(self):
        if self._model is not None:
            return
        from simple_lama_inpainting import SimpleLama
        t0 = time.perf_counter()
        self._model = SimpleLama(torch.device(self.device))
        # TorchScript 프로파일링 실행기는 처음 두 번이 느리다(실측 1.8초) - 미리 돌려 둔다.
        from PIL import Image
        blank = Image.new("RGB", (CROP_PX, CROP_PX), (128, 128, 128))
        m = np.zeros((CROP_PX, CROP_PX), np.uint8); m[100:300, 100:400] = 255
        for _ in range(3):
            self._model(blank, Image.fromarray(m))
        logger.info("LaMa 적재+워밍업 %.1fs", time.perf_counter() - t0)

    def inpaint(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """mask(0/255) 자리를 채운 BGR. 크기는 임의(내부에서 8의 배수로 패딩)."""
        from PIL import Image
        self._ensure()
        rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        with self._lock, torch.inference_mode():
            if self._stream is not None:
                with torch.cuda.stream(self._stream):
                    out = self._model(rgb, Image.fromarray(mask))
                self._stream.synchronize()
            else:
                out = self._model(rgb, Image.fromarray(mask))
        out = np.asarray(out)[:img_bgr.shape[0], :img_bgr.shape[1]]
        return cv2.cvtColor(np.ascontiguousarray(out), cv2.COLOR_RGB2BGR)


def build_forehead_asset(inpainter: ForeheadInpainter, frame_bgr: np.ndarray, cls: np.ndarray,
                         eye_l, eye_r, name: str, dilate: int = 7):
    """프레임 + 파싱 + 눈 2점 -> (얼굴 패치 HairAsset, 앞머리 px, 인페인트 ms).

    반환 에셋은 .face 가 자기 자신이다 - gpu_segmenter 의 그룸 분기는 asset.face /
    asset.ref_skin / asset.scale_adjust 만 보므로 2D 헤어 본체 없이 이걸 그대로 넘긴다.
    """
    h, w = frame_bgr.shape[:2]
    eye_l = np.asarray(eye_l, np.float32); eye_r = np.asarray(eye_r, np.float32)
    d = float(np.linalg.norm(eye_r - eye_l))
    c = (eye_l + eye_r) / 2.0
    zone = zone_mask(eye_l, eye_r, h, w)
    bangs = ((cls == CLS_HAIR) & zone).astype(np.uint8) * 255
    if dilate > 1:
        bangs = cv2.dilate(bangs, np.ones((dilate, dilate), np.uint8))
    bangs_px = int((bangs > 0).sum())

    bald = frame_bgr.copy()
    ms = 0.0
    if bangs_px > 0:
        # 얼굴 크롭 -> 512 -> 인페인트 -> 되돌려 붙임 (마스크 안만)
        S = int(round(d * CROP_EYES))
        x0 = int(round(c[0] - S / 2)); y0 = int(round(c[1] - S * 0.4))
        x0 = max(0, min(w - S, x0)); y0 = max(0, min(h - S, y0))
        S = min(S, w - x0, h - y0)
        crop = frame_bgr[y0:y0 + S, x0:x0 + S]
        mcrop = bangs[y0:y0 + S, x0:x0 + S]
        t0 = time.perf_counter()
        out = inpainter.inpaint(cv2.resize(crop, (CROP_PX, CROP_PX), interpolation=cv2.INTER_CUBIC),
                                cv2.resize(mcrop, (CROP_PX, CROP_PX), interpolation=cv2.INTER_NEAREST))
        ms = (time.perf_counter() - t0) * 1000
        out = cv2.resize(out, (S, S), interpolation=cv2.INTER_AREA)
        mm = mcrop > 0
        crop_out = crop.copy(); crop_out[mm] = out[mm]
        bald[y0:y0 + S, x0:x0 + S] = crop_out

    # 얼굴 패치 = 원래 얼굴 부위 ∪ 방금 채운 자리. asset_extract 와 같은 규칙, 같은 눈 앵커.
    face_mask = (((cls >= FACE_CLS_MIN) & (cls <= FACE_CLS_MAX)) | (bangs > 0)).astype(np.uint8)
    ref_skin = hair_asset.skin_mean(bald, cls == 1)
    asset, _, px = hair_asset.build_from_photo(bald, face_mask, eye_l, eye_r, name, ref_skin=ref_skin)
    if asset is None:
        return None, bangs_px, ms
    asset.face = asset
    asset.bald_bgr = bald            # 디버그/촬영용 (프레임 크기)
    return asset, bangs_px, ms
