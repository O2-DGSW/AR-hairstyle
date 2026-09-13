"""GAN 결과 이미지 -> 실시간 워핑용 에셋(머리 + 얼굴 패치).

server.py(라이브 뱅크/촬영)와 train/make_asset_bank.py(오프라인 뱅크)가 같은
물건을 만들어야 하는데, 지금까지는 각자 build_from_photo 를 따로 불러서
한쪽에만 얼굴 패치가 붙거나 크기 보정이 빠지는 식으로 어긋났다. 추출 규칙은
여기 한 곳에만 둔다.

GPU/모델을 직접 잡지 않는다 - 파싱 결과(cls)와 포즈는 호출부가 넘긴다.
서버는 CUDA 그래프 때문에 파싱을 전용 스레드에서 돌려야 하고, 오프라인
스크립트는 그냥 부르면 되므로 그 차이를 여기 들이지 않는다.
"""
from __future__ import annotations

import logging

import numpy as np

import hair_asset
from gpu_segmenter import CLS_HAIR, CLS_SKIN

logger = logging.getLogger("asset_extract")

#: 얼굴 패치로 쓰는 파싱 클래스 범위. 피부(1)부터 아랫입술(12)까지 - 머리(13),
#: 모자(14), 귀걸이/목/옷/배경은 뺀다.
FACE_CLS_MIN, FACE_CLS_MAX = 1, 12
#: 이보다 작은 머리 마스크는 실패로 본다(px).
MIN_HAIR_PX = 500
MIN_FACE_PX = 500


def extract(result_bgr: np.ndarray, cls: np.ndarray, eye_l, eye_r, name: str):
    """GAN 결과 + 파싱 클래스맵 + 눈 2점 -> (HairAsset | None, hair_px).

    머리(cls==13)를 본체로, 얼굴 부위(1~12)를 asset.face 로 오려낸다.
    둘 다 같은 눈 앵커를 쓰므로 런타임에 같은 닮음변환으로 정렬된다.

    얼굴 패치를 이마만 잘라내지 않는 이유: 앞머리가 어디까지 내려와 있었는지
    미리 알 수 없다. 넉넉히 오려 두고 덮을 자리는 런타임의 erase 마스크가
    정한다. 실제로 덮이는 건 원래 머리가 있던 자리뿐이다.
    """
    hair = (cls == CLS_HAIR).astype(np.uint8)
    # 이 결과의 피부색을 함께 기록해 두면, 나중에 조명이 달라져도 그 비율로
    # 헤어 색을 보정할 수 있다.
    ref_skin = hair_asset.skin_mean(result_bgr, cls == CLS_SKIN)

    asset, _, px = hair_asset.build_from_photo(
        result_bgr, hair, eye_l, eye_r, name, ref_skin=ref_skin)
    if asset is None or px < MIN_HAIR_PX:
        return None, px

    face_mask = ((cls >= FACE_CLS_MIN) & (cls <= FACE_CLS_MAX)).astype(np.uint8)
    face_asset, _, fpx = hair_asset.build_from_photo(
        result_bgr, face_mask, eye_l, eye_r, name + "#face", ref_skin=ref_skin)
    if face_asset is not None and fpx >= MIN_FACE_PX:
        asset.face = face_asset
    else:
        logger.warning("GAN 결과에서 얼굴 패치를 못 만들었습니다 (%s px) - "
                       "평균 살색으로 대체됩니다", fpx)
    return asset, px


def eye_len(asset) -> float:
    return float(np.linalg.norm(np.asarray(asset.eye_r) - np.asarray(asset.eye_l)))
