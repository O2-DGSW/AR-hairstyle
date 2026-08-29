"""MediaPipe multiclass selfie segmentation wrapper.

Categories (fixed order for this model): background, hair, body-skin,
face-skin, clothes, others.

이 서버는 더 이상 처리된 "영상"을 다시 WebRTC로 돌려보내지 않는다 (그게
왕복 인코딩/디코딩 때문에 지연의 가장 큰 원인이었음). 대신 마스크에서
바운딩박스만 뽑아서 아주 가벼운 JSON으로 DataChannel에 실어 보내고,
클라이언트가 자기 로컬(실시간) 비디오 위에 canvas로 그림 — server/server.py,
client/client.js 참고.

Perf notes:
- 매 프레임 추론하지 않고 `infer_every_n` 프레임마다만 실제로 모델을 돌리고,
  그 사이 프레임은 마지막 마스크를 재사용 (문서의 "무거운 재생성은 캐시로
  최소화" 원칙).
"""
import os
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "selfie_multiclass_256x256.tflite")

CATEGORY_BACKGROUND = 0
CATEGORY_HAIR = 1
CATEGORY_BODY_SKIN = 2
CATEGORY_FACE_SKIN = 3
CATEGORY_CLOTHES = 4
CATEGORY_OTHERS = 5


def _bbox(mask: np.ndarray):
    if not np.any(mask):
        return None
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return [int(x), int(y), int(w), int(h)]


class FaceHairSegmenter:
    def __init__(self, infer_every_n: int = 1):
        """infer_every_n=1 이면 매 프레임 추론(정확하지만 느림).
        2, 3 등으로 올리면 그 프레임 수마다 한 번만 추론하고 나머지는
        마지막 마스크를 재사용해서 bbox만 다시 계산 (빠르지만 빠르게
        움직일 때 살짝 지연됨)."""
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=True,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)
        self._start_ts = time.monotonic()
        self._infer_every_n = max(1, infer_every_n)
        self._frame_count = 0
        self._last_mask = None

    def _timestamp_ms(self) -> int:
        return int((time.monotonic() - self._start_ts) * 1000)

    def process(self, frame_bgr: np.ndarray):
        """Returns (result_dict, timings_dict). result_dict에는 hair_bbox /
        face_bbox([x,y,w,h] 또는 None)만 들어있음 - 그림은 클라이언트가 그림."""
        t0 = time.perf_counter()
        self._frame_count += 1
        do_infer = self._last_mask is None or self._frame_count % self._infer_every_n == 0

        infer_ms = None
        if do_infer:
            t1 = time.perf_counter()
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._segmenter.segment_for_video(mp_image, self._timestamp_ms())
            self._last_mask = result.category_mask.numpy_view()  # HxW uint8
            infer_ms = (time.perf_counter() - t1) * 1000

        category_mask = self._last_mask

        t2 = time.perf_counter()
        hair_bbox = _bbox(category_mask == CATEGORY_HAIR)
        face_bbox = _bbox(category_mask == CATEGORY_FACE_SKIN)
        composite_ms = (time.perf_counter() - t2) * 1000

        total_ms = (time.perf_counter() - t0) * 1000
        return (
            {"hair_bbox": hair_bbox, "face_bbox": face_bbox},
            {
                "did_infer": do_infer,
                "infer_ms": infer_ms,
                "composite_ms": composite_ms,
                "total_ms": total_ms,
            },
        )

    def close(self):
        self._segmenter.close()
