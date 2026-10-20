"""손 가림: 손이 3D 헤어 앞에 오면 헤어를 손 뒤로 보낸다.

셀카에서 머리 근처 손은 거의 항상 머리보다 카메라 쪽이라, 손 자리에서는 새 헤어 알파와 원래 머리 지우기를
뺀다(손이 그대로 보인다). 손은 MediaPipe 손 랜드마커 21점(VIDEO 추적 모드, CPU ~18ms, 입력 해상도와 무관)
으로 찾고, 점에서 손바닥 다각형 + 손가락 마디 띠 + 손목 토막을 그려 마스크를 만든다.

영상 경로를 막지 않게: 세션마다 HandTracker 하나를 앱의 hand_executor 스레드에서 돌리고(앞 작업이 끝났을 때만
새로 던진다), 프레임 경로는 끝나 있는 최신 결과만 쓴다(1~2 프레임 늦음 - 빠르게 움직이면 손 가장자리가 잠깐
헤어 뒤로 들어간다). 마스크는 그
스레드에서 1/4 해상도로 그려 별도 CUDA 스트림으로 올려 두므로 프레임 경로에 CPU<->GPU 동기화가 없다.
합성 쪽(gpu_segmenter)은 이 마스크에 '이 프레임 파서가 머리가 아니라고 한 자리'를 곱한다 - 마스크가 손보다
넓거나 늦어도 실제 머리 위에서는 헤어를 빼지 않는다(원래 머리가 비치지 않게).
"""
from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

TASK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")
#: 랜드마커 입력 긴 변 (속도는 거의 같고 - 모델 입력이 고정 - 줄이는 비용만 준다)
IN_PX = 480
#: 마스크 해상도 (프레임 대비)
MASK_SCALE = 0.25
#: 손가락 (마디 순서), 손바닥 다각형
FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
PALM = (0, 1, 2, 5, 9, 13, 17)


def draw_hands(hands, shape) -> np.ndarray:
    """손 점 목록 [(21,2) px] -> uint8 마스크 0/255 (shape). 손 크기 = 검지 뿌리~새끼 뿌리 거리 기준."""
    m = np.zeros(shape, np.uint8)
    big = 0.0
    for p in hands:
        p = np.asarray(p, np.float32)
        palm = float(np.linalg.norm(p[5] - p[17])) + 1e-3
        big = max(big, palm)
        fw = max(2, int(round(0.32 * palm)))                      # 손가락 굵기 (마디 띠 폭)
        cv2.fillPoly(m, [np.round(cv2.convexHull(p[list(PALM)]) * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
        for f in FINGERS:
            q = np.round(p[list(f)] * 4).astype(np.int32)
            cv2.polylines(m, [q], False, 255, fw, cv2.LINE_AA, shift=2)
            cv2.line(m, tuple(np.round(p[0] * 4).astype(int)), tuple(q[0]), 255, fw, cv2.LINE_AA, shift=2)
        # 손목 쪽 짧은 토막만 (손바닥 길이의 0.5배). 1.5배 팔뚝 띠는 손이 기울면 실제 팔이 아니라 얼굴 옆을
        # 지나가 그 자리 헤어를 뚫었다(Slovo 실측 - 얼굴 피부는 파서로 머리와 구별돼도 손과는 구별 못 한다)
        v = p[0] - p[9]
        e = p[0] + 0.5 * v
        cv2.line(m, tuple(np.round(p[0] * 4).astype(int)), tuple(np.round(e * 4).astype(int)), 255,
                 max(2, int(round(0.7 * palm))), cv2.LINE_AA, shift=2)
    k = int(0.08 * big) | 1
    if k > 1:
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
    return m


class HandTracker:
    """세션 하나의 손 추적 (VIDEO 모드라 상태가 있다 - 세션마다 하나). track() 은 한 스레드에서만 부른다."""

    def __init__(self, device: str = "cuda"):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        self.device = device
        self._lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=TASK), running_mode=vision.RunningMode.VIDEO,
            num_hands=2, min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4))
        self._stream = torch.cuda.Stream() if device == "cuda" else None
        self._last_ts = -1

    def track(self, frame_bgr: np.ndarray, t: float):
        """-> {"mask": (1,1,h/4,w/4) float GPU 0~1, "t", "n" 손 수, "ms"} (손 없으면 mask None)."""
        import mediapipe as mp
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        s = IN_PX / max(h, w)
        small = cv2.resize(frame_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        ts = max(self._last_ts + 1, int(t * 1000))          # VIDEO 모드는 시각이 단조 증가해야 한다
        self._last_ts = ts
        res = self._lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))),
            ts)
        mh, mw = max(1, round(h * MASK_SCALE)), max(1, round(w * MASK_SCALE))
        hands = [np.float32([[q.x * mw, q.y * mh] for q in hd]) for hd in res.hand_landmarks]
        out = {"t": t, "n": len(hands), "mask": None}
        if hands:
            # 결과는 한두 프레임 뒤에 쓰인다. 움직임만큼 앞 자리를 같이 그리거나 팽창하면 빠르게 움직일 때 손이
            # 없는 얼굴/머리 자리까지 헤어가 뚫렸다(Slovo 실측) - 손이 잠깐 헤어 뒤로 들어가는 쪽이 덜 거슬린다.
            draw = hands
            m = draw_hands(draw, (mh, mw))
            m = cv2.GaussianBlur(m, (0, 0), 1.0)
            mt = torch.from_numpy(m)
            if self._stream is not None:
                with torch.cuda.stream(self._stream):
                    g = mt.to(self.device).float().div_(255.0)[None, None]
                self._stream.synchronize()
            else:
                g = mt.float().div_(255.0)[None, None]
            out["mask"] = g
        out["ms"] = (time.perf_counter() - t0) * 1000
        return out

    def close(self):
        try:
            self._lm.close()
        except Exception:
            logger.exception("손 랜드마커 close 실패")
