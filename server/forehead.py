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
#: 인페인트/패치는 눈 위 이 높이(눈 간격 배수)까지만 - 이마+헤어라인. 그 위(정수리)는 3D 헤어가 덮고,
#: 거기까지 채우면 LaMa 가 배경/머리색으로 메운 어두운 영역이 고개 돌릴 때 헤어 밖으로 삐져나와
#: 검은 덩어리가 된다(실측). 원래 머리가 그 위에 남으면 런타임 플레이트/피부 평면이 처리한다.
BAND_UP = 1.4
#: 인페인트할 때 크롭 안의 머리 전체를 가려 머리 문맥을 없앤다(앞머리가 이마로 다시 그려지는 것 방지).
HIDE_HAIR_CONTEXT = True
#: 앞머리 자리 채우는 방법. "synth": 주변 실제 피부 확산 + 피부결 + 눈썹 그리기(기본).
#: "lama": LaMa 인페인트. 웹캠 해상도에서 이마만 한 영역은 LaMa 가 머리 결을 다시 그리거나(머리 문맥)
#: 배경/갈색 얼룩을 끌어와(문맥 제거) 둘 다 부자연스러웠다(2026-09-30 실측 비교).
FILL_METHOD = "synth"
#: 앞머리를 지우는 띠의 아래 끝 (눈 아래, 눈 간격 배수).
BANGS_DOWN = 0.35
#: 인페인터 입력 크롭: 눈 간격의 몇 배를 한 변으로 (얼굴 전체가 문맥으로 들어가야 색/음영이 맞는다).
CROP_EYES = 7.0
CROP_PX = 512
#: 얼굴 패치로 쓰는 파싱 클래스: 피부(1), 눈썹(2,3), 코(10). asset_extract 는 1~12 전부를 썼는데
#: 거기엔 안경(6)·귀(7,8,9)·눈(4,5)이 들어간다. 정면에서 구워진 안경테/귀가 고개를 돌리면 닮음변환으로
#: 엉뚱한 자리에 찍혀 검은 덩어리가 됐다(실측). 이마 채움엔 피부와 눈썹이면 충분하다.
FACE_CLS = (1, 2, 3, 10)
#: 3D 투영 패치에 더 담는 눈(4,5)·안경(6), 그리고 그 띠의 아래 끝(눈 아래, 눈 간격 배수).
EYE_CLS = (4, 5, 6)
EYE_BAND_DOWN = 0.6


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


def _pushpull(img: np.ndarray, w: np.ndarray) -> np.ndarray:
    """가중 w(0~1) 인 픽셀 값을 빈 곳으로 확산(피라미드 push-pull). img (h,w,3) float."""
    pyr = []
    c = img * w[..., None]; ww = w.copy()
    while min(ww.shape) > 4:
        pyr.append((c, ww))
        c = cv2.pyrDown(c); ww = cv2.pyrDown(ww)
    cur = c / np.maximum(ww, 1e-6)[..., None]
    for c, ww in reversed(pyr):
        up = cv2.resize(cur, (ww.shape[1], ww.shape[0]), interpolation=cv2.INTER_LINEAR)
        k = np.clip(ww * 50.0, 0.0, 1.0)[..., None]
        cur = (c / np.maximum(ww, 1e-6)[..., None]) * k + up * (1.0 - k)
    return cur


def synthesize_forehead(frame_bgr, cls, fill, eye_l, eye_r, brows=None):
    """앞머리 자리(fill)를 맨이마로 채운 프레임을 돌려준다. 생성 모델 없이 결정적으로:

    1) 색/음영: 주변 **실제 피부**(볼·코·보이는 이마, 머리 경계 헤일로는 깎음)를 push-pull 로 확산.
       빈 곳은 가까운 피부의 부드러운 연장이 된다 - 머리 결이 끼어들 여지가 없다.
    2) 피부결: 볼 피부의 고주파를 떼어 타일로 얹는다(확산만 하면 밀랍처럼 매끈하다).
    3) 눈썹: 앞머리에 가려 있던 눈썹을 메시 눈썹선으로 그린다. 눈썹 없는 이마는 매우 이상하다.
       보이는 눈썹은 원본 그대로라 fill 밖이면 건드리지 않는다.
    """
    h, w = frame_bgr.shape[:2]
    img = frame_bgr.astype(np.float32)
    d = float(np.linalg.norm(np.asarray(eye_r, np.float32) - np.asarray(eye_l, np.float32)))
    c = (np.asarray(eye_l, np.float32) + np.asarray(eye_r, np.float32)) / 2.0

    hair = (cls == CLS_HAIR).astype(np.uint8)
    halo = cv2.dilate(hair, np.ones((5, 5), np.uint8)) > 0          # 머리 경계 어두운 헤일로
    skin = np.isin(cls, (1, 10)) & ~halo & ~fill
    skin = cv2.erode(skin.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    base = _pushpull(img, skin.astype(np.float32))

    # 눈 자체는 절대 채우지 않는다: 파서가 앞머리+안경 아래 눈을 '머리'로 찍는 프레임이 있어
    # 그대로 두면 눈이 살색으로 지워졌다(실측). 눈마다 기하 타원(눈 간격 비례)을 뺀다.
    ys_, xs_ = np.mgrid[0:h, 0:w].astype(np.float32)
    for e in (eye_l, eye_r):
        r = ((xs_ - e[0]) / (0.42 * d)) ** 2 + ((ys_ - e[1]) / (0.26 * d)) ** 2
        fill = fill & (r > 1.0)
    # 피부결은 여기서 넣지 않는다 - 볼 조각을 타일로 깔았더니 안경테 곡선이 무늬로 반복됐다(실측).
    # 런타임이 카메라 노이즈(그레인)를 얹는다.

    out = img.copy()
    out[fill] = base[fill]

    if brows is not None:
        # 눈썹색: 보이는 눈썹 픽셀이 있으면 그 중앙값, 없으면 피부를 어둡게 + 머리색 쪽으로
        bm = np.isin(cls, (2, 3)) & ~fill
        if bm.sum() > 30:
            col = np.median(img[bm], axis=0)
        else:
            sk = np.median(img[skin], axis=0) if skin.any() else np.array([120, 140, 170], np.float32)
            hc = np.median(img[hair > 0], axis=0) if hair.any() else sk * 0.3
            col = 0.35 * sk * 0.5 + 0.65 * hc
        for up, lo in brows:
            m = np.zeros((h, w), np.float32)
            poly = np.concatenate([up, lo[::-1]], 0).astype(np.int32)
            cv2.fillPoly(m, [poly], 1.0, lineType=cv2.LINE_AA)
            # 안쪽(머리)이 진하고 꼬리로 옅어지게
            inner, outer = up[0], up[-1]
            ax = outer - inner; L = float(np.hypot(*ax)) + 1e-3
            t = np.clip(((xs_ - inner[0]) * ax[0] + (ys_ - inner[1]) * ax[1]) / (L * L), 0, 1)
            m = np.clip(cv2.GaussianBlur(m, (0, 0), max(0.8, 0.03 * d)) * 1.3, 0, 1) * (0.92 - 0.40 * t)
            m = m * cv2.GaussianBlur(fill.astype(np.float32), (0, 0), 1.5)   # 가려졌던 자리만
            out = out * (1 - m[..., None]) + col[None, None, :] * m[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def build_forehead_asset(inpainter: ForeheadInpainter, frame_bgr: np.ndarray, cls: np.ndarray,
                         eye_l, eye_r, name: str, dilate: int = 7, matrix=None, brows=None):
    """프레임 + 파싱 + 눈 2점 -> (얼굴 패치 HairAsset, 앞머리 px, 인페인트 ms).

    반환 에셋은 .face 가 자기 자신이다 - gpu_segmenter 의 그룸 분기는 asset.face /
    asset.ref_skin / asset.scale_adjust 만 보므로 2D 헤어 본체 없이 이걸 그대로 넘긴다.

    matrix: 이 프레임의 MediaPipe 4x4. 주어지면 런타임이 패치를 눈 앵커 닮음변환 대신 그룸의
    두상 오클루더에 **투영**해 붙인다(asset.build_matrix / bald_bgr / face_mask_full). 닮음변환은
    고개를 25° 만 돌려도 정면 이마가 평면 스티커처럼 어긋났다(실측).
    """
    h, w = frame_bgr.shape[:2]
    eye_l = np.asarray(eye_l, np.float32); eye_r = np.asarray(eye_r, np.float32)
    d = float(np.linalg.norm(eye_r - eye_l))
    c = (eye_l + eye_r) / 2.0
    zone = zone_mask(eye_l, eye_r, h, w)
    ys = np.arange(h, dtype=np.float32)[:, None]
    # 이마 띠: 눈선(+0.15d 여유) 부터 눈 위 BAND_UP·d 까지. 그 아래(눈·안경·볼·입)는 원래 머리가 덮지
    # 않는 자리라 채울 이유가 없고, 넣으면 정면 얼굴이 통째로 구워져 고개 돌릴 때 덧씌워진다(실측).
    band = (ys > c[1] - BAND_UP * d) & (ys < c[1] + 0.15 * d)
    # 앞머리 지우기 띠는 눈 아래까지 내린다: 눈썹·안경테를 덮고 내려온 앞머리가 남아 있었다(사용자
    # 피드백). 눈/안경 자체는 파싱상 머리가 아니므로 안 걸린다.
    bangs_band = (ys > c[1] - BAND_UP * d) & (ys < c[1] + BANGS_DOWN * d)
    bangs = ((cls == CLS_HAIR) & zone & bangs_band).astype(np.uint8) * 255
    if dilate > 1:
        bangs = cv2.dilate(bangs, np.ones((dilate, dilate), np.uint8))
    bangs_px = int((bangs > 0).sum())

    bald = frame_bgr.copy()
    ms = 0.0
    if bangs_px > 0 and FILL_METHOD == "synth":
        t0 = time.perf_counter()
        bald = synthesize_forehead(frame_bgr, cls, bangs > 0, eye_l, eye_r, brows)
        ms = (time.perf_counter() - t0) * 1000
    elif bangs_px > 0:
        # 얼굴 크롭 -> 512 -> 인페인트 -> 되돌려 붙임 (마스크 안만)
        S = int(round(d * CROP_EYES))
        x0 = int(round(c[0] - S / 2)); y0 = int(round(c[1] - S * 0.4))
        x0 = max(0, min(w - S, x0)); y0 = max(0, min(h - S, y0))
        S = min(S, w - x0, h - y0)
        crop = frame_bgr[y0:y0 + S, x0:x0 + S]
        mcrop = bangs[y0:y0 + S, x0:x0 + S]
        ctx = crop
        if HIDE_HAIR_CONTEXT:
            # 이마 띠 밖의 머리(정수리/옆머리)를 **살색으로 칠한** 입력을 준다 = 대머리 문맥.
            # 이마 띠만 가리면 그 위 머리가 문맥으로 남아 LaMa 가 앞머리를 이마로 다시 이어 그렸고
            # ("반투명 앞머리 유령", 실측), 머리 전체를 가리면 이번엔 배경색을 끌어와 이마가 어두워졌다.
            # 위아래가 다 피부면 피부로 잇는다. 결과는 face_mask(이마 띠) 안만 쓴다.
            hair_all = (cls[y0:y0 + S, x0:x0 + S] == CLS_HAIR).astype(np.uint8)
            if dilate > 1:
                hair_all = cv2.dilate(hair_all, np.ones((dilate, dilate), np.uint8))
            skin = cls[y0:y0 + S, x0:x0 + S] == 1
            if skin.sum() > 50:
                tone = crop[skin].reshape(-1, 3).astype(np.float32)
                tone = np.median(tone, axis=0)
                ctx = crop.copy()
                ctx[(hair_all > 0) & (mcrop == 0)] = tone.astype(np.uint8)
        t0 = time.perf_counter()
        out = inpainter.inpaint(cv2.resize(ctx, (CROP_PX, CROP_PX), interpolation=cv2.INTER_CUBIC),
                                cv2.resize(mcrop, (CROP_PX, CROP_PX), interpolation=cv2.INTER_NEAREST))
        ms = (time.perf_counter() - t0) * 1000
        out = cv2.resize(out, (S, S), interpolation=cv2.INTER_AREA)
        mm = bangs[y0:y0 + S, x0:x0 + S] > 0        # 붙여 넣는 건 이마 띠의 앞머리 자리만
        crop_out = crop.copy(); crop_out[mm] = out[mm]
        bald[y0:y0 + S, x0:x0 + S] = crop_out

    # 얼굴 패치 = 원래 얼굴 부위 ∪ 방금 채운 자리. asset_extract 와 같은 규칙, 같은 눈 앵커.
    face_mask = ((np.isin(cls, FACE_CLS) & band) | (bangs > 0)).astype(np.uint8)
    if matrix is not None:
        # 3D 투영 경로는 눈/안경 띠까지 담는다. 고개를 돌리면 옆으로 내려온 앞머리가 눈 위에 걸리고,
        # 그걸 지운 자리를 평면 살색으로 칠하면 눈이 뭉개져 이중노출처럼 보였다(실측, yaw 25°).
        # 투영은 두상 곡면을 따라가므로 닮음변환 때의 "안경이 엉뚱한 데 찍힘" 이 크게 줄고,
        # 이 패치는 지운 자리에만 쓰인다.
        eyes_band = (ys > c[1] - BAND_UP * d) & (ys < c[1] + EYE_BAND_DOWN * d)
        face_mask |= (np.isin(cls, FACE_CLS + EYE_CLS) & zone & eyes_band).astype(np.uint8)
    ref_skin = hair_asset.skin_mean(bald, cls == 1)
    asset, _, px = hair_asset.build_from_photo(bald, face_mask, eye_l, eye_r, name, ref_skin=ref_skin)
    if asset is None:
        return None, bangs_px, ms
    asset.face = asset
    asset.bald_bgr = bald            # 프레임 크기. 3D 투영 경로의 텍스처이기도 하다
    asset.build_matrix = None if matrix is None else np.asarray(matrix, np.float32).reshape(4, 4)
    # 투영 경로의 알파: 패치 마스크를 눈 간격 비례로 부드럽게 (닮음변환 경로의 에셋 알파와 같은 역할)
    k = max(3, int(round(0.12 * d)) | 1)
    asset.face_mask_full = cv2.GaussianBlur(face_mask.astype(np.float32), (k, k), 0)
    return asset, bangs_px, ms
