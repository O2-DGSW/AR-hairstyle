"""헤어 에셋: RGBA 이미지 + 눈 앵커 2점.

앵커로 눈 2점을 쓰는 이유: 두 점만으로 **닮음변환**(균등 스케일 + 회전 + 이동)이
정해진다. 3점 어파인은 전단(shear)까지 허용해서 세그멘테이션이 조금만 흔들려도
헤어가 기울어져 찌그러진다. 헤어에는 전단이 필요 없으므로 자유도를 줄이는 쪽이
훨씬 안정적이다.

에셋은 server/assets/*.png (RGBA) + 같은 이름의 .json 에서 읽는다:
    {"eyeL": [x, y], "eyeR": [x, y]}
파일이 없으면 절차적으로 생성한 기본 에셋을 쓴다.
"""
import json
import os
from collections import OrderedDict

import cv2
import numpy as np

from config import CONFIG

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
ASSET_SIZE = 512


# 증류 블렌더가 쓰는 눈 기준 정규화 크롭 규격.
# 학습(train/make_pairs.py)과 추론(gpu_segmenter)이 **반드시 같은 값**을 써야 한다.
# 어긋나면 학습 때와 다른 프레이밍이 들어가서 모델이 엉뚱하게 동작한다.
BLEND_CROP = 256
BLEND_EYE_Y = 0.42
BLEND_EYE_SPAN = 0.34


def blend_crop_matrix(eye_l, eye_r, size=BLEND_CROP):
    """눈 2점 -> 정규화 크롭으로 보내는 2x3 닮음변환."""
    dst_l = (size * (0.5 - BLEND_EYE_SPAN / 2), size * BLEND_EYE_Y)
    dst_r = (size * (0.5 + BLEND_EYE_SPAN / 2), size * BLEND_EYE_Y)
    return similarity_matrix(eye_l, eye_r, dst_l, dst_r)


def order_by_x(p, q):
    """두 점을 화면 x좌표 순으로 정렬해 (왼쪽, 오른쪽) 로 반환.

    좌우 라벨을 믿으면 안 되는 이유: CelebAMask-HQ의 l_eye/r_eye 는 **인물 기준**
    이라 화면에서는 뒤바뀌고, 모델이 각도/조명에 따라 두 눈 라벨을 헷갈리기도 한다.
    라벨이 한 번 뒤집히면 눈 방향벡터가 반대가 되어 닮음변환이 정확히 180° 돌아
    헤어가 뒤집힌다. 양쪽(에셋/얼굴)을 같은 기준으로 정렬하면 이 문제가 사라진다.
    """
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    return (p, q) if p[0] <= q[0] else (q, p)


class HairAsset:
    def __init__(self, name, rgba, eye_l, eye_r, ref_skin=None, yaw=None, bank=None,
                 scale_adjust=1.0):
        self.name = name
        # 다각도 뱅크용. yaw 가 있으면 런타임에 현재 각도와 비교해 고른다.
        self.yaw = yaw
        self.bank = bank
        # 에셋 자체의 크기 보정. HairFastGAN 은 머리보다 큰 헤어를 만드는
        # 경향이 있는데(참고사진 정규화로 6%만 줄어든다), 슬라이더로 매번
        # 맞추는 대신 에셋에 구워두면 다음 세션에도 유지된다.
        self.scale_adjust = float(scale_adjust)
        self.rgba = rgba                      # (H, W, 4) uint8, BGRA
        # 항상 화면 왼쪽 눈이 eye_l 이 되도록 정규화해서 보관한다.
        self.eye_l, self.eye_r = order_by_x(eye_l, eye_r)
        # 이 에셋을 뽑은 사진의 피부 평균색(BGR). 런타임에 현재 프레임의 피부색과
        # 비교해 조명/화이트밸런스 차이를 보정하는 기준으로 쓴다. 피부는 어느
        # 사진에나 있고 조명을 그대로 받으므로 조도의 대리 지표로 적합하다.
        self.ref_skin = None if ref_skin is None else np.asarray(ref_skin, dtype=np.float32)


def build_procedural() -> HairAsset:
    """기본 단발 에셋. 얼굴 비율 기준으로 좌표를 잡는다:
    눈 간격 D=112 를 1단위로 보고 헤어라인은 눈 위 1.0D, 정수리는 1.7D."""
    c = np.zeros((ASSET_SIZE, ASSET_SIZE, 4), dtype=np.uint8)
    eye_y, D = 300, 112
    cx = ASSET_SIZE // 2
    eye_l = (cx - D // 2, eye_y)
    eye_r = (cx + D // 2, eye_y)

    hairline = int(eye_y - 1.00 * D)
    crown = int(eye_y - 1.70 * D)
    half_out = int(1.38 * D)
    half_in = int(1.16 * D)
    bottom = int(eye_y + 1.05 * D)

    outer = np.array([[
        (cx - half_out, bottom),
        (cx - half_out - 6, int(eye_y - 0.35 * D)),
        (cx - int(0.95 * D), crown),
        (cx, crown - 10),
        (cx + int(0.95 * D), crown),
        (cx + half_out + 6, int(eye_y - 0.35 * D)),
        (cx + half_out, bottom),
    ]], dtype=np.int32)

    inner = np.array([[
        (cx - half_in, bottom),
        (cx - int(1.02 * D), int(eye_y - 0.30 * D)),
        (cx - int(0.55 * D), hairline),
        (cx, hairline - 8),
        (cx + int(0.55 * D), hairline),
        (cx + int(1.02 * D), int(eye_y - 0.30 * D)),
        (cx + half_in, bottom),
    ]], dtype=np.int32)

    mask = np.zeros((ASSET_SIZE, ASSET_SIZE), dtype=np.uint8)
    cv2.fillPoly(mask, outer, 255)
    cv2.fillPoly(mask, inner, 0)
    mask = cv2.GaussianBlur(mask, (0, 0), 3)   # 경계 부드럽게

    # 위에서 아래로 살짝 밝아지는 갈색 (BGR)
    shade = np.linspace(0.55, 1.0, ASSET_SIZE, dtype=np.float32)[:, None, None]
    base = np.array([34, 40, 62], dtype=np.float32) * shade
    c[..., :3] = np.broadcast_to(base, (ASSET_SIZE, ASSET_SIZE, 3)).clip(0, 255).astype(np.uint8)
    c[..., 3] = mask
    return HairAsset("procedural-bob", c, eye_l, eye_r)


def _read_asset(path):
    """<path>.png + 같은 이름의 .json -> HairAsset. 읽을 수 없으면 None."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    meta_path = os.path.splitext(path)[0] + ".json"
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        eye_l, eye_r = meta["eyeL"], meta["eyeR"]
        ref_skin = meta.get("refSkin")
        yaw, bank = meta.get("yaw"), meta.get("bank")
        scale_adjust = float(meta.get("scaleAdjust", 1.0))
    else:
        # 앵커 정보가 없으면 이미지 비율로 추정 (사용자가 슬라이더로 보정)
        h, w = img.shape[:2]
        eye_l, eye_r = (w * 0.39, h * 0.59), (w * 0.61, h * 0.59)
        ref_skin = None
        yaw = bank = None
        scale_adjust = 1.0
    name = os.path.splitext(os.path.basename(path))[0]
    return HairAsset(name, img, eye_l, eye_r, ref_skin, yaw, bank, scale_adjust)


def load_assets() -> dict:
    """server/assets 에서 에셋을 읽고, 항상 절차적 기본 에셋을 함께 제공.

    **여기서 CONFIG.generated_dir 를 읽지 않는 것은 의도적이다.** 생성 에셋은
    특정 사용자의 얼굴/피부톤/헤어라인이 그대로 구워진 생체정보에 가까운
    데이터다. 이걸 시작 시 공유 정적 목록에 올리면 다음에 접속한 남이
    목록에서 보고 자기 얼굴에 씌워볼 수 있다. 생성 에셋은 세션별 하위
    디렉터리에 남기고 load_asset_dir() 로 그 세션만 명시적으로 되살린다.
    """
    assets = {}
    default = build_procedural()
    assets[default.name] = default

    if not os.path.isdir(ASSET_DIR):
        return assets

    for fn in sorted(os.listdir(ASSET_DIR)):
        if not fn.lower().endswith(".png"):
            continue
        a = _read_asset(os.path.join(ASSET_DIR, fn))
        if a is not None:
            assets[a.name] = a
    return assets


def similarity_matrix(src0, src1, dst0, dst1, scale_mul=1.0, offset_up=0.0):
    """두 점 대응으로 닮음변환 2x3 행렬을 만든다.
    offset_up: 얼굴 기준 '위' 방향으로 픽셀 이동 (눈 간격 대비 비율이 아니라 px)."""
    sv = np.asarray(src1, np.float32) - np.asarray(src0, np.float32)
    dv = np.asarray(dst1, np.float32) - np.asarray(dst0, np.float32)
    s_len = float(np.linalg.norm(sv))
    d_len = float(np.linalg.norm(dv))
    if s_len < 1e-6 or d_len < 1e-6:
        return None

    scale = (d_len / s_len) * scale_mul
    ang = np.arctan2(dv[1], dv[0]) - np.arctan2(sv[1], sv[0])
    ca, sa = np.cos(ang) * scale, np.sin(ang) * scale
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)

    t = np.asarray(dst0, np.float32) - R @ np.asarray(src0, np.float32)
    if offset_up:
        # 얼굴 좌표계의 '위' = 눈 축에 수직인 방향
        ux, uy = dv / d_len
        t = t + np.array([uy, -ux], dtype=np.float32) * offset_up
    return np.hstack([R, t.reshape(2, 1)]).astype(np.float32)


#: 이어붙인 캔버스의 한 변 상한(px). 참고사진이 크면 워핑 결과가 커질 수 있다.
EXTEND_MAX_SIDE = 3000
#: 이음매 페더 폭. 눈 간격(D) 배수.
EXTEND_FEATHER_D = 0.25


def extend_hair_with_reference(gan_bgr, gan_hair, gan_eyes,
                               ref_bgr, ref_hair, ref_eyes,
                               feather_d: float = EXTEND_FEATHER_D):
    """GAN 결과의 헤어를 참고사진의 긴 머리로 아래쪽으로 이어붙인다.

    왜 필요한가
    -----------
    HairFastGAN 은 FFHQ 정렬된 1024 크롭 안에서만 동작한다. 그 크롭의 아래
    경계가 눈에서 약 2.5 D 아래인데(D = 눈 사이 거리), 가슴까지 오는 머리는
    4.6 D 까지 내려온다(실측). 즉 **머리 길이의 절반이 모델에 들어가기도 전에
    잘려 나간다.** 결과에 턱선 수평 절단이 보이는 이유다.

    크롭을 넓히는 건 답이 아니다 - FFHQ quad 배율을 1.15 에서 1.8 로만 올려도
    얼굴이 뭉개지고 2.4 면 완전히 무너진다(실측). StyleGAN2/e4e 가 FFHQ
    프레이밍을 전제로 학습돼서 벗어나면 잠재공간 인버전이 깨진다.

    그래서 잘린 아랫부분을 **참고사진의 실제 머리 픽셀**로 채운다. 두 이미지
    모두 눈 2점으로 정규화하면 같은 좌표계로 맞출 수 있다(=닮음변환).

    반환: (합성 BGR, 합성 머리 알파, 새 eye_l, 새 eye_r) 또는 실패 시 None
    """
    M = similarity_matrix(ref_eyes[0], ref_eyes[1], gan_eyes[0], gan_eyes[1])
    if M is None:
        return None

    gh, gw = gan_bgr.shape[:2]
    rh, rw = ref_bgr.shape[:2]

    # 워핑된 참고사진이 어디까지 뻗는지 보고 캔버스를 키운다.
    corners = np.array([[0, 0], [rw, 0], [rw, rh], [0, rh]], np.float32)
    warped = (M[:, :2] @ corners.T).T + M[:, 2]
    x0 = min(0.0, float(warped[:, 0].min())); x1 = max(float(gw), float(warped[:, 0].max()))
    y0 = min(0.0, float(warped[:, 1].min())); y1 = max(float(gh), float(warped[:, 1].max()))
    dx, dy = -x0, -y0
    W, H = int(np.ceil(x1 + dx)), int(np.ceil(y1 + dy))
    if W < 1 or H < 1 or max(W, H) > EXTEND_MAX_SIDE:
        return None

    # 캔버스 좌표 = GAN 좌표 + (dx, dy). 참고사진은 M 뒤에 같은 평행이동을 붙인다.
    Mt = M.copy(); Mt[0, 2] += dx; Mt[1, 2] += dy
    ref_rgb_w = cv2.warpAffine(ref_bgr, Mt, (W, H), flags=cv2.INTER_LINEAR)
    ref_a_w = cv2.warpAffine(ref_hair.astype(np.float32), Mt, (W, H),
                             flags=cv2.INTER_LINEAR)

    gan_rgb = np.zeros((H, W, 3), np.uint8)
    gan_a = np.zeros((H, W), np.float32)
    oy, ox = int(round(dy)), int(round(dx))
    gan_rgb[oy:oy + gh, ox:ox + gw] = gan_bgr
    gan_a[oy:oy + gh, ox:ox + gw] = gan_hair.astype(np.float32)

    eye_l = (float(gan_eyes[0][0]) + dx, float(gan_eyes[0][1]) + dy)
    eye_r = (float(gan_eyes[1][0]) + dx, float(gan_eyes[1][1]) + dy)
    D = float(np.hypot(eye_r[0] - eye_l[0], eye_r[1] - eye_l[1]))
    if D < 1e-3:
        return None

    # 이음매는 GAN 프레임의 **아래 경계**다. 거기서 머리가 잘렸으니까.
    band = max(4.0, feather_d * D)
    splice = oy + gh                       # GAN 이 끝나는 y
    ys = np.arange(H, dtype=np.float32).reshape(-1, 1)
    # splice-band 위: GAN(0), splice 아래: 참고사진(1)
    t = np.clip((ys - (splice - band)) / band, 0.0, 1.0)

    # 색 맞추기. 이음매 바로 위의 겹치는 구간에서 두 머리의 평균색 비를 쓴다.
    # GAN 은 색을 참고사진에서 가져오지만 조명이 달라 그대로는 띠가 보인다.
    lo, hi = max(0, int(splice - 3 * band)), min(H, int(splice))
    if hi > lo:
        gsel = gan_a[lo:hi] > 128
        rsel = ref_a_w[lo:hi] > 128
        if gsel.sum() > 200 and rsel.sum() > 200:
            gmean = gan_rgb[lo:hi][gsel].reshape(-1, 3).mean(0)
            rmean = ref_rgb_w[lo:hi][rsel].reshape(-1, 3).mean(0)
            ratio = np.clip(gmean / np.maximum(rmean, 1.0), 0.6, 1.7)
            ref_rgb_w = np.clip(ref_rgb_w.astype(np.float32) * ratio,
                                0, 255).astype(np.uint8)

    alpha = gan_a * (1.0 - t) + ref_a_w * t
    rgb = (gan_rgb.astype(np.float32) * (1.0 - t)[..., None]
           + ref_rgb_w.astype(np.float32) * t[..., None])
    # 알파가 거의 없는 곳은 색을 섞어봐야 의미가 없다. 있는 쪽 색을 그대로 쓴다.
    only_gan = (gan_a > 8) & (ref_a_w <= 8)
    only_ref = (ref_a_w > 8) & (gan_a <= 8)
    rgb[only_gan] = gan_rgb[only_gan]
    rgb[only_ref] = ref_rgb_w[only_ref]
    alpha[only_gan] = gan_a[only_gan]
    alpha[only_ref] = ref_a_w[only_ref]

    return (rgb.astype(np.uint8), np.clip(alpha, 0, 255).astype(np.uint8),
            eye_l, eye_r)


MIN_BLOB_RATIO = 0.08   # 최대 덩어리 대비 이보다 작은 조각은 잡음으로 버린다


def clean_hair_mask(mask: np.ndarray, feather: int = 3,
                    min_blob: float = MIN_BLOB_RATIO) -> np.ndarray:
    """hair 마스크 정리: 잡음 제거 -> 구멍 메우기 -> 경계 페더링."""
    m = (mask > 0).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest = areas.max()
        keep = np.zeros_like(m)
        for i, a in enumerate(areas, start=1):
            if a >= biggest * min_blob:
                keep[labels == i] = 1
        m = keep

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    alpha = (m * 255).astype(np.uint8)
    if feather > 0:
        # 하드 엣지로 합성하면 오려붙인 티가 심하게 난다.
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
    return alpha


def skin_mean(img_bgr, skin_mask):
    """피부 영역의 평균 BGR. 픽셀이 너무 적으면 None."""
    m = skin_mask.astype(bool)
    n = int(m.sum())
    if n < 200:
        return None
    return img_bgr[m].reshape(-1, 3).mean(axis=0).astype(np.float32)


def build_from_photo(img_bgr, hair_mask, eye_l, eye_r, name,
                     feather: int = 3, pad: float = 0.12,
                     min_blob: float = MIN_BLOB_RATIO, ref_skin=None):
    """사진 + 머리 마스크 + 눈 2점 -> HairAsset, 크롭된 눈 좌표.

    make_asset.py(오프라인 CLI)와 서버(GAN 결과에서 즉석 추출)가 공유한다.
    반환: (HairAsset, (eye_l, eye_r), alpha_px) / 머리를 못 찾으면 (None, None, 0)
    """
    alpha = clean_hair_mask(hair_mask, feather, min_blob)
    ys, xs = np.where(alpha > 8)
    if len(ys) == 0:
        return None, None, 0

    rgba = np.dstack([img_bgr, alpha])

    # 머리 영역 기준으로 여유를 두고 크롭 (여백이 있어야 경계가 자연스럽다)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    py, px = int((y1 - y0) * pad), int((x1 - x0) * pad)
    y0 = max(0, y0 - py); y1 = min(rgba.shape[0], y1 + py)
    x0 = max(0, x0 - px); x1 = min(rgba.shape[1], x1 + px)
    rgba = rgba[y0:y1, x0:x1]

    el = (float(eye_l[0] - x0), float(eye_l[1] - y0))
    er = (float(eye_r[0] - x0), float(eye_r[1] - y0))
    return (HairAsset(name, rgba, el, er, ref_skin), (el, er),
            int((alpha > 8).sum()))


def invert_affine(M):
    """2x3 어파인의 역변환. GPU 워핑은 '출력 픽셀 -> 입력 좌표' 방향이 필요하다."""
    A = M[:, :2]
    t = M[:, 2]
    Ainv = np.linalg.inv(A)
    return np.hstack([Ainv, (-Ainv @ t).reshape(2, 1)]).astype(np.float32)


def overlay(frame_bgr, asset: HairAsset, eye_l, eye_r, scale_mul=1.0, offset_up=0.0):
    """frame 위에 헤어 에셋을 닮음변환으로 얹어 알파 합성."""
    eye_l, eye_r = order_by_x(eye_l, eye_r)
    M = similarity_matrix(asset.eye_l, asset.eye_r, eye_l, eye_r, scale_mul, offset_up)
    if M is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    warped = cv2.warpAffine(asset.rgba, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = (warped[:, :, 3:4].astype(np.float32) / 255.0)
    out = frame_bgr.astype(np.float32) * (1 - a) + warped[:, :, :3].astype(np.float32) * a
    return out.clip(0, 255).astype(np.uint8)


def pick_by_yaw(assets: dict, bank: str, yaw: float):
    """뱅크에서 현재 yaw 에 가장 가까운 각도의 에셋을 고른다.

    각도별로 GAN 이 따로 생성한 것들이라, 고개를 돌리면 그 각도에서 실제로
    관측/생성된 헤어로 바뀐다. 닮음변환만으로는 만들 수 없는 평면 밖 회전이
    이 전환으로 근사된다.
    """
    cands = [a for a in assets.values() if a.bank == bank and a.yaw is not None]
    if not cands:
        return None
    return min(cands, key=lambda a: abs(a.yaw - yaw))


def list_banks(assets: dict):
    return sorted({a.bank for a in assets.values() if a.bank})


def pick_by_yaw_stable(assets: dict, bank: str, yaw: float, current=None,
                       margin: float = 3.0):
    """가장 가까운 각도의 에셋을 고르되, 경계에서 깜빡이지 않게 이력을 둔다.

    즉시 전환은 크로스페이드보다 또렷하다(섞으면 두 헤어가 반투명하게 겹쳐
    보인다). 대신 두 칸의 경계에 머리를 두면 매 프레임 왔다갔다 하며 깜빡인다.
    지금 쓰는 칸이 새 후보보다 margin 도 이상 나쁘지 않으면 그대로 유지해서
    이걸 막는다.
    """
    cands = [a for a in assets.values() if a.bank == bank and a.yaw is not None]
    if not cands:
        return None
    best = min(cands, key=lambda a: abs(a.yaw - yaw))
    if (current is not None and current.bank == bank and current.yaw is not None
            and abs(current.yaw - yaw) <= abs(best.yaw - yaw) + margin):
        return current
    return best


def pick_pair_by_yaw(assets: dict, bank: str, yaw: float):
    """현재 yaw 를 사이에 두는 두 에셋과 혼합 비율 t 를 반환.

    가장 가까운 하나만 고르면 10도 간격에서 하드 전환이 일어나 뚝뚝 끊긴다.
    양쪽을 각각 워핑해 t 로 섞으면 각도 변화가 연속적으로 보인다.
    반환: (a, b, t)  -> 결과 = a*(1-t) + b*t   (b 가 None 이면 a 만 사용)
    """
    cands = sorted([a for a in assets.values() if a.bank == bank and a.yaw is not None],
                   key=lambda a: a.yaw)
    if not cands:
        return None, None, 0.0
    if len(cands) == 1 or yaw <= cands[0].yaw:
        return cands[0], None, 0.0
    if yaw >= cands[-1].yaw:
        return cands[-1], None, 0.0
    for a, b in zip(cands, cands[1:]):
        if a.yaw <= yaw <= b.yaw:
            span = max(b.yaw - a.yaw, 1e-6)
            return a, b, float((yaw - a.yaw) / span)
    return cands[-1], None, 0.0


class AnchorSmoother:
    """눈 앵커를 (중심, 간격, 각도)로 분해해 각각 따로 시간축 평활화한다.

    왜 분해하는가
    -------------
    눈 두 점의 화면 좌표를 그대로 평활화하면 위치/크기/회전이 뒤섞여 뭉개진다.
    분해하면 성분별로 다른 세기를 줄 수 있는데, 이게 중요하다:

      위치  - 사람이 실제로 빠르게 움직이므로 반응이 빨라야 한다
      크기  - 실제로는 천천히만 변한다. 여기가 흔들리면 헤어가 커졌다 작아졌다
              하는 게 바로 눈에 띄므로 가장 강하게 잡는다
      각도  - 중간

    특히 크기는 두 곳에서 잡음이 들어온다. 세그멘테이션 눈 무게중심이 매 프레임
    떨리고, 거리 기반 보정 배율은 몇 프레임에 한 번만 갱신돼 계단식으로 튄다.
    최종 변환 성분에서 잡으면 두 원인을 한 번에 흡수한다.
    """

    # 성분별 "이 정도면 실제 움직임" 기준 속도(프레임당). 이보다 빠르면
    # 평활화를 거의 풀어 지연을 없애고, 느리면 강하게 잡아 떨림을 죽인다.
    #
    # 크기가 0 인 것은 적응을 끈다는 뜻이고, 의도적이다. 실측해 보면 사람이
    # 다가오는 속도(눈 간격 ~1.2px/프레임)가 관측 떨림(~1.1px)보다 느려서,
    # 속도로는 둘을 구분할 수가 없다. 적응을 켜면 떨림이 '움직임'으로 읽혀
    # 필터가 풀리고 오히려 더 떤다(실측 0.22 -> 0.62px). 크기만 고정으로 둔다.
    SPEED_REF = (5.0, 0.0, 0.08)      # 위치(px) / 크기(끔) / 각도(sin·cos 단위)

    def __init__(self, pos_a=0.12, scale_a=0.08, ang_a=0.10):
        self.base = (pos_a, scale_a, ang_a)
        self.pos_a, self.scale_a, self.ang_a = pos_a, scale_a, ang_a
        self.cx = self.cy = self.dist = None
        self.sin = self.cos = None
        # 속도 추정치. 관측 자체가 떨리므로 속도도 평활화해서 쓴다.
        self.v_pos = self.v_scale = self.v_ang = 0.0
        self._prev = None                 # 직전 '관측' 값 (평활화 전)

    def set_strength(self, k: float):
        """0 = 평활화 없음(가장 반응 빠름), 1 = 기본, 그 이상은 더 무겁게."""
        k = max(0.0, min(2.0, float(k)))
        if k <= 0:
            self.pos_a = self.scale_a = self.ang_a = 1.0
            return
        p, sc, a = self.base
        # k 가 클수록 알파가 작아진다(= 더 강한 평활화)
        self.pos_a = min(1.0, p / k)
        self.scale_a = min(1.0, sc / k)
        self.ang_a = min(1.0, a / k)

    def reset(self):
        self.cx = self.cy = self.dist = None
        self.sin = self.cos = None
        self.v_pos = self.v_scale = self.v_ang = 0.0
        self._prev = None

    def _adapt(self, alpha, speed, ref):
        """속도가 빠를수록 알파를 1 에 가깝게 올린다.

        고정 알파의 딜레마: 가만히 있을 때 떨림을 잡으려면 알파가 작아야 하고,
        움직일 때 안 늘어지려면 알파가 커야 한다. 한 값으로는 둘 다 안 된다.
        속도에 따라 바꾸면 멈춰 있을 땐 단단하고 움직일 땐 즉각적이다.
        """
        k = min(1.0, speed / ref) if ref > 0 else 0.0
        return alpha + (1.0 - alpha) * k * k    # 제곱: 미세한 떨림엔 반응하지 않게

    def update(self, eye_l, eye_r):
        el = np.asarray(eye_l, dtype=np.float32)
        er = np.asarray(eye_r, dtype=np.float32)
        cx, cy = float((el[0] + er[0]) / 2), float((el[1] + er[1]) / 2)
        vx, vy = float(er[0] - el[0]), float(er[1] - el[1])
        d = float(np.hypot(vx, vy))
        if d < 1e-3:
            return eye_l, eye_r
        s, c = vy / d, vx / d

        if self.cx is None:
            self.cx, self.cy, self.dist, self.sin, self.cos = cx, cy, d, s, c
            self._prev = (cx, cy, d, s, c)
        else:
            # 관측 속도를 먼저 갱신한다(관측값끼리의 차이여야 실제 움직임이다.
            # 평활화된 값과 비교하면 자기 지연을 속도로 착각한다)
            pcx, pcy, pd, ps, pc = self._prev
            self.v_pos += (float(np.hypot(cx - pcx, cy - pcy)) - self.v_pos) * 0.35
            self.v_scale += (abs(d - pd) - self.v_scale) * 0.35
            self.v_ang += (float(np.hypot(s - ps, c - pc)) - self.v_ang) * 0.35
            self._prev = (cx, cy, d, s, c)

            rp, rs, ra = self.SPEED_REF
            pa = self._adapt(self.pos_a, self.v_pos, rp)
            sa = self._adapt(self.scale_a, self.v_scale, rs)
            aa = self._adapt(self.ang_a, self.v_ang, ra)
            self.cx += (cx - self.cx) * pa
            self.cy += (cy - self.cy) * pa
            self.dist += (d - self.dist) * sa
            # 각도는 sin/cos 로 다뤄야 ±180도 경계에서 튀지 않는다
            self.sin += (s - self.sin) * aa
            self.cos += (c - self.cos) * aa
            n = float(np.hypot(self.sin, self.cos)) or 1.0
            self.sin, self.cos = self.sin / n, self.cos / n

        hx, hy = self.cos * self.dist / 2, self.sin * self.dist / 2
        return ((self.cx - hx, self.cy - hy), (self.cx + hx, self.cy + hy))


# ---------------------------------------------------------------------------
# 세션 스코프 에셋 레지스트리
# ---------------------------------------------------------------------------

class AssetRegistry:
    """정적 에셋(전 세션 공유, 읽기 전용) + 세션에서 생성된 에셋(세션 스코프, LRU).

    세션마다 하나씩 만든다. 생성 에셋은 이 세션에서만 보이고 세션이 끝나면 축출된다.

    왜 이렇게 나누는가
    ------------------
    원래는 모듈 전역 dict 하나를 모든 연결이 공유했다. 세 가지가 동시에 터진다:
      - GAN 이 만든 에셋이 계속 쌓이기만 하고 지워지지 않는다(프로세스 수명 내내).
      - A 가 만든 에셋에는 A 의 얼굴/피부톤/헤어라인이 그대로 구워져 있는데
        그게 B 의 에셋 목록에 뜬다. 남의 머리를 자기 얼굴에 씌워볼 수 있다.
      - 재시작하면 통째로 사라진다.
    정적 에셋은 공유해도 되는 것들뿐이므로 그대로 두고, 생성분만 세션에
    가둔다. 영속화는 save_asset()/load_asset_dir() 가 세션별 디렉터리로 한다.
    """

    def __init__(self, static: dict, max_session: int = None, on_evict=None):
        # static 은 서버 전체가 공유하는 dict 다. 여기서 절대 쓰지 않는다 -
        # 한 세션이 넣은 값이 다른 세션에 새는 경로가 바로 이것이었다.
        self._static = static if static is not None else {}
        self._max = int(CONFIG.session_asset_max if max_session is None else max_session)
        self._on_evict = on_evict
        self._session = OrderedDict()
        self._closed = False

    # ---- 조회 ----
    def get(self, name):
        a = self._session.get(name)
        if a is not None:
            self._session.move_to_end(name)      # 최근 사용 표시
            return a
        return self._static.get(name)

    def default(self):
        """첫 정적 에셋. 비어 있으면 None.

        예전 코드가 next(iter(ASSETS.values())) 를 썼는데, 에셋 디렉터리가
        비면 StopIteration 이 프레임 루프 한가운데서 터진다. 절차적 기본
        에셋이 항상 있긴 하지만 그건 load_assets() 의 구현 세부지 계약이 아니다.
        """
        for a in self._static.values():
            return a
        return None

    def __contains__(self, name):
        return name in self._session or name in self._static

    def __getitem__(self, name):
        a = self.get(name)
        if a is None:
            raise KeyError(name)
        return a

    def names(self) -> list:
        out = list(self._static.keys())
        out += [n for n in self._session if n not in self._static]
        return out

    def session_names(self) -> list:
        return list(self._session.keys())

    def banks(self) -> list:
        return sorted({a.bank for a in self.values() if a.bank})

    # dict 처럼 굴게 해서 pick_by_yaw() 계열과 list_banks() 에 그대로 넘길 수 있게 한다.
    # 그 함수들의 시그니처를 바꾸면 오프라인 스크립트와 테스트가 전부 깨진다.
    def keys(self):
        return self.names()

    def values(self):
        m = self._merged()
        return [m[n] for n in self.names()]

    def items(self):
        m = self._merged()
        return [(n, m[n]) for n in self.names()]

    def __iter__(self):
        return iter(self.names())

    def __len__(self):
        return len(self.names())

    def _merged(self):
        m = dict(self._static)
        m.update(self._session)      # 같은 이름이면 세션 것이 이긴다
        return m

    # ---- 등록 / 정리 ----
    def add(self, asset) -> None:
        self._session[asset.name] = asset
        self._session.move_to_end(asset.name)
        while len(self._session) > self._max:
            name, _ = self._session.popitem(last=False)
            self._evict(name)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while self._session:
            name, _ = self._session.popitem(last=False)
            self._evict(name)

    def _evict(self, name):
        """축출 훅은 실패해도 삼킨다. GPU 캐시 무효화가 던진다고 해서
        세션 정리 자체가 중단되면 나머지 에셋이 통째로 남는다."""
        if self._on_evict is None:
            return
        try:
            self._on_evict(name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 생성 에셋 영속화
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """파일명으로 쓸 수 있게 정리. 에셋 이름은 클라이언트가 보낸 문자열에서
    파생될 수 있어서, 그대로 경로에 붙이면 ../ 로 디렉터리를 빠져나간다."""
    base = os.path.basename(str(name)).strip()
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in base]
    out = "".join(keep).lstrip(".")
    return out or "asset"


def save_asset(asset, directory) -> str:
    """<name>.png(RGBA) + <name>.json 으로 저장하고 png 경로를 반환.

    JSON 은 load_assets() 가 읽는 {"eyeL", "eyeR"} 포맷 그대로에 필드를
    더한 것이다. 추가 필드는 전부 optional 이라 옛 파일도 그대로 읽힌다.
    """
    os.makedirs(directory, exist_ok=True)
    name = _safe_name(asset.name)
    png = os.path.join(directory, name + ".png")

    rgba = asset.rgba
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"에셋 {asset.name!r} 이 RGBA 가 아닙니다: {rgba.shape}")
    if not cv2.imwrite(png, rgba):
        raise IOError(f"에셋 PNG 저장 실패: {png}")

    meta = {
        "eyeL": [float(asset.eye_l[0]), float(asset.eye_l[1])],
        "eyeR": [float(asset.eye_r[0]), float(asset.eye_r[1])],
        "refSkin": (None if asset.ref_skin is None
                    else [float(v) for v in np.asarray(asset.ref_skin).reshape(-1)]),
        "yaw": None if asset.yaw is None else float(asset.yaw),
        "bank": asset.bank,
        "scaleAdjust": float(asset.scale_adjust),
    }
    with open(os.path.splitext(png)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return png


def load_asset_dir(directory) -> dict:
    """디렉터리 하나(=세션 하나)의 에셋을 dict 로 읽는다. 재귀하지 않는다.

    load_assets() 와 달리 절차적 기본 에셋을 끼워넣지 않는다 - 이건 이미
    레지스트리의 정적 쪽에 들어 있고, 여기 결과는 세션 쪽에 넣을 것이라
    같은 이름이 양쪽에 생기면 목록에 중복으로 보인다.
    """
    out = {}
    if not os.path.isdir(directory):
        return out
    for fn in sorted(os.listdir(directory)):
        if not fn.lower().endswith(".png"):
            continue
        try:
            a = _read_asset(os.path.join(directory, fn))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            # 저장 도중 프로세스가 죽으면 png 만 있고 json 이 잘려 있을 수 있다.
            # 한 장 때문에 세션 복원 전체를 포기할 이유는 없다.
            continue
        if a is not None:
            out[a.name] = a
    return out


def prune_dir(directory, max_mb) -> int:
    """오래된 것부터 지워 디렉터리를 상한 이하로 만든다. 지운 에셋 수를 반환.

    세션별 하위 디렉터리까지 재귀한다(생성 에셋이 거기 쌓인다). 이게 없으면
    DataChannel 명령 하나로 디스크를 무한히 채울 수 있다.
    """
    if not os.path.isdir(directory):
        return 0
    limit = int(max_mb) * 1024 * 1024

    entries, total = [], 0
    for root, _dirs, files in os.walk(directory):
        for fn in files:
            path = os.path.join(root, fn)
            try:
                st = os.stat(path)
            except OSError:
                continue
            total += st.st_size
            if fn.lower().endswith(".png"):
                meta = os.path.splitext(path)[0] + ".json"
                entries.append((st.st_mtime, path, meta))

    entries.sort()
    removed = 0
    for _mt, png, meta in entries:
        if total <= limit:
            break
        for path in (png, meta):
            try:
                total -= os.path.getsize(path)
                os.remove(path)
            except OSError:
                pass
        removed += 1

    # 다 비워진 세션 디렉터리는 같이 치운다. 남겨두면 세션 수만큼 빈 폴더가 쌓인다.
    for root, dirs, _files in os.walk(directory, topdown=False):
        if root == directory:
            continue
        try:
            if not os.listdir(root):
                os.rmdir(root)
        except OSError:
            pass
    return removed
