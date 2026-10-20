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
#: 포즈 뼈대 (POSE_IDS 안 순번): 어깨-어깨, 어깨-팔꿈치, 어깨-엉덩이
_P = {k: i for i, k in enumerate(POSE_IDS)}
BONES = ((_P[11], _P[12]), (_P[11], _P[13]), (_P[12], _P[14]), (_P[11], _P[23]), (_P[12], _P[24]))
#: 런타임 기억 쓰기 조건: 머리(확률 0.1)에서 떨어질 거리(d 배수), 지난 호출과의 색 차 상한(0~255 평균, 255=끔).
#: 안정 확인은 끔: 손은 hand_mask 가 거르고, 3Hz 에서는 두 번 연속 같은 색인 픽셀이 드물어 기억이 구멍의 1% 만
#: 남았다(Slovo, 정답 대용 L1 이득 0). 분할이 계속 놓친 머리는 안정 확인으로도 못 거른다.
MEM_HAIR_GAP, MEM_STABLE = 0.25, 255
#: 머리색 걸러내기: 이 프레임의 확실한 머리(확률 0.8) Lab 색 분포에서 마할라노비스 거리가 이보다 가까운 픽셀은
#: 기억에 안 쓴다 (0=끔). 셀피 분할이 계속 놓친 머리(어두운 니트 위 검은 머리)가 기억에 들어가 구멍에 되살아났다.
MEM_HAIR_COL = 3.0
MEM_HAIR_DARK = True
#: MediaPipe Pose 손 점 (손목, 새끼/검지/엄지)
HAND_IDS = (15, 16, 17, 18, 19, 20, 21, 22)
#: MediaPipe 손 랜드마커 (21점, CPU ~35ms @960). 없으면 포즈 손 점 원으로만 가린다.
HAND_TASK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")
#: 셀피 분류 (segmenter.CATEGORY_*)
S_BG, S_HAIR, S_BODY, S_FACE, S_CLOTH, S_OTHER = range(6)


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


def pose_guide(pts, vis, px: int = NET_PX) -> np.ndarray:
    """크롭 좌표 포즈 점 (len(POSE_IDS), 2) + 가시도 -> 뼈대 안내 (px, px) float32 (학습 torso_pairs.pose_guide 와
    같은 굵기/흐림, 흔들기만 없다)."""
    import cv2
    d = px / CROP_D
    g = np.zeros((px, px), np.float32)
    if pts is None:
        return g
    for a, b in BONES:
        if vis[a] > 0.3 and vis[b] > 0.3:
            cv2.line(g, tuple(np.round(pts[a] * 4).astype(int)), tuple(np.round(pts[b] * 4).astype(int)),
                     1.0, max(1, int(0.25 * d)), cv2.LINE_AA, shift=2)
    return cv2.GaussianBlur(g, (0, 0), 0.15 * d)


def hand_mask(hands: dict, skin: np.ndarray, d: float) -> np.ndarray:
    """손/팔뚝 마스크 (skin 과 같은 크기 bool). 기억에 손이 들어가면 구멍에 유령 손이 그려졌다(Slovo 실측).
    손 랜드마커 21점 볼록껍질 + 포즈 팔뚝(팔꿈치-손목) 띠 + 포즈 손 점 원(랜드마커가 놓친 손 - 흐린 손),
    0.15d 팽창 후 거기 닿는 셀피 '몸 피부' 덩어리(1d 안)로 넓힌다 - 랜드마커가 덜 덮은 손가락/흐린 손."""
    import cv2
    m = np.zeros(skin.shape, np.uint8)
    for hull in hands.get("hulls", []):
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(hull).astype(np.int32)), 1)
    for a in hands.get("arms", []):
        cv2.line(m, tuple(np.round(a[0]).astype(int)), tuple(np.round(a[1]).astype(int)), 1, max(1, int(0.5 * d)))
    for q in hands.get("pts", []):
        cv2.circle(m, tuple(np.round(q).astype(int)), int(0.5 * d), 1, -1)
    if not m.any():
        return m.astype(bool)
    m = cv2.dilate(m, np.ones((int(0.15 * d) | 1,) * 2, np.uint8))
    _, lab = cv2.connectedComponents(skin.astype(np.uint8))
    touch = np.unique(lab[(m > 0) & skin])
    near = cv2.dilate(m, np.ones((int(1.0 * d) | 1,) * 2, np.uint8)) > 0
    return (m > 0) | (np.isin(lab, touch[touch > 0]) & near)


def hair_color(img: np.ndarray, hair: np.ndarray, valid: np.ndarray, d: float, min_px: int = 200):
    """확실한 머리 픽셀(확률 0.8, 가장자리 0.05d 깎음)의 Lab 평균과 역공분산 -> (mean, icov) 또는 None."""
    import cv2
    k = int(0.05 * d) | 1
    m = cv2.erode(((hair > 0.8) & valid).astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    if m.sum() < min_px:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[m]
    cov = np.cov(lab.T) + np.eye(3, dtype=np.float32) * 4.0      # 균일한 머리에서 특이하지 않게
    return lab.mean(0), np.linalg.inv(cov).astype(np.float32)


def hair_like(img: np.ndarray, hcol, tau: float) -> np.ndarray:
    """머리색 분포에서 마할라노비스 거리 < tau 인 픽셀 (bool)."""
    import cv2
    mean, icov = hcol
    x = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) - mean
    if MEM_HAIR_DARK:
        # 머리보다 어두운 쪽은 거리로 안 친다: 확실한 머리(확률 0.8)는 주로 빛 받는 정수리라, 그늘진 아래쪽 검은
        # 머리가 분포 밖(너무 어두움)으로 나가 걸러지지 않았다(Slovo 실측)
        x[..., 0] = np.maximum(x[..., 0], 0.0)
    return np.einsum("...i,ij,...j->...", x, icov, x) < tau * tau


def _similarity(a0, a1, b0, b1):
    """두 점 (a0->b0, a1->b1) 을 맞추는 닮음 변환 (2x3, 배율) 또는 None."""
    va, vb = a1 - a0, b1 - b0
    na = float(np.hypot(*va))
    if na < 1e-3:
        return None
    s = float(np.hypot(*vb)) / na
    th = float(np.arctan2(vb[1], vb[0]) - np.arctan2(va[1], va[0]))
    R = s * np.float32([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.concatenate([R, (b0 - R @ a0)[:, None]], 1).astype(np.float32), s


class LearnedTorso:
    """런타임 래퍼. 긴 머리 분할(server._long_hair_mask, hairseg 백그라운드 스레드)과 같은 프레임으로 몇 Hz 부른다.

    프레임 + 셀피 6분류 확률(segmenter.selfie_confidences) + 눈/얼굴 윤곽 -> 눈 기준 크롭 384 에서 머리 구멍
    (학습 hole_from_conf 와 같은 연산)을 채운 결과. 포즈 뼈대는 여기서 PoseLandmarker(IMAGE) 로 구한다.
    기억(v2 가중치): 호출마다 머리가 아닌 픽셀을 크롭 좌표로 쌓아 두고, 다음 호출 때 어깨 두 점으로 새 크롭에
    맞춰 옮겨(몸을 따라감) 구멍 안 기억 채널로 넣는다. 신선도는 시간에 따라 줄고 오래되면 버린다.
    결과는 GPU 텐서 - 붙이기는 paste() (gpu_segmenter 가 결과마다 한 번).
    앱에 하나(넷/포즈 모델 공유, hairseg 스레드 하나라 직렬), 기억만 세션(key)별."""

    #: 구멍 닫기/팽창 (눈 간격 배수). 학습은 0.2~0.6 / 0.05~0.3 으로 흔들었다. 팽창은 넉넉히 - 런타임 지우기
    #: (gpu_segmenter hair_low: 프레임 높이 기준 닫기/팽창 + 파서 머리)를 덮어야 확산으로 떨어지는 자리가 없다.
    CLOSE_D, GROW_D = 0.4, 0.25

    def __init__(self, device="cuda", path=WEIGHTS, mem_tau=3.0, mem_reset=5.0, use_mem=True):
        self.device = device
        sd = torch.load(path, map_location=device)
        sd = {k: v.float() for k, v in sd.get("ema", sd.get("net", sd)).items()}
        self.net = TorsoNet(in_ch=sd["inc.0.weight"].shape[1]).to(device).eval()
        self.net.load_state_dict(sd)
        self.half = device == "cuda"
        if self.half:
            self.net.half()
        self.mem_ch = self.net.in_ch - 4 - len(GUIDES)
        self.mem_tau, self.mem_reset = mem_tau, mem_reset
        self.use_mem = use_mem and self.mem_ch > 0
        self._stream = torch.cuda.Stream() if device == "cuda" else None
        self._pose = None
        self._hand = None
        self._mem = {}

    def forget(self, key):
        """세션이 끝나면 그 기억을 버린다."""
        self._mem.pop(key, None)

    def _pose_pts(self, frame_bgr, M):
        """-> (크롭 좌표 포즈 점 (n,2) 또는 None, 가시도 (n,), 손 정보 dict - hand_mask 입력)."""
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        if self._pose is None:
            self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=os.path.join(os.path.dirname(WEIGHTS),
                                                                       "pose_landmarker_full.task")),
                running_mode=vision.RunningMode.IMAGE, num_poses=1))
        h, w = frame_bgr.shape[:2]
        s = min(1.0, 960.0 / max(h, w))      # 학습 전처리(prep_torso_data LM_PX)와 같은 크기
        small = frame_bgr
        if s < 1.0:
            small = cv2.resize(frame_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        res = self._pose.detect(img)
        tc = lambda q: (np.float32(q).reshape(-1, 2) * (w, h)) @ M[:, :2].T + M[:, 2]  # noqa: E731  정규 -> 크롭
        hands = {"hulls": [], "pts": np.zeros((0, 2), np.float32), "arms": []}
        if self._hand is None and os.path.isfile(HAND_TASK):
            self._hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=HAND_TASK), running_mode=vision.RunningMode.IMAGE,
                num_hands=2, min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))
        if self._hand is not None and self.use_mem:
            hands["hulls"] = [tc([(q.x, q.y) for q in hd]) for hd in self._hand.detect(img).hand_landmarks]
        if not res.pose_landmarks:
            return None, None, hands
        L = res.pose_landmarks[0]
        hands["pts"] = tc([(L[i].x, L[i].y) for i in HAND_IDS if L[i].visibility > 0.3])
        hands["arms"] = [tc([(L[e].x, L[e].y), (L[r].x, L[r].y)]) for e, r in ((13, 15), (14, 16))
                         if L[e].visibility > 0.3 and L[r].visibility > 0.3]
        return (tc([(L[i].x, L[i].y) for i in POSE_IDS]), np.float32([L[i].visibility for i in POSE_IDS]), hands)

    @torch.inference_mode()
    def run(self, frame_bgr: np.ndarray, conf6: np.ndarray, eye_l, eye_r, oval=None, t: float = 0.0, key=None):
        """-> {"rgb": (3,P,P) BGR 0~255 float, "alpha": (1,P,P) 구멍(부드럽게), "M": 프레임->크롭 2x3, "t", "ms",
        "mem": 구멍 중 기억이 있는 비율} 또는 None (구멍이 없음)."""
        import contextlib
        import time
        import cv2
        t0 = time.perf_counter()
        P = NET_PX
        d = P / CROP_D
        h, w = frame_bgr.shape[:2]
        box = crop_box(eye_l, eye_r)
        M = crop_matrix(box, P)
        # 크롭 배율로 먼저 INTER_AREA 축소 -> 평행이동만 (학습 전처리와 같은 축소 필터)
        kz = P / box[2]
        big, T = frame_bgr, M
        if kz < 1.0:
            big = cv2.resize(frame_bgr, (max(1, round(w * kz)), max(1, round(h * kz))), interpolation=cv2.INTER_AREA)
            T = np.float32([[1, 0, -box[0] * kz], [0, 1, -box[1] * kz]])
        img = cv2.warpAffine(big, T, (P, P), flags=cv2.INTER_LINEAR, borderValue=0)
        valid = cv2.warpAffine(np.full(big.shape[:2], 255, np.uint8), T, (P, P), flags=cv2.INTER_NEAREST,
                               borderValue=0) > 127
        # 분할 확률 (h', w') -> 크롭: 분할 픽셀 중심 -> 프레임 px -> 크롭
        ch, cw = conf6.shape[1:]
        sx, sy = w / cw, h / ch
        A = M[:, :2] @ np.float32([[sx, 0], [0, sy]])
        Ms = np.concatenate([A, (M[:, :2] @ np.float32([0.5 * (sx - 1), 0.5 * (sy - 1)]) + M[:, 2])[:, None]], 1)
        cf = np.stack([cv2.warpAffine(c, Ms.astype(np.float32), (P, P), flags=cv2.INTER_LINEAR, borderValue=0)
                       for c in conf6])
        cat = cf.argmax(0).astype(np.uint8)
        hair = cf[S_HAIR] * valid
        hl = np.clip((hair - 0.3) / 0.4, 0, 1)
        kc = max(3, int(round(self.CLOSE_D * d)) | 1)
        kl = max(3, int(round(self.GROW_D * d)) | 1)
        hl = cv2.erode(cv2.dilate(hl, np.ones((kc, kc), np.uint8)), np.ones((kc, kc), np.uint8))
        hole = cv2.dilate(hl, np.ones((kl, kl), np.uint8)) > 0.5
        face = np.zeros((P, P), np.uint8)
        if oval is not None:
            op = np.asarray(oval, np.float32) @ M[:, :2].T + M[:, 2]
            cv2.fillPoly(face, [np.round(op * 4).astype(np.int32)], 1, shift=2)
        face = face.astype(bool)
        hole &= ~face & valid
        if hole.sum() < 16:
            return None
        pts, vis, hands = self._pose_pts(frame_bgr, M)
        seen = cat.copy()
        seen[hole | ~valid] = 255
        g = [pose_guide(pts, vis, P), (seen == S_BG).astype(np.float32), (seen == S_CLOTH).astype(np.float32),
             ((seen == S_BODY) | (seen == S_FACE)).astype(np.float32)]
        to = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1  # noqa: E731
        # 기억: 지난 호출에서 쌓은 것을 어깨 두 점으로 이 크롭에 맞춰 옮기고 신선도를 줄인다
        sh = pts[3:5] if pts is not None and vis[3] > 0.5 and vis[4] > 0.5 else None
        mem = np.zeros((P, P, 3), np.uint8)
        mconf = np.zeros((P, P), np.float32)
        last = lput = None
        prev = self._mem.get(key) if self.use_mem else None
        if prev is not None and 0 <= t - prev[3] < self.mem_reset:
            m0, c0, sh0, t_prev, l0, lp0 = prev
            Aw = np.float32([[1, 0, 0], [0, 1, 0]])      # 어깨를 못 보면 눈 기준 크롭 그대로 (머리를 따라감)
            if sh is not None and sh0 is not None:
                r = _similarity(sh0[0], sh0[1], sh[0], sh[1])
                if r is not None and 0.8 < r[1] < 1.25:
                    Aw = r[0]
            mem = cv2.warpAffine(m0, Aw, (P, P), flags=cv2.INTER_LINEAR, borderValue=0)
            mconf = cv2.warpAffine(c0, Aw, (P, P), flags=cv2.INTER_LINEAR, borderValue=0)
            mconf *= float(np.exp(-(t - t_prev) / self.mem_tau))
            last = cv2.warpAffine(l0, Aw, (P, P), flags=cv2.INTER_LINEAR, borderValue=0)
            lput = cv2.warpAffine(lp0.astype(np.uint8), Aw, (P, P), flags=cv2.INTER_NEAREST, borderValue=0) > 0
        mem_used = (mconf > 0.05) & hole        # 이번 입력에 기억이 들어간 자리 (결과에 같이 돌려준다 - 진단용)
        if self.mem_ch:
            mh = mem_used.astype(np.float32)
            g += list(to(mem).transpose(2, 0, 1) * mh) + [mconf * mh]
        G = torch.from_numpy(np.stack(g)).to(self.device)[None]
        X = torch.from_numpy(to(img)).to(self.device).permute(2, 0, 1)[None]
        Mk = torch.from_numpy(hole.astype(np.float32)).to(self.device)[None, None]
        a = torch.from_numpy(cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 1.5)).to(self.device)[None]
        dt = torch.float16 if self.half else torch.float32
        with torch.cuda.stream(self._stream) if self._stream is not None else contextlib.nullcontext():
            comp, _ = self.net(X.to(dt), Mk.to(dt), G.to(dt))
            rgb = ((comp[0].float().clamp(-1, 1) + 1) * 127.5).flip(0).contiguous()      # RGB -> BGR
        if self._stream is not None:
            self._stream.synchronize()
        # 기억 갱신: 확실히 머리가 아닌 픽셀만 (머리 경계에서 MEM_HAIR_GAP 떨어진 곳 - 경계 가닥이 섞이면 유령 머리),
        # 손 둘레 제외, 그리고 **지난 호출에도 머리가 아니었고 색이 거의 같은 픽셀만** (안정 확인). 확인 없이
        # 쌓았더니 실제 영상(Slovo)에서 지나가던 손과, 분할이 한 프레임 놓친 검은 머리가 기억에 들어가 구멍에
        # 다시 그려졌다(2026-10-20 실측).
        near = cv2.dilate((hair > 0.1).astype(np.uint8), np.ones((int(MEM_HAIR_GAP * d) | 1,) * 2, np.uint8)) > 0
        put = valid & ~near & ~face
        put &= ~hand_mask(hands, cat == S_BODY, d)
        hcol = hair_color(img, hair, valid, d)
        if MEM_HAIR_COL > 0 and hcol is not None:
            put &= ~hair_like(img, hcol, MEM_HAIR_COL)
        ok = put
        if MEM_STABLE < 255:        # 255 = 안정 확인 끔
            ok = np.zeros_like(put)
            if last is not None:
                ok = put & lput & (np.abs(img.astype(np.int16) - last.astype(np.int16)).mean(-1) < MEM_STABLE)
        mem[ok] = img[ok]
        mconf[ok] = 1.0
        if self.use_mem:
            self._mem[key] = (mem, mconf, sh, t, img, put)
        return {"rgb": rgb, "alpha": a, "M": M, "t": t, "ms": (time.perf_counter() - t0) * 1000,
                "mem": float(mem_used.sum()) / float(hole.sum()), "mem_used": mem_used, "hole": hole,
                "hcol": hcol}


def paste(res: dict, h: int, w: int):
    """LearnedTorso.run 결과 -> 프레임 좌표 (fill (h,w,3) BGR float, alpha (h,w)). 결과와 같은 장치의 텐서."""
    import torch.nn.functional as F
    rgb, a, M = res["rgb"], res["alpha"], res["M"]
    P = rgb.shape[-1]
    ys, xs = torch.meshgrid(torch.arange(h, device=rgb.device, dtype=torch.float32),
                            torch.arange(w, device=rgb.device, dtype=torch.float32), indexing="ij")
    u = float(M[0, 0]) * xs + float(M[0, 1]) * ys + float(M[0, 2])
    v = float(M[1, 0]) * xs + float(M[1, 1]) * ys + float(M[1, 2])
    grid = torch.stack([(2 * u + 1) / P - 1, (2 * v + 1) / P - 1], -1)[None]
    o = F.grid_sample(torch.cat([rgb, a], 0)[None], grid, mode="bilinear", padding_mode="zeros",
                      align_corners=False)[0]
    return o[:3].permute(1, 2, 0), o[3]
