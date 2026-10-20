"""긴 머리 밑 인페인터 학습 쌍 (prep_torso_data.py 출력 -> 입력/정답/구멍/안내).

    python train/torso_pairs.py --data ../data/torso --preview out.jpg     # 쌍 8개 미리보기

정답 = 머리가 턱 아래로 안 내려온 크롭(hair_below 작음). 구멍 = 다른 사람(긴 머리 기증자)의 셀피 머리
확률을 런타임과 같은 연산(gpu_segmenter hair_low 경로: 0.3~0.7 램프 -> 닫기 -> 팽창)으로 만든 마스크.
크롭이 눈 기준이라 기증자 머리가 그대로 정답의 얼굴 둘레에 얹힌다. 정답 얼굴(윤곽 안)은 구멍에서 뺀다 -
런타임도 얼굴은 이마 패치/파서 몫이다.

손실 가중 W: 원본 밖(무효)은 0, 정답 자신의 머리(짧은 머리)도 0 - 구멍이 그 위에 걸려도 머리를 그리는
법은 배우지 않게(이마 VALID 와 같은 생각).
남은 가닥: 런타임 구멍 둘레에는 지우다 남은 머리 가닥이 붙어 있고, 범용 인페인터는 그걸 구멍 안으로
이어 그렸다(유령 가닥). 기증자의 옅은 머리 자리(구멍 밖)에 기증자 픽셀을 그대로 붙여 넣어, 그걸 보고도
구멍 안은 정답(맨 옷/배경)으로 채우게 한다.
"""
from __future__ import annotations

import argparse, json, os, random, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torso_net import STORE_PX, CROP_D, INVALID, POSE_IDS   # noqa: E402

#: 정답: 턱+0.3d 아래 셀피 머리 넓이(d^2) 상한, 턱 아래가 원본 안에 든 비율 하한
GT_MAX, GT_VALID = 0.05, 0.5
#: 기증자: 턱 아래 머리 넓이 하한
DONOR_MIN = 1.0
#: 기억 흉내 (v2): 기억이 아예 없는 비율, 틀린(낡은) 기억을 섞는 비율
MEM_NONE, MEM_STALE = 0.4, 0.3
#: 셀피 분류 (segmenter.CATEGORY_*)
S_BG, S_HAIR, S_BODY, S_FACE, S_CLOTH, S_OTHER = range(6)
#: 포즈 뼈대 (POSE_IDS 안 순번): 어깨-어깨, 어깨-팔꿈치, 어깨-엉덩이
_P = {k: i for i, k in enumerate(POSE_IDS)}
BONES = ((_P[11], _P[12]), (_P[11], _P[13]), (_P[12], _P[14]), (_P[11], _P[23]), (_P[12], _P[24]))


def load_meta(root):
    """meta.jsonl + meta_<i>.jsonl (prep --shard). 죽은 프로세스가 반쯤 쓴 줄은 건너뛴다."""
    rows = []
    for fn in sorted(os.listdir(root)):
        if fn.startswith("meta") and fn.endswith(".jsonl"):
            with open(os.path.join(root, fn), encoding="utf-8") as f:
                for l in f:
                    try:
                        rows.append(json.loads(l))
                    except ValueError:
                        pass
    return rows


class TorsoPairs:
    def __init__(self, root, px=384, sets=("train",), seed=0, cache=False):
        self.root, self.px = root, px
        rows = [r for r in load_meta(root) if r["set"] in sets]
        self.gt = [r for r in rows if r["hair_below"] <= GT_MAX and r["valid_below"] >= GT_VALID]
        self.donor = [r for r in rows if r["hair_below"] >= DONOR_MIN]
        self.rng = random.Random(seed)
        #: 줄인 크기로 디코딩한 것을 메모리에 (384 기준 장당 ~1MB, 2만 장 ~20GB - 학교 서버 램 180GB)
        self.cache = {} if cache else None

    def __repr__(self):
        return f"TorsoPairs(gt={len(self.gt)}, donor={len(self.donor)}, px={self.px})"

    def _load(self, r):
        if self.cache is not None:
            v = self.cache.get(r["id"])
            if v is None:
                v = self.cache[r["id"]] = self._read(r)
            return v
        return self._read(r)

    def _read(self, r):
        p = self.px
        im = cv2.imread(os.path.join(self.root, "img", r["id"] + ".jpg"))
        lab = cv2.imread(os.path.join(self.root, "lab", r["id"] + ".png"), cv2.IMREAD_UNCHANGED)
        hair = cv2.imread(os.path.join(self.root, "hair", r["id"] + ".png"), cv2.IMREAD_GRAYSCALE)
        if p != STORE_PX:
            im = cv2.resize(im, (p, p), interpolation=cv2.INTER_AREA)
            lab = cv2.resize(lab, (p, p), interpolation=cv2.INTER_NEAREST)
            hair = cv2.resize(hair, (p, p), interpolation=cv2.INTER_AREA)
        return im, lab, hair.astype(np.float32) / 255.0

    def _jitter(self, rng):
        """기증자 -> 정답 정렬 흔들기: 눈 기준 크롭이라 거의 맞지만 체형/자세가 다르다."""
        p = self.px
        s = rng.uniform(0.9, 1.12)
        ang = rng.uniform(-6, 6)
        M = cv2.getRotationMatrix2D((p / 2, p / CROP_D), ang, s)     # 눈 중심 둘레
        M[0, 2] += rng.uniform(-0.15, 0.15) * p / CROP_D
        M[1, 2] += rng.uniform(-0.1, 0.25) * p / CROP_D
        return M, rng.random() < 0.5

    def _warp(self, a, M, flip, interp):
        p = self.px
        if flip:
            a = a[:, ::-1]
        return cv2.warpAffine(np.ascontiguousarray(a), M, (p, p), flags=interp,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def hole_from_conf(self, conf, rng=None):
        """셀피 머리 확률 -> 런타임과 같은 구멍 (gpu_segmenter: 램프 0.3~0.7, 닫기 0.04h, 팽창 long_hair_grow).
        런타임 프레임 높이 h ≈ 10d(폰 세로) -> 닫기 0.4d, 팽창 0.15d. 학습은 둘 다 흔든다."""
        d = self.px / CROP_D
        rng = rng or self.rng
        hl = np.clip((conf - 0.3) / 0.4, 0, 1)
        kc = max(3, int(round(rng.uniform(0.2, 0.6) * d)) | 1)
        kl = max(3, int(round(rng.uniform(0.05, 0.3) * d)) | 1)
        hl = cv2.erode(cv2.dilate(hl, np.ones((kc, kc), np.uint8)), np.ones((kc, kc), np.uint8))
        hl = cv2.dilate(hl, np.ones((kl, kl), np.uint8))
        return hl > 0.5

    def face_fill(self, r):
        p = self.px
        k = p / STORE_PX
        m = np.zeros((p, p), np.uint8)
        cv2.fillPoly(m, [np.round(np.asarray(r["oval"]) * k * 4).astype(np.int32)], 1, shift=2)
        return m.astype(bool)

    def pose_guide(self, r, rng=None):
        """포즈 뼈대(어깨선/팔/몸통) 를 굵은 선으로 -> float32 (p, p). 런타임 포즈(머리에 가린 어깨를 추정)
        보다 정답 사진 포즈가 정확하니 점을 흔든다."""
        p = self.px
        d = p / CROP_D
        g = np.zeros((p, p), np.float32)
        if not r.get("pose"):
            return g
        rng = rng or self.rng
        k = p / STORE_PX
        P = [(np.array(q[:2]) * k + np.array([rng.gauss(0, 0.08 * d), rng.gauss(0, 0.08 * d)]), q[2])
             for q in r["pose"]]
        for a, b in BONES:
            if P[a][1] > 0.3 and P[b][1] > 0.3:
                cv2.line(g, tuple(np.round(P[a][0] * 4).astype(int)), tuple(np.round(P[b][0] * 4).astype(int)),
                         1.0, max(1, int(0.25 * d)), cv2.LINE_AA, shift=2)
        return cv2.GaussianBlur(g, (0, 0), 0.15 * d)

    @staticmethod
    def guides(s) -> np.ndarray:
        """sample() -> 안내 (4, p, p) float32: torso_net.GUIDES 순서 (포즈, 배경, 옷, 피부). 분류는 구멍 밖만."""
        sc = s["seen"]
        return np.stack([s["pose"], (sc == S_BG).astype(np.float32), (sc == S_CLOTH).astype(np.float32),
                         ((sc == S_BODY) | (sc == S_FACE)).astype(np.float32)])

    def sample(self, rng=None):
        """-> dict: img(정답 BGR uint8), inp(구멍 밖 = 정답, 남은 가닥 붙임), hole(bool), W(float 손실 가중),
        seen(구멍 밖 셀피 분류, 무효 255), pose(안내 float)."""
        rng = rng or self.rng
        g = rng.choice(self.gt)
        while True:
            dn = rng.choice(self.donor)
            if dn["user"] != g["user"]:
                break
        img, lab, ghair = self._load(g)
        dimg, _, dconf = self._load(dn)
        M, flip = self._jitter(rng)
        dconf = self._warp(dconf, M, flip, cv2.INTER_LINEAR)
        dimg = self._warp(dimg, M, flip, cv2.INTER_LINEAR)
        valid = lab[..., 0] != INVALID
        face = self.face_fill(g)
        hole = self.hole_from_conf(dconf, rng) & ~face & valid
        inp = img.copy()
        # 남은 가닥: 구멍 밖 기증자의 옅은 머리 자리에 기증자 픽셀 (절반만)
        stray = np.zeros_like(hole)
        if rng.random() < 0.5:
            stray = (dconf > rng.uniform(0.08, 0.3)) & ~hole & ~face & valid
            inp[stray] = dimg[stray]
        target = img.copy()
        target[stray] = dimg[stray]
        W = valid.astype(np.float32)
        W[ghair > 0.3] = 0.0
        seen = lab[..., 1].copy()
        seen[hole] = INVALID
        return {"img": target, "inp": inp, "hole": hole, "W": W, "seen": seen, "pose": self.pose_guide(g, rng),
                "cat": lab[..., 1], "other": dimg, "gt_id": g["id"], "donor_id": dn["id"]}

    def _blobs(self, nr, size_d):
        """매끈한 무작위 얼룩장 [0,1] (얼룩 크기 ~ size_d·d)."""
        p = self.px
        k = max(2, int(round(CROP_D / size_d)))
        return cv2.resize(nr.random((k, k)).astype(np.float32), (p, p), interpolation=cv2.INTER_CUBIC)

    def memory(self, img, hole, cat, other, rng=None):
        """이전 프레임 기억 흉내 -> (mem BGR uint8, conf float32, good bool), 구멍 안만.

        런타임 기억 = 머리가 비켜 실제로 보였던 픽셀을 카메라 움직임을 맞춰 쌓아 둔 버퍼와 그 신선도.
        배경은 고개/머리가 움직이면 자주 드러나고 머리 밑 옷은 드물다 -> 드러나는 자리를 배경과 구멍
        가장자리(머리가 흔들려 생기는 틈) 쪽으로 기울인다. 정렬 오차(작은 아핀), 조명 변화(색 이득), 가끔
        틀린 내용(다른 사람 사진/엉뚱한 자리, 낮은 신선도)을 넣어 무조건 베끼지 않고 맞춰 쓰게 한다.
        good = 정답과 같은 내용이 들어간 자리 (학습이 L1 을 거는 곳)."""
        p = self.px
        d = p / CROP_D
        rng = rng or self.rng
        mem = np.zeros_like(img)
        conf = np.zeros((p, p), np.float32)
        if rng.random() < MEM_NONE or hole.sum() < 16:
            return mem, conf, np.zeros((p, p), bool)
        nr = np.random.default_rng(rng.randrange(1 << 30))
        n = self._blobs(nr, rng.uniform(0.3, 1.2)) + 0.35 * (cat == S_BG) - 0.15 * (cat == S_CLOTH)
        dist = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 5)
        n = n + 0.5 * np.clip(1 - dist / (rng.uniform(0.1, 0.6) * d), 0, 1)
        seen = hole & (n > np.quantile(n[hole], 1 - rng.uniform(0.1, 0.9)))
        # 정렬 오차 + 조명 변화 + 가끔 움직임 흐림
        M = cv2.getRotationMatrix2D((p / 2, p / CROP_D), rng.uniform(-2.5, 2.5), rng.uniform(0.97, 1.03))
        M[0, 2] += rng.gauss(0, 0.05 * d); M[1, 2] += rng.gauss(0, 0.05 * d)
        src = cv2.warpAffine(img, M, (p, p), borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        gain = np.float32([rng.uniform(0.92, 1.08) for _ in range(3)]) * rng.uniform(0.9, 1.1)
        src = np.clip(src * gain + rng.uniform(-10, 10), 0, 255).astype(np.uint8)
        if rng.random() < 0.3:
            src = cv2.GaussianBlur(src, (0, 0), rng.uniform(0.5, 1.5))
        mem[seen] = src[seen]
        conf[seen] = rng.uniform(0.4, 1.0)
        good = seen.copy()
        if rng.random() < MEM_STALE:
            bad = seen & (self._blobs(nr, rng.uniform(0.3, 1.0)) > rng.uniform(0.4, 0.8))
            if rng.random() < 0.5:
                wrong = other
            else:
                T = np.float32([[1, 0, rng.choice((-1, 1)) * rng.uniform(0.6, 1.5) * d],
                                [0, 1, rng.uniform(-0.5, 0.5) * d]])
                wrong = cv2.warpAffine(img, T, (p, p), borderMode=cv2.BORDER_REFLECT)
            mem[bad] = wrong[bad]
            conf[bad] = rng.uniform(0.1, 0.6)
            good &= ~bad
        return mem, conf, good


def preview(ds, out, n=8, seed=1):
    rng = random.Random(seed)
    PAL = np.uint8([[0, 0, 0], [0, 200, 255], [200, 150, 120], [120, 120, 220], [220, 80, 80], [80, 200, 80]])
    tiles = []
    for _ in range(n):
        s = ds.sample(rng)
        a = s["inp"].copy()
        a[s["hole"]] = 128
        b = s["img"]
        sc = s["seen"].copy()
        c = np.where(sc[..., None] == INVALID, np.uint8([60, 60, 60]), PAL[np.minimum(sc, 5)])
        c = np.maximum(c, (s["pose"][..., None] * 255).astype(np.uint8))
        w = (np.stack([s["W"]] * 3, -1) * 255).astype(np.uint8)
        # 기억: 구멍 회색 위에 기억 픽셀, 틀린 기억은 빨간 테
        mem, conf, good = ds.memory(s["img"], s["hole"], s["cat"], s["other"], rng)
        m = a.copy()
        m[conf > 0] = mem[conf > 0]
        bad = (conf > 0) & ~good
        m[bad & (cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~cv2.erode(
            bad.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)] = (0, 0, 255)
        tiles.append(np.concatenate([a, m, b, c, w], 1))
    cv2.imwrite(out, cv2.resize(np.concatenate(tiles, 0), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("..", "data", "torso"))
    ap.add_argument("--sets", default="train")
    ap.add_argument("--preview", default="")
    a = ap.parse_args()
    ds = TorsoPairs(a.data, sets=tuple(a.sets.split(",")))
    print(ds)
    if a.preview:
        preview(ds, a.preview)
        print(a.preview)
