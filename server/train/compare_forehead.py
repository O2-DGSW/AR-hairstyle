"""실제 웹캠 프레임에서 이마 채우기 방법 비교: 원본 | synth(규칙 기반) | learned(학습 모델).

    python train/compare_forehead.py train/frames/0816_015751 --frames 60,190 --out out.png

forehead.build_forehead_asset 를 같은 프레임/파싱/랜드마크로 두 번 돌려 asset.bald_bgr(앞머리를 걷어낸
맨이마 프레임)의 얼굴 둘레를 잘라 나란히 놓는다. 구멍(앞머리) 윤곽은 초록 선으로 표시.
"""
import argparse, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forehead                                         # noqa: E402
from gpu_segmenter import GpuFaceParser, CLS_HAIR      # noqa: E402
from face_pose import landmarks_image                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--frames-idx", "--frames", dest="idx", default="60,190")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.frames) if f.endswith(".png"))
    seg = GpuFaceParser()
    inp = forehead.ForeheadInpainter(seg.device)
    rows = []
    for i in [int(x) for x in a.idx.split(",")]:
        img = cv2.imread(os.path.join(a.frames, files[i]))
        cls = seg.class_map(img)
        lm = landmarks_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not lm:
            print("no face", i); continue
        outs = []
        for method in ("synth", "learned"):
            forehead.FILL_METHOD = method
            asset, bangs_px, ms = forehead.build_forehead_asset(
                inp, img, cls, lm["eye_l"], lm["eye_r"], "cmp", 7, lm.get("matrix"), lm.get("brows"))
            print(i, method, "bangs px", bangs_px, f"{ms:.0f}ms")
            outs.append(asset.bald_bgr if asset is not None else img)
        el, er = np.asarray(lm["eye_l"]), np.asarray(lm["eye_r"])
        c = (el + er) / 2; d = float(np.linalg.norm(er - el))
        x0, y0 = int(c[0] - 2.2 * d), int(c[1] - 2.2 * d); x1, y1 = int(c[0] + 2.2 * d), int(c[1] + 1.6 * d)
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
        hole = ((np.abs(outs[0].astype(int) - img.astype(int)).sum(-1) > 6) |
                (np.abs(outs[1].astype(int) - img.astype(int)).sum(-1) > 6)).astype(np.uint8)
        cnt, _ = cv2.findContours(hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        orig = img.copy(); cv2.drawContours(orig, cnt, -1, (0, 255, 0), 1)
        tiles = [t[y0:y1, x0:x1] for t in (orig, outs[0], outs[1])]
        tiles = [cv2.resize(t, (360, int(360 * t.shape[0] / t.shape[1]))) for t in tiles]
        rows.append(np.hstack(tiles))
    w = max(r.shape[1] for r in rows)
    cv2.imwrite(a.out, np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]))


if __name__ == "__main__":
    main()
