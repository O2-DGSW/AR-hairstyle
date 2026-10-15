"""FFHQ -> 이마 인페인팅 학습 데이터 (눈 기준 얼굴 크롭 256 + 파싱 맵).

    python train/prep_forehead_data.py --n 24000 --out ../data/forehead

각 이미지를 512 로 줄여 얼굴 파서(GpuFaceParser)와 MediaPipe 랜드마크(눈, 눈썹)를 돌리고, 런타임과
같은 정의(forehead_net.crop_box)로 눈 기준 정사각 크롭을 256 으로 저장한다.
  <out>/img/<id>.jpg   크롭 RGB
  <out>/cls/<id>.png   크롭 파싱 클래스 (0~18)
  <out>/meta.jsonl     id, 이마 띠의 피부/머리 비율, 안경 여부 - 학습 쪽이 '맨이마 정답'(피부 비율 높음)과
                       '앞머리 모양 기증자'(머리 비율 높음)를 고른다
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_segmenter import GpuFaceParser, CLS_HAIR, CLS_EYE_G   # noqa: E402
from face_pose import landmarks_image                           # noqa: E402
from forehead_net import crop_box, band_mask, NET_PX, CROP_D, EYE_Y            # noqa: E402


def load(path):
    im = cv2.imread(path)
    return None if im is None else cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join("..", "data", "ffhq", "images"))
    ap.add_argument("--out", default=os.path.join("..", "data", "forehead"))
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    for d in ("img", "cls"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    seg = GpuFaceParser()
    ids = [f"{i:05d}" for i in range(a.start, a.start + a.n)]
    done = set()
    mp = os.path.join(a.out, "meta.jsonl")
    if os.path.isfile(mp):
        with open(mp, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    ids = [i for i in ids if i not in done]
    print("todo", len(ids), "done", len(done), flush=True)
    pool = ThreadPoolExecutor(6)
    futs = {}
    with open(mp, "a", encoding="utf-8") as meta:
        for k, i in enumerate(ids):
            for j in ids[k:k + 12]:
                if j not in futs:
                    futs[j] = pool.submit(load, os.path.join(a.src, j + ".png"))
            im = futs.pop(i).result()
            if im is None:
                continue
            lm = landmarks_image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            if not lm:
                continue
            cls = seg.class_map(im)
            box = crop_box(lm["eye_l"], lm["eye_r"])
            x0, y0, S = box
            M = np.float32([[NET_PX / S, 0, -x0 * NET_PX / S], [0, NET_PX / S, -y0 * NET_PX / S]])
            ci = cv2.warpAffine(im, M, (NET_PX, NET_PX), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
            cc = cv2.warpAffine(cls.astype(np.uint8), M, (NET_PX, NET_PX), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            band = band_mask()
            # 판정은 눈썹 위 이마만 (띠 아래쪽은 눈/안경이라 피부 비율이 낮게 나온다)
            dd = NET_PX / CROP_D
            ys = np.arange(NET_PX)[:, None]
            xs = np.arange(NET_PX)[None, :]
            fh = ((ys > EYE_Y * NET_PX - 1.0 * dd) & (ys < EYE_Y * NET_PX - 0.3 * dd)
                  & (np.abs(xs - NET_PX / 2) < 1.0 * dd))
            skin = float(np.isin(cc[fh], (1, 2, 3, 10)).mean())
            hair = float((cc[fh] == CLS_HAIR).mean())
            glasses = bool((cc == CLS_EYE_G).sum() > 150)
            cv2.imwrite(os.path.join(a.out, "img", i + ".jpg"), ci, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(a.out, "cls", i + ".png"), cc)
            meta.write(json.dumps({"id": i, "skin": round(skin, 3), "hair": round(hair, 3),
                                   "glasses": glasses}) + "\n")
            if k % 500 == 0:
                meta.flush()
                print(k, i, f"skin={skin:.2f} hair={hair:.2f}", flush=True)


if __name__ == "__main__":
    main()
