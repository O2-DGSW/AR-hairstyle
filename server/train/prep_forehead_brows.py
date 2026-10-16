"""이마 학습 데이터에 MediaPipe 눈썹선/눈 윤곽 추가 (크롭 좌표).

    python train/prep_forehead_brows.py --data ../data/forehead --out lm.jsonl

prep_forehead_data.py 로 만든 크롭마다 같은 FFHQ 원본에서 랜드마크를 다시 구해 눈썹 윗선/아랫선
(눈썹마다 5점씩)을 크롭 256 좌표로 바꿔 <data>/brows.jsonl 에 쓴다. 학습 때 이 선을 그린 마스크를
네트워크의 5번째 입력(눈썹 위치 안내)으로 준다. 런타임도 앞머리에 가린 눈썹을 메시가 추정한 같은 선을
준다(face_pose.landmarks_image / FacePose 의 "brows").
"""
import argparse, json, os, sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def work(args):
    src, i = args
    from face_pose import landmarks_image
    from forehead_net import crop_box, NET_PX
    im = cv2.imread(os.path.join(src, i + ".png"))
    if im is None:
        return None
    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
    lm = landmarks_image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    if not lm:
        return None
    x0, y0, S = crop_box(lm["eye_l"], lm["eye_r"])
    k = NET_PX / S
    br = [[((p - (x0, y0)) * k).round(2).tolist() for p in pair] for pair in lm["brows"]]
    ey = [((ring - (x0, y0)) * k).round(2).tolist() for ring in lm["eyes"]]
    return json.dumps({"id": i, "brows": br, "eyes": ey})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join("..", "data", "ffhq", "images"))
    ap.add_argument("--data", default=os.path.join("..", "data", "forehead"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default="brows.jsonl")
    a = ap.parse_args()
    ids = [json.loads(l)["id"] for l in open(os.path.join(a.data, "meta.jsonl"), encoding="utf-8") if l.strip()]
    out = os.path.join(a.data, a.out)
    done = set()
    if os.path.isfile(out):
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8") if l.strip()}
    ids = [i for i in ids if i not in done]
    print("todo", len(ids), "done", len(done), flush=True)
    with Pool(a.procs) as pool, open(out, "a", encoding="utf-8") as f:
        for k, r in enumerate(pool.imap(work, [(a.src, i) for i in ids], chunksize=32)):
            if r:
                f.write(r + "\n")
            if k % 2000 == 0:
                f.flush(); print(k, flush=True)


if __name__ == "__main__":
    main()
