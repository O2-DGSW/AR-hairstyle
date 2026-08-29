"""이미지 디렉토리의 얼굴 yaw 를 라벨링한다 (MediaPipe, IMAGE 모드).

왜 FacePose 를 그대로 안 쓰는가
------------------------------
FacePose 는 VIDEO 모드다. 연속 프레임을 가정하고 프레임 간 추적을 하므로,
서로 무관한 정지 이미지 수만 장을 밀어 넣으면 앞 장의 얼굴 위치가 다음 장의
추정에 새어 든다. 여기서는 IMAGE 모드로 매 장을 독립 추정한다.

각도 부호 규약은 FacePose._euler 를 **그대로 재사용**한다. 뱅크를 굽는 쪽
(make_asset_bank)과 런타임 칸 선택(pick_by_yaw)이 모두 이 규약을 쓰므로
학습 라벨만 다른 규약을 쓰면 나중에 조용히 어긋난다. 이 규약의 yaw 부호는
실제와 뒤집혀 있지만 **일관되게** 뒤집혀 있어 기능상 문제가 없다
(face_pose.py 의 _euler 독스트링 참고). 게다가 여기서 쓰는 건 Δyaw 의
절댓값과 버킷이라 부호와 더 무관하다.

사용:
    python label_yaw.py <이미지디렉토리> [--out yaw.json] [--limit N]

이어서 돌릴 수 있다 - 기존 out 파일에 있는 항목은 건너뛴다.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from face_pose import FacePose, MODEL_PATH, EYE_L_OUT, EYE_L_IN, EYE_R_IN, EYE_R_OUT

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def list_images(root):
    out = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(EXTS):
                out.append(os.path.relpath(os.path.join(dirpath, f), root).replace("\\", "/"))
    out.sort()
    return out


def make_landmarker():
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    ))


def label_one(lm, rgb):
    """-> dict 또는 None(얼굴 없음)."""
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks or not res.facial_transformation_matrixes:
        return None

    h, w = rgb.shape[:2]
    pts = res.face_landmarks[0]

    def px(i):
        return np.array([pts[i].x * w, pts[i].y * h], dtype=np.float32)

    e0 = (px(EYE_L_OUT) + px(EYE_L_IN)) / 2.0
    e1 = (px(EYE_R_IN) + px(EYE_R_OUT)) / 2.0
    eye_l, eye_r = (e0, e1) if e0[0] <= e1[0] else (e1, e0)

    m = np.asarray(res.facial_transformation_matrixes[0].data, dtype=np.float32).reshape(-1)
    yaw, pitch, roll = FacePose._euler(m)
    return {
        "yaw": round(float(yaw), 3),
        "pitch": round(float(pitch), 3),
        "roll": round(float(roll), 3),
        "d": round(float(np.linalg.norm(eye_r - eye_l)), 2),
        "eye_l": [round(float(eye_l[0]), 1), round(float(eye_l[1]), 1)],
        "eye_r": [round(float(eye_r[0]), 1), round(float(eye_r[1]), 1)],
    }


def main(args):
    files = list_images(args.image_dir)
    if args.limit:
        files = files[: args.limit]

    labels = {}
    if os.path.isfile(args.out):
        with open(args.out, encoding="utf-8") as f:
            labels = json.load(f)
        print(f"기존 라벨 {len(labels)}개를 이어받는다")

    todo = [f for f in files if f not in labels]
    print(f"이미지 {len(files)}장 중 {len(todo)}장 라벨링")

    lm = make_landmarker()
    t0 = time.perf_counter()
    fails = 0
    try:
        for i, rel in enumerate(todo, 1):
            try:
                with Image.open(os.path.join(args.image_dir, rel)) as im:
                    rgb = np.asarray(im.convert("RGB"))
                rec = label_one(lm, rgb)
            except Exception as e:                      # 깨진 파일 하나가 전체를 멈추면 안 된다
                rec, e_ = None, e
                print(f"  [읽기실패] {rel}: {type(e_).__name__}")
            if rec is None:
                fails += 1
                rec = {"yaw": None}
            labels[rel] = rec

            if i % 500 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(todo)}  {el:.0f}s  ({i/max(el,1e-9):.0f}/s)  얼굴없음 {fails}", flush=True)
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(labels, f)
    finally:
        lm.close()
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(labels, f)

    ok = [v["yaw"] for v in labels.values() if v.get("yaw") is not None]
    print(f"\n완료: {len(ok)}장 라벨 / {len(labels)-len(ok)}장 실패 -> {args.out}")
    if ok:
        a = np.abs(np.array(ok))
        print("|yaw| 분포:")
        for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 90)):
            n = int(((a >= lo) & (a < hi)).sum())
            print(f"  {lo:2d}~{hi:2d}도  {n:6d}  ({n/len(ok)*100:5.1f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="이미지 디렉토리 yaw 라벨링")
    p.add_argument("image_dir")
    p.add_argument("--out", default="yaw.json")
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
