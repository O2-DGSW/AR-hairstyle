"""eval_rotate.py 결과 두 개를 Δyaw 구간별로 비교한다.

    python compare_eval.py baseline.json after.json

같은 pairs.json 으로 잰 결과여야 의미가 있다(다르면 경고한다).
방향을 지표마다 따로 표시한다 - pose_err/cycle_mse 는 낮을수록,
id_cos 는 높을수록 좋다.
"""
import argparse
import json
import os

# (키, 낮을수록 좋은가)
METRICS = [
    ("pose_err", True),
    ("pose_excess", True),      # 인버전 오차를 뺀 값. Rotate 자체의 기여.
    ("id_cos", False),
    ("cycle_mse", True),
    ("hair_ratio", None),       # 1 에 가까울수록 좋다 - 방향이 아니라 거리
]


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main(a):
    base, aft = load(a.baseline), load(a.after)
    if base.get("pairs") != aft.get("pairs"):
        print("[경고] 두 결과의 페어 목록이 다르다. 직접 비교하면 안 된다.")
        print(f"  baseline: {base.get('pairs')}")
        print(f"  after   : {aft.get('pairs')}")

    print(f"baseline: {base['checkpoint']}")
    print(f"after   : {aft['checkpoint']}\n")

    for key, lower_better in METRICS:
        print(f"--- {key} " + ("(낮을수록 좋음)" if lower_better is True else
                               "(높을수록 좋음)" if lower_better is False else
                               "(1에 가까울수록 좋음)"))
        print(f"{'|dyaw|':>10} {'before':>10} {'after':>10} {'delta':>10} {'':>6}")
        for bucket, bv in base["summary"].items():
            av = aft["summary"].get(bucket)
            if av is None:
                continue
            b, c = bv["median"][key], av["median"][key]
            d = c - b
            if lower_better is None:
                mark = "개선" if abs(c - 1) < abs(b - 1) else ("악화" if abs(c - 1) > abs(b - 1) else "")
            else:
                good = (d < 0) if lower_better else (d > 0)
                mark = "" if d == 0 else ("개선" if good else "악화")
            pct = f"{d / b * 100:+.1f}%" if b else ""
            print(f"{bucket:>10} {b:>10.4f} {c:>10.4f} {d:>+10.4f} {mark:>6} {pct}")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rotate 평가 결과 비교")
    p.add_argument("baseline")
    p.add_argument("after")
    a = p.parse_args()
    a.baseline, a.after = os.path.abspath(a.baseline), os.path.abspath(a.after)
    main(a)
