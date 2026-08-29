"""HairFastGAN Rotate 모듈을 Δyaw 구간별로 평가한다.

왜 필요한가
-----------
"±40도에서 이상하다"를 숫자로 바꾸지 않으면 파인튜닝이 나아졌는지 알 수 없다.
학습 전에 베이스라인을 찍어 두고, 학습 후 같은 페어로 다시 재서 비교한다.
페어 목록은 파일로 저장해 재사용하므로 before/after 조건이 정확히 같다.

무엇을 재는가
-------------
소스 A(정체성/머리를 유지할 쪽)와 타깃 B(포즈만 가져올 쪽)에 대해
    rotate_to = R(W_A[:6], W_B[:6]);   I_gen = G([rotate_to, W_A[6:]])

  pose_err   kp(I_gen) vs kp(B) 랜드마크 MSE. 낮을수록 좋다. **주 지표**
             (학습의 'mse points to' 손실과 같은 추정기, 같은 식)
  id_cos     ArcFace(I_gen) vs ArcFace(A) 코사인. 높을수록 좋다.
  cycle_mse  I_gen 을 다시 A 포즈로 되돌렸을 때의 잠재 MSE. 학습의 hair loss
             와 같고, 머리 보존의 대리 지표다.
  hair_ratio I_gen 의 머리 픽셀 수 / A 의 머리 픽셀 수. 머리가 사라지거나
             부풀지 않았는지 본다. 포즈가 달라 정확히 1 이 목표는 아니고,
             구간별 추세를 본다.

인버전 천장 - 이 두 열이 결정적이다
-----------------------------------
  pose_err_inv  kp(G(W_B)) vs kp(B).  Rotate 를 거치지 않고 B 를 e4e 로
                넣었다 뺀 것만의 포즈 오차.
  id_cos_inv    ArcFace(G(W_A)) vs ArcFace(A).

큰 각도에서 pose_err 가 나쁜데 **pose_err_inv 도 같이 나쁘면** 범인은 Rotate 가
아니라 e4e 인버전이다. 그러면 Rotate 를 파인튜닝해도 안 낫는다(e4e 는 학습
스크립트에 없다). 반대로 pose_err 만 나쁘면 Rotate 학습 분포 문제이고, 그때
파인튜닝이 답이 된다. 두 열을 반드시 같이 보고 판단할 것.

사용:
    python eval_rotate.py --images <FFHQ디렉토리> --yaw yaw.json \
        --pairs pairs.json --out baseline.json
    python eval_rotate.py ... --rotate_checkpoint runs/.../best.pth --out after.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
HF = os.path.abspath(os.path.join(HERE, "..", "..", "..", "external", "HairFastGAN"))
sys.path.insert(0, HERE)
sys.path.insert(0, HF)

from vendor_compat import prepare                # noqa: E402
prepare()                                         # scripts.* import 전에 반드시 호출

ORIG_CWD = os.getcwd()     # chdir 이 import 시점에 일어나므로 먼저 잡아 둔다
os.chdir(HF)                                      # vendored 코드가 상대경로로 가중치를 읽는다

import torchvision.transforms.functional as TF                     # noqa: E402
from scripts.rotate_train import Trainer                           # noqa: E402
from vendor_compat import install_star_landmarks_arg   # noqa: E402
install_star_landmarks_arg(HF)
from models.face_parsing.model import BiSeNet, seg_mean, seg_std   # noqa: E402
from models.Encoders import RotateModel                            # noqa: E402

def resolve(p):
    """사용자가 준 경로는 **스크립트 실행 위치** 기준으로 푼다.

    이 모듈은 import 시점에 HairFastGAN 으로 chdir 한다(vendored 코드가
    상대경로로 가중치를 읽기 때문). 그래서 os.path.abspath 를 그냥 쓰면
    사용자가 준 --images 같은 경로가 HairFastGAN 안쪽으로 잘못 풀린다.
    """
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(ORIG_CWD, p))


BUCKETS = ((0, 10), (10, 20), (20, 30), (30, 40), (40, 200))
KEYS = ["pose_err", "pose_err_inv", "pose_excess", "id_cos", "rot_norm", "cycle_mse", "hair_ratio"]


def bucket_of(d):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= d < hi:
            return i
    return len(BUCKETS) - 1


def build_pairs(yaw_path, n_per_bucket, seed, restrict=None):
    with open(yaw_path, encoding="utf-8") as f:
        labels = json.load(f)
    items = [(k, v["yaw"]) for k, v in labels.items() if v.get("yaw") is not None]
    if restrict:
        # 학습에 안 쓴 이미지로만 평가한다. 안 그러면 파인튜닝 후의 개선이
        # 일반화인지 암기인지 구분할 수 없다.
        with open(restrict, encoding="utf-8") as f:
            keep = set(json.load(f))
        items = [it for it in items if it[0] in keep]
        print(f"홀드아웃 {len(items)}장으로 제한 -> {restrict}")
    items.sort()
    rng = np.random.default_rng(seed)

    # 구간이 찰 때까지 무작위 추출한다. 극단 구간은 후보가 드물어 요청 수를
    # 못 채울 수 있고, 그 사실 자체가 정면 편중의 증거이므로 그대로 보고한다.
    pairs, need = [], {i: n_per_bucket for i in range(len(BUCKETS))}
    tries, max_tries = 0, n_per_bucket * len(BUCKETS) * 20000
    while any(v > 0 for v in need.values()) and tries < max_tries:
        tries += 1
        i, j = (int(x) for x in rng.integers(0, len(items), size=2))
        if i == j:
            continue
        (fa, ya), (fb, yb) = items[i], items[j]
        b = bucket_of(abs(ya - yb))
        if need[b] > 0:
            need[b] -= 1
            pairs.append({"src": fa, "dst": fb, "yaw_src": ya, "yaw_dst": yb,
                          "dyaw": round(abs(ya - yb), 2), "bucket": b})
    for b, v in need.items():
        if v:
            print(f"  [경고] 구간 {BUCKETS[b][0]}~{BUCKETS[b][1]}도: {v}쌍 부족 "
                  f"(데이터에 해당 각도 조합이 드물다)")
    return pairs


@torch.no_grad()
def main(args):
    if os.path.isfile(args.pairs):
        with open(args.pairs, encoding="utf-8") as f:
            pairs = json.load(f)
        print(f"기존 페어 {len(pairs)}쌍 재사용 -> {args.pairs}")
    else:
        pairs = build_pairs(args.yaw, args.n_per_bucket, args.seed, args.restrict)
        with open(args.pairs, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=1)
        print(f"페어 {len(pairs)}쌍 생성 -> {args.pairs}")

    t = Trainer()                       # net(생성기) / e4e / arcface / STAR 랜드마크
    rot = RotateModel().to("cuda").eval()
    rot.load_state_dict(torch.load(args.rotate_checkpoint, map_location="cuda")["model_state_dict"])
    print(f"Rotate 체크포인트: {args.rotate_checkpoint}")

    seg = BiSeNet(n_classes=16).to("cuda").eval()
    seg.load_state_dict(torch.load("pretrained_models/BiSeNet/seg.pth"))

    def gen_from(latent):
        img, _ = t.net.generator([latent], input_is_latent=True, return_latents=False)
        return t.downsample_256(((img + 1) / 2)).clip(0, 1)

    def hair_px(img_256):
        """머리 픽셀 수와 전체 픽셀 수를 함께 준다(비율 판정에 필요)."""
        big = F.interpolate(img_256, size=(1024, 1024), mode="bilinear", align_corners=False)
        x = (t.downsample_512(big) - seg_mean) / seg_std
        m = torch.argmax(seg(x)[0], dim=1)
        return float((m == 10).sum().item()), float(m.numel())

    def cos(x, y):
        return float(F.cosine_similarity(t.arc_face(t.toArcface(x)),
                                         t.arc_face(t.toArcface(y))).mean())

    def load(rel):
        with Image.open(os.path.join(args.images, rel)) as im:
            x = TF.to_tensor(im.convert("RGB")).unsqueeze(0).to("cuda")
        if x.shape[-1] != 1024:
            x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        return t.downsample_256(x).clip(0, 1)

    rows, t0 = [], time.perf_counter()
    for n, p in enumerate(pairs, 1):
        try:
            A, B = load(p["src"]), load(p["dst"])
            W_A, W_B = t.generate_latents(A * 2 - 1), t.generate_latents(B * 2 - 1)

            lat_in = torch.cat((rot(W_A[:, :6], W_B[:, :6]), W_A[:, 6:]), dim=1)
            gen, recA, recB = gen_from(lat_in), gen_from(W_A), gen_from(W_B)

            kp_gen, kp_B, kp_recB = (t.generate_key_points(x) for x in (gen, B, recB))
            cyc = rot(lat_in[:, :6], W_A[:, :6])
            hp_a, tot = hair_px(A)
            hp_g, _ = hair_px(gen)

            pe = float(F.mse_loss(kp_gen, kp_B))
            pe_inv = float(F.mse_loss(kp_recB, kp_B))
            # RotateModel 은 output = latent_from + 0.1*dt_latent 라, dt_latent 가
            # 0 으로 줄면 "회전 안 함"(항등함수)으로 붕괴한다. 그러면 타깃 포즈를
            # 못 맞추면서(pose_err 상승) 원본 정체성은 그대로 남아(id_cos 상승)
            # 지표만 보면 반쯤 좋아진 것처럼 보인다. 실제 회전량을 재서 구분한다.
            rot_norm = float((lat_in[:, :6] - W_A[:, :6]).norm() / W_A[:, :6].norm())
            rows.append({
                "bucket": p["bucket"], "dyaw": p["dyaw"],
                "pose_err": pe,
                "pose_err_inv": pe_inv,
                # 같은 이미지 쌍에서 뺀 값이라 이미지별 랜드마크 잡음이 상쇄된다.
                # Rotate 모듈 자체의 기여만 남으므로 이게 진짜 주 지표다.
                "pose_excess": pe - pe_inv,
                "id_cos": cos(gen, A),
                "id_cos_inv": cos(recA, A),
                "cycle_mse": float(F.mse_loss(cyc, W_A[:, :6])),
                "rot_norm": rot_norm,
                # 소스에 머리가 거의 없으면(모자/대머리/분할 실패) 비율이 폭발한다.
                # 전체의 1% 미만이면 표본에서 뺀다.
                "hair_ratio": (hp_g / hp_a) if hp_a > 0.01 * tot else float("nan"),
            })
        except Exception as e:
            print(f"  [실패] {p['src']} -> {p['dst']}: {type(e).__name__}: {e}")
        if n % 25 == 0:
            print(f"  {n}/{len(pairs)}  {time.perf_counter() - t0:.0f}s", flush=True)

    # 평균이 아니라 **중앙값**으로 집계한다. 랜드마크 추출이 가끔 크게 빗나가고
    # (pose_err 가 수십배로 튄다) 그런 소수 표본이 평균을 통째로 지배한다.
    # 실제로 첫 베이스라인에서 hair_ratio 평균이 12.2 로 나왔는데 중앙값은
    # 1 근처였다. 비교 지표로는 중앙값이 옳다.
    print(f"\n체크포인트: {args.rotate_checkpoint}   페어 {len(rows)}쌍  (중앙값)\n")
    hdr = f"{'|dyaw|':>10} {'n':>4} " + " ".join(f"{k:>13}" for k in KEYS)
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for b, (lo, hi) in enumerate(BUCKETS):
        sel = [r for r in rows if r["bucket"] == b]
        if not sel:
            continue
        med = {k: float(np.nanmedian([r[k] for r in sel])) for k in KEYS}
        avg = {k: float(np.nanmean([r[k] for r in sel])) for k in KEYS}
        summary[f"{lo}-{hi}"] = {"n": len(sel),
                                 "median": {k: round(v, 5) for k, v in med.items()},
                                 "mean": {k: round(v, 5) for k, v in avg.items()}}
        print(f"{lo:4d}~{hi:<4d} {len(sel):>4} " + " ".join(f"{med[k]:13.5f}" for k in KEYS))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.rotate_checkpoint, "pairs": args.pairs,
                   "summary": summary, "rows": rows}, f, indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rotate 모듈 Δyaw 구간별 평가")
    p.add_argument("--images", required=True)
    p.add_argument("--yaw", default="yaw.json")
    p.add_argument("--pairs", default="pairs.json")
    p.add_argument("--out", default="eval.json")
    p.add_argument("--rotate_checkpoint", default="pretrained_models/Rotate/rotate_best.pth")
    p.add_argument("--n-per-bucket", type=int, default=40)
    p.add_argument("--restrict", default="", help="이 JSON 목록의 이미지로만 평가 (홀드아웃)")
    p.add_argument("--seed", type=int, default=3407)
    a = p.parse_args()
    for k in ("images", "yaw", "pairs", "out"):
        setattr(a, k, resolve(getattr(a, k)))
    if a.restrict:
        a.restrict = resolve(a.restrict)
    if not os.path.isabs(a.rotate_checkpoint) and os.path.isfile(resolve(a.rotate_checkpoint)):
        a.rotate_checkpoint = resolve(a.rotate_checkpoint)   # 사용자 경로는 원래 cwd 기준
    main(a)
