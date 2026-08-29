"""Rotate 학습용 데이터셋을 만든다 (yaw 라벨 포함).

원본 scripts/rotate_gen.py 를 왜 그대로 안 쓰는가
------------------------------------------------
1. **메모리**. 원본은 joblib 로 1024x1024 이미지 전체를 float32 리스트로
   한 번에 올린다(load_dataset_images). 장당 12.6MB 이므로 1만 장이면
   약 126GB 다. 이 머신은 15.6GB 라 시작도 못 한다.
   -> 여기서는 배치 단위로 읽어 256 으로 줄이고 **uint8 로 보관**한다
      (1만 장 약 2.0GB). 이미지는 ArcFace 손실의 타깃으로만 쓰이므로
      uint8 로 충분하다.
2. **yaw 라벨이 없다**. 원본은 파일명을 버려서 나중에 각도로 쌍을 고를 수
   없다. 여기서는 파일명과 yaw 를 함께 저장한다 - 그게 이 실험의 핵심이다.

출력 dict:
    images      uint8  [N,3,256,256]
    key_points  float32[N,76,2]
    latents     float32[N,18,512]   e4e W+
    files       list[str]
    yaws        float32[N]

사용:
    python gen_rotate_data.py --images <FFHQ디렉토리> --yaw yaw.json \
        --size 10000 --out rotate_dataset.pt
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
HF = os.path.abspath(os.path.join(HERE, "..", "..", "..", "external", "HairFastGAN"))
sys.path.insert(0, HERE)
sys.path.insert(0, HF)

from vendor_compat import prepare                # noqa: E402
prepare()

ORIG_CWD = os.getcwd()     # chdir 이 import 시점에 일어나므로 먼저 잡아 둔다
os.chdir(HF)

import torchvision.transforms.functional as TF   # noqa: E402
from scripts.rotate_train import Trainer         # noqa: E402
from vendor_compat import install_star_landmarks_arg   # noqa: E402
install_star_landmarks_arg(HF)


def resolve(p):
    """사용자가 준 경로는 **스크립트 실행 위치** 기준으로 푼다.

    이 모듈은 import 시점에 HairFastGAN 으로 chdir 한다(vendored 코드가
    상대경로로 가중치를 읽기 때문). os.path.abspath 를 그냥 쓰면 --images
    같은 경로가 HairFastGAN 안쪽으로 잘못 풀린다.
    """
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(ORIG_CWD, p))


def main(args):
    with open(args.yaw, encoding="utf-8") as f:
        labels = json.load(f)
    files = sorted(k for k, v in labels.items() if v.get("yaw") is not None)
    if args.exclude:
        # 평가 전용 홀드아웃은 학습 데이터에서 뺀다 (eval_rotate --restrict 와 짝)
        with open(args.exclude, encoding="utf-8") as f:
            drop = set(json.load(f))
        before = len(files)
        files = [f_ for f_ in files if f_ not in drop]
        print(f"홀드아웃 {before - len(files)}장 제외 -> {args.exclude}")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(files)
    files = files[: args.size]
    yaws = np.array([labels[f]["yaw"] for f in files], dtype=np.float32)
    print(f"이미지 {len(files)}장 (yaw 라벨 있는 것만)")

    t = Trainer()
    images, key_points, latents = [], [], []
    t0 = time.perf_counter()

    for s in range(0, len(files), args.batch):
        chunk = files[s: s + args.batch]
        arr = []
        for rel in chunk:
            with Image.open(os.path.join(args.images, rel)) as im:
                arr.append(TF.to_tensor(im.convert("RGB")))
        batch = torch.stack(arr).to("cuda")          # [B,3,1024,1024]
        del arr

        with torch.no_grad():
            im256 = t.downsample_256(batch).clip(0, 1)
            latents.append(t.generate_latents(im256 * 2 - 1).cpu())
            key_points.append(t.generate_key_points(batch).cpu())
            images.append((im256 * 255).round().to(torch.uint8).cpu())
        del batch, im256

        done = min(s + args.batch, len(files))
        if done % (args.batch * 20) == 0 or done == len(files):
            el = time.perf_counter() - t0
            print(f"  {done}/{len(files)}  {el:.0f}s  ({done/max(el,1e-9):.1f}/s)  "
                  f"VRAM peak {torch.cuda.max_memory_allocated()/2**20:.0f}MiB", flush=True)

    out = {
        "images": torch.cat(images),
        "key_points": torch.cat(key_points),
        "latents": torch.cat(latents),
        "files": files,
        "yaws": torch.from_numpy(yaws),
    }
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k:11s} {tuple(v.shape)} {v.dtype} "
                  f"{v.element_size()*v.nelement()/2**20:.0f}MiB")
    torch.save(out, args.out)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rotate 학습 데이터 생성 (yaw 포함)")
    p.add_argument("--images", required=True)
    p.add_argument("--yaw", default="yaw.json")
    p.add_argument("--out", default="rotate_dataset.pt")
    p.add_argument("--size", type=int, default=10000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--exclude", default="", help="이 JSON 목록의 이미지를 학습에서 제외 (홀드아웃)")
    p.add_argument("--seed", type=int, default=3407)
    a = p.parse_args()
    for k in ("images", "yaw", "out"):
        setattr(a, k, resolve(getattr(a, k)))
    if a.exclude:
        a.exclude = resolve(a.exclude)
    main(a)
