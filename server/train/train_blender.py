"""블렌딩 네트워크 학습 — 워핑 합성본 -> GAN 품질.

    python train/train_blender.py train/frames/0816_015751_pairs --epochs 60

손실 구성
---------
  L1          픽셀 정합. 색/밝기를 맞추는 기본기.
  LPIPS       지각 손실. **이게 핵심이다.** L1만 쓰면 안전한 평균값으로 수렴해
              결과가 흐려진다. 머리카락 같은 고주파 질감은 픽셀 위치가 조금만
              달라도 L1이 크게 벌하므로, 모델은 '뭉개는' 쪽을 택한다.
              LPIPS는 사람이 느끼는 유사도로 재기 때문에 질감을 살리는 방향으로
              민다. FSGAN 계열 블렌딩이 지각 손실을 쓰는 이유가 이것이다.

적대 손실(GAN loss)은 일부러 넣지 않았다. 데이터가 한 사람뿐이라 판별자가
쉽게 과적합되고, 학습이 불안정해지는 대가에 비해 얻을 게 불확실하다.
증류가 효과 있다는 게 먼저 확인되면 그때 고려할 항목.
"""
import argparse
import glob
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from blender_net import BlenderUNet   # noqa: E402


class PairSet(Dataset):
    def __init__(self, root, files, augment=False):
        self.root, self.files, self.augment = root, files, augment

    def __len__(self):
        return len(self.files)

    def _load(self, sub, name):
        img = cv2.imread(os.path.join(self.root, sub, name), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def __getitem__(self, i):
        name = self.files[i]
        x, y = self._load("inp", name), self._load("tgt", name)
        if self.augment:
            if random.random() < 0.5:                    # 좌우 반전
                x, y = x[:, ::-1].copy(), y[:, ::-1].copy()
            if random.random() < 0.3:                    # 밝기 흔들기
                g = random.uniform(0.85, 1.15)
                x = np.clip(x * g, 0, 1); y = np.clip(y * g, 0, 1)
        to_t = lambda a: torch.from_numpy(a.transpose(2, 0, 1))
        return to_t(x), to_t(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_dir")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lpips-w", type=float, default=0.8)
    ap.add_argument("--out", default=os.path.join(HERE, "blender.pt"))
    args = ap.parse_args()

    names = sorted(os.path.basename(p) for p in
                   glob.glob(os.path.join(args.pairs_dir, "inp", "*.png")))
    if len(names) < 20:
        print(f"짝이 너무 적습니다: {len(names)}"); return 1
    random.Random(0).shuffle(names)
    n_val = max(4, len(names) // 10)
    val_names, tr_names = names[:n_val], names[n_val:]
    print(f"학습 {len(tr_names)} / 검증 {len(val_names)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DataLoader(PairSet(args.pairs_dir, tr_names, augment=True),
                    batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    va = DataLoader(PairSet(args.pairs_dir, val_names), batch_size=args.batch)

    net = BlenderUNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    l1 = nn.L1Loss()

    import lpips
    percep = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in percep.parameters():
        p.requires_grad_(False)

    def total_loss(pred, tgt):
        # LPIPS 는 [-1,1] 입력을 기대한다
        lp = percep(pred * 2 - 1, tgt * 2 - 1).mean()
        return l1(pred, tgt) + args.lpips_w * lp, lp

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        net.train(); t0 = time.perf_counter(); tot = 0.0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            loss, _ = total_loss(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()

        net.eval(); vl = vlp = 0.0; vb = 0
        # 입력을 그대로 통과시켰을 때의 손실(=아무것도 안 한 기준선).
        # 이보다 낮아지지 않으면 학습이 무의미하다.
        base = 0.0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(dev), y.to(dev)
                l, lp = total_loss(net(x), y)
                b, _ = total_loss(x, y)
                vl += l.item(); vlp += lp.item(); base += b.item(); vb += 1
        vl /= vb; vlp /= vb; base /= vb

        mark = ""
        if vl < best:
            best = vl
            torch.save({"model": net.state_dict(), "crop": 256}, args.out)
            mark = "  <- 저장"
        print(f"ep {ep:3d}  train {tot/len(tr):.4f}  val {vl:.4f} "
              f"(lpips {vlp:.4f})  기준선 {base:.4f}  {time.perf_counter()-t0:.0f}s{mark}")

    print(f"\n최종 검증 손실 {best:.4f} (기준선 {base:.4f})")
    print(f"저장: {args.out}")
    if best >= base:
        print("경고: 기준선보다 나아지지 않았습니다 — 증류가 효과 없다는 뜻입니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
