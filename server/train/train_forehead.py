"""이마 인페인터 학습 (forehead_net.ForeheadNet).

    python train/train_forehead.py --data ../data/forehead --steps 30000

자기지도: '맨이마' FFHQ 크롭(정답)에 '다른 사람의 앞머리' 모양 마스크(기증자의 머리 ∩ 앞머리 띠)를 씌워
원래 이마를 복원한다. 정답이 항상 맨이마라 네트워크는 머리카락을 이마에 다시 그리는 법을 배우지 않는다
(LaMa 가 실패한 지점). 눈/안경은 런타임처럼 구멍에서 뺀다.

손실: L1(구멍 가중) + LPIPS(VGG) + 약한 PatchGAN(hinge, adv_start 이후) - L1 만으로는 피부결 없이 뭉갠
밀랍 이마가 된다(규칙 기반 synth 의 문제 그대로). EMA 가중치를 저장한다.
출력: server/models/forehead_net.pt, <out>/samples_<step>.jpg (입력 | 출력 | 정답)
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forehead_net import ForeheadNet, band_mask, NET_PX, CROP_D, EYE_Y, WEIGHTS   # noqa: E402

CLS_HAIR, EYES = 13, (4, 5, 6)


class PatchD(nn.Module):
    def __init__(self, c=48):
        super().__init__()
        L = []
        cin = 3
        for i, co in enumerate((c, c * 2, c * 4, c * 8)):
            L += [nn.utils.spectral_norm(nn.Conv2d(cin, co, 4, stride=2 if i < 3 else 1, padding=1)),
                  nn.LeakyReLU(0.2)]
            cin = co
        L += [nn.utils.spectral_norm(nn.Conv2d(cin, 1, 4, padding=1))]
        self.m = nn.Sequential(*L)

    def forward(self, x):
        return self.m(x)


def load_pair(root, i):
    im = cv2.imread(os.path.join(root, "img", i + ".jpg"))
    cl = cv2.imread(os.path.join(root, "cls", i + ".png"), cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), cl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("..", "data", "forehead"))
    ap.add_argument("--out", default=os.path.join("..", "data", "forehead_runs"))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--adv-start", type=int, default=4000)
    ap.add_argument("--adv-w", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True

    meta = [json.loads(l) for l in open(os.path.join(a.data, "meta.jsonl"), encoding="utf-8") if l.strip()]
    targets = [m["id"] for m in meta if m["skin"] > 0.88 and m["hair"] < 0.06]
    donors = [m["id"] for m in meta if m["hair"] > 0.25]
    random.seed(0); random.shuffle(targets)
    val, targets = targets[:64], targets[64:]
    print(f"targets {len(targets)} donors {len(donors)} val {len(val)}", flush=True)

    pool = ThreadPoolExecutor(8)
    T = list(pool.map(lambda i: load_pair(a.data, i), targets))
    V = list(pool.map(lambda i: load_pair(a.data, i), val))
    band = band_mask()
    Dm = []
    for i, (_, cl) in zip(donors, pool.map(lambda i: load_pair(a.data, i), donors)):
        m = ((cl == CLS_HAIR) & band).astype(np.uint8)
        if m.sum() > 300:
            Dm.append(m)
    print("loaded", len(T), "donor masks", len(Dm), flush=True)

    rng = np.random.default_rng(0)
    d = NET_PX / CROP_D

    def make_hole(cl):
        r = rng.random()
        if r < 0.75:
            m = Dm[rng.integers(len(Dm))].copy()
            if rng.random() < 0.5:
                m = m[:, ::-1].copy()
            s = rng.uniform(0.9, 1.12); tx = rng.uniform(-0.15, 0.15) * d; ty = rng.uniform(-0.15, 0.25) * d
            M = np.float32([[s, 0, (1 - s) * NET_PX / 2 + tx], [0, s, (1 - s) * EYE_Y * NET_PX + ty]])
            m = cv2.warpAffine(m, M, (NET_PX, NET_PX), flags=cv2.INTER_NEAREST)
            m = cv2.dilate(m, np.ones((5, 5), np.uint8))
        else:
            # 임의 띠/덩어리 (일반화): 앞머리 띠 안의 굵은 붓질
            m = np.zeros((NET_PX, NET_PX), np.uint8)
            for _ in range(rng.integers(2, 6)):
                x0, y0 = rng.uniform(0.15, 0.85) * NET_PX, (EYE_Y - rng.uniform(0.1, 0.32)) * NET_PX
                x1, y1 = x0 + rng.uniform(-0.3, 0.3) * NET_PX, y0 + rng.uniform(-0.1, 0.15) * NET_PX
                cv2.line(m, (int(x0), int(y0)), (int(x1), int(y1)), 1, int(rng.uniform(0.15, 0.45) * d))
        m = m.astype(bool) & band & ~np.isin(cl, EYES)
        return m

    def batch(src, n, train=True):
        X, Mk, W = [], [], []
        for _ in range(n):
            im, cl = src[rng.integers(len(src))] if train else src[_ % len(src)]
            if train and rng.random() < 0.5:
                im = im[:, ::-1]; cl = cl[:, ::-1]
            m = make_hole(cl)
            # 정답에서 '머리/배경' 인 자리는 손실에서 뺀다(헤어라인 위쪽 - 맨이마 복원이 목적)
            w = np.isin(cl, (1, 2, 3, 10)) | ~m
            X.append(im.astype(np.float32) / 127.5 - 1); Mk.append(m); W.append(w)
        X = torch.from_numpy(np.stack(X)).permute(0, 3, 1, 2).to(dev)
        Mk = torch.from_numpy(np.stack(Mk).astype(np.float32))[:, None].to(dev)
        W = torch.from_numpy(np.stack(W).astype(np.float32))[:, None].to(dev)
        if train:   # 밝기/색 지터 (웹캠 조명 다양성)
            g = torch.empty(X.shape[0], 3, 1, 1, device=dev).uniform_(0.85, 1.15)
            b = torch.empty(X.shape[0], 1, 1, 1, device=dev).uniform_(-0.12, 0.12)
            X = ((X + 1) * g - 1 + b).clamp(-1, 1)
        return X, Mk, W

    net = ForeheadNet().to(dev)
    ema = ForeheadNet().to(dev).eval(); ema.load_state_dict(net.state_dict())
    D = PatchD().to(dev)
    import lpips
    lp = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.5, 0.99), weight_decay=1e-4)
    optd = torch.optim.AdamW(D.parameters(), lr=a.lr, betas=(0.5, 0.99))
    step0 = 0
    ck = os.path.join(a.out, "last.pt")
    if a.resume and os.path.isfile(ck):
        s = torch.load(ck, map_location=dev)
        net.load_state_dict(s["net"]); ema.load_state_dict(s["ema"]); D.load_state_dict(s["D"])
        opt.load_state_dict(s["opt"]); optd.load_state_dict(s["optd"]); step0 = s["step"]
        print("resume", step0, flush=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.05, last_epoch=step0 - 1) \
        if step0 else torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.05)
    scaler = torch.amp.GradScaler()
    t0 = time.time()
    for step in range(step0, a.steps):
        X, Mk, W = batch(T, a.bs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            comp, out = net(X, Mk)
        out = out.float(); comp = comp.float()
        l1 = ((out - X).abs() * W * (1 + 5 * Mk)).mean()
        # LPIPS/판별기도 bf16: fp32 로 돌렸더니 판별기가 붙은 뒤 VRAM 이 12GB 를 넘어 공유 메모리로
        # 넘치며 스텝이 0.45 -> 2초로 느려졌다(실측).
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lpl = lp(comp, X).mean().float()
        loss = l1 + 0.8 * lpl
        use_adv = step >= a.adv_start
        if use_adv:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss + a.adv_w * (-D(comp).float().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if use_adv:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dr = D(X).float(); df = D(comp.detach()).float()
            ld = F.relu(1 - dr).mean() + F.relu(1 + df).mean()
            optd.zero_grad(set_to_none=True); ld.backward(); optd.step()
        with torch.no_grad():
            k = 0.999 if step > 1000 else 0.99
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(k).add_(pn.detach(), alpha=1 - k)
        if step % 50 == 0:
            print(f"{step} l1={l1.item():.4f} lpips={lpl.item():.4f} "
                  f"{(time.time() - t0) / max(1, step - step0 + 1):.3f}s/it", flush=True)
        if step % 2000 == 0 or step == a.steps - 1:
            with torch.no_grad():
                rng_state = rng.bit_generator.state
                Xv, Mv, _ = batch(V, 8, train=False)
                rng.bit_generator.state = rng_state
                cv_, _ = ema(Xv, Mv)
                inp = Xv * (1 - Mv) + Mv * torch.tensor([0.0, 1.0, 0.0], device=dev).view(1, 3, 1, 1)
                grid = torch.cat([torch.cat(list(t), 2) for t in (inp, cv_, Xv)], 1)
                g = ((grid.permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
                cv2.imwrite(os.path.join(a.out, f"samples_{step:06d}.jpg"), cv2.cvtColor(g, cv2.COLOR_RGB2BGR))
            os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
            torch.save({"ema": ema.state_dict()}, WEIGHTS)
            torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "D": D.state_dict(),
                        "opt": opt.state_dict(), "optd": optd.state_dict(), "step": step + 1}, ck)


if __name__ == "__main__":
    main()
