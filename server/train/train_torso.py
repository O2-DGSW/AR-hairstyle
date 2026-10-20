"""긴 머리 밑 인페인터 학습 (torso_net.TorsoNet).

    python train/train_torso.py --data ../data/torso --out ../data/torso_runs/v2 --init torso_net_v1.pt

자기지도: 머리가 턱 아래로 안 내려온 상반신 크롭(정답)에 다른 사람의 긴 머리 모양 구멍을 씌워 원래
목/어깨/옷/배경을 복원한다(쌍 만들기는 torso_pairs.py). 정답에 머리가 없으니 머리를 그리는 법을 배우지
않는다 - LaMa/MI-GAN 의 유령 가닥이 그 반대였다.

손실 (v2, LaMa 설정을 따름): v1 은 구멍 안 L1 을 6배로 걸고 판별기는 0.05 로 약했다 - 구멍 안은 원래
알 수 없으니 픽셀 손실이 '그럴듯한 정답들의 평균' = 흐린 그림을 가르쳤다(옷 무늬가 단색으로 뭉개짐).
LaMa(lama-fourier.yaml)는 구멍 안 L1 = 0, 구멍 밖 10, 판별기 10, 판별기 특징 맞추기 100, 지각 30 이다.
  L1: 구멍 밖(--l1-known) + 기억이 맞게 들어간 구멍 안(--l1-mem) 만. 구멍 안 나머지는 --l1-hole(기본 0).
  LPIPS(VGG), 판별기(SN PatchGAN, softplus): 구멍 패치만 가짜 - 구멍 밖 패치는 정답과 같으니 진짜로 둔다
  (LaMa mask_as_fake_target). 생성기 적대 손실도 구멍 패치만. 판별기 중간 특징 맞추기(--fm-w).
  v1 가중치에서 시작하므로 적대 손실은 처음부터, --adv-ramp 스텝에 걸쳐 키운다.
v1 설정(참고): L1 = W(1+5·구멍), LPIPS 0.8, hinge PatchD(train_forehead) 0.05 를 5000 스텝부터, 기억 없음.

W = 0 인 자리(원본 밖, 정답 자기 머리)는 L1 에서 빠지고 LPIPS/판별기 입력에서는 정답으로 덮는다.
안내 채널(torso_net.GUIDES): 포즈 뼈대(머리에 가린 어깨선 위치), 구멍 밖에 보이는 배경/옷/피부 - 구멍 안을
'어느 쪽 재질로 이어 채울지' 알려 준다. 기억 채널(torso_net.MEM_CH, --mem): 이전 프레임에 실제로 보였던
구멍 안 픽셀 + 신선도 (흉내는 torso_pairs.memory). 웹캠 열화(축소/노이즈/JPEG)를 섞는다.

검증은 val 세트(다른 사람들)에서 고정 구멍 64개. 출력: --weights-out (기본 server/models/torso_net.pt),
<out>/samples_<step>.jpg (입력(기억 포함) | 출력 | 정답).
"""
import argparse, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torso_net import TorsoNet, NET_PX, CROP_D, GUIDES, MEM_CH, WEIGHTS   # noqa: E402
from torso_pairs import TorsoPairs                                          # noqa: E402


class MaskD(nn.Module):
    """SN PatchGAN. 384 에서 22x22 패치, 수용 영역 ~140px(≈2.4d - 구멍이 어깨 폭이라 이마 PatchD 보다 넓게).
    -> (로짓, 중간 특징들) - 특징 맞추기 손실용."""

    def __init__(self, c=64):
        super().__init__()
        self.blocks = nn.ModuleList()
        cin = 3
        for i, co in enumerate((c, c * 2, c * 4, c * 8, c * 8)):
            self.blocks.append(nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(cin, co, 4, stride=2 if i < 4 else 1, padding=1)),
                nn.LeakyReLU(0.2)))
            cin = co
        self.head = nn.utils.spectral_norm(nn.Conv2d(cin, 1, 4, padding=1))

    def forward(self, x):
        f = []
        for b in self.blocks:
            x = b(x)
            f.append(x)
        return self.head(x), f


def load_init(net, path, dev):
    """가중치(ema) 불러오기. 입력 채널이 늘었으면(v1 -> 기억 채널) 새 채널 가중치는 0 - 처음엔 v1 과 같다."""
    s = torch.load(path, map_location=dev)
    sd = {k: v.float() for k, v in s.get("ema", s.get("net", s)).items()}
    own = net.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape != v.shape and v.dim() == 4 and own[k].shape[1] > v.shape[1]:
            w = torch.zeros_like(own[k])
            w[:, :v.shape[1]] = v
            sd[k] = w
            print("init: 입력 채널", v.shape[1], "->", w.shape[1], k, flush=True)
    net.load_state_dict(sd)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("..", "data", "torso"))
    ap.add_argument("--out", default=os.path.join("..", "data", "torso_runs", "v2"))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--accum", type=int, default=2, help="배치를 나눠 쌓기 (실효 배치 = bs x accum)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--l1-known", type=float, default=1.0, help="구멍 밖 L1 (출력 원본 기준, 이음매 연속성)")
    ap.add_argument("--l1-hole", type=float, default=0.0, help="구멍 안 L1 (v1 은 6 - 흐림의 원인)")
    ap.add_argument("--l1-mem", type=float, default=1.0, help="기억이 맞게 들어간 구멍 안 L1 (기억을 쓰게)")
    ap.add_argument("--lpips-w", type=float, default=1.0)
    ap.add_argument("--adv-w", type=float, default=0.5)
    ap.add_argument("--fm-w", type=float, default=5.0, help="판별기 특징 맞추기")
    ap.add_argument("--adv-start", type=int, default=0)
    ap.add_argument("--adv-ramp", type=int, default=2000, help="적대/특징 손실을 이 스텝에 걸쳐 0 에서 키운다")
    ap.add_argument("--mem", type=int, default=1, help="기억 채널 (0 이면 v1 과 같은 입력)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default="", help="이 가중치(ema)에서 시작")
    ap.add_argument("--weights-out", default=WEIGHTS)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--degrade", type=float, default=0.7, help="웹캠 열화를 거는 확률")
    ap.add_argument("--strokes", type=float, default=0.15, help="기증자 구멍에 임의 붓질을 더하는 확률 (일반화)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    P = NET_PX
    d = P / CROP_D

    ds = TorsoPairs(a.data, px=P, sets=("train",), cache=True)
    vs = TorsoPairs(a.data, px=P, sets=("val",), cache=True)
    print("train", ds, "val", vs, flush=True)
    pool = ThreadPoolExecutor(a.workers)

    def strokes(rng):
        """가슴/어깨 높이 굵은 붓질 몇 개 - 기증자 머리 모양에만 맞추지 않게."""
        m = np.zeros((P, P), np.uint8)
        for _ in range(rng.randint(1, 4)):
            x0, y0 = rng.uniform(0.05, 0.95) * P, rng.uniform(0.3, 0.95) * P
            x1, y1 = x0 + rng.uniform(-0.4, 0.4) * P, y0 + rng.uniform(-0.3, 0.3) * P
            cv2.line(m, (int(x0), int(y0)), (int(x1), int(y1)), 1, int(rng.uniform(0.2, 0.7) * d))
        return m.astype(bool)

    def degrade(im, rng):
        """웹캠 열화: 축소 후 확대(흐림), 센서 노이즈, JPEG. 정답 자체에 걸어 출력도 웹캠 질감을 따른다."""
        s = rng.uniform(0.4, 1.0)
        if s < 0.95:
            k = max(16, int(P * s))
            im = cv2.resize(cv2.resize(im, (k, k), interpolation=cv2.INTER_AREA), (P, P),
                            interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.6:
            im = np.clip(im + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(1, 6), im.shape),
                         0, 255).astype(np.uint8)
        if rng.random() < 0.7:
            ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(35, 92))])
            im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        return im

    def one(src, seed, train):
        rng = random.Random(seed)
        s = src.sample(rng)
        hole = s["hole"]
        if train and rng.random() < a.strokes:
            extra = strokes(rng) & (s["seen"] != 255)
            hole = hole | extra
            s["seen"] = s["seen"].copy(); s["seen"][extra] = 255
        im = s["img"]
        if train and rng.random() < a.degrade:
            im = degrade(im, rng)
        g = TorsoPairs.guides(s)
        to = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1  # noqa: E731
        if a.mem:
            # 기억은 열화된 프레임에서 (런타임 기억도 같은 카메라 화질)
            mem, conf, good = src.memory(im, hole, s["cat"], s["other"], rng)
            g = np.concatenate([g, to(mem).transpose(2, 0, 1) * (conf > 0), conf[None]])
        else:
            good = np.zeros_like(hole)
        return to(im), hole, s["W"], g.astype(np.float32), good

    def batch(src, n, train=True, base=None):
        seeds = [random.randrange(1 << 30) for _ in range(n)] if base is None else [base + i for i in range(n)]
        items = list(pool.map(lambda sd: one(src, sd, train), seeds))
        X = torch.from_numpy(np.stack([t[0] for t in items])).permute(0, 3, 1, 2).to(dev)
        Mk = torch.from_numpy(np.stack([t[1] for t in items]).astype(np.float32))[:, None].to(dev)
        W = torch.from_numpy(np.stack([t[2] for t in items]))[:, None].to(dev)
        G = torch.from_numpy(np.stack([t[3] for t in items])).to(dev)
        Mm = torch.from_numpy(np.stack([t[4] for t in items]).astype(np.float32))[:, None].to(dev)
        if train:   # 밝기/색 지터 (웹캠 조명 다양성) - 기억도 같은 장면이니 같이
            gm = torch.empty(n, 3, 1, 1, device=dev).uniform_(0.85, 1.15)
            b = torch.empty(n, 1, 1, 1, device=dev).uniform_(-0.12, 0.12)
            X = ((X + 1) * gm - 1 + b).clamp(-1, 1)
            if a.mem:
                g0 = len(GUIDES)
                seen = (G[:, -1:] > 0).float()
                G[:, g0:g0 + 3] = ((G[:, g0:g0 + 3] + 1) * gm - 1 + b).clamp(-1, 1) * seen
        return X, Mk, W, G, Mm

    t0 = time.time()
    Vb = batch(vs, 64, train=False, base=123)   # 고정 검증 구멍 (+ 기억)
    print(f"val batch {time.time() - t0:.1f}s", flush=True)

    def val_scores(Xv, Mv, Wv, cv_):
        hm = Wv * Mv
        vl1 = ((cv_ - Xv).abs() * hm).sum() / (3 * hm.sum()).clamp(min=1) * 3
        cw = cv_ * Wv + Xv * (1 - Wv)
        vlp = torch.cat([lp(cw[i:i + 8], Xv[i:i + 8]).flatten() for i in range(0, len(Xv), 8)]).mean()
        return f"l1={vl1.item():.4f} lpips={vlp.item():.4f}"

    def run_ema(Xv, Mv, Gv):
        """검증 64장을 8장씩 (한 번에 돌리면 384 에서 VRAM 이 모자랐다)."""
        return torch.cat([ema(Xv[i:i + 8], Mv[i:i + 8], Gv[i:i + 8])[0] for i in range(0, len(Xv), 8)])

    in_ch = 4 + len(GUIDES) + (MEM_CH if a.mem else 0)
    net = TorsoNet(in_ch=in_ch).to(dev)
    ema = TorsoNet(in_ch=in_ch).to(dev).eval(); ema.load_state_dict(net.state_dict())
    D = MaskD().to(dev)
    print(f"params {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M "
          f"D {sum(p.numel() for p in D.parameters()) / 1e6:.1f}M", flush=True)
    import lpips
    lp = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.5, 0.99), weight_decay=1e-4)
    optd = torch.optim.AdamW(D.parameters(), lr=a.lr, betas=(0.5, 0.99))
    step0 = 0
    ck = os.path.join(a.out, "last.pt")
    if a.init and not (a.resume and os.path.isfile(ck)):
        ema.load_state_dict(load_init(net, a.init, dev))
        print("init", a.init, flush=True)
    if a.resume and os.path.isfile(ck):
        s = torch.load(ck, map_location=dev)
        net.load_state_dict(s["net"]); ema.load_state_dict(s["ema"]); D.load_state_dict(s["D"])
        opt.load_state_dict(s["opt"]); optd.load_state_dict(s["optd"]); step0 = s["step"]
        print("resume", step0, flush=True)
    if a.eval_only:
        with torch.no_grad():
            Xv, Mv, Wv, Gv, _ = Vb
            cv_ = run_ema(Xv, Mv, Gv)
            print("val", val_scores(Xv, Mv, Wv, cv_))
        return
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.05, last_epoch=step0 - 1) \
        if step0 else torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.05)
    t0 = time.time()
    # 다음 배치를 미리 만든다 (CPU 쌍 만들기가 GPU 스텝과 겹치게). batch 안에서 pool 을 쓰므로 따로 둔다.
    pre = ThreadPoolExecutor(1)
    nxt = pre.submit(batch, ds, a.bs)
    for step in range(step0, a.steps):
        ramp = min(1.0, (step - a.adv_start + 1) / max(1, a.adv_ramp)) if step >= a.adv_start else 0.0
        use_adv = ramp > 0
        opt.zero_grad(set_to_none=True)
        if use_adv:
            optd.zero_grad(set_to_none=True)
        # --accum: 학교 3090 은 남이 나눠 써서 빈 VRAM 이 ~10GB 뿐 - 384 는 배치 6 이 한계라 나눠 쌓는다
        for _ in range(a.accum):
            X, Mk, W, G, Mm = nxt.result()
            nxt = pre.submit(batch, ds, a.bs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                comp, out = net(X, Mk, G)
            out = out.float(); comp = comp.float()
            wpx = a.l1_known * (1 - Mk) + a.l1_hole * Mk + a.l1_mem * Mm
            l1 = ((out - X).abs() * W * wpx).mean()
            comp_w = comp * W + X * (1 - W)
            # LPIPS/판별기도 bf16 (이마 학습에서 fp32 는 VRAM 이 넘쳐 4배 느려졌다)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lpl = lp(comp_w, X).mean().float()
            loss = l1 + a.lpips_w * lpl
            gadv = fm = torch.zeros((), device=dev)
            if use_adv:
                # 생성기 쪽 적대 손실이 판별기 기울기에 쌓이지 않게 (쌓기 중엔 zero_grad 로 지울 수 없다)
                D.requires_grad_(False)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lf, ff = D(comp_w)
                    with torch.no_grad():
                        _, fr = D(X)
                md = F.adaptive_avg_pool2d(Mk, lf.shape[-2:])
                gadv = (F.softplus(-lf.float()) * md).sum() / md.sum().clamp(min=1)
                fm = sum((x.float() - y.float()).abs().mean() for x, y in zip(ff, fr)) / len(ff)
                loss = loss + ramp * (a.adv_w * gadv + a.fm_w * fm)
            (loss / a.accum).backward()
            if use_adv:
                D.requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    dr, _ = D(X)
                    df, _ = D(comp_w.detach())
                dr = dr.float(); df = df.float()
                # 구멍 패치만 가짜, 구멍 밖 패치(정답과 같은 픽셀)는 진짜 (LaMa mask_as_fake_target)
                ld = F.softplus(-dr).mean() + (F.softplus(df) * md + F.softplus(-df) * (1 - md)).mean()
                (ld / a.accum).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if use_adv:
            optd.step()
        with torch.no_grad():
            k = 0.999 if step - step0 > 1000 or a.init else 0.99
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(k).add_(pn.detach(), alpha=1 - k)
        if step % 50 == 0:
            print(f"{step} l1={l1.item():.4f} lpips={lpl.item():.4f} adv={gadv.item():.3f} fm={fm.item():.4f} "
                  f"d={ld.item() if use_adv else 0:.3f} cache={len(ds.cache)} "
                  f"{(time.time() - t0) / max(1, step - step0 + 1):.3f}s/it", flush=True)
        if step % 2000 == 0 or step == a.steps - 1:
            with torch.no_grad():
                Xv, Mv, Wv, Gv, _ = Vb
                cv_ = run_ema(Xv, Mv, Gv)
                print(f"val {step}", val_scores(Xv, Mv, Wv, cv_), flush=True)
                inp = Xv * (1 - Mv) + Mv * torch.tensor([0.0, 1.0, 0.0], device=dev).view(1, 3, 1, 1)
                gg = 0.5 * Gv[:, :1] * Mv     # 포즈 안내는 가린 자리에서만 빨갛게
                inp = inp * (1 - gg) + gg * torch.tensor([1.0, -1.0, -1.0], device=dev).view(1, 3, 1, 1)
                if a.mem:   # 기억은 구멍 위에 그대로
                    sm = (Gv[:, -1:] > 0).float()
                    g0 = len(GUIDES)
                    inp = inp * (1 - sm) + Gv[:, g0:g0 + 3] * sm
                grid = torch.cat([torch.cat(list(t[:8]), 2) for t in (inp, cv_, Xv)], 1)
                g = ((grid.permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
                g = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(a.out, f"samples_{step:06d}.jpg"), cv2.cvtColor(g, cv2.COLOR_RGB2BGR))
            os.makedirs(os.path.dirname(os.path.abspath(a.weights_out)), exist_ok=True)
            torch.save({"ema": {k: v.half() for k, v in ema.state_dict().items()}}, a.weights_out)
            torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "D": D.state_dict(),
                        "opt": opt.state_dict(), "optd": optd.state_dict(), "step": step + 1}, ck)


if __name__ == "__main__":
    main()
