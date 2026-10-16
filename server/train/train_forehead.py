"""이마 인페인터 학습 (forehead_net.ForeheadNet).

    python train/train_forehead.py --data ../data/forehead --steps 30000

자기지도: '맨이마' FFHQ 크롭(정답)에 '다른 사람의 앞머리' 모양 마스크(기증자의 머리 ∩ 앞머리 띠)를 씌워
원래 이마를 복원한다. 정답이 항상 맨이마라 네트워크는 머리카락을 이마에 다시 그리는 법을 배우지 않는다
(LaMa 가 실패한 지점). 눈/안경은 런타임처럼 구멍에서 뺀다.

손실: L1(구멍 가중) + LPIPS(VGG) + 약한 PatchGAN(hinge, adv_start 이후) - L1 만으로는 피부결 없이 뭉갠
밀랍 이마가 된다(규칙 기반 synth 의 문제 그대로). EMA 가중치를 저장한다.
--hair-mask: 정답에서 머리/배경인 구멍 자리(헤어라인 위)를 LPIPS/판별기 입력에서 정답으로 덮고 피부 평균색으로
당긴다. 헤어라인 머리결을 없애려던 실험(v2)인데, 실제 웹캠 프레임에서 관자놀이에 평평한 살색 판과 앞머리 잔상
줄무늬가 생겨 v1 보다 나빴다(2026-10-16) - 기본 꺼짐. 웹캠 열화(축소/노이즈/JPEG)를 섞어 FFHQ 고화질과의
차이를 줄인다.

    재학습(이어서): --init ../server/models/forehead_net.pt --steps 15000 --adv-start 1000 --lr 1e-4

눈썹(v4): 앞머리를 지우며 눈썹을 살린다. (1) v1~v3 는 손실 가중 W 의 '피부' 클래스에 눈썹(6,7)이 빠져 있어
구멍 안 눈썹이 L1 에서 통째로 무시됐고, 구멍에서 빼는 '눈' 클래스에 왼눈썹(6)이 잘못 들어가 왼눈썹은 아예
구멍에 들어간 적이 없었다 - 눈썹을 그리는 법을 배운 적이 없다. (2) 눈썹을 일부러 가리는 구멍을 섞고
(--brow-hide) 구멍 안 눈썹 픽셀에 가중(--brow-w)을 준다. (3) MediaPipe 눈썹선 마스크를 5번째 입력으로
준다(prep_forehead_brows.py 의 brows.jsonl). 4ch 가중치에서 --init 하면 새 채널은 0 으로 시작한다.

눈(v5, --guides 2): 눈을 찌르는 앞머리도 지운다. 런타임이 눈 둘레 타원을 구멍에서 빼서 눈 위 머리카락이
남았고, 학습도 눈을 구멍에 넣은 적이 없었다. 눈/안경을 구멍에서 빼지 않고, 눈을 가로지르는 가는 머리
가닥(가끔 눈 전체를 덮는 덩어리)을 구멍에 섞고(--eye-strands), MediaPipe 눈 윤곽을 6번째 입력으로 준다
(prep_forehead_brows.py --out lm.jsonl). 구멍 안 눈 픽셀에 가중(--eye-w).
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
from forehead_net import ForeheadNet, band_mask, guides, NET_PX, CROP_D, EYE_Y, WEIGHTS   # noqa: E402

CLS_HAIR = 13
#: 구멍에서 빼는 것: 안경(3)/눈(4,5) - 런타임도 파서상 머리가 아니라 구멍에 안 들어간다
NOHOLE = (3, 4, 5)
#: 정답으로 복원할 얼굴 클래스: 피부, 코, 눈썹(6,7), 귀, 입/입술 (머리/배경/모자/옷은 뺀다)
VALID = (1, 2, 6, 7, 8, 9, 10, 11, 12)
BROWS = (6, 7)
EYES = (4, 5)


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
    ap.add_argument("--init", default="", help="이 가중치(ema)에서 시작 (재학습)")
    ap.add_argument("--weights-out", default=WEIGHTS)
    ap.add_argument("--eval-only", action="store_true", help="--init 가중치의 검증 점수만 출력")
    ap.add_argument("--hair-mask", action="store_true",
                    help="헤어라인 위를 LPIPS/판별기에서 빼고 피부색으로 당김 (v2 실험 - 실측에서 더 나빴다)")
    ap.add_argument("--brow-hide", type=float, default=0.5, help="눈썹을 일부러 덮는 구멍 비율")
    ap.add_argument("--brow-w", type=float, default=4.0, help="구멍 안 눈썹 픽셀 L1 추가 가중")
    ap.add_argument("--guides", type=int, default=2, help="안내 채널 수: 1=눈썹(v4), 2=눈썹+눈(v5)")
    ap.add_argument("--lm-file", default="lm.jsonl", help="랜드마크 파일 (guides 1 이면 brows.jsonl 도 됨)")
    ap.add_argument("--eye-strands", type=float, default=0.5, help="눈을 가로지르는 머리 가닥을 넣는 비율")
    ap.add_argument("--eye-w", type=float, default=4.0, help="구멍 안 눈 픽셀 L1 추가 가중")
    ap.add_argument("--degrade", type=float, default=0.7, help="웹캠 열화를 거는 확률")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True

    meta = [json.loads(l) for l in open(os.path.join(a.data, "meta.jsonl"), encoding="utf-8") if l.strip()]
    global NOHOLE, VALID
    if a.guides >= 2:   # 눈/안경도 구멍에 들어가고 정답으로 복원한다
        NOHOLE = ()
        VALID = VALID + (3, 4, 5)
    bl = {}
    for l in open(os.path.join(a.data, a.lm_file), encoding="utf-8"):
        if l.strip():
            r = json.loads(l); bl[r["id"]] = (r["brows"], r.get("eyes"))
    meta = [m for m in meta if m["id"] in bl]
    targets = [m["id"] for m in meta if m["skin"] > 0.88 and m["hair"] < 0.06]
    donors = [m["id"] for m in meta if m["hair"] > 0.25]
    # 검증 64장은 v1(FFHQ 0~29999) 과 같은 것으로 고정 - 데이터를 늘려도 모델끼리 비교가 된다
    v1 = [t for t in targets if int(t) < 30000]
    random.seed(0); random.shuffle(v1)
    val = v1[:64]
    vs = set(val)
    targets = [t for t in targets if t not in vs]
    print(f"targets {len(targets)} donors {len(donors)} val {len(val)}", flush=True)

    pool = ThreadPoolExecutor(8)
    def load_t(i):
        im, cl = load_pair(a.data, i)
        return im, cl, guides(a.guides, *bl[i])
    # 정답은 눈썹이 보이는 얼굴만 (눈썹을 복원하는 법을 배워야 하므로)
    T = [t for t in pool.map(load_t, targets) if np.isin(t[1], BROWS).sum() > 60]
    V = list(pool.map(load_t, val))
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
        if rng.random() < a.brow_hide:
            # 눈썹을 덮는 구멍: 양쪽 또는 한쪽 눈썹 전체(+여유). 앞머리가 눈썹을 다 덮은 프레임과 같은 상황.
            bm = np.isin(cl, BROWS).astype(np.uint8)
            if rng.random() < 0.4:
                half = np.zeros_like(bm)
                if rng.random() < 0.5:
                    half[:, :NET_PX // 2] = 1
                else:
                    half[:, NET_PX // 2:] = 1
                bm = bm * half
            k = int(rng.integers(5, 15))
            m = m.astype(bool) | (cv2.dilate(bm, np.ones((k, k), np.uint8)) > 0)
        if a.guides >= 2 and rng.random() < a.eye_strands:
            m = m.astype(bool) | eye_strands(cl)
        m = m.astype(bool) & band & ~np.isin(cl, NOHOLE)
        return m

    def eye_strands(cl):
        """눈을 찌르는 앞머리: 이마에서 내려와 눈을 가로지르는 가는 가닥 몇 개 (가끔 눈 하나를 덮는 덩어리)."""
        m = np.zeros((NET_PX, NET_PX), np.uint8)
        ey = np.isin(cl, EYES)
        if ey.sum() < 20:
            return m.astype(bool)
        ys, xs = np.nonzero(ey)
        x_lo, x_hi = xs.min() - 0.15 * d, xs.max() + 0.15 * d
        y_eye = ys.mean()
        for _ in range(int(rng.integers(1, 7))):
            x0 = rng.uniform(x_lo, x_hi); y0 = y_eye - rng.uniform(0.4, 1.0) * d
            x1 = x0 + rng.uniform(-0.35, 0.35) * d; y1 = y_eye + rng.uniform(-0.05, 0.35) * d
            xm = (x0 + x1) / 2 + rng.uniform(-0.15, 0.15) * d; ym = (y0 + y1) / 2
            t = np.linspace(0, 1, 12)[:, None]
            pts = (1 - t) ** 2 * [x0, y0] + 2 * (1 - t) * t * [xm, ym] + t ** 2 * [x1, y1]
            th = int(rng.integers(2, 6)) if rng.random() < 0.8 else int(rng.integers(6, 13))
            cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 1, th)
        if rng.random() < 0.2:   # 눈 하나를 다 덮는 덩어리
            one = ey & ((np.arange(NET_PX)[None, :] < NET_PX // 2) == (rng.random() < 0.5))
            m |= cv2.dilate(one.astype(np.uint8), np.ones((9, 9), np.uint8))
        return m.astype(bool)

    def batch(src, n, train=True):
        X, Mk, W, Sk, G, Bw, Ew = [], [], [], [], [], [], []
        for _ in range(n):
            im, cl, gd = src[rng.integers(len(src))] if train else src[_ % len(src)]
            if train and rng.random() < 0.5:
                im = im[:, ::-1]; cl = cl[:, ::-1]; gd = gd[:, :, ::-1]
            if train and rng.random() < a.degrade:
                im = degrade(im)
            m = make_hole(cl)
            # 정답에서 '머리/배경' 인 자리는 손실에서 뺀다(헤어라인 위쪽 - 맨이마 복원이 목적)
            skin = np.isin(cl, VALID)
            w = skin | ~m
            fs = skin & band
            sk = im[fs].mean(0) if fs.sum() > 50 else im[skin].mean(0) if skin.any() else im.reshape(-1, 3).mean(0)
            X.append(im.astype(np.float32) / 127.5 - 1); Mk.append(m); W.append(w)
            Sk.append(sk.astype(np.float32) / 127.5 - 1)
            G.append(gd); Bw.append(np.isin(cl, BROWS) & m); Ew.append(np.isin(cl, EYES) & m)
        X = torch.from_numpy(np.stack(X)).permute(0, 3, 1, 2).to(dev)
        Mk = torch.from_numpy(np.stack(Mk).astype(np.float32))[:, None].to(dev)
        W = torch.from_numpy(np.stack(W).astype(np.float32))[:, None].to(dev)
        Sk = torch.from_numpy(np.stack(Sk))[:, :, None, None].to(dev)
        G = torch.from_numpy(np.stack(G).astype(np.float32)).to(dev)
        Bw = torch.from_numpy(np.stack(Bw).astype(np.float32))[:, None].to(dev)
        Ew = torch.from_numpy(np.stack(Ew).astype(np.float32))[:, None].to(dev)
        if train:   # 밝기/색 지터 (웹캠 조명 다양성)
            g = torch.empty(X.shape[0], 3, 1, 1, device=dev).uniform_(0.85, 1.15)
            b = torch.empty(X.shape[0], 1, 1, 1, device=dev).uniform_(-0.12, 0.12)
            X = ((X + 1) * g - 1 + b).clamp(-1, 1)
            Sk = ((Sk + 1) * g - 1 + b).clamp(-1, 1)
        return X, Mk, W, Sk, G, Bw, Ew

    def degrade(im):
        """웹캠 열화: 축소 후 확대(흐림), 센서 노이즈, JPEG. 입력과 정답에 똑같이 - 출력도 웹캠 질감을 따른다."""
        im = np.ascontiguousarray(im)
        s = rng.uniform(0.35, 1.0)
        if s < 0.95:
            k = max(8, int(NET_PX * s))
            im = cv2.resize(cv2.resize(im, (k, k), interpolation=cv2.INTER_AREA), (NET_PX, NET_PX),
                            interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.6:
            im = np.clip(im + rng.normal(0, rng.uniform(1, 6), im.shape), 0, 255).astype(np.uint8)
        if rng.random() < 0.7:
            ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(35, 92))])
            im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        return im

    def val_batch():
        nonlocal rng
        keep, rng = rng, np.random.default_rng(123)   # 매번 같은 구멍
        Xv, Mv, Wv, _, Gv, Bv, Ev = batch(V, 64, train=False)
        rng = keep
        return Xv, Mv, Wv, Gv, Bv, Ev

    def val_scores(Xv, Mv, Wv, Bv, Ev, cv_):
        vl1 = ((cv_ - Xv).abs() * Wv * Mv).sum() / (Wv * Mv).sum().clamp(min=1)
        vb = ((cv_ - Xv).abs() * Bv).sum() / (3 * Bv.sum()).clamp(min=1) * 3
        ve = ((cv_ - Xv).abs() * Ev).sum() / (3 * Ev.sum()).clamp(min=1) * 3
        vlp = lp(cv_ * Wv + Xv * (1 - Wv), Xv).mean()
        return f"l1={vl1.item():.4f} brow_l1={vb.item():.4f} eye_l1={ve.item():.4f} lpips={vlp.item():.4f}"

    in_ch = 4 + a.guides
    net = ForeheadNet(in_ch=in_ch).to(dev)
    ema = ForeheadNet(in_ch=in_ch).to(dev).eval(); ema.load_state_dict(net.state_dict())
    D = PatchD().to(dev)
    import lpips
    lp = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.5, 0.99), weight_decay=1e-4)
    optd = torch.optim.AdamW(D.parameters(), lr=a.lr, betas=(0.5, 0.99))
    step0 = 0
    ck = os.path.join(a.out, "last.pt")
    if a.init and not (a.resume and os.path.isfile(ck)):
        s = torch.load(a.init, map_location=dev)
        sd = {k: v.float() for k, v in s.get("ema", s.get("net", s)).items()}
        w0 = sd["inc.0.weight"]
        if w0.shape[1] < in_ch:   # 4ch(v1~v3)/5ch(v4) -> 새 안내 채널은 0 에서 시작
            sd["inc.0.weight"] = torch.cat([w0, torch.zeros_like(w0[:, :1]).repeat(1, in_ch - w0.shape[1], 1, 1)], 1)
        net.load_state_dict(sd); ema.load_state_dict(sd)
        print("init", a.init, flush=True)
    if a.resume and os.path.isfile(ck):
        s = torch.load(ck, map_location=dev)
        net.load_state_dict(s["net"]); ema.load_state_dict(s["ema"]); D.load_state_dict(s["D"])
        opt.load_state_dict(s["opt"]); optd.load_state_dict(s["optd"]); step0 = s["step"]
        print("resume", step0, flush=True)
    if a.eval_only:
        with torch.no_grad():
            Xv, Mv, Wv, Gv, Bv, Ev = val_batch()
            cv_, _ = ema(Xv, Mv, Gv)
            print("val", val_scores(Xv, Mv, Wv, Bv, Ev, cv_))
        return
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.05, last_epoch=step0 - 1) \
        if step0 else torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.05)
    scaler = torch.amp.GradScaler()
    t0 = time.time()
    for step in range(step0, a.steps):
        X, Mk, W, Sk, G, Bw, Ew = batch(T, a.bs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            comp, out = net(X, Mk, G)
        out = out.float(); comp = comp.float()
        l1 = ((out - X).abs() * (W * (1 + 5 * Mk) + a.brow_w * Bw + a.eye_w * Ew)).mean()
        if a.hair_mask:
            ex = Mk * (1 - W)   # 정답이 머리/배경인 구멍 자리: 피부 평균색으로 약하게
            l1 = l1 + 0.3 * ((out - Sk).abs() * ex).mean()
            comp = comp * (1 - ex) + X * ex
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
                Xv, Mv, Wv, Gv, Bv, Ev = val_batch()
                cv_, _ = ema(Xv, Mv, Gv)
                print(f"val {step}", val_scores(Xv, Mv, Wv, Bv, Ev, cv_), flush=True)
                inp = Xv * (1 - Mv) + Mv * torch.tensor([0.0, 1.0, 0.0], device=dev).view(1, 3, 1, 1)
                for gi, col in enumerate(([1.0, -1.0, -1.0], [-1.0, -1.0, 1.0])[:Gv.shape[1]]):
                    gg = 0.5 * Gv[:, gi:gi + 1] * Mv   # 안내는 가린 자리에서만 보이게
                    inp = inp * (1 - gg) + gg * torch.tensor(col, device=dev).view(1, 3, 1, 1)
                grid = torch.cat([torch.cat(list(t[:8]), 2) for t in (inp, cv_, Xv)], 1)
                g = ((grid.permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
                cv2.imwrite(os.path.join(a.out, f"samples_{step:06d}.jpg"), cv2.cvtColor(g, cv2.COLOR_RGB2BGR))
            os.makedirs(os.path.dirname(os.path.abspath(a.weights_out)), exist_ok=True)
            torch.save({"ema": {k: v.half() for k, v in ema.state_dict().items()}}, a.weights_out)
            torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "D": D.state_dict(),
                        "opt": opt.state_dict(), "optd": optd.state_dict(), "step": step + 1}, ck)


if __name__ == "__main__":
    main()
