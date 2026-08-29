"""Rotate 모듈을 Δyaw 층화 페어링으로 파인튜닝한다.

가설
----
±40도에서 헤어가 무너지는 건 아키텍처가 아니라 **학습 쌍의 각도 분포** 탓이다.
원본 scripts/rotate_train.py 는 같은 데이터로더의 두 번째 이터레이터에서 짝을
뽑아 완전 무작위로 페어링한다(rotate_train.py:203). FFHQ 는 정면 편중이라
|Δyaw|>30도 쌍이 학습에서 거의 안 나온다. 논문도 "포즈 차이가 크면 여전히
문제"라고 인정했다.

그래서 손실과 모델은 그대로 두고 **쌍을 뽑는 방식만** 바꾼다. 이게 이 실험의
전부다 - 다른 변수를 같이 건드리면 무엇이 효과를 냈는지 알 수 없다.

원본 대비 바뀐 것
-----------------
1. Δyaw 버킷을 균등(기본)하게 샘플링. --dyaw-weights 로 조절.
2. pretrained_models/Rotate/rotate_best.pth 에서 warm start (파인튜닝이므로).
3. WandbLogger -> LocalLogger (wandb 는 protobuf 가 mediapipe 와 충돌하고
   WANDB_KEY 를 강제한다. locallog.py 참고).
4. batch 2 + 그래디언트 누적 8 = 유효 배치 16. 12GB 실측 결과 batch 2 가
   최적이고(3.9GB/234ms), 4 이상은 공유메모리로 넘어가 오히려 느려진다.

주의: 큰 |Δyaw| 쌍을 강제로 뽑으면 FFHQ 에 드문 극단 각도 이미지가 반복
사용된다. 학습 후 실제 분포를 출력하니 특정 소수 이미지에 과적합되는지
확인할 것.

사용:
    python finetune_rotate.py --dataset rotate_dataset.pt --name flat-dyaw
"""
import argparse
import os
import sys
import time
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
HF = os.path.abspath(os.path.join(HERE, "..", "..", "..", "external", "HairFastGAN"))
sys.path.insert(0, HERE)
sys.path.insert(0, HF)

from locallog import LocalLogger                        # noqa: E402
from vendor_compat import prepare                        # noqa: E402
prepare()

ORIG_CWD = os.getcwd()     # chdir 이 import 시점에 일어나므로 먼저 잡아 둔다
os.chdir(HF)

from tqdm.auto import tqdm                              # noqa: E402
import scripts.rotate_train as RT                       # noqa: E402
from vendor_compat import install_star_landmarks_arg   # noqa: E402
install_star_landmarks_arg(HF)
from models.Encoders import RotateModel                 # noqa: E402
from utils.train import seed_everything                 # noqa: E402

def resolve(p):
    """사용자가 준 경로는 **스크립트 실행 위치** 기준으로 푼다.

    이 모듈은 import 시점에 HairFastGAN 으로 chdir 한다(vendored 코드가
    상대경로로 가중치를 읽기 때문). 그래서 os.path.abspath 를 그냥 쓰면
    사용자가 준 --images 같은 경로가 HairFastGAN 안쪽으로 잘못 풀린다.
    """
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(ORIG_CWD, p))


BUCKETS = ((0, 10), (10, 20), (20, 30), (30, 40), (40, 200))


class YawPairDataset(Dataset):
    """Δyaw 분포를 제어해 (소스, 타깃포즈) 쌍을 만든다.

    소스(anchor)는 정체성과 머리를 유지할 쪽, 타깃은 포즈만 가져올 쪽이다.
    학습 손실이 그렇게 정의돼 있다(rotate_train.calc_loss).
    """

    def __init__(self, images, key_points, latents, yaws, idx, weights, seed, fixed=False):
        self.images, self.key_points, self.latents = images, key_points, latents
        self.yaws = yaws
        self.idx = np.asarray(idx)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights /= self.weights.sum()
        self.fixed = fixed
        self.seed = seed

        ys = self.yaws[self.idx]
        o = np.argsort(ys)
        self.order = self.idx[o]              # yaw 오름차순 전역 인덱스
        self.sorted_yaws = ys[o]
        self._rng = np.random.default_rng(seed)
        # 검증셋은 매 에폭 같은 쌍이어야 비교가 된다
        self._fixed_pairs = [self._sample(i, np.random.default_rng(seed + i))
                             for i in range(len(self.idx))] if fixed else None

    def __len__(self):
        return len(self.idx)

    def _sample(self, i, rng):
        a = int(self.idx[i])
        y = float(self.yaws[a])
        b = int(rng.choice(len(BUCKETS), p=self.weights))
        lo, hi = BUCKETS[b]
        for sign in rng.permutation(np.array([1, -1])):
            t0, t1 = sorted((y + sign * lo, y + sign * hi))
            s = int(np.searchsorted(self.sorted_yaws, t0, "left"))
            e = int(np.searchsorted(self.sorted_yaws, t1, "right"))
            if e > s:
                return a, int(self.order[rng.integers(s, e)])
        # 해당 Δyaw 후보가 아예 없으면 가장 먼 얼굴을 쓴다
        far = self.order[0] if abs(self.sorted_yaws[0] - y) > abs(self.sorted_yaws[-1] - y) \
            else self.order[-1]
        return a, int(far)

    def _get(self, k):
        return self.images[k].float() / 255.0, self.key_points[k], self.latents[k]

    def __getitem__(self, i):
        a, p = self._fixed_pairs[i] if self.fixed else self._sample(i, self._rng)
        return (*self._get(a), *self._get(p))


class FTTrainer(RT.Trainer):
    """원본 Trainer 를 그대로 쓰되 페어링과 그래디언트 누적만 바꾼다."""

    def __init__(self, *a, accum=8, max_steps=0, **kw):
        super().__init__(*a, **kw)
        self.accum = accum
        self.max_steps = max_steps

    def train_one_epoch(self):
        self.model.to(self.device).train()
        sum_losses = lambda x, y: {k: y.get(k, 0) + x.get(k, 0) for k in set(x) | set(y)}

        self.optimizer.zero_grad()
        for step, batch in enumerate(tqdm(self.train_dataloader), start=1):
            (I_from, kp_from, lat_from,
             I_to, kp_to, lat_to) = (x.to(self.device) for x in batch)

            loss, info, _, gen_latent = self.calc_loss(
                I_to, I_from, kp_to, lat_from, lat_to, ret_images=True)

            if self.args.use_hair_loss:
                hair_loss, info2 = self.calc_hair_loss(gen_latent, lat_from)
                loss = loss + hair_loss
                info = sum_losses(info, info2)

            (loss / self.accum).backward()
            self.MAL.update(info)

            if step % self.accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.logger.log("grad", gn.item())

            self.logger.next_step()
            for k, v in info.items():
                self.logger.log(k, v)

            if self.max_steps and step >= self.max_steps:
                print(f"  [max-steps {self.max_steps} 도달, 에폭 조기 종료]")
                break


def report_distribution(ds, n=4000, tag=""):
    """실제로 뽑히는 Δyaw 분포와 이미지 재사용 정도를 확인한다."""
    rng = np.random.default_rng(0)
    ds_yaws, used = [], []
    for i in rng.integers(0, len(ds), size=min(n, len(ds))):
        a, p = ds._sample(int(i), rng)
        ds_yaws.append(abs(float(ds.yaws[a]) - float(ds.yaws[p])))
        used.append(p)
    ds_yaws = np.array(ds_yaws)
    uniq = len(set(used))
    print(f"  [{tag}] 표본 {len(ds_yaws)}쌍, 타깃 이미지 고유 {uniq}개 "
          f"({uniq/len(ds_yaws)*100:.0f}%)")
    for lo, hi in BUCKETS:
        c = int(((ds_yaws >= lo) & (ds_yaws < hi)).sum())
        print(f"     |dyaw| {lo:3d}~{hi:<3d}  {c:5d}  ({c/len(ds_yaws)*100:5.1f}%)")


def main(args):
    seed_everything()
    data = torch.load(args.dataset, weights_only=False)
    images, kps, lats = data["images"], data["key_points"], data["latents"]
    yaws = data["yaws"].numpy()
    n = len(images)
    print(f"데이터셋 {n}장  images={tuple(images.shape)} {images.dtype}")

    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    test_idx, train_idx = perm[: args.test_size], perm[args.test_size:]

    w = [float(x) for x in args.dyaw_weights.split(",")]
    assert len(w) == len(BUCKETS), f"--dyaw-weights 는 {len(BUCKETS)}개여야 한다"

    train_ds = YawPairDataset(images, kps, lats, yaws, train_idx, w, seed=3407)
    test_ds = YawPairDataset(images, kps, lats, yaws, test_idx, w, seed=1234, fixed=True)
    print(f"train {len(train_ds)} / test {len(test_ds)}   Δyaw 가중치 {w}")
    report_distribution(train_ds, tag="train")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=True, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=0, pin_memory=True)

    logger = LocalLogger(name=args.name, project="HairFast-Rotate").start_logging()

    model = RotateModel()
    if args.init:
        sd = torch.load(args.init, map_location="cpu")["model_state_dict"]
        model.load_state_dict(sd)
        print(f"warm start <- {args.init}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    targs = Namespace(use_hair_loss=args.use_hair_loss)
    RT.args = targs          # 원본 validate() 가 모듈 전역 args 를 참조한다

    tr = FTTrainer(model, targs, optimizer, None, train_dl, test_dl, logger,
                   accum=args.accum, max_steps=args.max_steps)
    t0 = time.perf_counter()
    tr.train_loop(args.epochs)
    print(f"\n총 {time.perf_counter()-t0:.0f}s  -> {logger.run_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rotate 파인튜닝 (Δyaw 층화)")
    p.add_argument("--dataset", default="rotate_dataset.pt")
    p.add_argument("--name", default="flat-dyaw")
    p.add_argument("--init", default="pretrained_models/Rotate/rotate_best.pth")
    p.add_argument("--dyaw-weights", default="1,1,1,1,1",
                   help="Δyaw 버킷별 샘플링 가중치 (0-10,10-20,20-30,30-40,40+)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--test-size", type=int, default=512)
    p.add_argument("--max-steps", type=int, default=0, help="에폭당 스텝 제한 (스모크 테스트용)")
    p.add_argument("--no-hair-loss", dest="use_hair_loss", action="store_false")
    a = p.parse_args()
    a.dataset = resolve(a.dataset)
    if a.init and not os.path.isabs(a.init) and os.path.isfile(resolve(a.init)):
        a.init = resolve(a.init)
    main(a)
