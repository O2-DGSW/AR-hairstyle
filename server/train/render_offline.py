"""오프라인 렌더 하네스 - GAN 없이 실시간 합성 경로를 디스크 프레임으로 재현한다.

    python train/render_offline.py train/frames/0816_015751 \
        --bank assets_generated/03f3b49a91a6 --out /tmp/render_A \
        --render 60,120,180,240

왜 필요한가
-----------
합성 품질을 고치는 작업은 "바꾸고 → 서버 띄우고 → 웹캠 앞에 앉아 고개를
돌려 보고" 를 반복하게 된다. 그 루프는 느리고, 무엇보다 **같은 입력으로
전/후를 비교할 수 없다.** 여기서는 녹화된 원본 프레임과 이미 구워진 에셋
뱅크로 서버의 프레임 루프(server.py _compose -> gpu_segmenter.process)를
그대로 돌려 결정적인 출력을 만든다. 같은 프레임의 전/후 png 를 나란히 놓고
보면 된다.

서버와 같은 것: SegFormer 파싱, 배경 플레이트 누적(앞 프레임들을 순서대로
흘려 채운다), MediaPipe 포즈 + 거리 보정, 앵커 평활화, yaw EMA 로 뱅크 칸
선택, harmonize/shadow 기본값.
서버와 다른 것: WebRTC/VP8 인코딩이 없다(그 손실은 여기서 안 보인다).

GPU 를 쓴다. 서버가 떠 있어도 SegFormer 한 벌(수백 MB)만 더 올리므로 같이
돌릴 수 있다.
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
sys.path.insert(0, SERVER)

import hair_asset                                            # noqa: E402
from config import CONFIG                                    # noqa: E402


def parse_indices(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), min(n, int(b) + 1)))
        else:
            out.append(int(part))
    return [i for i in out if 0 <= i < n]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", help="원본 프레임 디렉터리 (png, 이름순)")
    ap.add_argument("--bank", required=True,
                    help="에셋 뱅크 디렉터리 (assets_generated/<sid>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--render", default="",
                    help="합성 결과를 저장할 프레임 인덱스 (예: 10,20,50-60). 비우면 전부")
    ap.add_argument("--mode", default="tryon", choices=["tryon", "remove", "seg", "raw"])
    ap.add_argument("--harmonize", type=int, default=1)
    ap.add_argument("--shadow", type=float, default=0.35)
    ap.add_argument("--scale", type=float, default=1.0, help="scale_mul")
    ap.add_argument("--offset", type=float, default=0.0, help="offset_up (px)")
    ap.add_argument("--side-by-side", type=int, default=1,
                    help="원본|합성 을 가로로 붙인 png 도 저장")
    ap.add_argument("--no-graph", action="store_true",
                    help="CUDA 그래프 없이 (서버와 동시에 돌리다 문제가 생기면)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.frames, "*.png")))
    if not paths:
        sys.exit(f"프레임이 없습니다: {args.frames}")
    render = set(parse_indices(args.render, len(paths)))
    os.makedirs(args.out, exist_ok=True)

    from gpu_segmenter import GpuFaceParser, SessionPlate
    from face_pose import FacePose

    seg = GpuFaceParser(use_cuda_graph=not args.no_graph)
    plate = SessionPlate(seg.device)
    smoother = hair_asset.AnchorSmoother()

    assets = hair_asset.load_asset_dir(args.bank)
    if not assets:
        sys.exit(f"에셋이 없습니다: {args.bank}")
    banks = hair_asset.list_banks(assets)
    bank = banks[0] if banks else None
    print(f"에셋 {len(assets)}개, 뱅크 {banks}")
    # 서버가 등록 직후 하는 것처럼 피라미드를 미리 굽는다(실시간 스파이크 방지).
    for a in assets.values():
        seg.warm_asset(a)

    yaw_ema = None
    cur = None
    times = []
    with FacePose() as poser:
        for i, p in enumerate(paths):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            pose = poser.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), i * 33)
            if pose is not None:
                y = float(pose["yaw"])
                yaw_ema = y if yaw_ema is None else yaw_ema * 0.75 + y * 0.25

            asset = None
            if bank is not None and yaw_ema is not None:
                a = hair_asset.pick_by_yaw_stable(assets, bank, yaw_ema, cur)
                if a is not None:
                    asset = cur = a
            if asset is None:
                asset = next(iter(assets.values()))

            # 렌더 대상이 아니어도 process 는 돌린다 - 플레이트를 채워야 하니까.
            mode = args.mode if i in render else "raw"
            t = time.perf_counter()
            out, info = seg.process(
                img, plate, mode, asset, args.scale, args.offset, pose,
                bool(args.harmonize), args.shadow, 0.0, None, 0.0, smoother)
            if i in render:
                times.append((time.perf_counter() - t) * 1000)
                name = os.path.splitext(os.path.basename(p))[0]
                cv2.imwrite(os.path.join(args.out, f"{name}.png"), out)
                if args.side_by_side:
                    cv2.imwrite(os.path.join(args.out, f"{name}_sbs.png"),
                                np.hstack([img, out]))
                print(f"[{i}] {name}: yaw={yaw_ema:+.1f} asset={asset.name} "
                      f"anchor={info['anchor']} post={info['post_ms']:.1f}ms "
                      f"cov={info['coverage']}")
    if times:
        print(f"process() 평균 {np.mean(times):.1f}ms (p95 {np.percentile(times, 95):.1f})")


if __name__ == "__main__":
    main()
