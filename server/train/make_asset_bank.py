"""다각도 헤어 에셋 뱅크 생성.

    python train/make_asset_bank.py train/frames/0816_015751 --reference korean-frontal

왜 필요한가
-----------
닮음변환은 자유도가 4개(이동/평면내 회전/크기)뿐이라 **평면 밖 회전(yaw)** 을
표현할 수 없다. 그래서 고개를 좌우로 돌리면 헤어는 계속 정면을 향한 채 남는다.

3D 메시로 만들어도 이 문제는 안 풀린다. 문제는 기하가 아니라 **관측**이기
때문이다 - 정면 사진 한 장에는 옆에서 본 머리 데이터가 애초에 없다. 3D로
돌리면 한 번도 본 적 없는 면이 드러나고, 거기 채울 정보가 없다.

그래서 지어내는 일은 GAN 에게 맡긴다. 각도별 프레임으로 GAN 을 여러 번 돌려
그 각도의 헤어를 각각 생성해두고, 런타임에는 측정된 yaw 로 골라 쓴다.

출력: server/assets/<ref>_yawNNN.png + .json (yaw 기록됨)
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
sys.path.insert(0, SERVER)

ASSET_DIR = os.path.join(SERVER, "assets")
MAX_TRIES = 4       # 각 각도에서 시도할 후보 프레임 수


def main():
    ap = argparse.ArgumentParser(description="각도별 헤어 에셋 뱅크 생성")
    ap.add_argument("frames_dir")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--buckets", default="-30,-20,-10,0,10,20,30",
                    help="생성할 yaw 각도(도), 쉼표 구분")
    ap.add_argument("--tol", type=float, default=6.0,
                    help="각 구간에서 이 오차 안의 프레임만 후보로 삼는다")
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not frames:
        print(f"프레임 없음: {args.frames_dir}"); return 1

    import gan_worker
    import hair_asset
    from gpu_segmenter import GpuFaceParser, CLS_HAIR, CLS_SKIN
    from face_pose import FacePose

    refs = gan_worker.list_references()
    if args.reference not in refs:
        print(f"참고사진 없음: {args.reference}"); return 1
    ref_path = refs[args.reference]

    print("모델 로딩...", flush=True)
    parser = GpuFaceParser()
    poser_f = FacePose()      # 웹캠 프레임 전용
    poser_t = FacePose()      # GAN 출력 전용 (인스턴스를 섞으면 추적이 깨진다)
    gan = gan_worker.GanWorker()

    # --- 1) 모든 프레임의 yaw/pitch 측정 ---
    print("프레임 포즈 측정 중...", flush=True)
    poses = []
    for f in frames:
        r = poser_f.process(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))
        if r is not None:
            poses.append((f, float(r["yaw"]), float(r["pitch"])))
    print(f"  얼굴 검출 {len(poses)}/{len(frames)}장", flush=True)

    # --- 2) 구간별 대표 프레임 선정 ---
    targets = [float(v) for v in args.buckets.split(",")]
    picks = []
    for t in targets:
        cand = [(abs(y - t) + 0.3 * abs(p), f, y)
                for f, y, p in poses if abs(y - t) <= args.tol]
        if not cand:
            print(f"  yaw {t:+.0f}°: 후보 프레임 없음 - 건너뜀")
            continue
        cand.sort()
        picks.append((t, [(c[1], c[2]) for c in cand[:MAX_TRIES]]))
        print(f"  yaw {t:+.0f}°: 후보 {min(len(cand), MAX_TRIES)}개 "
              f"(1순위 {os.path.basename(cand[0][1])}, 실측 {cand[0][2]:+.1f}°)")

    if not picks:
        print("생성할 각도가 없습니다."); return 1

    # --- 3) 각도별 GAN 실행 -> 에셋 추출 ---
    os.makedirs(ASSET_DIR, exist_ok=True)
    made = 0
    for t, cands in picks:
        name = f"{args.reference}_yaw{int(round(t)):+03d}"
        print(f"\n[{name}] GAN 실행...", flush=True)
        t0 = time.perf_counter()

        # 후보를 순서대로 시도한다. 특정 프레임은 흔들림/가림 때문에 정렬이
        # 실패하는데, 그것 때문에 그 각도를 통째로 포기할 이유는 없다.
        asset = eyes = None
        px = 0
        yaw = None
        for fpath, yaw in cands:
            base = os.path.basename(fpath)
            try:
                res, _ = gan.swap(cv2.imread(fpath), ref_path)
            except Exception as e:
                print(f"  {base}: {str(e)[:45]} - 다음 후보"); continue

            cls = parser.class_map(res)
            pose = poser_t.process(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            if pose is None:
                print(f"  {base}: GAN 결과 얼굴 미검출 - 다음 후보"); continue

            asset, eyes, px = hair_asset.build_from_photo(
                res, (cls == CLS_HAIR).astype(np.uint8), pose["eye_l"], pose["eye_r"], name,
                ref_skin=hair_asset.skin_mean(res, cls == CLS_SKIN))
            if asset is not None and px >= 500:
                break
            print(f"  {base}: 머리 추출 실패 - 다음 후보")
            asset = None

        if asset is None:
            print("  모든 후보 실패 - 이 각도 건너뜀"); continue

        cv2.imwrite(os.path.join(ASSET_DIR, name + ".png"), asset.rgba)
        with open(os.path.join(ASSET_DIR, name + ".json"), "w", encoding="utf-8") as f:
            json.dump({
                "eyeL": list(eyes[0]), "eyeR": list(eyes[1]),
                "refSkin": None if asset.ref_skin is None else [float(v) for v in asset.ref_skin],
                "yaw": t, "measuredYaw": yaw, "bank": args.reference,
                "source": os.path.basename(fpath),
            }, f, ensure_ascii=False, indent=2)
        made += 1
        print(f"  완료: {asset.rgba.shape[1]}x{asset.rgba.shape[0]}  머리 {px:,}px  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

    poser_f.close(); poser_t.close()
    print(f"\n뱅크 생성 완료: {made}개 각도 -> {ASSET_DIR}")
    print("서버를 재시작하면 자동으로 각도에 맞춰 전환됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
