"""다각도 헤어 에셋 뱅크를 오프라인으로 굽는다 - 라이브 뱅크의 배치 버전.

    python train/make_asset_bank.py train/frames/0816_015751 --reference korean-frontal
    python train/make_asset_bank.py train/frames/0816_015751 --reference korean-layered \
        --out assets_generated/offline-layered

왜 필요한가
-----------
닮음변환은 자유도가 4개(이동/평면내 회전/크기)뿐이라 **평면 밖 회전(yaw)** 을
표현할 수 없다. 그래서 고개를 좌우로 돌리면 헤어는 계속 정면을 향한 채 남는다.
3D 메시로 만들어도 이 문제는 안 풀린다 - 정면 사진 한 장에는 옆에서 본 머리
데이터가 애초에 없다. 그래서 지어내는 일은 GAN 에게 맡긴다. 각도별 프레임으로
GAN 을 여러 번 돌려 그 각도의 헤어를 각각 생성해 두고, 런타임에는 측정된 yaw
로 골라 쓴다.

서버의 라이브 뱅크(server.py LiveBank / build_asset_from_result)와 **같은
물건**을 만든다: 같은 추출 규칙(asset_extract.extract - 머리 + 얼굴 패치),
같은 크기 정규화(정면 칸 눈간격 기준 + gan_asset_scale), 같은 저장 포맷
(hair_asset.save_asset). 그래서 결과 디렉터리를 train/render_offline.py 의
--bank 로 그대로 넣어 서버 없이 합성 품질을 검증할 수 있다.

서버가 떠 있으면 GAN 이 VRAM 을 두 벌 잡아 OOM 이 난다. 먼저 내리고 돌릴 것.
"""
import argparse
import glob
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
sys.path.insert(0, SERVER)

from config import CONFIG                                    # noqa: E402

MAX_TRIES = 4       # 각 각도에서 시도할 후보 프레임 수


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--buckets", default=",".join(str(int(t)) for t in CONFIG.live_targets),
                    help="생성할 yaw 각도(도), 쉼표 구분. 기본은 CONFIG.live_targets")
    ap.add_argument("--tol", type=float, default=CONFIG.live_tol,
                    help="각 구간에서 이 오차 안의 프레임만 후보로 삼는다")
    ap.add_argument("--out", default="",
                    help="저장 디렉터리. 기본 assets_generated/offline-<ref>-<시각>")
    ap.add_argument("--save-gan", action="store_true",
                    help="GAN 원본 결과(1024 png)도 같이 남긴다")
    ap.add_argument("--no-prefill", action="store_true",
                    help="앞머리 사전 제거를 끈다 (기본은 CONFIG.gan_prefill_forehead)")
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not frames:
        print(f"프레임 없음: {args.frames_dir}"); return 1

    import asset_extract
    import gan_input
    import gan_worker
    import hair_asset
    from gpu_segmenter import GpuFaceParser
    from face_pose import FacePose, landmarks_image

    refs = gan_worker.list_references()
    if args.reference not in refs:
        print(f"참고사진 없음: {args.reference} (있는 것: {sorted(refs)})"); return 1
    ref_path = refs[args.reference]

    out_dir = args.out or os.path.join(
        CONFIG.generated_dir, f"offline-{args.reference}-{int(time.time()) % 100000}")
    bank_name = os.path.basename(os.path.normpath(out_dir))

    print("모델 로딩...", flush=True)
    parser = GpuFaceParser()   # class_map 이 입력을 512 로 맞추므로 1024 결과도 그대로 된다
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
    targets = [float(v) for v in args.buckets.split(",") if v.strip()]
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
    # 정면부터. 첫 칸의 눈 간격이 나머지 칸의 크기 기준이 된다(서버와 동일).
    picks.sort(key=lambda tc: abs(tc[0]))
    os.makedirs(out_dir, exist_ok=True)
    made = 0
    ref_eye_len = None
    for t, cands in picks:
        name = f"gan-{args.reference}-yaw{int(round(t)):+03d}"
        print(f"\n[{name}] GAN 실행...", flush=True)
        t0 = time.perf_counter()

        # 후보를 순서대로 시도한다. 특정 프레임은 흔들림/가림 때문에 정렬이
        # 실패하는데, 그것 때문에 그 각도를 통째로 포기할 이유는 없다.
        asset = None
        yaw = None
        for fpath, yaw in cands:
            base = os.path.basename(fpath)
            src = cv2.imread(fpath)
            if CONFIG.gan_prefill_forehead and not args.no_prefill:
                # 서버(run_bank_bucket)와 같은 전처리. 파싱은 여기서, 눈은 MediaPipe.
                lm = landmarks_image(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
                if lm is not None:
                    src, npx = gan_input.prefill_forehead(
                        src, parser.class_map(src), lm["eye_l"], lm["eye_r"])
                    if npx:
                        print(f"  {base}: 앞머리 {npx:,}px 사전 제거")
            try:
                res, gan_s = gan.swap(src, ref_path)
            except Exception as e:
                print(f"  {base}: {str(e)[:60]} - 다음 후보"); continue

            cls = parser.class_map(res)
            pose = poser_t.process(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            if pose is None:
                print(f"  {base}: GAN 결과 얼굴 미검출 - 다음 후보"); continue

            asset, px = asset_extract.extract(res, cls, pose["eye_l"], pose["eye_r"], name)
            if asset is not None:
                if args.save_gan:
                    # 에셋 파일명과 같은 규칙(+ -> _)으로 맞춰야 옆에 나란히 놓인다.
                    stem = os.path.join(out_dir, hair_asset._safe_name(name))
                    cv2.imwrite(stem + ".gan.png", res)
                    cv2.imwrite(stem + ".src.png", src)
                break
            print(f"  {base}: 머리 추출 실패({px}px) - 다음 후보")

        if asset is None:
            print("  모든 후보 실패 - 이 각도 건너뜀"); continue

        asset.yaw = float(yaw)          # 라벨이 아니라 **측정된** yaw (서버와 동일)
        asset.bank = bank_name
        d = asset_extract.eye_len(asset)
        if ref_eye_len is None:
            ref_eye_len = d
        asset.scale_adjust = (d / ref_eye_len) * CONFIG.gan_asset_scale

        hair_asset.save_asset(asset, out_dir)
        made += 1
        print(f"  완료: {asset.rgba.shape[1]}x{asset.rgba.shape[0]}  머리 {px:,}px  "
              f"얼굴패치 {'O' if asset.face is not None else 'X'}  "
              f"GAN {gan_s:.2f}s  ({time.perf_counter()-t0:.1f}s)", flush=True)

    poser_f.close(); poser_t.close(); gan.close()
    print(f"\n뱅크 생성 완료: {made}개 각도 -> {out_dir}")
    print(f"검증: python train/render_offline.py {args.frames_dir} --bank {out_dir} --out <dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
