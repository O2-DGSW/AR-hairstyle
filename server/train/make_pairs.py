"""학습 짝 생성 — (싸구려 워핑 합성본) → (GAN 품질 결과).

    python train/make_pairs.py train/frames/0816_0130 --reference korean-frontal

무엇을 만드나
-------------
증류(distillation) 데이터다. GAN을 교사로 삼아, 실시간 워핑이 만든 결과를
GAN 품질로 끌어올리는 매핑을 배우게 한다. 헤어 이식은 정답 이미지가 없어서
지도학습이 어려운데, GAN이 정답을 만들어주므로 그 문제가 풀린다.

추론 상황을 그대로 재현하는 게 핵심이다. 실제 서비스에서는:
  - 에셋은 '촬영 시점(포즈 P0)'에 한 번 만들어져 고정되고
  - 매 프레임 '다른 포즈(P1)'의 얼굴에 워핑된다  -> 여기서 이질감이 생긴다
그래서 학습 짝도 반드시 **에셋을 만든 프레임과 적용하는 프레임을 다르게** 잡는다.
같은 프레임으로 만들면 입력과 목표가 거의 같아져서 모델이 항등함수를 배운다.

좌표계
------
GAN 출력은 정렬된 1024 크롭이라 웹캠 프레임과 공간이 다르다. 양쪽을 **눈 기준
정규화 크롭**으로 맞춘다. 이 크롭은 눈 2점만 있으면 만들 수 있어서 추론 때도
프레임당 1ms 미만으로 동일하게 재현 가능하다(정렬에 dlib을 쓰면 100ms라 불가).

출력: <out>/inp/NNNNN.png, <out>/tgt/NNNNN.png  (둘 다 CROP x CROP)
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

CROP = 256          # 학습/추론 공용 해상도
EYE_Y = 0.42        # 크롭 안에서 눈 높이 비율
EYE_SPAN = 0.34     # 크롭 폭 대비 눈 간격 비율


def eye_crop_matrix(eye_l, eye_r, size=CROP):
    """눈 2점 -> 정규화 크롭으로 보내는 2x3 닮음변환."""
    import hair_asset
    dst_l = (size * (0.5 - EYE_SPAN / 2), size * EYE_Y)
    dst_r = (size * (0.5 + EYE_SPAN / 2), size * EYE_Y)
    return hair_asset.similarity_matrix(eye_l, eye_r, dst_l, dst_r)


def to_crop(img, eye_l, eye_r, size=CROP):
    M = eye_crop_matrix(eye_l, eye_r, size)
    if M is None:
        return None
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser(description="증류 학습용 (입력, 목표) 짝 생성")
    ap.add_argument("frames_dir", help="원본 프레임 디렉터리")
    ap.add_argument("--reference", required=True, help="references/ 의 헤어스타일 이름")
    ap.add_argument("--out", default=None, help="출력 디렉터리 (기본: <frames_dir>_pairs)")
    ap.add_argument("--asset-frame", type=int, default=0,
                    help="에셋을 뽑을 프레임 인덱스. 나머지 프레임에 이걸 적용한다")
    ap.add_argument("--limit", type=int, default=0, help="최대 처리 장수 (0=전부)")
    ap.add_argument("--list", default=None, help="처리할 프레임 경로 목록 파일")
    args = ap.parse_args()

    if args.list:
        with open(args.list, encoding="utf-8") as f:
            frames = [ln.strip() for ln in f if ln.strip()]
    else:
        frames = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not frames:
        print(f"프레임이 없습니다: {args.frames_dir}"); return 1
    if args.asset_frame >= len(frames):
        print("--asset-frame 인덱스가 범위를 벗어났습니다"); return 1

    out_dir = args.out or (args.frames_dir.rstrip("/\\") + "_pairs")
    os.makedirs(os.path.join(out_dir, "inp"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tgt"), exist_ok=True)

    import gan_worker
    import hair_asset
    from gpu_segmenter import GpuFaceParser, SessionPlate, CLS_HAIR, CLS_SKIN
    from face_pose import FacePose

    refs = gan_worker.list_references()
    if args.reference not in refs:
        print(f"참고사진 없음: {args.reference} (가능: {list(refs)})"); return 1
    ref_path = refs[args.reference]

    print("모델 로딩...", flush=True)
    parser = GpuFaceParser()
    plate = SessionPlate(parser.device)
    # 추정기를 역할별로 분리한다. MediaPipe VIDEO 모드는 직전 프레임의 얼굴
    # 위치로 추적을 이어가는데, 웹캠 프레임(640x480)과 GAN 출력(1024x1024)을
    # 한 인스턴스에 번갈아 넣으면 추적이 깨져서 검출률이 급락한다.
    poser = FacePose()        # 웹캠 프레임 전용
    poser_tgt = FacePose()    # GAN 출력 전용
    gan = gan_worker.GanWorker()

    # --- 1) 에셋 프레임에서 GAN 한 번 돌려 에셋 생성 (= 추론 때의 '촬영' 단계) ---
    a_path = frames[args.asset_frame]
    print(f"에셋 생성용 프레임: {os.path.basename(a_path)}", flush=True)
    a_img = cv2.imread(a_path)
    a_res, _ = gan.swap(a_img, ref_path)
    a_cls = parser.class_map(a_res)
    a_pose = poser_tgt.process(cv2.cvtColor(a_res, cv2.COLOR_BGR2RGB))
    if a_pose is None:
        print("에셋 프레임의 GAN 결과에서 얼굴을 못 찾았습니다"); return 1
    asset, _, apx = hair_asset.build_from_photo(
        a_res, (a_cls == CLS_HAIR).astype(np.uint8), a_pose["eye_l"], a_pose["eye_r"],
        "train-asset", ref_skin=hair_asset.skin_mean(a_res, a_cls == CLS_SKIN))
    if asset is None:
        print("에셋 추출 실패"); return 1
    print(f"에셋 준비 완료 ({apx:,} px)", flush=True)

    # --- 2) 나머지 프레임: 입력=워핑 합성, 목표=GAN ---
    todo = [f for i, f in enumerate(frames) if i != args.asset_frame]
    if args.limit:
        todo = todo[:args.limit]

    made = skipped = 0
    reasons = {}
    t0 = time.perf_counter()
    for i, fp in enumerate(todo):
        img = cv2.imread(fp)
        if img is None:
            skipped += 1; continue
        try:
            # 입력: 지금 파이프라인이 실제로 만드는 결과 (워핑 + 정합 + 그림자)
            cls = parser.class_map(img)
            pose = poser.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if pose is None:
                reasons['프레임 얼굴 미검출'] = reasons.get('프레임 얼굴 미검출', 0) + 1
                skipped += 1; continue
            # plate 를 None 으로 넘기면 안 된다. tryon 분기가 plate 를 요구하므로
            # 조건이 빠지면서 '세그멘테이션 색칠(마젠타)' 분기로 떨어지고, 그 위에
            # 헤어가 얹혀 학습 입력이 통째로 오염된다.
            comp, _ = parser.process(
                img, plate, "tryon", asset, 1.0, 0.0, pose, True, 0.35)

            # 목표: 같은 프레임의 GAN 결과 (교사)
            tgt, _ = gan.swap(img, ref_path)
            t_pose = poser_tgt.process(cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB))
            if t_pose is None:
                reasons['GAN 결과 얼굴 미검출'] = reasons.get('GAN 결과 얼굴 미검출', 0) + 1
                skipped += 1; continue

            inp_c = to_crop(comp, pose["eye_l"], pose["eye_r"])
            tgt_c = to_crop(tgt, t_pose["eye_l"], t_pose["eye_r"])
            if inp_c is None or tgt_c is None:
                reasons['크롭 실패'] = reasons.get('크롭 실패', 0) + 1
                skipped += 1; continue

            cv2.imwrite(os.path.join(out_dir, "inp", f"{made:05d}.png"), inp_c)
            cv2.imwrite(os.path.join(out_dir, "tgt", f"{made:05d}.png"), tgt_c)
            made += 1
        except Exception as e:
            key = 'GAN 실패: ' + str(e)[:40]
            reasons[key] = reasons.get(key, 0) + 1
            skipped += 1

        if (i + 1) % 5 == 0:
            el = time.perf_counter() - t0
            eta = el / (i + 1) * (len(todo) - i - 1)
            print(f"  {i+1}/{len(todo)}  생성 {made}  건너뜀 {skipped}  "
                  f"경과 {el/60:.1f}분  남은 예상 {eta/60:.1f}분", flush=True)

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"reference": args.reference, "asset_frame": a_path,
                   "pairs": made, "crop": CROP,
                   "eye_y": EYE_Y, "eye_span": EYE_SPAN}, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {made}쌍 (건너뜀 {skipped}) -> {out_dir}")
    poser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
