"""헤어스타일 사진 -> 재사용 가능한 에셋(RGBA PNG + 앵커 JSON).

    python make_asset.py photo.jpg --name bob-short
    python make_asset.py photos/*.jpg          # 여러 장 한 번에

파이프라인:
  1) GPU 얼굴파싱(19클래스)으로 hair 마스크 추출
  2) 마스크 정리 - 작은 조각 제거, 구멍 메우기, 경계 페더링
  3) MediaPipe로 눈 2점 측정 -> 앵커로 기록
  4) server/assets/<name>.png (RGBA) + <name>.json 저장

앵커를 눈 2점으로 기록하는 게 핵심이다. 런타임에는 이 눈 간격과 대상 얼굴의
눈 간격을 맞추는 닮음변환을 쓰므로, **원본 사진을 얼마나 가까이/멀리서
찍었든 자동으로 정규화된다.** 문서에 적힌 "참고사진과 대상사진의 얼굴-프레임
비율 정규화가 부정확" 문제가 여기서 해결된다.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
# 가장 큰 덩어리 대비 이 비율보다 작으면 잡음으로 버린다. 어두운 옷깃/배경이
# 머리로 잘못 분류되어 몸통 근처에 떨어진 덩어리로 남는 일이 흔하다.
import hair_asset

MIN_BLOB_RATIO = hair_asset.MIN_BLOB_RATIO


def extract(path: str, name: str, parser, poser, feather: int, pad: float, min_blob: float):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return f"읽기 실패: {path}"

    # 너무 큰 사진은 줄인다 (세그멘테이션 품질에 영향 없고 처리만 느려짐)
    max_side = 1024
    if max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    cls = parser.class_map(img)
    from gpu_segmenter import CLS_HAIR, CLS_SKIN
    hair = (cls == CLS_HAIR).astype(np.uint8)
    ref_skin = hair_asset.skin_mean(img, (cls == CLS_SKIN))
    if hair.sum() < 500:
        return f"머리를 찾지 못함: {path}"

    pose = poser.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
    if pose is None:
        return f"얼굴을 찾지 못함(정면 사진이 필요): {path}"

    asset, eyes, hair_px = hair_asset.build_from_photo(
        img, hair, pose["eye_l"], pose["eye_r"], name,
        feather=feather, pad=pad, min_blob=min_blob, ref_skin=ref_skin)
    if asset is None:
        return f"머리 마스크가 비어 있음: {path}"
    rgba = asset.rgba
    eye_l, eye_r = eyes

    os.makedirs(ASSET_DIR, exist_ok=True)
    out_png = os.path.join(ASSET_DIR, name + ".png")
    out_json = os.path.join(ASSET_DIR, name + ".json")
    cv2.imwrite(out_png, rgba)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "eyeL": eye_l, "eyeR": eye_r,
            "source": os.path.basename(path),
            "interocular_px": float(pose["d_measured"]),
            "refSkin": None if ref_skin is None else [float(v) for v in ref_skin],
            "yaw": round(pose["yaw"], 1),
            "pitch": round(pose["pitch"], 1),
            "roll": round(pose["roll"], 1),
        }, f, ensure_ascii=False, indent=2)

    warn = ""
    if abs(pose["yaw"]) > 15:
        warn = f"  [주의] yaw={pose['yaw']:.0f}° - 정면 사진일수록 결과가 좋습니다"
    return (f"{name}: {rgba.shape[1]}x{rgba.shape[0]}  "
            f"눈간격={pose['d_measured']:.0f}px  머리={hair_px:,}px{warn}")


def main():
    ap = argparse.ArgumentParser(description="헤어스타일 사진에서 에셋 추출")
    ap.add_argument("photos", nargs="+", help="사진 경로 (glob 가능)")
    ap.add_argument("--name", help="에셋 이름 (사진 1장일 때만)")
    ap.add_argument("--feather", type=int, default=3, help="경계 페더링 강도 (0=끔)")
    ap.add_argument("--pad", type=float, default=0.12, help="크롭 여백 비율")
    ap.add_argument("--min-blob", type=float, default=MIN_BLOB_RATIO,
                    help="가장 큰 덩어리 대비 이 비율보다 작은 조각은 버림")
    args = ap.parse_args()

    paths = []
    for p in args.photos:
        paths.extend(glob.glob(p) if any(c in p for c in "*?[") else [p])
    if not paths:
        print("사진을 찾지 못했습니다."); return
    if args.name and len(paths) > 1:
        print("--name 은 사진 1장일 때만 쓸 수 있습니다."); return

    from gpu_segmenter import GpuFaceParser
    from face_pose import FacePose
    print("모델 로딩 중...")
    parser = GpuFaceParser(use_cuda_graph=False)   # 사진 크기가 제각각이라 그래프 미사용
    poser = FacePose()

    for p in paths:
        name = args.name or os.path.splitext(os.path.basename(p))[0]
        try:
            print(extract(p, name, parser, poser, args.feather, args.pad, args.min_blob))
        except Exception as e:
            print(f"{p}: 실패 - {e}")

    poser.close()
    print(f"\n저장 위치: {ASSET_DIR}")
    print("서버를 재시작하면 에셋 목록에 자동으로 뜹니다.")


if __name__ == "__main__":
    main()
