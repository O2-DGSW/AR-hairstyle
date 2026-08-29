"""에셋의 크기 보정값을 파일에 구워 넣는다.

    python train/set_scale.py korean-frontal 0.9     # 뱅크 전체를 90%로
    python train/set_scale.py korean-frontal_yaw+00 0.88

왜 필요한가
-----------
HairFastGAN 은 머리 대비 큰 헤어를 만드는 경향이 있다. 참고사진을 얼굴 사진과
같은 기준으로 정규화하면 6%쯤 줄지만(실측), 나머지는 모델 성향이라 남는다.
[크기] 슬라이더로 매번 맞출 수도 있지만 세션이 끝나면 사라지므로, 적당한 값을
찾았으면 에셋 JSON 에 적어두어 다음부터 기본으로 적용되게 한다.
"""
import glob
import json
import os
import sys

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    target, value = sys.argv[1], float(sys.argv[2])
    if not (0.3 <= value <= 2.0):
        print("보정값은 0.3~2.0 범위로 주세요."); return 1

    # 정확히 일치하는 에셋, 없으면 그 이름으로 시작하는 뱅크 전체
    paths = [os.path.join(ASSET_DIR, target + ".json")]
    if not os.path.isfile(paths[0]):
        paths = sorted(glob.glob(os.path.join(ASSET_DIR, target + "*.json")))
    if not paths:
        print(f"대상을 찾지 못했습니다: {target}"); return 1

    for p in paths:
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
        meta["scaleAdjust"] = value
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"  {os.path.basename(p)} -> scaleAdjust={value}")

    print(f"\n{len(paths)}개 적용. 서버를 재시작하면 반영됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
