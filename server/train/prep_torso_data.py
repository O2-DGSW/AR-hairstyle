"""EasyPortrait -> 긴 머리 밑 인페인팅 학습 데이터 (눈 기준 상반신 크롭 512 + 라벨 + 랜드마크).

    python train/prep_torso_data.py --zip ../data/easyportrait/val.zip --out ../data/torso
    python train/prep_torso_data.py --zip ../data/easyportrait/train.zip --out ../data/torso

크롭 정의는 torso_net.crop_box (런타임과 공유). zip 에서 바로 읽는다(풀지 않는다 - 68GB).
  <out>/img/<id>.jpg   크롭 RGB 512 (원본 밖은 검정)
  <out>/lab/<id>.png   3ch uint8: [얼굴 파서 0~18, 셀피 분할 0~5, EasyPortrait 사람 라벨 0~8], 원본 밖 255
  <out>/hair/<id>.png  셀피 분할의 머리 확률 x255
  <out>/meta.jsonl     id, user, set, d(원본 px), box, 크롭 좌표 랜드마크(눈/입/얼굴 윤곽/포즈), 통계
통계 hair_below = 턱 + 0.3d 아래 머리 픽셀(셀피 머리>0.5)의 넓이, d^2 단위. 학습 쪽이 이걸로
'정답'(머리가 어깨로 안 내려옴, 거의 0)과 '긴 머리 기증자'(큼)를 고른다.

얼굴 파서(CelebAMask-HQ)는 옷 위로 내려온 머리를 옷(18)으로 찍는다(실측) - 그래서 셀피 분할을 같이 둔다.
"""
import argparse, csv, io, json, os, sys, zipfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_segmenter import GpuFaceParser, CLS_HAIR                                   # noqa: E402
from face_pose import landmarks_image                                               # noqa: E402
from segmenter import MODEL_PATH as SELFIE_MODEL, CATEGORY_HAIR                     # noqa: E402
from torso_net import crop_box, STORE_PX, CROP_D, INVALID, POSE_IDS                 # noqa: E402

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
#: 랜드마크/포즈는 긴 변을 이 크기로 줄여서 돌린다
LM_PX = 960


def decode(buf):
    im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--ann", default=os.path.join("..", "data", "easyportrait", "annotations.zip"))
    ap.add_argument("--meta", default=os.path.join("..", "data", "easyportrait", "meta.zip"))
    ap.add_argument("--out", default=os.path.join("..", "data", "torso"))
    ap.add_argument("--n", type=int, default=0, help="0 = 전부")
    ap.add_argument("--shard", default="", help="i/k: k 개 프로세스로 나눠 돌릴 때 i 번째 (meta_i.jsonl 에 쓴다, "
                                                "끝나면 meta.jsonl 에 합친다)")
    a = ap.parse_args()
    for d in ("img", "lab", "hair"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    with zipfile.ZipFile(a.meta) as z:
        users = {r["image_name"]: (r["user_id"], r["set"])
                 for r in csv.DictReader(io.StringIO(z.read("meta.tsv").decode()))}
    zi = zipfile.ZipFile(a.zip)
    za = zipfile.ZipFile(a.ann)
    ann = {os.path.splitext(os.path.basename(n))[0]: n for n in za.namelist()
           if n.endswith(".png") and "checkpoint" not in n}
    names = sorted(n for n in zi.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))
                   and "checkpoint" not in n)
    mp_ = os.path.join(a.out, "meta.jsonl")
    done = set()
    for fn in os.listdir(a.out):
        if fn.startswith("meta") and fn.endswith(".jsonl"):
            with open(os.path.join(a.out, fn), encoding="utf-8") as f:
                for l in f:
                    try:
                        done.add(json.loads(l)["id"])
                    except ValueError:      # 죽은 프로세스가 반쯤 쓴 마지막 줄
                        pass
    if a.shard:
        # 나누기는 '끝난 것 빼기' 전에: 뒤에 하면 이어 돌릴 때마다 나누는 몫이 달라져 빠지는 사진이 생긴다
        i, k = map(int, a.shard.split("/"))
        names = names[i::k]
        mp_ = os.path.join(a.out, f"meta_{i}.jsonl")
    names = [n for n in names if os.path.splitext(os.path.basename(n))[0] not in done]
    if a.n:
        names = names[:a.n]
    print("todo", len(names), "done", len(done), flush=True)

    seg = GpuFaceParser()
    selfie = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=SELFIE_MODEL), running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True, output_confidence_masks=True))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "pose_landmarker_full.task")),
        running_mode=vision.RunningMode.IMAGE, num_poses=1))

    pool = ThreadPoolExecutor(6)
    futs = {}
    skipped = 0
    with open(mp_, "a", encoding="utf-8") as meta:
        for k, n in enumerate(names):
            for j in names[k:k + 12]:
                if j not in futs:
                    futs[j] = pool.submit(decode, zi.read(j))
            im = futs.pop(n).result()
            iid = os.path.splitext(os.path.basename(n))[0]
            if im is None:
                skipped += 1
                continue
            H, W = im.shape[:2]
            s = LM_PX / max(H, W)
            small = cv2.resize(im, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
            srgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            lm = landmarks_image(srgb)
            if not lm:
                skipped += 1
                continue
            box = crop_box(lm["eye_l"] / s, lm["eye_r"] / s)
            x0, y0, S = box
            d = S / CROP_D
            if d < 40:
                skipped += 1
                continue
            # 크롭 배율로 먼저 INTER_AREA 축소 -> 평행이동만 (warpAffine 은 AREA 축소가 없다)
            kz = STORE_PX / S
            big = cv2.resize(im, (max(1, round(W * kz)), max(1, round(H * kz))), interpolation=cv2.INTER_AREA)
            T = np.float32([[1, 0, -x0 * kz], [0, 1, -y0 * kz]])
            crop = cv2.warpAffine(big, T, (STORE_PX, STORE_PX), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            valid = cv2.warpAffine(np.full(big.shape[:2], 255, np.uint8), T, (STORE_PX, STORE_PX),
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127

            # 원본 좌표 -> 크롭 좌표 (s 로 줄인 랜드마크 좌표 기준)
            def tc(p, from_small=True):
                p = np.asarray(p, np.float32) / (s if from_small else 1.0)
                return np.stack([(p[..., 0] - x0) * kz, (p[..., 1] - y0) * kz], -1)

            cls = seg.class_map(crop)
            res = selfie.segment(mp.Image(image_format=mp.ImageFormat.SRGB,
                                          data=np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
            scat = np.array(res.category_mask.numpy_view(), np.uint8)
            hair = np.array(res.confidence_masks[CATEGORY_HAIR].numpy_view(), np.float32)
            if scat.shape != (STORE_PX, STORE_PX):
                scat = cv2.resize(scat, (STORE_PX, STORE_PX), interpolation=cv2.INTER_NEAREST)
                hair = cv2.resize(hair, (STORE_PX, STORE_PX), interpolation=cv2.INTER_LINEAR)
            ep = np.zeros((STORE_PX, STORE_PX), np.uint8)
            if iid in ann:
                ea = cv2.imdecode(np.frombuffer(za.read(ann[iid]), np.uint8), cv2.IMREAD_GRAYSCALE)
                if ea is not None:
                    if ea.shape[:2] != (H, W):
                        ea = cv2.resize(ea, (W, H), interpolation=cv2.INTER_NEAREST)
                    ea = cv2.resize(ea, (big.shape[1], big.shape[0]), interpolation=cv2.INTER_NEAREST)
                    ep = cv2.warpAffine(ea, T, (STORE_PX, STORE_PX), flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            lab = np.stack([cls, scat, ep], -1)
            lab[~valid] = INVALID
            hair[~valid] = 0

            pres = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=srgb))
            pts = None
            if pres.pose_landmarks:
                P = pres.pose_landmarks[0]
                sw, sh = srgb.shape[1], srgb.shape[0]
                pts = [[*np.round(tc([P[i].x * sw, P[i].y * sh]), 1).tolist(), round(float(P[i].visibility), 3)]
                       for i in POSE_IDS]

            oval = tc(lm["oval"])
            chin = float(oval[:, 1].max())
            dd = STORE_PX / CROP_D
            # 판정은 셀피 머리만: 파서는 카키/갈색 옷을 머리로 찍는다(실측, 민소매 남자 hb 3.9)
            below = np.arange(STORE_PX)[:, None] > chin + 0.3 * dd
            hair_below = float(((hair > 0.5) & below & valid).sum()) / dd ** 2
            hair_below_parser = float(((cls == CLS_HAIR) & below & valid).sum()) / dd ** 2
            # 정답 후보로 쓸 수 있는지: 턱 아래가 원본 안에 얼마나 들어오나
            valid_below = float((valid & below).sum()) / max(1.0, float(below.sum()) * STORE_PX)

            cv2.imwrite(os.path.join(a.out, "img", iid + ".jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(a.out, "lab", iid + ".png"), lab)
            cv2.imwrite(os.path.join(a.out, "hair", iid + ".png"), np.round(hair * 255).astype(np.uint8))
            u, st = users.get(iid, ("", ""))
            r2 = lambda v: np.round(np.asarray(v, np.float32), 1).tolist()  # noqa: E731
            meta.write(json.dumps({
                "id": iid, "user": u, "set": st, "src": [W, H], "d": round(d, 1),
                "box": [round(x0, 1), round(y0, 1), round(S, 1)],
                "eye_l": r2(tc(lm["eye_l"])), "eye_r": r2(tc(lm["eye_r"])),
                "mouth_l": r2(tc(lm["mouth_l"])), "mouth_r": r2(tc(lm["mouth_r"])),
                "oval": r2(oval), "pose": pts,
                "chin": round(chin, 1), "hair_below": round(hair_below, 3),
                "hair_below_parser": round(hair_below_parser, 3), "valid_below": round(valid_below, 3),
            }) + "\n")
            if k % 200 == 0:
                meta.flush()
                print(k, iid, f"d={d:.0f} hair_below={hair_below:.2f} valid_below={valid_below:.2f}",
                      f"skipped={skipped}", flush=True)
    print("DONE skipped", skipped, flush=True)


if __name__ == "__main__":
    main()
