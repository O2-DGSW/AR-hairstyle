"""3D 그룸 합성을 서버 없이 실제 웹캠 프레임으로 검증한다.

    python train/render_groom_offline.py server/train/frames/0816_015751 --groom short-swept \
        --out /tmp/groom_test --render 0,60,190,110

앞 프레임들로 배경 플레이트를 쌓고(raw 모드도 플레이트는 갱신된다), --render 프레임을
tryon 으로 합성해 before/after 를 나란히 저장한다. GAN/서버 불필요. 결과는 눈으로 볼 것.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu_segmenter import GpuFaceParser, SessionPlate      # noqa: E402
from face_pose import FacePose                              # noqa: E402
import hair_asset                                           # noqa: E402
import server                                               # noqa: E402  (list_grooms / GROOM_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--groom", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--render", default="0,60,190")
    ap.add_argument("--warm", type=int, default=40, help="플레이트를 쌓을 프레임 수")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--up", type=float, default=0.0, help="cm")
    ap.add_argument("--fwd", type=float, default=0.0, help="cm")
    ap.add_argument("--color", default="", help="#rrggbb (비우면 내 머리색)")
    ap.add_argument("--forehead", type=int, default=-1,
                    help="이 프레임으로 맨이마 패치(LaMa)를 만들어 이마에 쓴다. -1 이면 살색 평면")
    ap.add_argument("--fh3d", type=int, default=1, help="1: 이마 패치를 3D 두상에 투영 (0: 예전 닮음변환)")
    ap.add_argument("--debug", action="store_true",
                    help="groom_dbg_*.png: 빨강=안 지워진 원래 머리, 파랑=지웠는데 새 헤어가 안 덮는 자리")
    ap.add_argument("--dyn", type=float, default=0.0,
                    help="2차 운동 세기(1=기본). 프레임 순서대로 30fps 로 스프링을 굴린다")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    files = sorted(f for f in os.listdir(a.frames) if f.endswith(".png"))
    grooms = server.list_grooms()
    obj = {"name": a.groom, "path": os.path.join(server.GROOM_DIR, a.groom + ".glb"),
           "meta": grooms[a.groom]}
    color = server._parse_bgr(a.color) if a.color else None

    seg = GpuFaceParser()
    seg.keep_debug = a.debug
    plate = SessionPlate(seg.device)
    poser = FacePose()
    smoother = hair_asset.AnchorSmoother()
    t0 = time.perf_counter()
    seg.warm_groom(obj)
    print(f"groom warm {(time.perf_counter() - t0) * 1000:.0f}ms")

    fh_asset = None
    if a.forehead >= 0:
        import forehead
        from face_pose import landmarks_image
        img = cv2.imread(os.path.join(a.frames, files[a.forehead]))
        cls = seg.class_map(img)
        lm = landmarks_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fh_asset, bangs_px, ms = forehead.build_forehead_asset(
            forehead.ForeheadInpainter(seg.device), img, cls, lm["eye_l"], lm["eye_r"], "forehead",
            7, lm.get("matrix") if a.fh3d else None, lm.get("brows"))
        seg.warm_asset(fh_asset)
        cv2.imwrite(os.path.join(a.out, "forehead_bald.png"), fh_asset.bald_bgr)
        print(f"forehead: 앞머리 {bangs_px}px, 인페인트 {ms:.0f}ms")

    from hair_dynamics import HairDynamics
    hd = HairDynamics()

    def run(idx, mode):
        img = cv2.imread(os.path.join(a.frames, files[idx]))
        pose = poser.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), idx * 33)
        dyn = None
        if a.dyn > 0 and pose is not None and pose.get("matrix") is not None:
            dyn = hd.step(pose["matrix"], 1 / 30)
            dyn["tip"] = a.dyn
        out, t = seg.process(img, plate, mode, fh_asset, 1.0, 0.0, pose, True, 0.35, 0.0,
                             None, 0.0, smoother, obj, (a.scale, a.up, a.fwd), color, dyn)
        return img, out, t, pose

    for i in range(min(a.warm, len(files))):
        run(i, "raw")
    ts = []
    want = [int(x) for x in a.render.split(",")]
    # 2차 운동은 연속성이 필요하다: 첫 렌더 프레임까지 그 앞 프레임들을 raw 로 순서대로 굴린다
    for i in range(a.warm, want[0]):
        run(i, "raw")     # 랜드마커(VIDEO 모드) 추적 연속성도 필요하다
    for idx in want:
        img, out, t, pose = run(idx, "tryon")
        ts.append(t)
        side = np.hstack([img, out])
        cv2.imwrite(os.path.join(a.out, f"groom_{idx:05d}.png"), side)
        if a.debug and seg.debug_maps is not None:
            m = {k: (v.float().cpu().numpy() if v is not None else None) for k, v in seg.debug_maps.items()}
            na = m["new_a"] if m["new_a"] is not None else np.zeros_like(m["hair_a"])
            left = m["hair_a"] * (1 - na) * (1 - m["a"])      # 원래 머리가 그대로 보이는 양
            hole = m["a"] * (1 - na)                           # 지웠는데 새 헤어가 안 덮음
            dbg = out.astype(np.float32) * 0.5
            dbg[..., 2] += 255 * 0.5 * left
            dbg[..., 0] += 255 * 0.5 * hole
            cv2.imwrite(os.path.join(a.out, f"groom_dbg_{idx:05d}.png"), dbg.clip(0, 255).astype(np.uint8))
            print(f"   left(원래머리 잔존) {left.sum():.0f}px  hole(지운 빈자리) {hole.sum():.0f}px")
        print(f"frame {idx}: total {t['total_ms']:.1f}ms groom {t['groom_ms']:.2f}ms "
              f"yaw {pose['yaw'] if pose else None}")
    print("->", a.out)


if __name__ == "__main__":
    main()
