"""합성 품질 관련 순수 로직 회귀 테스트.

GPU/모델 없이 cv2/numpy 만으로 검증한다:
  - 얼굴 패치 영속화 (save_asset / load_asset_dir 왕복)
  - 앞머리 사전 제거 마스크의 기하 (gan_input.prefill_forehead)
  - 밉맵 레벨 선택 규칙 (gpu_segmenter._warp_asset 이 쓰는 것과 같은 식)
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import hair_asset as ha


def _skip_without_torch():
    try:
        import torch  # noqa: F401
        return False
    except ImportError:
        return True


class FacePatchPersistTest(unittest.TestCase):
    def test_round_trip_keeps_face_patch(self):
        a = ha.HairAsset("t", np.zeros((8, 8, 4), np.uint8), (2, 4), (6, 4),
                         ref_skin=[1, 2, 3], yaw=3.0, bank="b")
        a.face = ha.HairAsset("t#face", np.full((6, 6, 4), 7, np.uint8), (1, 3), (5, 3))
        d = tempfile.mkdtemp()
        ha.save_asset(a, d)
        self.assertEqual(sorted(os.listdir(d)), ["t.face.png", "t.json", "t.png"])
        out = ha.load_asset_dir(d)
        # 얼굴 패치 PNG 가 독립 에셋으로 등록되면 안 된다.
        self.assertEqual(list(out), ["t"])
        f = out["t"].face
        self.assertIsNotNone(f)
        self.assertEqual(f.rgba.shape, (6, 6, 4))
        np.testing.assert_allclose(f.eye_l, [1, 3])
        np.testing.assert_allclose(f.eye_r, [5, 3])

    def test_prune_removes_face_png_with_body(self):
        """prune_dir 가 본체를 지울 때 <name>.face.png 도 같이 지운다(고아 방지)."""
        import time
        rng = np.random.default_rng(1)
        d = tempfile.mkdtemp()
        for i in range(2):
            # 압축이 안 되는 무작위 픽셀로 세트당 ~2MB(본체 1MB + 패치 1MB).
            a = ha.HairAsset(f"b{i}", rng.integers(0, 255, (512, 512, 4), dtype=np.uint8),
                             (2, 4), (6, 4))
            a.face = ha.HairAsset(f"b{i}#face",
                                  rng.integers(0, 255, (512, 512, 4), dtype=np.uint8),
                                  (1, 3), (5, 3))
            ha.save_asset(a, d)
            time.sleep(0.05)     # mtime 순서를 확실히
        size = {f: os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)}
        total = sum(size.values())
        set0 = size["b0.png"] + size["b0.json"] + size["b0.face.png"]
        # 상한은 정수 MB 다. 첫 세트만 지우면 들어가고, 안 지우면 넘치는 값을 고른다.
        limit_mb = (total - set0) // (1024 * 1024) + 1
        self.assertLess(total - set0, limit_mb * 1024 * 1024)
        self.assertGreater(total, limit_mb * 1024 * 1024)
        removed = ha.prune_dir(d, max_mb=limit_mb)
        self.assertEqual(removed, 1)
        self.assertEqual(sorted(os.listdir(d)), ["b1.face.png", "b1.json", "b1.png"])

    def test_loaded_face_name_matches_extract_rule(self):
        a = ha.HairAsset("n2", np.zeros((8, 8, 4), np.uint8), (2, 4), (6, 4))
        a.face = ha.HairAsset("n2#face", np.zeros((6, 6, 4), np.uint8), (1, 3), (5, 3))
        d = tempfile.mkdtemp()
        ha.save_asset(a, d)
        self.assertEqual(ha.load_asset_dir(d)["n2"].face.name, "n2#face")

    def test_asset_without_face_loads(self):
        a = ha.HairAsset("n", np.zeros((8, 8, 4), np.uint8), (2, 4), (6, 4))
        d = tempfile.mkdtemp()
        ha.save_asset(a, d)
        out = ha.load_asset_dir(d)
        self.assertIsNone(out["n"].face)


@unittest.skipIf(_skip_without_torch(), "torch 필요 (gpu_segmenter import)")
class PrefillForeheadTest(unittest.TestCase):
    """눈 간격 D=50, 눈 중심 (100,120)/(150,120) 인 200x250 합성 프레임."""

    def _frame(self):
        import gan_input
        from gpu_segmenter import CLS_HAIR, CLS_SKIN
        h, w = 200, 250
        img = np.full((h, w, 3), 200, np.uint8)
        cls = np.zeros((h, w), np.uint8)
        cls[60:190, 70:180] = CLS_SKIN          # 얼굴
        cls[20:110, 60:190] = CLS_HAIR          # 머리 (이마 y<110 까지 앞머리)
        img[cls == CLS_HAIR] = 30
        img[cls == CLS_SKIN] = (150, 170, 210)
        return gan_input, img, cls, (100.0, 120.0), (150.0, 120.0)

    def test_fills_bangs_above_limit_only(self):
        gi, img, cls, el, er = self._frame()
        out, n = gi.prefill_forehead(img, cls, el, er)
        self.assertGreater(n, 0)
        d = 50.0
        # 아래 한계(눈 위 BROW_D*D) 아래는 절대 안 건드린다.
        limit_y = int(120 - gi.BROW_D * d)
        np.testing.assert_array_equal(out[limit_y + 1:], img[limit_y + 1:])
        # 이마 한가운데(눈 위 0.9D)는 피부색으로 바뀌었다.
        y = int(120 - 0.9 * d)
        self.assertLess(abs(int(out[y, 125, 0]) - 150), 25)
        self.assertNotEqual(int(out[y, 125, 0]), 30)
        # 정수리(눈 위 1.6D)는 남는다.
        y = int(120 - 1.6 * d)
        self.assertEqual(int(out[y, 125, 0]), 30)

    def test_no_hair_no_change(self):
        gi, img, cls, el, er = self._frame()
        from gpu_segmenter import CLS_HAIR, CLS_SKIN
        cls[cls == CLS_HAIR] = CLS_SKIN
        out, n = gi.prefill_forehead(img, cls, el, er)
        self.assertEqual(n, 0)
        self.assertIs(out, img)


class MipLevelRuleTest(unittest.TestCase):
    """_warp_asset 의 레벨 선택: 배율 scale 에 대해 0.5^lvl >= scale > 0.5^(lvl+1)."""

    @staticmethod
    def pick(scale, n_levels):
        lvl = 0
        while lvl + 1 < n_levels and scale <= 0.5 ** (lvl + 1):
            lvl += 1
        return lvl

    def test_levels(self):
        self.assertEqual(self.pick(1.0, 5), 0)
        self.assertEqual(self.pick(0.6, 5), 0)
        self.assertEqual(self.pick(0.5, 5), 1)
        self.assertEqual(self.pick(0.3, 5), 1)
        self.assertEqual(self.pick(0.23, 5), 2)      # 웹캠 480p 전형값 (50/220)
        self.assertEqual(self.pick(0.1, 5), 3)
        self.assertEqual(self.pick(0.01, 5), 4)      # 상한에서 멈춘다
        self.assertEqual(self.pick(0.01, 1), 0)


class SharpnessPickTest(unittest.TestCase):
    def test_blurred_frame_scores_lower(self):
        import cv2
        try:
            import server
        except Exception as e:      # aiortc 등 서버 의존성이 없으면 건너뛴다
            self.skipTest(f"server import 불가: {e}")
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        pose = {"eye_l": np.array([140.0, 120.0]), "eye_r": np.array([180.0, 120.0])}
        sharp = server.SegmentedVideoTrack._sharpness(img, pose)
        blurred = server.SegmentedVideoTrack._sharpness(cv2.GaussianBlur(img, (0, 0), 2), pose)
        self.assertGreater(sharp, blurred * 3)


if __name__ == "__main__":
    unittest.main()
