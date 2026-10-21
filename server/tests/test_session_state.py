"""세그멘터(세션 공용)의 세션별 상태가 세션끼리 섞이지 않는지 (GpuFaceParser._use_session), 그리고
_pushpull 을 줄인 식이 예전 식과 같은 값을 내는지.

2세션이면 머리색 EMA·그룸 자동 맞춤이 두 사람 것으로 섞이고 캐시가 프레임마다 서로 덮어써졌다.
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                   # noqa: E402
import torch.nn.functional as F                                # noqa: E402

from gpu_segmenter import GpuFaceParser, SessionPlate          # noqa: E402


def _fake_parser():
    """모델 없이 _use_session 만 쓸 수 있는 껍데기."""
    p = types.SimpleNamespace(_sess=None, **GpuFaceParser._SESSION_STATE)
    p._SESSION_STATE = GpuFaceParser._SESSION_STATE
    p.use = lambda plate: GpuFaceParser._use_session(p, plate)
    return p


class TestSessionState(unittest.TestCase):
    def test_states_do_not_mix(self):
        seg = _fake_parser()
        a, b = SessionPlate("cpu"), SessionPlate("cpu")
        seg.use(a)
        seg._hair_color, seg._autofit = "A색", {"n": 3}
        seg.use(b)
        self.assertIsNone(seg._hair_color)              # 새 세션은 기본값부터
        self.assertIsNone(seg._autofit)
        seg._hair_color = "B색"
        seg.use(a)
        self.assertEqual(seg._hair_color, "A색")        # A 것이 그대로 돌아온다
        self.assertEqual(seg._autofit, {"n": 3})
        seg.use(b)
        self.assertEqual(seg._hair_color, "B색")

    def test_same_session_and_none_keep_state(self):
        seg = _fake_parser()
        a = SessionPlate("cpu")
        seg.use(a)
        seg._stat_frame = 42
        seg.use(a)
        seg.use(None)                                   # plate 없는 호출(class_map 등)은 안 바꾼다
        self.assertEqual(seg._stat_frame, 42)


def _pushpull_ref(img, m, levels=7):
    """줄이기 전 _pushpull (기준)."""
    pyr = []
    c, w_ = img * m, m
    for _ in range(levels):
        pyr.append((c, w_))
        if min(c.shape[-2:]) < 4:
            break
        c = F.avg_pool2d(c, 2, ceil_mode=True)
        w_ = F.avg_pool2d(w_, 2, ceil_mode=True)
    cur = c / w_.clamp(min=1e-6)
    for c, w_ in reversed(pyr[:-1]):
        up = F.interpolate(cur, size=c.shape[-2:], mode="bilinear", align_corners=False)
        k = w_.clamp(0.0, 1e-2) * 100.0
        cur = (c / w_.clamp(min=1e-6)) * k + up * (1.0 - k)
    return cur


class TestPushPull(unittest.TestCase):
    def test_matches_reference(self):
        g = torch.Generator().manual_seed(0)
        for shp, ch in [((97, 131), 3), ((61, 40), 4), ((5, 7), 3), ((3, 3), 1), ((300, 260), 1)]:
            img = torch.rand(1, ch, *shp, generator=g) * 255
            m = (torch.rand(1, 1, *shp, generator=g) > 0.7).float() * torch.rand(1, 1, *shp, generator=g) * 1.3
            out = GpuFaceParser._pushpull(img, m)
            self.assertEqual(out.shape, img.shape)
            self.assertLess((out - _pushpull_ref(img, m)).abs().max().item(), 1e-3, shp)


if __name__ == "__main__":
    unittest.main()
