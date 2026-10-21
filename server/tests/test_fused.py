"""fused.py 의 식이 gpu_segmenter 에 있던 원래 식과 같은 값을 내는지 (eager 경로).

torch.compile 은 이 식들을 그대로 합칠 뿐이라, 여기서 식이 같으면 켜도 결과가 같다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                   # noqa: E402

import fused                                                   # noqa: E402


def _r(*s, g):
    return torch.rand(*s, generator=g)


class TestFused(unittest.TestCase):
    def setUp(self):
        self.g = torch.Generator().manual_seed(0)

    def test_blend_under(self):
        g = self.g
        rgb, a, pp, e = _r(30, 20, 3, g=g) * 255, _r(30, 20, 1, g=g), _r(30, 20, 3, g=g) * 255, _r(30, 20, 1, g=g)
        tot = a + e * (1.0 - a)
        ref = (rgb * a + pp * e * (1.0 - a)) / tot.clamp(min=1e-4)
        out, t2 = fused.blend_under(rgb, a, pp, e)
        self.assertTrue(torch.allclose(out, ref) and torch.allclose(t2, tot))

    def test_extend_weight_matches_original(self):
        g = self.g
        h, w = 40, 30
        hair_a, a, near, below, ff, ahead = (_r(h, w, g=g) for _ in range(6))
        ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32),
                                indexing="ij")
        ey, cx, d, ek, fw, gain = 25.0, 14.0, 9.0, 0.3, 0.8, 1.0
        # 원래 _groom_extend 의 식
        above = ((ey - ek * d - ys) / (0.25 * d)).clamp(0.0, 1.0)
        inner = ((fw * d - (xs - cx).abs()) / (0.25 * d)).clamp(0.0, 1.0)
        forehead = torch.maximum(torch.maximum((below * inner).clamp(0.0, 1.0), ff), ahead)
        ref = hair_a * (1.0 - a) * near * above * (1.0 - forehead) * gain
        out = fused.extend_weight(hair_a, a, near, below, ys, xs, ff, ahead, ey, cx, d, ek, fw, gain)
        self.assertTrue(torch.allclose(out, ref))

    def test_extend_final_without_detail_is_plain_blend(self):
        g = self.g
        rgb, a, pp, e = _r(10, 8, 3, g=g) * 255, _r(10, 8, 1, g=g), _r(10, 8, 3, g=g) * 255, _r(10, 8, 1, g=g)
        one = torch.ones_like(a)
        out, tot = fused.extend_final(rgb, a, pp, e, one, one, 0.0)
        ref, rt = fused.blend_under(rgb, a, pp, e)
        self.assertTrue(torch.allclose(out, ref.clamp(0, 255)) and torch.allclose(tot, rt))


if __name__ == "__main__":
    unittest.main()
