"""gpu_segmenter 의 분리형 필터(_maxk, _box)가 정사각 풀링과 같은 값을 내는지.

해상도를 올리면 정사각 k x k 풀링이 O(k^2) 로 비싸져 분리형으로 바꿨다. 화질이 그대로여야
하므로 "같은 결과" 를 고정해 둔다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                   # noqa: E402
import torch.nn.functional as F                                # noqa: E402

from gpu_segmenter import GpuFaceParser                       # noqa: E402


class TestSeparablePooling(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        # 이진 마스크(연장/틈 메우기 입력)와 연속값(알파) 둘 다, 가장자리 패딩이 걸리는 크기로
        self.inputs = [(torch.rand(1, 1, 61, 83, generator=g) > 0.7).float(),
                       torch.rand(1, 3, 47, 52, generator=g)]

    def test_maxk_equals_square_max_pool_exactly(self):
        for x in self.inputs:
            for k in (3, 5, 9, 21, 41):
                ref = F.max_pool2d(x, k, stride=1, padding=k // 2)
                self.assertTrue(torch.equal(GpuFaceParser._maxk(x, k), ref), k)

    def test_closing_with_maxk_equals_square(self):
        # 틈 메우기의 닫힘(팽창 후 침식)
        x, k = self.inputs[0], 15
        ref = -F.max_pool2d(-F.max_pool2d(x, k, stride=1, padding=k // 2), k, stride=1, padding=k // 2)
        got = -GpuFaceParser._maxk(-GpuFaceParser._maxk(x, k), k)
        self.assertTrue(torch.equal(got, ref))

    def test_box_matches_square_avg_pool(self):
        # 그림자 띠. 작은 커널은 합 순서만 달라 부동소수 오차 수준으로 같다.
        # 큰 커널(box_ds_k 의 2배 이상)은 줄인 해상도에서 흐리므로 근사 - 무작위 이진 마스크처럼
        # 고주파가 최대인 입력에서도 값(0~1)의 몇 % 안쪽이다.
        from config import CONFIG
        for x in self.inputs:
            for k in (3, 9, 31):
                ref = F.avg_pool2d(x, k, stride=1, padding=k // 2)
                approx = CONFIG.box_ds_k > 0 and k // CONFIG.box_ds_k >= 2
                tol = dict(rtol=0, atol=0.03) if approx else dict(rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(GpuFaceParser._box(x, k), ref, **tol)

    def test_even_and_tiny_kernels_follow_the_odd_rule(self):
        x = self.inputs[1]
        self.assertIs(GpuFaceParser._maxk(x, 1), x)
        self.assertTrue(torch.equal(GpuFaceParser._maxk(x, 4), GpuFaceParser._maxk(x, 5)))


if __name__ == "__main__":
    unittest.main()
