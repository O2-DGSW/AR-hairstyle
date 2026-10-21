"""groom_renderer._hug_front: 얼굴 앞 가닥만 두상 표면 쪽으로 당기는지."""
import dataclasses
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh

import groom_renderer as gr
from config import CONFIG

R = 9.0     # 구 두상 반지름(cm)


def _sphere():
    s = trimesh.creation.icosphere(subdivisions=4, radius=R)
    return np.asarray(s.vertices, np.float32), np.asarray(s.faces, np.int32)


def _strand(x, y, z):
    return np.array([[[x, y, z]] * 4], np.float32)        # (1, 4, 3)


class TestHugFront(unittest.TestCase):
    def _run(self, c, **kw):
        kw = {"groom_front_hug": 0.3, "groom_front_hug_cm": 0.6, "groom_front_hug_y": (-1.0, 9.0, 12.0), **kw}
        cfg = dataclasses.replace(CONFIG, **kw)
        with mock.patch.object(gr, "CONFIG", cfg):
            return gr._hug_front(c, *_sphere())

    def test_front_fringe_is_pulled_toward_forehead(self):
        # 왜 중요: 이마 앞 4cm 뜬 앞머리가 옆모습에서 차양처럼 길게 보였다 - knee + (4 - knee) * k 로 줄어야 한다.
        q = self._run(_strand(0.0, 3.0, np.sqrt(R * R - 9.0) + 4.0))
        gap = np.linalg.norm(q[0, 0]) - R
        self.assertAlmostEqual(gap, 0.6 + 3.4 * 0.3, delta=0.15)

    def test_back_and_low_hair_untouched(self):
        # 뒷머리(볼륨은 groom_volume 몫)와 코 아래로 늘어진 긴 머리는 그대로.
        back = _strand(0.0, 3.0, -(R + 4.0))
        low = _strand(0.0, -6.0, R + 4.0)
        for c in (back, low):
            q = self._run(c)
            self.assertTrue(q is None or np.allclose(q, c))

    def test_off_when_k_is_one(self):
        self.assertIsNone(self._run(_strand(0.0, 3.0, R + 4.0), groom_front_hug=1.0))


if __name__ == "__main__":
    unittest.main()
