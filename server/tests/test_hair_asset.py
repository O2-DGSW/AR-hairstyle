"""hair_asset 의 순수 기하/선택/평활화 로직 회귀 테스트.

GPU 도 모델도 쓰지 않는다. cv2/numpy 만 필요하다.
"""
import math
import os
import sys
import unittest

# `python -m unittest discover -s tests` 는 tests/ 를 top-level 로 잡아서 패키지로
# 인식하지 않는다(상대 import 가 깨진다). 그래서 자기 위치 기준으로 server/ 를
# 직접 sys.path 에 넣는다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import hair_asset as ha


def apply(M, p):
    """2x3 어파인을 점 하나에 적용."""
    M = np.asarray(M, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return M[:, :2] @ p + M[:, 2]


def fake_asset(name, yaw=None, bank=None, size=8):
    """작은 RGBA 배열로 만든 가짜 에셋. 선택 로직만 보므로 픽셀 내용은 무의미."""
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    return ha.HairAsset(name, rgba, (2, 4), (6, 4), yaw=yaw, bank=bank)


def fake_bank(bank, yaws):
    out = {}
    for y in yaws:
        name = "%s_%+03.0f" % (bank, y)
        out[name] = fake_asset(name, yaw=y, bank=bank)
    return out


# ---------------------------------------------------------------- order_by_x

class TestOrderByX(unittest.TestCase):
    def test_order_by_x_is_independent_of_argument_order(self):
        # 왜 중요: CelebAMask-HQ 의 l_eye/r_eye 는 인물 기준이라 화면에서는 뒤바뀐다.
        # 인자 순서가 결과를 바꾸면 그 라벨을 그대로 믿는 것과 같아진다.
        left, right = (10.0, 50.0), (90.0, 52.0)
        a = ha.order_by_x(left, right)
        b = ha.order_by_x(right, left)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])
        np.testing.assert_allclose(a[0], np.asarray(left, np.float32))
        np.testing.assert_allclose(a[1], np.asarray(right, np.float32))

    def test_order_by_x_returns_screen_left_first(self):
        # 왜 중요: 이 함수의 유일한 사후조건. 항상 x 가 작은 쪽이 먼저 나와야 한다.
        for p, q in [((0, 0), (1, 0)), ((5, 9), (-5, 9)), ((-3, 1), (-1, 1))]:
            l, r = ha.order_by_x(p, q)
            self.assertLessEqual(float(l[0]), float(r[0]))

    def test_order_by_x_with_equal_x_keeps_both_points(self):
        # 왜 중요: 두 눈의 x 가 같은 퇴화 입력에서 점을 잃거나 예외를 내면 안 된다
        # (어느 쪽이 먼저인지는 규정하지 않는다).
        l, r = ha.order_by_x((7.0, 1.0), (7.0, 9.0))
        self.assertEqual(sorted([float(l[1]), float(r[1])]), [1.0, 9.0])
        self.assertEqual(float(l[0]), 7.0)
        self.assertEqual(float(r[0]), 7.0)

    def test_order_by_x_accepts_tuples_lists_and_arrays(self):
        # 왜 중요: 호출부가 mediapipe(np.float32 배열)/JSON(list)/상수(tuple) 를 섞어 넘긴다.
        for p, q in [((80.0, 1.0), (4.0, 2.0)),
                     ([80.0, 1.0], [4.0, 2.0]),
                     (np.array([80.0, 1.0], np.float32), np.array([4.0, 2.0], np.float32))]:
            l, r = ha.order_by_x(p, q)
            self.assertIsInstance(l, np.ndarray)
            self.assertIsInstance(r, np.ndarray)
            self.assertEqual((float(l[0]), float(r[0])), (4.0, 80.0))

    def test_order_by_x_cancels_a_flipped_eye_label(self):
        # 왜 중요: 과거 사고 그 자체. 라벨이 한 번 뒤집히면 눈 방향벡터가 반대가 되어
        # 닮음변환이 정확히 180도 돌고 헤어가 뒤집혔다. 양쪽을 order_by_x 로
        # 정규화하면 라벨이 뒤집혀도 같은 행렬이 나와야 한다.
        src = ((100.0, 300.0), (212.0, 300.0))
        good = ((300.0, 250.0), (400.0, 250.0))
        flipped = (good[1], good[0])          # 세그멘터가 좌우 라벨을 헷갈린 프레임

        sl, sr = ha.order_by_x(*src)
        m_good = ha.similarity_matrix(sl, sr, *ha.order_by_x(*good))
        m_flip = ha.similarity_matrix(sl, sr, *ha.order_by_x(*flipped))
        np.testing.assert_allclose(m_good, m_flip, atol=1e-5)

    def test_unordered_flipped_labels_really_do_rotate_180_degrees(self):
        # 왜 중요: 위 테스트가 막고 있는 사고가 실재함을 고정한다(대조군).
        # order_by_x 를 건너뛰면 회전 성분의 부호가 통째로 뒤집힌다(= 180도 회전).
        src = ((100.0, 300.0), (212.0, 300.0))
        d0, d1 = (300.0, 250.0), (400.0, 250.0)
        m_good = ha.similarity_matrix(src[0], src[1], d0, d1)
        m_flip = ha.similarity_matrix(src[0], src[1], d1, d0)
        np.testing.assert_allclose(m_flip[:, :2], -m_good[:, :2], atol=1e-5)

    def test_hair_asset_stores_eyes_in_screen_order(self):
        # 왜 중요: 에셋 json 의 eyeL/eyeR 도 사람이 손으로 찍은 것이라 뒤집힐 수 있다.
        rgba = np.zeros((8, 8, 4), np.uint8)
        a = ha.HairAsset("x", rgba, (6, 4), (2, 4))   # 일부러 뒤집어 넣는다
        self.assertLessEqual(float(a.eye_l[0]), float(a.eye_r[0]))
        self.assertEqual(float(a.eye_l[0]), 2.0)


# -------------------------------------------------------- similarity_matrix

class TestSimilarityMatrix(unittest.TestCase):
    SRC = ((100.0, 300.0), (212.0, 300.0))
    DST = ((320.0, 250.0), (410.0, 285.0))     # 기울어진 눈 축

    def test_similarity_matrix_maps_src_points_exactly_onto_dst(self):
        # 왜 중요: scale_mul=1, offset_up=0 이면 두 대응점은 정확히 일치해야 한다.
        M = ha.similarity_matrix(self.SRC[0], self.SRC[1], self.DST[0], self.DST[1])
        np.testing.assert_allclose(apply(M, self.SRC[0]), self.DST[0], atol=1e-3)
        np.testing.assert_allclose(apply(M, self.SRC[1]), self.DST[1], atol=1e-3)

    def test_similarity_matrix_has_no_shear(self):
        # 왜 중요: 이 함수가 존재하는 이유. 3점 어파인은 전단을 허용해서 세그멘테이션이
        # 조금만 흔들려도 헤어가 찌그러진다. 좌상단 2x2 는 [[a,-b],[b,a]] 여야 한다.
        for scale_mul in (0.6, 1.0, 1.7):
            M = ha.similarity_matrix(self.SRC[0], self.SRC[1], self.DST[0], self.DST[1],
                                     scale_mul=scale_mul)
            A = np.asarray(M, np.float64)[:, :2]
            self.assertAlmostEqual(A[0, 0], A[1, 1], places=5)      # a == a
            self.assertAlmostEqual(A[0, 1], -A[1, 0], places=5)     # -b, b
            # 등가 표현: A^T A 가 스칼라 * 단위행렬 (직교 + 균등 스케일)
            s2 = A[0, 0] ** 2 + A[1, 0] ** 2
            np.testing.assert_allclose(A.T @ A, np.eye(2) * s2, atol=1e-4 * max(1.0, s2))

    def test_similarity_matrix_scales_every_direction_equally(self):
        # 왜 중요: 전단이 없다는 것의 관측 가능한 형태. 어떤 방향의 선분이든
        # 같은 배율로 늘어나야 헤어가 기울어져 보이지 않는다.
        M = ha.similarity_matrix(self.SRC[0], self.SRC[1], self.DST[0], self.DST[1])
        base = np.asarray(self.SRC[0], np.float64)
        ratios = []
        for ang in range(0, 360, 30):
            v = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))]) * 37.0
            ratios.append(np.linalg.norm(apply(M, base + v) - apply(M, base)) / 37.0)
        np.testing.assert_allclose(ratios, [ratios[0]] * len(ratios), rtol=1e-4)

    def test_similarity_matrix_scale_mul_scales_distance_proportionally(self):
        # 왜 중요: 슬라이더/에셋 scale_adjust 가 이 인자로 들어간다. 2배는 정확히 2배여야 한다.
        M1 = ha.similarity_matrix(self.SRC[0], self.SRC[1], self.DST[0], self.DST[1],
                                  scale_mul=1.0)
        M2 = ha.similarity_matrix(self.SRC[0], self.SRC[1], self.DST[0], self.DST[1],
                                  scale_mul=2.0)
        d1 = np.linalg.norm(apply(M1, self.SRC[1]) - apply(M1, self.SRC[0]))
        d2 = np.linalg.norm(apply(M2, self.SRC[1]) - apply(M2, self.SRC[0]))
        self.assertAlmostEqual(d2 / d1, 2.0, places=4)

    def test_similarity_matrix_offset_up_translates_only_in_y_for_level_eyes(self):
        # 왜 중요: 눈이 수평이면 '얼굴 기준 위' 는 화면 위(-y)다. x 가 움직이면 헤어가 옆으로 샌다.
        src = ((0.0, 0.0), (10.0, 0.0))
        dst = ((100.0, 200.0), (200.0, 200.0))
        base = ha.similarity_matrix(src[0], src[1], dst[0], dst[1])
        up = ha.similarity_matrix(src[0], src[1], dst[0], dst[1], offset_up=7.0)
        delta = apply(up, src[0]) - apply(base, src[0])
        self.assertAlmostEqual(delta[0], 0.0, places=4)
        self.assertAlmostEqual(delta[1], -7.0, places=4)   # 위 = y 감소

    def test_similarity_matrix_offset_up_is_pure_translation(self):
        # 왜 중요: offset_up 은 위치만 바꿔야 한다. 회전/스케일이 딸려 오면
        # 슬라이더를 만질 때마다 헤어 크기가 같이 변한다.
        base = ha.similarity_matrix(self.SRC[0], self.SRC[1], self.DST[0], self.DST[1])
        up = ha.similarity_matrix(self.SRC[0], self.SRC[1], self.DST[0], self.DST[1],
                                  offset_up=13.0)
        np.testing.assert_allclose(np.asarray(up)[:, :2], np.asarray(base)[:, :2], atol=1e-5)

    def test_similarity_matrix_offset_up_is_perpendicular_to_the_eye_axis(self):
        # 왜 중요: 고개를 기울이면 '위' 도 같이 기울어야 헤어가 정수리에 붙어 있다.
        base = ha.similarity_matrix(self.SRC[0], self.SRC[1], self.DST[0], self.DST[1])
        up = ha.similarity_matrix(self.SRC[0], self.SRC[1], self.DST[0], self.DST[1],
                                  offset_up=9.0)
        delta = apply(up, self.SRC[0]) - apply(base, self.SRC[0])
        axis = np.asarray(self.DST[1], np.float64) - np.asarray(self.DST[0], np.float64)
        self.assertAlmostEqual(float(np.linalg.norm(delta)), 9.0, places=3)
        self.assertAlmostEqual(float(delta @ axis), 0.0, places=2)

    def test_similarity_matrix_returns_none_for_degenerate_input(self):
        # 왜 중요: 세그멘테이션이 두 눈을 한 점으로 뭉개는 프레임이 실제로 나온다.
        # 예외 대신 None 을 돌려주고 호출부가 프레임을 건너뛰는 계약이다.
        self.assertIsNone(ha.similarity_matrix((5, 5), (5, 5), (0, 0), (10, 0)))
        self.assertIsNone(ha.similarity_matrix((0, 0), (10, 0), (3, 3), (3, 3)))
        self.assertIsNone(ha.similarity_matrix((0, 0), (0, 0), (0, 0), (0, 0)))

    def test_similarity_matrix_is_float32_2x3(self):
        # 왜 중요: cv2.warpAffine / GPU 워퍼가 2x3 float32 를 기대한다.
        M = ha.similarity_matrix(self.SRC[0], self.SRC[1], self.DST[0], self.DST[1])
        self.assertEqual(M.shape, (2, 3))
        self.assertEqual(M.dtype, np.float32)


# ------------------------------------------------------------ invert_affine

class TestInvertAffine(unittest.TestCase):
    def test_invert_affine_round_trips_points(self):
        # 왜 중요: GPU 워핑은 '출력 픽셀 -> 입력 좌표' 방향을 쓴다. 왕복이 안 맞으면
        # 헤어와 마스크가 서로 다른 좌표계로 합성돼 어긋난다.
        M = ha.similarity_matrix((100, 300), (212, 300), (320, 250), (410, 285),
                                 scale_mul=1.3, offset_up=11.0)
        Minv = ha.invert_affine(M)
        for p in [(0.0, 0.0), (100.0, 300.0), (511.0, 7.0), (-40.0, 260.0)]:
            np.testing.assert_allclose(apply(Minv, apply(M, p)), p, atol=1e-2)

    def test_invert_affine_composes_to_identity(self):
        # 왜 중요: 왕복을 행렬 수준에서 고정한다(점 몇 개가 우연히 맞는 것과 구분).
        M = ha.similarity_matrix((10, 20), (60, 90), (300, 40), (120, 200))
        Minv = np.asarray(ha.invert_affine(M), np.float64)
        M = np.asarray(M, np.float64)
        comp_A = Minv[:, :2] @ M[:, :2]
        comp_t = Minv[:, :2] @ M[:, 2] + Minv[:, 2]
        np.testing.assert_allclose(comp_A, np.eye(2), atol=1e-4)
        np.testing.assert_allclose(comp_t, np.zeros(2), atol=1e-2)

    def test_invert_affine_of_a_similarity_is_also_shear_free(self):
        # 왜 중요: 역변환 쪽에서 전단이 새로 생기면 워핑 결과가 찌그러진다.
        M = ha.similarity_matrix((10, 20), (60, 90), (300, 40), (120, 200))
        A = np.asarray(ha.invert_affine(M), np.float64)[:, :2]
        self.assertAlmostEqual(A[0, 0], A[1, 1], places=5)
        self.assertAlmostEqual(A[0, 1], -A[1, 0], places=5)

    def test_invert_affine_returns_float32_2x3(self):
        M = ha.similarity_matrix((0, 0), (10, 0), (0, 0), (20, 0))
        Minv = ha.invert_affine(M)
        self.assertEqual(Minv.shape, (2, 3))
        self.assertEqual(Minv.dtype, np.float32)


# -------------------------------------------------------- blend_crop_matrix

class TestBlendCropMatrix(unittest.TestCase):
    def test_blend_crop_constants_match_the_trained_blender(self):
        # 왜 중요: 이 세 상수는 학습(train/make_pairs.py)과 추론(gpu_segmenter)이
        # 공유하는 크롭 규격이다. 값이 바뀌면 이미 학습된 blender.pt 와 프레이밍이
        # 어긋나 모델이 엉뚱하게 동작한다. 바꾸려면 blender 를 다시 학습해야 하고,
        # 그때 이 테스트를 같이 고쳐야 한다(= 조용히 바뀌는 걸 막는 잠금장치).
        self.assertEqual(ha.BLEND_CROP, 256)
        self.assertAlmostEqual(ha.BLEND_EYE_Y, 0.42, places=6)
        self.assertAlmostEqual(ha.BLEND_EYE_SPAN, 0.34, places=6)

    def test_blend_crop_matrix_places_eyes_at_the_trained_positions(self):
        # 왜 중요: 어떤 프레임의 눈이 들어와도 크롭 안에서는 항상 같은 자리에 와야
        # 블렌더가 학습 때와 같은 입력을 본다.
        c = ha.BLEND_CROP
        want_l = (c * (0.5 - ha.BLEND_EYE_SPAN / 2), c * ha.BLEND_EYE_Y)
        want_r = (c * (0.5 + ha.BLEND_EYE_SPAN / 2), c * ha.BLEND_EYE_Y)
        for eye_l, eye_r in [((100.0, 300.0), (212.0, 300.0)),
                             ((640.0, 120.0), (700.0, 190.0)),      # 기울고 작은 얼굴
                             ((30.0, 400.0), (330.0, 380.0))]:      # 크고 반대로 기운 얼굴
            M = ha.blend_crop_matrix(eye_l, eye_r)
            np.testing.assert_allclose(apply(M, eye_l), want_l, atol=1e-2)
            np.testing.assert_allclose(apply(M, eye_r), want_r, atol=1e-2)

    def test_blend_crop_eye_span_is_the_fraction_of_the_crop_width(self):
        # 왜 중요: 눈 간격이 크롭 폭의 BLEND_EYE_SPAN 배가 되는 것이 규격의 핵심.
        M = ha.blend_crop_matrix((100.0, 300.0), (212.0, 300.0))
        d = np.linalg.norm(apply(M, (212.0, 300.0)) - apply(M, (100.0, 300.0)))
        self.assertAlmostEqual(d / ha.BLEND_CROP, ha.BLEND_EYE_SPAN, places=5)

    def test_blend_crop_matrix_scales_with_the_size_argument(self):
        # 왜 중요: size 를 바꿔도 상대 배치는 그대로여야 한다(해상도만 다른 같은 규격).
        p = (150.0, 280.0)
        m1 = apply(ha.blend_crop_matrix((100.0, 300.0), (212.0, 300.0), size=256), p)
        m2 = apply(ha.blend_crop_matrix((100.0, 300.0), (212.0, 300.0), size=512), p)
        np.testing.assert_allclose(m2, m1 * 2.0, atol=1e-2)

    def test_blend_crop_matrix_returns_none_for_degenerate_eyes(self):
        # 왜 중요: 퇴화 프레임에서 예외 대신 None (호출부가 건너뛴다).
        self.assertIsNone(ha.blend_crop_matrix((7.0, 7.0), (7.0, 7.0)))


# --------------------------------------------------------------- yaw 선택

class TestPickByYaw(unittest.TestCase):
    def setUp(self):
        self.assets = fake_bank("bob", [-36, -24, -12, 0, 12, 24, 36])
        self.assets["procedural"] = fake_asset("procedural")            # yaw/bank 없음
        self.assets.update(fake_bank("long", [-20, 0, 20]))

    def test_pick_by_yaw_returns_the_nearest_bin(self):
        # 왜 중요: 뱅크 전환의 기본 계약.
        for yaw, want in [(0.0, 0.0), (5.0, 0.0), (7.0, 12.0), (-13.0, -12.0), (23.0, 24.0)]:
            self.assertEqual(ha.pick_by_yaw(self.assets, "bob", yaw).yaw, want)

    def test_pick_by_yaw_clamps_beyond_the_bank_extremes(self):
        # 왜 중요: 뱅크가 덜 찼거나 검출 한계(+-40도)를 넘는 각도에서도 항상
        # 무언가를 돌려줘야 한다. None 이면 헤어가 통째로 사라진다.
        sparse = fake_bank("half", [0, 12, 24])
        self.assertEqual(ha.pick_by_yaw(sparse, "half", 90.0).yaw, 24)
        self.assertEqual(ha.pick_by_yaw(sparse, "half", -90.0).yaw, 0)

    def test_pick_by_yaw_ignores_other_banks_and_yawless_assets(self):
        # 왜 중요: 정적 에셋(yaw=None)이 섞여 들어오면 min() 이 TypeError 로 죽는다.
        got = ha.pick_by_yaw(self.assets, "long", 100.0)
        self.assertEqual(got.bank, "long")
        self.assertEqual(got.yaw, 20)

    def test_pick_by_yaw_returns_none_for_an_unknown_bank(self):
        self.assertIsNone(ha.pick_by_yaw(self.assets, "nope", 0.0))
        self.assertIsNone(ha.pick_by_yaw({}, "bob", 0.0))

    def test_list_banks_returns_sorted_unique_bank_names(self):
        self.assertEqual(ha.list_banks(self.assets), ["bob", "long"])


class TestPickByYawStable(unittest.TestCase):
    def setUp(self):
        self.assets = fake_bank("bob", [-12, 0, 12])
        self.a_0 = self.assets["bob_+00"]
        self.a_12 = self.assets["bob_+12"]

    def test_stable_pick_matches_plain_pick_without_history(self):
        # 왜 중요: current 가 없으면 히스테리시스가 개입할 여지가 없다.
        for yaw in (-20.0, -5.0, 0.0, 5.0, 20.0):
            self.assertIs(ha.pick_by_yaw_stable(self.assets, "bob", yaw),
                          ha.pick_by_yaw(self.assets, "bob", yaw))

    def test_stable_pick_keeps_current_inside_the_hysteresis_margin(self):
        # 왜 중요: 깜빡임 방지가 이 함수의 존재 이유. 경계(yaw=6)를 막 넘은
        # yaw=7 에서 현재 칸(0도, 오차 7)은 최선(12도, 오차 5)보다
        # margin(3도) 이내로만 나쁘므로 그대로 유지해야 한다.
        got = ha.pick_by_yaw_stable(self.assets, "bob", 7.0, current=self.a_0, margin=3.0)
        self.assertIs(got, self.a_0)

    def test_stable_pick_switches_once_the_margin_is_exceeded(self):
        # 왜 중요: 히스테리시스가 영원히 붙잡고 있으면 뱅크 전환 자체가 죽는다.
        # yaw=10 -> 현재(0도) 오차 10, 최선(12도) 오차 2. 10 > 2+3 이므로 전환.
        got = ha.pick_by_yaw_stable(self.assets, "bob", 10.0, current=self.a_0, margin=3.0)
        self.assertIs(got, self.a_12)

    def test_stable_pick_does_not_flicker_while_yaw_jitters_on_a_boundary(self):
        # 왜 중요: 실제 증상 재현. 경계(6도) 위에서 +-1.4도로 떨리는 yaw 시퀀스를
        # 흘려도 전환은 많아야 한 번이어야 한다.
        seq = [6.0 + 1.4 * (1 if i % 2 else -1) for i in range(40)]
        cur = ha.pick_by_yaw_stable(self.assets, "bob", seq[0])
        switches = 0
        for yaw in seq[1:]:
            nxt = ha.pick_by_yaw_stable(self.assets, "bob", yaw, current=cur)
            if nxt is not cur:
                switches += 1
            cur = nxt
        self.assertLessEqual(switches, 1)

    def test_plain_pick_would_flicker_on_the_same_sequence(self):
        # 왜 중요: 위 테스트가 실제로 무언가를 막고 있음을 보인다(대조군).
        seq = [6.0 + 1.4 * (1 if i % 2 else -1) for i in range(40)]
        picks = [ha.pick_by_yaw(self.assets, "bob", y) for y in seq]
        switches = sum(1 for a, b in zip(picks, picks[1:]) if a is not b)
        self.assertGreater(switches, 10)

    def test_stable_pick_drops_a_current_from_another_bank(self):
        # 왜 중요: 사용자가 헤어를 바꾸면 이전 뱅크의 에셋이 current 로 남아 있다.
        # 그걸 유지하면 다른 머리가 계속 붙어 있게 된다.
        other = fake_asset("other", yaw=0, bank="long")
        got = ha.pick_by_yaw_stable(self.assets, "bob", 0.0, current=other)
        self.assertIs(got, self.a_0)

    def test_stable_pick_drops_a_yawless_current(self):
        # 왜 중요: 정적 에셋(yaw=None)이 current 로 들어오면 abs(None - yaw) 로 죽는다.
        static = fake_asset("static", yaw=None, bank="bob")
        got = ha.pick_by_yaw_stable(self.assets, "bob", 12.0, current=static)
        self.assertIs(got, self.a_12)

    def test_stable_pick_returns_none_for_an_empty_bank(self):
        self.assertIsNone(ha.pick_by_yaw_stable({}, "bob", 0.0, current=self.a_0))


class TestPickPairByYaw(unittest.TestCase):
    def setUp(self):
        self.assets = fake_bank("bob", [-24, -12, 0, 12, 24])

    def test_pick_pair_brackets_the_current_yaw(self):
        # 왜 중요: 두 칸을 섞어야 각도 변화가 연속으로 보인다. 감싸는 두 개를 골라야 한다.
        a, b, t = ha.pick_pair_by_yaw(self.assets, "bob", 6.0)
        self.assertEqual((a.yaw, b.yaw), (0, 12))
        self.assertAlmostEqual(t, 0.5, places=6)

    def test_pick_pair_t_is_one_at_the_upper_anchor(self):
        # 왜 중요: t 의 정의(결과 = a*(1-t) + b*t)를 끝점에서 고정한다.
        a, b, t = ha.pick_pair_by_yaw(self.assets, "bob", 12.0)
        self.assertEqual((a.yaw, b.yaw), (0, 12))     # [0,12] 구간이 먼저 잡힌다
        self.assertAlmostEqual(t, 1.0, places=6)

    def test_pick_pair_t_is_monotonic_inside_one_interval(self):
        # 왜 중요: t 가 단조가 아니면 고개를 한 방향으로 돌리는데 헤어가 되돌아간다.
        # 앵커에 정확히 걸친 값은 (-12,0,t=1) 과 (0,12,t=0) 두 표현이 모두 옳으므로
        # 구간 내부만 본다.
        ts = []
        for yaw in np.linspace(0.1, 11.9, 25):
            a, b, t = ha.pick_pair_by_yaw(self.assets, "bob", float(yaw))
            self.assertEqual((a.yaw, b.yaw), (0, 12))
            ts.append(t)
        self.assertTrue(all(x < y for x, y in zip(ts, ts[1:])))

    def test_pick_pair_blended_yaw_is_monotonic_over_the_whole_range(self):
        # 왜 중요: 표현(어느 구간으로 잡히는지)과 무관하게, 혼합 결과의 유효 각도는
        # yaw 를 한 방향으로 돌리는 동안 절대 되돌아가면 안 된다. 이게 진짜 계약이다.
        def blended(yaw):
            a, b, t = ha.pick_pair_by_yaw(self.assets, "bob", yaw)
            return a.yaw if b is None else a.yaw * (1 - t) + b.yaw * t
        vals = [blended(float(y)) for y in np.linspace(-40.0, 40.0, 400)]
        self.assertTrue(all(x <= y + 1e-9 for x, y in zip(vals, vals[1:])))
        self.assertAlmostEqual(vals[0], -24.0, places=6)     # 아래쪽 클램프
        self.assertAlmostEqual(vals[-1], 24.0, places=6)     # 위쪽 클램프

    def test_pick_pair_clamps_outside_the_bank_range(self):
        # 왜 중요: 뱅크가 덜 찼을 때 양 끝으로 클램프하는 것이 docstring 의 보장.
        # 바깥에서는 섞을 짝이 없으므로 b 는 None 이어야 한다.
        a, b, t = ha.pick_pair_by_yaw(self.assets, "bob", 90.0)
        self.assertEqual((a.yaw, b, t), (24, None, 0.0))
        a, b, t = ha.pick_pair_by_yaw(self.assets, "bob", -90.0)
        self.assertEqual((a.yaw, b, t), (-24, None, 0.0))

    def test_pick_pair_with_a_single_entry_returns_it_alone(self):
        one = fake_bank("solo", [5])
        a, b, t = ha.pick_pair_by_yaw(one, "solo", -100.0)
        self.assertEqual((a.yaw, b, t), (5, None, 0.0))

    def test_pick_pair_returns_a_none_triple_for_an_empty_bank(self):
        # 왜 중요: 호출부가 3-튜플 언패킹을 하므로 None 하나만 돌려주면 죽는다.
        self.assertEqual(ha.pick_pair_by_yaw({}, "bob", 0.0), (None, None, 0.0))
        self.assertEqual(ha.pick_pair_by_yaw(self.assets, "nope", 0.0), (None, None, 0.0))

    def test_pick_pair_is_continuous_across_an_anchor(self):
        # 왜 중요: 앵커를 넘는 순간 혼합 결과가 튀면 안 된다. 앵커 직전/직후의
        # 유효 각도(가중 평균 yaw)가 이어져야 한다.
        def blended(yaw):
            a, b, t = ha.pick_pair_by_yaw(self.assets, "bob", yaw)
            return a.yaw if b is None else a.yaw * (1 - t) + b.yaw * t
        self.assertAlmostEqual(blended(11.999), blended(12.001), places=2)


# ---------------------------------------------------------- AnchorSmoother

class TestAnchorSmoother(unittest.TestCase):
    L, R = (100.0, 200.0), (212.0, 200.0)

    def test_first_sample_passes_through_unchanged(self):
        # 왜 중요: 세션 첫 프레임이 원점 등에서 끌려오면 헤어가 날아 들어온다.
        s = ha.AnchorSmoother()
        l, r = s.update(self.L, self.R)
        np.testing.assert_allclose(l, self.L, atol=1e-3)
        np.testing.assert_allclose(r, self.R, atol=1e-3)

    def test_reset_makes_the_next_sample_pass_through_again(self):
        # 왜 중요: 헤어/세션을 바꿀 때 reset 을 부른다. 이전 상태가 남으면
        # 새 얼굴 위치로 서서히 기어간다.
        s = ha.AnchorSmoother()
        for _ in range(30):
            s.update(self.L, self.R)
        s.reset()
        far_l, far_r = (400.0, 90.0), (460.0, 130.0)
        l, r = s.update(far_l, far_r)
        np.testing.assert_allclose(l, far_l, atol=1e-3)
        np.testing.assert_allclose(r, far_r, atol=1e-3)

    def test_strength_zero_disables_smoothing(self):
        # 왜 중요: 슬라이더 0 = "평활화 없음" 이 계약이다. 조금이라도 남으면
        # 지연을 못 없앤다.
        s = ha.AnchorSmoother()
        s.set_strength(0.0)
        s.update(self.L, self.R)
        for l_in, r_in in [((140.0, 210.0), (250.0, 214.0)),
                           ((90.0, 190.0), (200.0, 205.0)),
                           ((300.0, 100.0), (360.0, 160.0))]:
            l, r = s.update(l_in, r_in)
            np.testing.assert_allclose(l, l_in, atol=1e-2)
            np.testing.assert_allclose(r, r_in, atol=1e-2)

    def test_stationary_input_converges_to_that_input(self):
        # 왜 중요: 가만히 있으면 정확히 그 자리에 수렴해야 한다. 편향이 남으면
        # 헤어가 얼굴에서 살짝 어긋난 채 고정된다.
        s = ha.AnchorSmoother()
        s.update((0.0, 0.0), (100.0, 0.0))     # 다른 곳에서 시작
        for _ in range(400):
            l, r = s.update(self.L, self.R)
        np.testing.assert_allclose(l, self.L, atol=1e-2)
        np.testing.assert_allclose(r, self.R, atol=1e-2)

    def test_small_jitter_is_attenuated(self):
        # 왜 중요: 미세 떨림 억제가 존재 이유. 1px 흔들림이 그대로 통과하면 안 된다.
        s = ha.AnchorSmoother()
        s.update(self.L, self.R)
        l, _ = s.update((self.L[0] + 1.0, self.L[1]), (self.R[0] + 1.0, self.R[1]))
        moved = l[0] - self.L[0]
        self.assertGreaterEqual(moved, 0.0)
        self.assertLess(moved, 0.5)            # 1px 입력의 절반 미만만 따라간다

    def test_large_motion_is_followed_more_closely_than_small_jitter(self):
        # 왜 중요: 적응형 알파의 계약. 빠른 움직임에는 필터가 풀려야 지연이 없다.
        def follow_fraction(step):
            s = ha.AnchorSmoother()
            s.update(self.L, self.R)
            l, _ = s.update((self.L[0] + step, self.L[1]), (self.R[0] + step, self.R[1]))
            return (l[0] - self.L[0]) / step
        self.assertLess(follow_fraction(1.0), follow_fraction(60.0))
        self.assertGreater(follow_fraction(60.0), 0.9)

    def test_degenerate_zero_distance_input_is_returned_unchanged(self):
        # 왜 중요: 두 눈이 한 점으로 뭉개진 프레임에서 0으로 나누면 NaN 이 상태에
        # 눌러앉아 이후 모든 프레임이 망가진다.
        s = ha.AnchorSmoother()
        l, r = s.update((50.0, 50.0), (50.0, 50.0))
        self.assertEqual((tuple(l), tuple(r)), ((50.0, 50.0), (50.0, 50.0)))
        self.assertIsNone(s.cx)                # 상태가 오염되지 않았다

    def test_output_never_becomes_nan(self):
        # 왜 중요: NaN 이 한 번 들어가면 EMA 상태가 영구히 NaN 이 된다.
        s = ha.AnchorSmoother()
        rng = np.random.default_rng(0)
        for _ in range(200):
            n = rng.normal(0, 2.0, 4)
            l, r = s.update((self.L[0] + n[0], self.L[1] + n[1]),
                            (self.R[0] + n[2], self.R[1] + n[3]))
            self.assertTrue(np.isfinite(l).all() and np.isfinite(r).all())

    def test_set_strength_is_clamped_and_monotonic(self):
        # 왜 중요: 슬라이더 값이 범위를 벗어나도 알파가 1 을 넘거나 음수가 되면 발산한다.
        s = ha.AnchorSmoother()
        for k in (-5.0, 0.0, 0.5, 1.0, 2.0, 99.0):
            s.set_strength(k)
            for a in (s.pos_a, s.scale_a, s.ang_a):
                self.assertGreater(a, 0.0)
                self.assertLessEqual(a, 1.0)
        s.set_strength(1.0)
        weak = s.pos_a
        s.set_strength(2.0)
        self.assertLess(s.pos_a, weak)         # k 가 클수록 더 강한 평활화

    def test_angle_smoothing_survives_the_180_degree_wrap(self):
        # 왜 중요: 각도를 도 단위로 평활화하면 +179 -> -179 에서 통째로 한 바퀴 돈다.
        # sin/cos 로 다루므로 그런 튐이 없어야 한다.
        s = ha.AnchorSmoother()
        c, half = np.array([200.0, 200.0]), 56.0
        l = r = None
        for deg in [179.0, -179.5, 179.5, -179.0] * 10:
            a = math.radians(deg)
            u = np.array([math.cos(a), math.sin(a)])
            l, r = s.update(tuple(c - u * half), tuple(c + u * half))
        v = np.asarray(r) - np.asarray(l)
        ang = math.degrees(math.atan2(v[1], v[0]))
        self.assertGreater(abs(ang), 170.0)    # 0도 근처로 무너지지 않았다
        np.testing.assert_allclose(np.linalg.norm(v), 2 * half, atol=1.0)


# ------------------------------------------------- clean_hair_mask / skin_mean

class TestMaskHelpers(unittest.TestCase):
    def big_mask(self):
        m = np.zeros((100, 100), np.uint8)
        m[20:60, 20:60] = 255          # 1600 px
        return m

    def test_clean_hair_mask_drops_speckles_below_the_ratio(self):
        # 왜 중요: 세그멘터가 배경/옷에 찍는 작은 오검출이 헤어로 합성되면 눈에 띈다.
        m = self.big_mask()
        m[80:83, 80:83] = 255          # 9 px = 최대 덩어리의 0.6% < 8%
        out = ha.clean_hair_mask(m, feather=0)
        self.assertEqual(int(out[81, 81]), 0)
        self.assertEqual(int(out[40, 40]), 255)

    def test_clean_hair_mask_keeps_blobs_above_the_ratio(self):
        # 왜 중요: 옆머리처럼 본체와 떨어진 큰 조각까지 지우면 헤어가 잘려 나간다.
        m = self.big_mask()
        m[70:90, 70:90] = 255          # 400 px = 25% > 8%
        out = ha.clean_hair_mask(m, feather=0)
        self.assertEqual(int(out[80, 80]), 255)

    def test_clean_hair_mask_fills_small_holes(self):
        # 왜 중요: 머리카락 사이로 배경이 뚫려 보이면 합성이 지저분해진다.
        m = self.big_mask()
        m[38:41, 38:41] = 0
        out = ha.clean_hair_mask(m, feather=0)
        self.assertEqual(int(out[39, 39]), 255)

    def test_clean_hair_mask_without_feather_is_strictly_binary(self):
        out = ha.clean_hair_mask(self.big_mask(), feather=0)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(sorted(np.unique(out).tolist()), [0, 255])

    def test_clean_hair_mask_feathers_the_boundary(self):
        # 왜 중요: 하드 엣지로 합성하면 오려붙인 티가 난다(중간값이 존재해야 한다).
        out = ha.clean_hair_mask(self.big_mask(), feather=3)
        mids = out[(out > 0) & (out < 255)]
        self.assertGreater(mids.size, 0)
        self.assertEqual(int(out[40, 40]), 255)          # 내부는 그대로 불투명

    def test_clean_hair_mask_on_an_empty_mask_stays_empty(self):
        # 왜 중요: 얼굴/머리를 못 찾은 프레임에서 예외 없이 빈 알파를 돌려줘야 한다.
        out = ha.clean_hair_mask(np.zeros((20, 20), np.uint8), feather=3)
        self.assertEqual(int(out.max()), 0)

    def test_clean_hair_mask_does_not_mutate_the_input(self):
        # 왜 중요: 같은 마스크를 조명 보정 등 다른 단계가 또 쓴다.
        m = self.big_mask()
        before = m.copy()
        ha.clean_hair_mask(m)
        np.testing.assert_array_equal(m, before)

    def test_skin_mean_averages_only_the_masked_pixels(self):
        # 왜 중요: 배경 픽셀이 섞이면 조명 보정 배율이 통째로 어긋난다.
        img = np.zeros((50, 50, 3), np.uint8)
        img[:, :, 0], img[:, :, 1], img[:, :, 2] = 10, 20, 30
        img[30:, :, :] = 250                      # 마스크 밖은 완전히 다른 색
        mask = np.zeros((50, 50), np.uint8)
        mask[:20, :20] = 1                        # 400 px
        np.testing.assert_allclose(ha.skin_mean(img, mask), [10, 20, 30], atol=1e-4)

    def test_skin_mean_returns_none_when_too_few_pixels(self):
        # 왜 중요: 픽셀이 몇 개뿐이면 평균이 잡음이라, 그걸로 색을 맞추면 헤어 색이 튄다.
        img = np.zeros((50, 50, 3), np.uint8)
        mask = np.zeros((50, 50), np.uint8)
        mask[:10, :10] = 1                        # 100 px < 200
        self.assertIsNone(ha.skin_mean(img, mask))

    def test_skin_mean_accepts_a_0_255_mask(self):
        # 왜 중요: 호출부가 0/1 과 0/255 를 섞어 넘긴다(astype(bool) 로 둘 다 받는다).
        img = np.full((50, 50, 3), 7, np.uint8)
        mask = np.zeros((50, 50), np.uint8)
        mask[:20, :20] = 255
        np.testing.assert_allclose(ha.skin_mean(img, mask), [7, 7, 7], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
