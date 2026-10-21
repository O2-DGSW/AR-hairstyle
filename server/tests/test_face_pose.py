"""face_pose 의 GPU/모델 없이 검증 가능한 부분.

**FacePose() 인스턴스화는 하지 않는다** - MediaPipe 모델 파일을 열고 네이티브
그래프를 띄운다. _euler 는 staticmethod 라 인스턴스 없이 부를 수 있고,
eyes_scaled 는 dict 만 받는 순수 함수다.
"""
import math
import os
import sys
import unittest

# `python -m unittest discover -s tests` 는 tests/ 를 패키지로 인식하지 않는다.
# 자기 위치 기준으로 server/ 를 sys.path 에 직접 넣는다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from face_pose import FacePose, eyes_scaled


# --- 회전행렬 헬퍼 -----------------------------------------------------------
# _euler 는 m[0], m[1], m[2], m[6], m[10] 을 읽는다. 이는 4x4 를 **column-major**
# 로 편 배열에서 각각 R[0,0], R[1,0], R[2,0], R[2,1], R[2,2] 다(MediaPipe 의
# facial_transformation_matrixes[i].data 규약). 아래 flatten 이 그 규약이다.

def Rx(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def Ry(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def Rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def flat_col_major(R=None, t=(0.0, 0.0, 0.0)):
    """3x3 회전 + 평행이동 -> mediapipe 가 주는 column-major 4x4 flat 배열."""
    M = np.eye(4, dtype=np.float32)
    if R is not None:
        M[:3, :3] = R
    M[:3, 3] = t
    return M.T.reshape(-1)          # 전치 후 row-major 로 펴면 column-major


class TestEuler(unittest.TestCase):
    def test_identity_matrix_gives_zero_angles(self):
        # 왜 중요: 정면 얼굴에서 각도가 0 이 아니면 거리 캘리브레이션
        # (|yaw| <= frontal_yaw_deg 일 때만 갱신)이 아예 안 돌거나 늘 돈다.
        yaw, pitch, roll = FacePose._euler(flat_col_major())
        for v in (yaw, pitch, roll):
            self.assertAlmostEqual(v, 0.0, places=6)

    def test_pure_yaw_rotation_is_reported_as_yaw(self):
        # 왜 중요: yaw 는 뱅크 선택과 캘리브레이션 게이트에 직접 쓰인다.
        # 부호 규약: 화면 좌표계의 Y축 회전 각도가 그대로 나온다(Ry(+30) -> +30).
        for deg in (-40.0, -30.0, -12.0, 12.0, 30.0, 40.0):
            yaw, pitch, roll = FacePose._euler(flat_col_major(Ry(deg)))
            self.assertAlmostEqual(yaw, deg, places=4)
            self.assertAlmostEqual(pitch, 0.0, places=4)
            self.assertAlmostEqual(roll, 0.0, places=4)

    def test_pure_pitch_rotation_is_reported_as_pitch(self):
        # 왜 중요: pitch/roll 이 yaw 로 새면 고개를 끄덕일 때 뱅크가 바뀐다.
        for deg in (-25.0, 20.0):
            yaw, pitch, roll = FacePose._euler(flat_col_major(Rx(deg)))
            self.assertAlmostEqual(pitch, deg, places=4)
            self.assertAlmostEqual(yaw, 0.0, places=4)
            self.assertAlmostEqual(roll, 0.0, places=4)

    def test_pure_roll_rotation_is_reported_as_roll(self):
        for deg in (-15.0, 15.0):
            yaw, pitch, roll = FacePose._euler(flat_col_major(Rz(deg)))
            self.assertAlmostEqual(roll, deg, places=4)
            self.assertAlmostEqual(yaw, 0.0, places=4)
            self.assertAlmostEqual(pitch, 0.0, places=4)

    def test_combined_rotation_recovers_all_three_angles(self):
        # 왜 중요: 분해 순서 규약을 고정한다. 이 구현은 R = Rz(roll) Ry(yaw) Rx(pitch)
        # (intrinsic Z-Y-X) 의 역분해다. 순서가 바뀌면 세 각도가 서로 섞여 나온다.
        for roll, yaw, pitch in [(15.0, 30.0, 20.0), (-10.0, -25.0, 8.0), (5.0, 0.0, -30.0)]:
            R = Rz(roll) @ Ry(yaw) @ Rx(pitch)
            got_yaw, got_pitch, got_roll = FacePose._euler(flat_col_major(R))
            self.assertAlmostEqual(got_yaw, yaw, places=4)
            self.assertAlmostEqual(got_pitch, pitch, places=4)
            self.assertAlmostEqual(got_roll, roll, places=4)

    def test_yaw_is_monotonic_in_the_head_turn(self):
        # 왜 중요: pick_by_yaw 계열이 yaw 의 단조성을 전제한다. 어딘가에서
        # 되감기면 고개를 계속 돌리는데 뱅크가 되돌아간다.
        yaws = [FacePose._euler(flat_col_major(Ry(d)))[0] for d in np.linspace(-80, 80, 161)]
        self.assertTrue(all(a < b for a, b in zip(yaws, yaws[1:])))

    def test_translation_does_not_affect_the_angles(self):
        # 왜 중요: 각도는 회전만의 함수여야 한다. 이 테스트는 동시에 인덱싱 규약이
        # column-major 임을 못박는다 - row-major 로 읽으면 평행이동이 m[3],m[7],m[11]
        # 이 아니라 회전 자리에 들어와 각도가 흔들린다.
        R = Rz(7.0) @ Ry(23.0) @ Rx(-11.0)
        base = FacePose._euler(flat_col_major(R))
        moved = FacePose._euler(flat_col_major(R, t=(120.0, -45.0, -600.0)))
        np.testing.assert_allclose(base, moved, atol=1e-4)

    def test_tz_lives_at_index_14_of_the_column_major_flat(self):
        # 왜 중요: process() 가 tz = abs(m[14]) 로 거리를 읽는다. 이 인덱스가
        # 틀리면 스케일 정규화가 통째로 엉뚱한 값을 쓴다. 규약을 고정한다.
        m = flat_col_major(Ry(30.0), t=(1.0, 2.0, -55.0))
        self.assertAlmostEqual(float(m[12]), 1.0, places=5)
        self.assertAlmostEqual(float(m[13]), 2.0, places=5)
        self.assertAlmostEqual(abs(float(m[14])), 55.0, places=5)

    def test_gimbal_lock_at_90_degrees_does_not_raise(self):
        # 왜 중요: 옆모습(|yaw|=90)에서 hypot(r21,r22)=0 이 되어 atan2(0,0) 이 불린다.
        # 예외 없이 유한한 값을 돌려줘야 프레임 루프가 안 죽는다.
        for deg in (90.0, -90.0):
            angles = FacePose._euler(flat_col_major(Ry(deg)))
            self.assertTrue(all(math.isfinite(v) for v in angles))
            self.assertAlmostEqual(abs(angles[0]), 90.0, places=3)

    def test_euler_accepts_a_plain_python_list(self):
        # 왜 중요: mediapipe 의 .data 는 파이썬 리스트로 올 수도 있다.
        # 구현이 인덱싱만 하므로 리스트여도 같은 값이 나와야 한다.
        m = flat_col_major(Ry(17.0))
        self.assertAlmostEqual(FacePose._euler(list(map(float, m)))[0], 17.0, places=4)


class TestEyesScaled(unittest.TestCase):
    @staticmethod
    def pose(eye_l, eye_r, d_corrected):
        return {"eye_l": np.asarray(eye_l, np.float32),
                "eye_r": np.asarray(eye_r, np.float32),
                "d_corrected": float(d_corrected)}

    def test_eyes_scaled_preserves_the_midpoint(self):
        # 왜 중요: 계약의 절반. 위치는 관측값을 그대로 따라가야 헤어가 얼굴에 붙어 있다.
        p = self.pose((100.0, 200.0), (180.0, 230.0), 200.0)
        l, r = eyes_scaled(p)
        want = (p["eye_l"] + p["eye_r"]) / 2.0
        np.testing.assert_allclose((np.asarray(l) + np.asarray(r)) / 2.0, want, atol=1e-4)

    def test_eyes_scaled_preserves_the_eye_axis_direction(self):
        # 왜 중요: 계약의 나머지 절반. 방향이 바뀌면 헤어가 회전한다.
        p = self.pose((100.0, 200.0), (180.0, 230.0), 200.0)
        l, r = eyes_scaled(p)
        u_in = p["eye_r"] - p["eye_l"]
        u_out = np.asarray(r) - np.asarray(l)
        u_in = u_in / np.linalg.norm(u_in)
        u_out = u_out / np.linalg.norm(u_out)
        np.testing.assert_allclose(u_out, u_in, atol=1e-5)

    def test_eyes_scaled_sets_the_distance_to_d_corrected(self):
        # 왜 중요: 이 함수가 존재하는 이유. 거리만 보정값으로 갈아끼운다.
        p = self.pose((100.0, 200.0), (180.0, 230.0), 200.0)
        l, r = eyes_scaled(p)
        self.assertAlmostEqual(float(np.linalg.norm(np.asarray(r) - np.asarray(l))),
                               200.0, places=3)

    def test_eyes_scaled_applies_base_gain_multiplicatively(self):
        # 왜 중요: 사용자 크기 슬라이더가 base_gain 으로 들어간다.
        p = self.pose((100.0, 200.0), (180.0, 230.0), 120.0)
        for gain in (0.5, 1.0, 1.75):
            l, r = eyes_scaled(p, base_gain=gain)
            d = float(np.linalg.norm(np.asarray(r) - np.asarray(l)))
            self.assertAlmostEqual(d, 120.0 * gain, places=3)

    def test_eyes_scaled_is_independent_of_the_measured_distance(self):
        # 왜 중요: yaw 단축 보정 그 자체. 고개를 돌려 관측 눈 간격(d_measured)이
        # 줄어도, 중점/방향이 같고 d_corrected 가 같으면 결과가 **완전히 같아야**
        # 한다. 아니면 고개를 돌릴 때마다 헤어가 작아진다.
        c = np.array([300.0, 220.0], dtype=np.float32)
        u = np.array([math.cos(math.radians(20.0)), math.sin(math.radians(20.0))],
                     dtype=np.float32)
        outs = []
        for d_measured in (120.0, 96.0, 60.0):      # cos(yaw) 로 단축된 관측값들
            half = d_measured / 2.0
            outs.append(eyes_scaled(self.pose(c - u * half, c + u * half, 130.0)))
        for l, r in outs[1:]:
            np.testing.assert_allclose(l, outs[0][0], atol=1e-3)
            np.testing.assert_allclose(r, outs[0][1], atol=1e-3)

    def test_eyes_scaled_keeps_screen_left_on_the_left(self):
        # 왜 중요: 출력 순서가 뒤집히면 뒤이은 닮음변환이 180도 돈다(과거 사고).
        p = self.pose((100.0, 200.0), (180.0, 230.0), 200.0)
        l, r = eyes_scaled(p)
        self.assertLess(float(l[0]), float(r[0]))

    def test_eyes_scaled_returns_the_input_for_degenerate_eyes(self):
        # 왜 중요: 두 눈이 한 점이면 방향을 못 정한다. 0으로 나눠 NaN 을 뱉는 대신
        # 관측값을 그대로 돌려주는 것이 계약이다.
        p = self.pose((50.0, 50.0), (50.0, 50.0), 130.0)
        l, r = eyes_scaled(p)
        np.testing.assert_allclose(l, [50.0, 50.0])
        np.testing.assert_allclose(r, [50.0, 50.0])

    def test_eyes_scaled_handles_a_zero_corrected_distance(self):
        # 왜 중요: 캘리브레이션 직전 d_corrected 가 0 에 가까울 수 있다.
        # NaN 없이 두 점이 중점으로 모이기만 해야 한다.
        p = self.pose((100.0, 200.0), (180.0, 200.0), 0.0)
        l, r = eyes_scaled(p)
        self.assertTrue(np.isfinite(l).all() and np.isfinite(r).all())
        np.testing.assert_allclose(l, r, atol=1e-6)

    def test_eyes_scaled_does_not_mutate_the_pose_dict(self):
        # 왜 중요: 같은 pose dict 를 로깅/메트릭/GAN 경로가 다시 읽는다.
        p = self.pose((100.0, 200.0), (180.0, 230.0), 200.0)
        before = (p["eye_l"].copy(), p["eye_r"].copy(), p["d_corrected"])
        eyes_scaled(p, base_gain=1.4)
        np.testing.assert_array_equal(p["eye_l"], before[0])
        np.testing.assert_array_equal(p["eye_r"], before[1])
        self.assertEqual(p["d_corrected"], before[2])


class TestModuleConstants(unittest.TestCase):
    def test_eye_landmark_indices_are_the_mediapipe_corners(self):
        # 왜 중요: 이 인덱스가 바뀌면 눈 중심이 엉뚱한 곳으로 가고, 그걸 기준으로
        # 만든 모든 닮음변환이 어긋난다. 조용히 바뀌는 걸 막는 잠금장치.
        import face_pose as fp
        self.assertEqual((fp.EYE_L_OUT, fp.EYE_L_IN), (33, 133))
        self.assertEqual((fp.EYE_R_IN, fp.EYE_R_OUT), (362, 263))


if __name__ == "__main__":
    unittest.main()
