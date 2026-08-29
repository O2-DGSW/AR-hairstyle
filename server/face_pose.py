"""얼굴 3D 포즈 + 거리 기반 스케일 정규화.

왜 필요한가
-----------
눈 간격(interocular)으로만 헤어 크기를 정하면 버그가 있다. 눈 간격은 2D
투영값이라 **고개를 옆으로 돌리면 cos(yaw)만큼 줄어든다.** 그러면 헤어도
같이 작아진다 - 머리 크기는 그대로인데.

MediaPipe Face Landmarker는 정규 3D 얼굴 모델로 PnP를 풀어서 4x4 변환행렬을
준다. 여기서 얻는 tz(카메라로부터의 거리)는 **머리를 돌려도 변하지 않으므로**
스케일 기준으로 쓰기에 적합하다.

자동 캘리브레이션
-----------------
tz만으로는 픽셀 크기를 못 정한다(초점거리와 개인별 두상 크기를 모름). 대신
정면일 때 관측된 눈 간격으로 상수 K를 역산해서 계속 갱신한다:

    정면(|yaw| 작음)일 때:  K <- EMA(D_measured * tz)
    항상:                    D_corrected = K / tz

이러면 세 가지가 한꺼번에 해결된다:
  - 거리 변화       -> tz 가 반영
  - yaw 단축        -> tz 는 회전 불변이므로 영향 없음
  - 개인별 두상 크기 -> K 가 사람마다 알아서 맞춰짐
"""
import math
import os

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from config import CONFIG

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")

# MediaPipe face mesh 눈 코너 인덱스 (눈 중심 = 양 코너의 중점)
EYE_L_OUT, EYE_L_IN = 33, 133
EYE_R_IN, EYE_R_OUT = 362, 263


class FacePose:
    def __init__(self):
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._k = None          # 캘리브레이션 상수 (D_measured * tz)
        self._last_ts = -1      # VIDEO 모드는 타임스탬프가 단조증가해야 한다
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @staticmethod
    def _euler(m):
        """4x4 flat -> (yaw, pitch, roll) 도 단위.

        알려진 문제 (지금 고치지 않는다)
        --------------------------------
        아래는 column-major 를 가정하고 m[0], m[1], m[2], m[6], m[10] 을 읽는데,
        MediaPipe 가 주는 행렬은 실제로는 **row-major** 다(tz 주석 참고).
        그래서 회전행렬의 **전치**를 읽고 있고, 전치는 곧 역회전이라
        **yaw/pitch/roll 의 부호가 뒤집혀 있다**(크기는 맞다).

        그런데 이 부호는 지금 **일관되게** 뒤집혀 있다. 뱅크를 구울 때도
        (make_asset_bank), 런타임에 칸을 고를 때도(pick_by_yaw_stable) 같은
        함수를 쓰므로 같은 실제 포즈에 같은 에셋이 붙는다. 즉 기능상 문제가
        없다.

        여기서 부호만 고치면 런타임 yaw 가 뒤집혀서 **이미 구워진 뱅크의 라벨과
        어긋난다** - 왼쪽으로 돌렸는데 오른쪽 칸이 붙는다. 고치려면 뱅크 JSON 의
        yaw/measuredYaw 와 파일명을 함께 뒤집어야 하므로, 뱅크를 다시 구울 때
        같이 처리한다.
        """
        r00, r10, r20 = m[0], m[1], m[2]
        r21, r22 = m[6], m[10]
        deg = 180.0 / math.pi
        return (
            math.atan2(-r20, math.hypot(r21, r22)) * deg,   # yaw
            math.atan2(r21, r22) * deg,                     # pitch
            math.atan2(r10, r00) * deg,                     # roll
        )

    def process(self, frame_rgb: np.ndarray, ts_ms: int = None):
        """RGB 프레임 -> dict 또는 None(얼굴 없음).

        ts_ms 를 생략하거나 이전보다 작은 값을 주면 자동으로 증가시킨다.
        VIDEO 모드는 타임스탬프 단조증가를 강제하는데, 정지 이미지 용도(에셋
        추출, GAN 입력 준비 등)에서는 보통 0을 넘기게 되고 그러면 두 번째
        호출부터 예외가 난다.
        """
        if self._closed:
            # 닫힌 네이티브 그래프에 프레임을 밀어 넣으면 mediapipe 가 조용히
            # 이상한 값을 주거나 프로세스째 죽는다. 여기서 명시적으로 끊는다.
            raise RuntimeError("FacePose 가 이미 close() 되었습니다")
        if ts_ms is None or ts_ms <= self._last_ts:
            ts_ms = self._last_ts + 33
        self._last_ts = ts_ms

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = self._lm.detect_for_video(image, ts_ms)
        if not res.face_landmarks:
            return None

        h, w = frame_rgb.shape[:2]
        lm = res.face_landmarks[0]

        def px(i):
            return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

        e0 = (px(EYE_L_OUT) + px(EYE_L_IN)) / 2.0
        e1 = (px(EYE_R_IN) + px(EYE_R_OUT)) / 2.0
        # 항상 (화면 왼쪽, 화면 오른쪽) 순으로 맞춘다 - 좌우가 바뀌면 이 값을
        # 쓰는 닮음변환이 180° 돌아 헤어가 뒤집힌다.
        eye_l, eye_r = (e0, e1) if e0[0] <= e1[0] else (e1, e0)
        d_measured = float(np.linalg.norm(eye_r - eye_l))

        yaw = pitch = roll = 0.0
        tz = None
        if res.facial_transformation_matrixes:
            m = np.asarray(res.facial_transformation_matrixes[0].data,
                           dtype=np.float32).reshape(-1)
            yaw, pitch, roll = self._euler(m)
            # MediaPipe 의 4x4 는 **row-major** 다. 평행이동이 마지막 '열'
            # (인덱스 3, 7, 11)에 들어 있다. 실측으로 확인한 실제 행렬:
            #     마지막 행 [0, 0, 0, 1]      <- 여기가 m[12..15]
            #     마지막 열 [4.13, -2.08, -51.76, 1]
            #
            # 예전에는 column-major 로 보고 m[14] 를 읽었는데, 그 자리는 바닥 행의
            # 0 이라 **tz 가 항상 0.0** 이었다. 그러면 아래 캘리브레이션 조건
            # (tz > 1e-3)이 영원히 거짓이라 K 가 안 잡히고 d_corrected 가
            # d_measured 를 그대로 통과했다. 즉 §4.7 의 거리 보정이 통째로
            # 죽은 채로 돌고 있었다.
            #
            # 그 결과가 "고개를 완전히 돌리면 헤어가 작아진다" 였다. 눈 간격은
            # 투영값이라 yaw 로 짧아지는데(실측: 정면 50.9px -> +60도에서 22.8px,
            # 44.8%) 되돌릴 방법이 없으니 헤어가 그대로 반토막 났다.
            # m[11] 로 고치면 전 구간 92.8~100.4% 로 유지된다.
            tz = abs(float(m[11]))

        # --- 거리 기반 스케일 정규화 ---
        d_corrected = d_measured
        if tz and tz > 1e-3 and d_measured > 1e-3:
            if abs(yaw) <= CONFIG.frontal_yaw_deg:
                k_obs = d_measured * tz
                a = CONFIG.cal_alpha
                self._k = k_obs if self._k is None else self._k * (1 - a) + k_obs * a
            if self._k is not None:
                d_corrected = self._k / tz

        return {
            "eye_l": eye_l, "eye_r": eye_r,
            "d_measured": d_measured,
            "d_corrected": d_corrected,
            "yaw": yaw, "pitch": pitch, "roll": roll,
            "tz": tz,
            "calibrated": self._k is not None,
        }

    def close(self):
        """네이티브 MediaPipe 그래프를 해제한다. 여러 번 불러도 안전하다.

        GC 에 맡기면 안 되는 자원이다 - 파이썬 객체가 사라져도 그래프
        스레드와 텐서 버퍼는 남는다. 세션마다 하나씩 만드는 구조라
        정리를 빠뜨리면 몇 시간 돌린 서버에서 메모리가 계속 오른다.
        서버는 정상 종료/에러 경로 양쪽에서 부르므로 idempotent 여야 한다.
        """
        if self._closed:
            return
        self._closed = True
        self._lm.close()


def eyes_scaled(pose, base_gain=1.0):
    """포즈 결과에서 '보정된 눈 2점'을 만든다.

    눈 중점과 눈 축 방향은 관측값 그대로 쓰고, 두 점 사이 거리만 거리 기반
    보정값으로 바꾼다. 이러면 위치/회전은 실제 얼굴을 따라가면서 크기만
    yaw에 흔들리지 않는다.
    """
    c = (pose["eye_l"] + pose["eye_r"]) / 2.0
    v = pose["eye_r"] - pose["eye_l"]
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return pose["eye_l"], pose["eye_r"]
    half = (pose["d_corrected"] * base_gain) / 2.0
    u = v / n
    return c - u * half, c + u * half
