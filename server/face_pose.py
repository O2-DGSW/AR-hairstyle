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
# 입꼬리. FFHQ 정렬 사각형이 눈 2점 + 입꼬리 2점만으로 정해지므로
# (shape_predictor.py 의 eye_to_eye / eye_to_mouth) 이 둘이 있어야
# dlib 68점을 대체할 수 있다.
MOUTH_L, MOUTH_R = 61, 291


_IMAGE_LM = None


#: MediaPipe 눈썹 인덱스 (윗선, 아랫선), 안쪽 -> 바깥. R = 피사체 오른쪽(화면 왼쪽).
BROW_R = ((107, 66, 105, 63, 70), (55, 65, 52, 53, 46))
BROW_L = ((336, 296, 334, 293, 300), (285, 295, 282, 283, 276))
#: 얼굴 윤곽(실루엣) 고리. 앞머리에 가린 관자놀이까지 메시가 추정한다(forehead.py 앞머리 옆 끝).
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
#: 눈 윤곽 (닫힌 고리 16점). 앞머리가 눈을 찔러 덮어도 메시가 눈 자리를 추정한다 - 이마 인페인터가
#: 눈 위 머리카락을 지우고 눈을 다시 그릴 때 위치 안내로 쓴다(forehead_net, v5).
EYE_R_RING = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)
EYE_L_RING = (263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249)


#: 그룸 자동 맞춤에 쓰는 MediaPipe 랜드마크 (mp_flame_corr.npz 로 FLAME head_prior 정점과 대응)
FIT_LM = (54, 284, 234, 454, 151, 152)


def face_box(pts, w: int, h: int):
    """랜드마크 한 사람분 -> 얼굴 윤곽 외접 사각형의 (중심 x, 중심 y, 크기 = 긴 변) px.

    눈 간격과 달리 고개를 돌려도 크게 줄지 않아 여러 얼굴 중 '가장 큰(가까운)' 을 고르고, 머리를 사람별로
    나눌 때(gpu_segmenter._owner_mask) 거리 정규화에 쓴다."""
    xs = [pts[i].x * w for i in FACE_OVAL]
    ys = [pts[i].y * h for i in FACE_OVAL]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1), max(x1 - x0, y1 - y0))


def pick_wearer(boxes, track, ts_ms, hold_ms, max_jump, frame_w=None, center_w=0.0,
                switch_ratio=1.3, switch_ms=700.0, size_mem_ms=5000.0):
    """화면의 얼굴들(face_box) 중 헤어를 씌울 사람. -> (인덱스 또는 None, 새 track, 새 사람인지)

    track = (x, y, 크기, 마지막으로 본 ts_ms, 도전자가 나타난 ts_ms 또는 None, 기억 크기) 또는 None.
    랜드마커는 얼굴 순서를 보장하지 않고 num_faces=1 이면 아무나 하나를 잡는다 - 2인 프레임에서 옆 사람(사진)을
    잡아 그쪽에 헤어를 씌우고 정작 사용자 머리는 지웠다(실측).
      - 처음엔 점수(크기, 크기가 비슷하면 가운데 쪽 - center_w: 화면 끝이면 크기를 이만큼 깎는다)가 가장 높은 얼굴.
      - 그 뒤로는 직전 착용자와 가까운 얼굴을 따라간다. 놓쳐도(손으로 가림, 옆모습) hold_ms 동안은 다른 사람에게
        넘기지 않는다(None) - 넘겼다 돌아오면 헤어가 두 사람 사이를 오간다.
      - 점수가 착용자의 switch_ratio 배를 넘는 얼굴이 switch_ms 동안 계속 보이면 그 사람으로 넘긴다. 착용자가 오래
        옆을 보는 사이 옆 사람에게 넘어간 뒤, 돌아온 착용자를 되찾는 길이다(실측: 이게 없으면 끝까지 옆 사람).
        착용자 점수는 지금 크기가 아니라 최근 최대 크기(size_mem_ms 로 서서히 잊는다)로 잰다 - 고개를 젖히거나
        숙이면 얼굴 상자가 반토막 나 그 사이 옆 사람에게 빼앗겼다(실측, 서버 경로).
    """
    if track is not None and ts_ms - track[3] > hold_ms:
        track = None
    if not boxes:
        return None, track, False

    def score(b, size=None):
        s = b[2] if size is None else size
        if not frame_w:
            return s
        off = min(1.0, abs(b[0] - 0.5 * frame_w) / (0.5 * frame_w))
        return s * (1.0 - center_w * off)

    best = max(range(len(boxes)), key=lambda i: score(boxes[i]))
    chal = None
    if track is None:
        k, new = best, True
    else:
        dist = [math.hypot(b[0] - track[0], b[1] - track[1]) for b in boxes]
        k = min(range(len(boxes)), key=dist.__getitem__)
        if dist[k] > max_jump * track[2]:
            return None, track, False
        new = False
        ref = max(boxes[k][2], track[5] * math.exp(-(ts_ms - track[3]) / max(size_mem_ms, 1.0)))
        if best != k and score(boxes[best]) > switch_ratio * score(boxes[k], ref):
            chal = track[4] if track[4] is not None else ts_ms
            if ts_ms - chal >= switch_ms:
                k, new, chal = best, True, None
    b = boxes[k]
    ref = b[2] if new else ref
    return k, (b[0], b[1], b[2], ts_ms, chal, ref), new


def landmarks_image(frame_rgb: np.ndarray, near=None):
    """정지 이미지 1장의 눈/입꼬리. -> dict 또는 None.

    **FacePose 를 쓰면 안 되는 자리다.** 그건 VIDEO 모드라 프레임 간 추적을
    하는데, 서로 무관한 정지 이미지를 넣으면 직전 이미지의 기하에 끌려간
    랜드마크가 나온다. 실제로 FFHQ 정렬에 FacePose 를 재사용했더니 잘린
    이미지에서 얼굴이 기울고 중심이 어긋난 크롭이 나왔다.

    랜드마커는 한 번만 만들어 재사용한다(생성 비용이 크다). 프로세스당 하나이고
    IMAGE 모드는 상태가 없어 호출 순서에 영향받지 않는다.

    near: (x, y) px - 여러 얼굴이면 얼굴 중심이 여기에 가장 가까운 사람(실시간 착용자). 없으면 pick_wearer 의
    첫 선택과 같은 사람(크기 x 가운데) - 그냥 가장 큰 얼굴이면 첫 이마 패치를 옆 사람 얼굴로 만들 수 있다.
    """
    global _IMAGE_LM
    if _IMAGE_LM is None:
        _IMAGE_LM = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=max(1, CONFIG.max_faces),
                output_facial_transformation_matrixes=True,
            ))
    res = _IMAGE_LM.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=np.ascontiguousarray(frame_rgb)))
    if not res.face_landmarks:
        return None
    h, w = frame_rgb.shape[:2]
    boxes = [face_box(f, w, h) for f in res.face_landmarks]
    if near is not None:
        k = min(range(len(boxes)), key=lambda i: math.hypot(boxes[i][0] - near[0], boxes[i][1] - near[1]))
    else:
        k, _, _ = pick_wearer(boxes, None, 0, 0.0, 0.0, w, CONFIG.wearer_center_weight)
    pts = res.face_landmarks[k]

    def px(i):
        return np.array([pts[i].x * w, pts[i].y * h], dtype=np.float32)

    e0 = (px(EYE_L_OUT) + px(EYE_L_IN)) / 2.0
    e1 = (px(EYE_R_IN) + px(EYE_R_OUT)) / 2.0
    m0, m1 = px(MOUTH_L), px(MOUTH_R)
    matrix = None
    if res.facial_transformation_matrixes:
        # FacePose.process 와 같은 row-major 4x4 (정규 얼굴 cm -> 카메라). 맨이마 패치를 3D 두상에
        # 투영해 붙일 때 "만든 순간의 포즈" 로 쓴다(forehead.py).
        matrix = np.asarray(res.facial_transformation_matrixes[k].data, np.float32).reshape(4, 4).copy()
    # 눈썹 윗선/아랫선 (안쪽 -> 바깥, 각 5점). 앞머리에 가려진 눈썹도 메시가 추정해 준다 -
    # 맨이마를 합성할 때 눈썹을 그려 넣는 데 쓴다(forehead.py).
    brows = [(np.array([px(i) for i in up], np.float32), np.array([px(i) for i in lo], np.float32))
             for up, lo in (BROW_R, BROW_L)]
    eyes = [np.array([px(i) for i in ring], np.float32) for ring in (EYE_R_RING, EYE_L_RING)]
    return {
        "matrix": matrix,
        "brows": brows,
        "eyes": eyes,
        "oval": np.array([px(i) for i in FACE_OVAL], np.float32),
        "eye_l": e0 if e0[0] <= e1[0] else e1,
        "eye_r": e1 if e0[0] <= e1[0] else e0,
        "mouth_l": m0 if m0[0] <= m1[0] else m1,
        "mouth_r": m1 if m0[0] <= m1[0] else m0,
    }


class FacePose:
    def __init__(self):
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            # 2 이상이어야 화면에 사람이 여럿일 때 누구에게 씌울지 고를 수 있다(pick_wearer). 1인 화면에서
            # 1 -> 2 는 +1ms(검출기를 매 프레임 돈다), 2인 화면은 +5ms (실측 640x480, 별도 스레드).
            num_faces=max(1, CONFIG.max_faces),
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._k = None          # 캘리브레이션 상수 (D_measured * tz)
        self._last_ts = -1      # VIDEO 모드는 타임스탬프가 단조증가해야 한다
        self._closed = False
        self._track = None      # 착용자 (x, y, 크기, 마지막 ts) - pick_wearer
        self._face_id = 0       # 착용자가 (새로) 정해질 때마다 +1. 서버가 이마 패치를 그 사람 것으로 다시 만든다

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
        h, w = frame_rgb.shape[:2]
        boxes = [face_box(f, w, h) for f in res.face_landmarks]
        k, self._track, new = pick_wearer(boxes, self._track, ts_ms, CONFIG.wearer_hold_ms,
                                          CONFIG.wearer_max_jump, w, CONFIG.wearer_center_weight,
                                          CONFIG.wearer_switch_ratio, CONFIG.wearer_switch_ms,
                                          CONFIG.wearer_size_mem_ms)
        if k is None:
            return None
        if new:
            self._face_id += 1
            self._k = None      # 거리 캘리브레이션 상수는 두상 크기라 사람마다 다르다
        lm = res.face_landmarks[k]

        def px(i):
            return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

        e0 = (px(EYE_L_OUT) + px(EYE_L_IN)) / 2.0
        e1 = (px(EYE_R_IN) + px(EYE_R_OUT)) / 2.0
        # 항상 (화면 왼쪽, 화면 오른쪽) 순으로 맞춘다 - 좌우가 바뀌면 이 값을
        # 쓰는 닮음변환이 180° 돌아 헤어가 뒤집힌다.
        eye_l, eye_r = (e0, e1) if e0[0] <= e1[0] else (e1, e0)
        d_measured = float(np.linalg.norm(eye_r - eye_l))

        m0, m1 = px(MOUTH_L), px(MOUTH_R)
        mouth_l, mouth_r = (m0, m1) if m0[0] <= m1[0] else (m1, m0)

        yaw = pitch = roll = 0.0
        tz = None
        matrix = None
        if res.facial_transformation_matrixes:
            m = np.asarray(res.facial_transformation_matrixes[k].data,
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
            # 3D 그룸 렌더용 원본 4x4 (row-major, 정규 얼굴 cm -> 카메라 공간; 카메라는 -z 를
            # 보고 y 위, 얼굴은 z<0). groom_renderer.py 가 이걸 모델 행렬로 그대로 쓴다.
            matrix = m.reshape(4, 4).copy()

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
            "mouth_l": mouth_l, "mouth_r": mouth_r,
            "d_measured": d_measured,
            "d_corrected": d_corrected,
            "yaw": yaw, "pitch": pitch, "roll": roll,
            "tz": tz,
            "matrix": matrix,
            "calibrated": self._k is not None,
            # 3D 그룸 자동 맞춤 대응점 (관자놀이 54/284, 광대 옆 234/454, 이마 151, 턱 152) - 이미지 px
            "fit_lm": {i: px(i) for i in FIT_LM},
            # 얼굴 윤곽 (px) - 긴 머리 밑 인페인터(torso_net)가 구멍에서 얼굴을 뺀다
            "oval": np.array([px(i) for i in FACE_OVAL], np.float32),
            # 착용자 얼굴 상자와 화면의 다른 얼굴들 (x, y, 크기) px - 옆 사람 머리는 안 지운다(_owner_mask)
            "face_box": boxes[k],
            "others": [b for i, b in enumerate(boxes) if i != k],
            "face_id": self._face_id,
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
