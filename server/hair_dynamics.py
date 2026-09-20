"""3D 헤어 2차 운동(secondary motion): 중력 늘어짐 + 관성 지연 + 흔들림.

왜 필요한가
-----------
그룸을 두상 행렬로 강체 렌더하면 고개를 끄덕이거나 기울여도 머리카락이 두상과 함께 '딱'
돌아간다. 실제 머리카락은 뿌리만 두피를 따르고 끝은 중력 쪽으로 처지며 한 박자 늦게
따라와 흔들리다 가라앉는다. 이게 없으면 가발을 쓴 것처럼 보인다(사용자 피드백).

어떻게
------
스트랜드 단위 물리가 아니라 **두상 전체에 대한 '지연된 자세' 하나**를 스프링-댐퍼로 굴리고,
버텍스 셰이더에서 뿌리 기준 상대좌표를 [현재 자세 → 지연 자세] 로 끝쪽 가중치(s²)만큼 섞는다
(groom_renderer 의 u_lag/u_shift). 게임의 '스프링본/다이내믹본'과 같은 원리이고 프레임당
비용은 CPU 몇 줄 + 유니폼 몇 개다. hair3d.js 의 스프링본을 서버 렌더에 옮긴 것.

- 중력: 끝의 목표 자세는 현재 두상 회전에서 pitch/roll 을 gravity 만큼 '세운' 것(yaw 는
  유지). 고개를 기울이면 뿌리는 기울고 끝은 아래로 처진다.
- 관성: 목표를 향해 회전/이동 스프링(임계감쇠 비율 damping_ratio, 1 미만이면 한 번 넘쳤다
  돌아온다). 빨리 돌리면 끝이 늦게 따라오고, 멈추면 흔들리다 선다.
- 이동 지연은 상한을 둔다(두상에서 너무 떨어지면 잘린 것처럼 보인다).

좌표: MediaPipe 행렬(카메라 공간, cm, y 위). 반환은 두상(정규) 공간의 상대 변환이라
셰이더가 뿌리 기준 상대좌표에 바로 곱한다.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def _rotvec(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].reshape(3)


def _rotmat(v: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))[0]


def _yaw_only(R: np.ndarray) -> np.ndarray:
    """R 에서 y축(수직) 회전만 남긴 회전. 두상의 '앞' 방향을 수평면에 투영해 만든다."""
    fwd = R @ np.array([0.0, 0.0, 1.0])
    fwd[1] = 0.0
    n = np.linalg.norm(fwd)
    if n < 1e-6:
        return np.eye(3)
    fwd /= n
    yaw = math.atan2(fwd[0], fwd[2])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class HairDynamics:
    def __init__(self, stiffness: float = 140.0, damping_ratio: float = 0.55,
                 gravity: float = 0.5, tip: float = 1.0, max_shift_cm: float = 2.5,
                 idle_cm: float = 0.12):
        self.k = float(stiffness)              # 스프링 상수 (1/s²)
        self.c = 2.0 * math.sqrt(self.k) * float(damping_ratio)
        self.gravity = float(gravity)          # 0=두상 그대로, 1=끝이 완전히 수직으로 처짐
        self.tip = float(tip)                  # 끝 가중치 배율 (셰이더 w = tip * s²)
        self.max_shift = float(max_shift_cm)
        self.idle = float(idle_cm)             # 미세 흔들림 진폭(cm, 끝 기준)
        self.q = None                          # 지연된 회전 (rotvec, 카메라 공간)
        self.w = np.zeros(3)                   # 각속도
        self.p = None                          # 지연된 위치
        self.v = np.zeros(3)
        self.t = 0.0

    def reset(self):
        self.q = None; self.p = None
        self.w[:] = 0; self.v[:] = 0

    def step(self, matrix: np.ndarray, dt: float) -> dict:
        """MediaPipe 4x4(row-major) + dt(초) -> 셰이더 유니폼 dict.

        lag: 3x3 두상 공간 회전(끝에 곱함), shift: 두상 공간 이동(cm, 끝에 s 배로 더함),
        tip, idle, time.
        """
        M = np.asarray(matrix, np.float64).reshape(4, 4)
        R, t = M[:3, :3], M[:3, 3]
        dt = float(min(max(dt, 1e-3), 0.1))    # 프레임 드롭 시 폭주 방지
        self.t += dt

        # 끝의 목표 자세: pitch/roll 을 gravity 비율만큼 세운다 (yaw 유지)
        R_up = _yaw_only(R)
        d = _rotvec(R_up @ R.T)                # R -> R_up 로 가는 회전
        R_target = _rotmat(d * self.gravity) @ R

        if self.q is None:
            self.q = _rotvec(R_target); self.p = t.copy()
        # 회전 스프링 (rotvec 오차 공간에서 선형화 - 각도가 작아 충분)
        err = _rotvec(R_target @ _rotmat(self.q).T)
        self.w += (self.k * err - self.c * self.w) * dt
        self.q = _rotvec(_rotmat(self.w * dt) @ _rotmat(self.q))
        # 이동 스프링
        self.v += (self.k * (t - self.p) - self.c * self.v) * dt
        self.p = self.p + self.v * dt
        shift_cam = self.p - t
        n = np.linalg.norm(shift_cam)
        if n > self.max_shift:
            shift_cam *= self.max_shift / n

        Rt = R.T
        lag = Rt @ _rotmat(self.q)             # 두상 공간: 현재 -> 지연 자세
        shift = Rt @ shift_cam
        return {"lag": lag.astype(np.float32), "shift": shift.astype(np.float32),
                "tip": self.tip, "idle": self.idle, "time": self.t}
