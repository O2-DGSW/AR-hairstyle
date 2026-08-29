"""HairFastGAN 학습 스크립트의 WandbLogger 를 대체하는 로컬 로거.

왜 wandb 를 안 쓰는가
--------------------
1. wandb 0.29 는 protobuf>=5 를 끌어오는데 mediapipe 0.10.14 는 protobuf<5 를
   요구한다. 같은 venv 에 둘 수 없다(실제로 설치했다가 mediapipe 가 깨졌다).
2. WandbLogger.start_logging() 이 os.environ['WANDB_KEY'] 를 강제한다.
   계정 없이는 학습 자체가 시작이 안 된다.

대신 run 디렉토리에 그대로 남긴다:
    runs/<name>/scalars.jsonl  스텝별 손실
    runs/<name>/*.pth          체크포인트
    runs/<name>/val_*.png      검증 이미지 그리드

WandbLogger 와 같은 인터페이스라 학습 스크립트를 그대로 쓸 수 있다.
"""
import json
import os
import shutil
import time

import numpy as np


class LocalLogger:
    def __init__(self, name="base-name", project="HairFast", root=None):
        self.name = name
        self.project = project
        self.root = root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
        self.run_dir = os.path.join(self.root, f"{project}_{name}")
        self.train_step = 0
        self._pending = {}
        self._fh = None

    def start_logging(self):
        os.makedirs(self.run_dir, exist_ok=True)
        self._fh = open(os.path.join(self.run_dir, "scalars.jsonl"), "a", encoding="utf-8")
        print(f"[locallog] run dir: {self.run_dir}")
        return self

    # --- WandbLogger 호환 API ---
    def log(self, scalar_name, scalar):
        # 이미지 리스트 등 스칼라가 아닌 것은 PNG 로 떨군다
        if isinstance(scalar, (list, tuple)):
            self._save_images(scalar_name, scalar)
            return
        try:
            self._pending[scalar_name] = float(scalar)
        except (TypeError, ValueError):
            pass

    def log_scalars(self, scalars: dict):
        for k, v in scalars.items():
            self.log(k, v)

    def next_step(self):
        self._flush()
        self.train_step += 1

    def save(self, file_path, save_online=True):
        os.makedirs(self.run_dir, exist_ok=True)
        dst = os.path.join(self.run_dir, os.path.basename(file_path))
        shutil.copy2(file_path, dst)

    def _flush(self):
        # CSV 가 아니라 JSONL 인 이유: 스텝마다 기록되는 키가 다르다. 예를 들어
        # grad 는 그래디언트 누적 경계에서만 나오는데, CSV 는 첫 행에서 헤더가
        # 고정되므로 나중에 처음 등장하는 키가 영영 누락된다.
        if not self._pending or self._fh is None:
            self._pending = {}
            return
        row = {"step": self.train_step, "time": round(time.time(), 1), **self._pending}
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        self._pending = {}

    def _save_images(self, tag, images):
        from PIL import Image
        d = os.path.join(self.run_dir, "val")
        os.makedirs(d, exist_ok=True)
        for i, im in enumerate(images[:16]):
            im = getattr(im, "image", im)          # wandb.Image 호환
            if isinstance(im, np.ndarray):
                im = Image.fromarray(im)
            if hasattr(im, "save"):
                im.save(os.path.join(d, f"step{self.train_step:07d}_{tag.replace(' ','_')}_{i}.png"))

    def close(self):
        self._flush()
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
