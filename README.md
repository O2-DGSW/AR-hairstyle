# heddy-v2

브라우저 웹캠 영상에 헤어스타일을 실시간으로 합성해 보여주는 로컬 프로토타입.
WebRTC 로 프레임을 받아 GPU 에서 얼굴 파싱(SegFormer) → 워핑 → 합성한 뒤 다시
내보낸다. 정지 프레임 고화질 합성에는 HairFastGAN 을 별도 경로로 쓴다.

구조·설계 판단·성능 수치는 전부 **[docs/hair-ar-pipeline.md](docs/hair-ar-pipeline.md)**
에 있다. 이 문서는 "어떻게 돌리는가"만 다룬다.

---

## 요구 환경

- Windows 11
- NVIDIA GPU + 최신 드라이버 (검증: RTX 4070 SUPER, 드라이버 591.86)
  - CUDA **툴킷**은 필요 없다. torch 휠에 런타임이 들어 있다.
    (툴킷/MSVC 가 없어서 생기는 제약은 아래 "알려진 제약" 참고)
- Python 3.12 (검증: 3.12.0)
- 디스크 여유 25GB 이상 — venv 6GB, HairFastGAN 가중치 7GB,
  그 가중치를 받는 데 쓰는 git 클론이 다시 7GB

---

## 설치

### 1. 가상환경

```bash
cd heddy-v2
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
```

### 2. 의존성

`server/requirements.txt` 안에 CUDA 휠 인덱스(`--extra-index-url .../cu124`)가
지시자로 들어 있으므로 보통은 이 한 줄로 끝난다.

```bash
python -m pip install -r server/requirements.txt
```

인덱스 지시자를 무시하는 도구를 쓴다면 직접 넘긴다:

```bash
python -m pip install -r server/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124
```

설치 후 CUDA 가 실제로 잡혔는지 확인한다. `False` 가 나오면 CPU 휠이 깔린
것이므로 torch/torchvision 을 지우고 인덱스를 지정해 다시 설치해야 한다.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 기대: 2.6.0+cu124 True
```

개발/학습용은 추가로:

```bash
python -m pip install -r server/requirements-dev.txt
```

### 3. 모델 가중치 (리포에 없다 — 따로 받아야 한다)

가중치는 바이너리라 버전관리에서 제외했다(`.gitignore` 참고). 아래를 직접 채운다.

**`server/models/face_landmarker.task`** (약 3.7MB)
MediaPipe Tasks 의 얼굴 랜드마커. `face_pose.py` 가 요구하는 경로다.
Google 배포본을 받아 그 이름 그대로 놓는다.

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

**`server/models/selfie_multiclass_256x256.tflite`** (약 16MB)
구 세그멘터(`segmenter.py`)가 쓰는 MediaPipe 셀피 멀티클래스 모델.
실시간 경로는 이미 `gpu_segmenter.py` 의 SegFormer 로 넘어갔으므로 이건
비교/폴백용이다. 새로 세팅한다면 건너뛰어도 서버는 뜬다.

```
https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
```

**SegFormer 얼굴 파싱 — 받을 필요 없다.**
`jonathandinu/face-parsing` 을 첫 실행 때 Hugging Face 허브에서 자동으로
내려받아 로컬 HF 캐시에 넣는다. 리비전은
`758b82e15a0178c9db39c1ff666a8b56e3a550c8` 로 고정돼 있다
(`server/config.py` 의 `model_revision`). 핀이 없으면 업스트림이 바뀐 날
결과가 조용히 달라지기 때문이다. 즉 **첫 실행에는 인터넷이 필요하다.**

**`server/train/blender.pt`** (약 1.9MB)
학습된 블렌딩 네트워크. 없으면 서버는 뜨지만 학습형 블렌딩이 빠진다.
`server/train/train_blender.py` 로 직접 학습해서 만든다.

**HairFastGAN pretrained_models** (약 7GB)
`external/HairFastGAN/README.md` 의 절차를 따른다. 요약하면 HF 에 올라온
가중치 리포를 git-lfs 로 클론해서 `pretrained_models/` 와 `input/` 을
`external/HairFastGAN/` 안으로 옮기는 것이다.

```bash
cd external
git clone https://huggingface.co/AIRI-Institute/HairFastGAN hf-weights
cd hf-weights && git lfs pull && cd ..
mv hf-weights/pretrained_models HairFastGAN/pretrained_models
mv hf-weights/input           HairFastGAN/input
```

`hf-weights/.git` 이 LFS 객체 때문에 7GB 를 더 차지한다. 옮기고 나면 지워도 된다.
그리고 이 PC 에서는 StyleGAN2 커스텀 CUDA 연산을 컴파일할 수 없으므로 한 번 더:

```bash
python external/patch_stylegan_ops.py
```

이유와 되돌리는 법은 `external/patch_stylegan_ops.py` 의 docstring 에 있다.

---

## 실행

```bash
cd server
python server.py --port 8080 --preload
```

- `--preload` 는 첫 연결을 기다리지 않고 시작 시점에 모델을 올린다.
  SegFormer 적재에 수 초가 걸려서, 이게 없으면 첫 접속자가 그 시간을 다 뒤집어쓴다.
- 접속: `http://localhost:8080/`

HairFastGAN(정지 프레임 고화질 합성)은 기본적으로 **별도 프로세스**로 뜬다
(`HEDDY_GAN_BACKEND=process`). 실시간 경로가 GAN 의 적재(약 90초)나 추론(약 9초)에
절대 막히지 않고, GAN 이 죽어도 서버가 같이 죽지 않게 하려는 것이다.
프로세스 기동에 실패하면 조용히 죽는 대신 같은 프로세스의 전용 스레드
(in-process)로 **자동 폴백**한다 — 느려도 도는 쪽이 낫다는 판단이다.
스레드 방식을 강제하려면 `HEDDY_GAN_BACKEND=thread` 로 띄운다.
그러면 GAN 적재/추론 동안 서버 프로세스가 함께 무거워진다.

### ⚠ HTTPS 나 localhost 가 아니면 웹캠이 안 켜진다

브라우저는 **보안 컨텍스트**(HTTPS 또는 `localhost`)에서만 카메라를 허용한다.
평문 HTTP 로 IP 주소에 접속하면 `navigator.mediaDevices` 가 **아예 존재하지
않는다.** 버튼은 눌리는데 아무 일도 안 일어나거나 예외만 나므로, 서버 문제로
착각하기 딱 좋다. 실제로 여기서 가장 많이 막힌다.

`client/preflight.js` 가 이 상황을 감지해서 페이지의 `#status` 에 직접 원인을
써 준다. 화면에 빨간 글씨가 뜨면 그걸 먼저 읽을 것.

- 같은 PC 에서 테스트: `http://localhost:8080/` 을 쓴다. `127.0.0.1` 도 된다.
- 다른 기기(휴대폰 등)에서 테스트: `http://192.168.x.x:8080/` 은 안 된다.
  Chrome 주소창에 `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
  를 열고 해당 origin 을 등록한 뒤 브라우저를 재시작하거나, 앞단에 HTTPS 를 둔다.

또 하나: 클라이언트는 three.js / MediaPipe 를 `cdn.jsdelivr.net` 에서 받는다.
CDN 이 막힌 망이면 모듈이 통째로 안 올라가 버튼에 핸들러조차 붙지 않는다.
이 경우도 `preflight.js` 가 9초 뒤에 알려준다.

---

## 테스트

GPU 도 모델도 건드리지 않는 순수 로직(기하·상태) 회귀 테스트다.
표준 라이브러리 `unittest` 만 쓰므로 추가 설치 없이 돈다.

```bash
cd server
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

## 설정

튜닝 값은 전부 `server/config.py` 한 곳에 모여 있고, 코드를 고치지 않고
**환경변수로 덮어쓴다.** 규칙은 `HEDDY_` + 대문자 필드명이다.
(이 서버는 재시작 비용이 크다 — SegFormer 수 초, HairFastGAN 약 90초.)

```bash
HEDDY_MAX_SESSIONS=3 HEDDY_GAN_BACKEND=thread python server.py
```

자주 건드리는 것 몇 개:

| 환경변수 | 기본값 | 무엇 |
|---|---|---|
| `HEDDY_MAX_SESSIONS` | `2` | 동시 접속 상한. GPU 워커가 1개라 초과분은 거절하는 편이 다 같이 느려지는 것보다 낫다 |
| `HEDDY_GAN_BACKEND` | `process` | `process`(별도 프로세스, 권장) / `thread`. GAN 이 죽어도 서버가 안 죽게 하려는 것 |
| `HEDDY_INPUT_SIZE` | `512` | 세그멘테이션 입력 한 변(px). CUDA 그래프가 이 크기로 캡처되므로 런타임에 못 바꾼다. 바꾸려면 재시작 |
| `HEDDY_USE_CUDA_GRAPH` | `true` | CUDA 그래프 재생. 끄면 세그멘테이션이 9ms → 50ms 로 느려진다. 디버깅용 |
| `HEDDY_ALLOW_FRAME_RECORDING` | `false` | 얼굴 프레임을 디스크에 남기는 학습 데이터 수집. **생체정보라 기본은 꺼져 있다** |

전체 목록과 각 값의 근거는 `server/config.py` 주석에 있다.
`bool` 은 `1/true/yes/on` 을 참으로 보고, 튜플 필드는 쉼표로 구분한다
(`HEDDY_LIVE_TARGETS=-30,0,30`).

---

## 알려진 제약

- **인증이 없다.** `/offer`, `/references`, `/captures/{name}` 전부 무인증이고
  기본 바인드가 `0.0.0.0` 이다. 같은 네트워크의 누구나 접속해서 GPU 와 디스크를
  쓸 수 있다. **개발 전용이다. 공개망에 그대로 올리지 말 것.**
  (디스크 쿼터와 세션 상한은 있지만 그건 사고 방지지 접근 제어가 아니다.)
- 이 서버는 사람 얼굴을 다룬다. `server/captures/`, `server/train/frames/`,
  `server/assets_generated/` 에 실제 얼굴 이미지가 쌓인다. 전부 `.gitignore`
  대상이다. 실수로 `git add -f` 하지 말 것.
- MSVC / CUDA 툴킷이 없으면 StyleGAN2 의 fused CUDA 연산이 순수 PyTorch
  폴백으로 돈다(1.5~3배 느림). HairFastGAN 은 프레임당이 아니라 오프라인
  경로에만 쓰이므로 감수하고 있다.
- numpy 는 1.x 에 묶여 있다. 이유는 `server/requirements.txt` 주석 참고.
