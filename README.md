# AR-hairstyle

실시간 헤어 시뮬레이터 서버입니다. 웹캠 영상을 WebRTC 로 받아 GPU 에서 3D 헤어를 합성한 뒤
다시 돌려보냅니다. 합성이 전부 서버에서 끝나므로 클라이언트는 받은 영상을 그대로 재생하기만
하면 되고, 영상과 헤어가 서로 어긋날 일이 없습니다.

## 구조

2026-09-19 부터 기본 경로는 **3D 헤어카드(GLB)를 얼굴 포즈에 맞춰 렌더하는 방식**입니다.
예전의 2D 워핑/HairFastGAN 경로는 고개를 돌리면 헤어가 정면인 채로 남거나 각도 칸 전환이
끊겨서 껐습니다(`gan_enabled=False`, 코드는 남아 있고 UI 는 `?debug=1` 에서만 보입니다).

프레임마다 다음 순서로 처리합니다.

1. **SegFormer 얼굴 파싱** — 19클래스 분할로 원래 머리·얼굴 영역을 구합니다 (CUDA 그래프, ~9ms)
2. **MediaPipe Face Landmarker** — 얼굴 포즈 4x4 행렬과 랜드마크를 구합니다 (`face_pose.py`)
3. **원래 머리 지우기** — 시간축 배경 플레이트와 두상 맵(두피 베이스·피부톤)으로 채웁니다
4. **3D 그룸 렌더** — `groom_renderer.py` 가 moderngl 로 GLB 를 렌더합니다
   (수직 화각 63° 가상카메라, 두상 오클루더 깊이, 2패스 알파 + MSAA)
5. **맞춤·후처리** — 정면 프레임에서 두상 크기를 자동으로 맞추고(`groom_autofit`), 머리색·대비를
   사용자 머리에 맞추며, 부피 압축·빈 공간/틈 메우기·소프트닝·라이트 랩을 적용합니다
6. **맨이마** — 앞머리 자리를 학습 이마 인페인터(`forehead_net.py`, FFHQ 로 학습한 UNet)로
   채웁니다. 가중치가 없으면 주변 피부 확산(synth) 방식으로 자동 전환됩니다
7. **2차 운동** — `hair_dynamics.py` 의 스프링-댐퍼로 고개를 돌릴 때 머리끝이 살짝 따라옵니다

640x480 기준 프레임당 약 20ms 입니다(RTX 4070 SUPER). 자세한 내용은
`docs/hair-ar-pipeline.md` §7 을 참고해 주세요.

## 사용자 플로우

1. 앱(또는 웹 페이지)에서 헤어스타일을 고릅니다 — 썸네일과 라벨은 `GET /grooms` 에서 옵니다
2. 정면을 잠깐 보면 두상 크기가 자동으로 맞춰집니다
3. 이후 고개를 돌리면 3D 헤어가 그 각도 그대로 실시간으로 따라붙습니다
4. 필요하면 크기·위치 슬라이더와 머리색(자동/수동)으로 미세 조정합니다

## 실행

```bash
pip install -r server/requirements.txt

cd server
python server.py --port 8080
```

- 세그멘터는 기본으로 미리 올립니다(`--no-preload` 로 끌 수 있습니다).
- `server/tls/` 에 인증서가 있으면 HTTPS(8443)도 함께 엽니다. 브라우저 웹캠은 HTTPS 또는
  localhost 에서만 동작하므로 다른 기기에서 접속할 때는 8443 을 써 주세요.
  자체서명 인증서는 `server/tls/make_cert.sh`, Tailscale 인증서는 `server/tls/renew_ts_cert.sh`
  (90일 만료)로 만듭니다. 두 인증서는 SNI 로 나눠 씁니다.

죽어도 자동으로 다시 뜨게 하려면 다음과 같이 실행합니다.

```bash
python supervise.py --port 8080
python supervise.py --status
python supervise.py --stop
```

### 서버 여러 대 (학교 GPU 서버로 넘기기)

앱은 이 서버(Funnel 주소) 하나만 알고 있습니다. 이 서버가 `HEDDY_MAX_SESSIONS` 만큼 차면
`/offer` 를 `HEDDY_REMOTE_BACKENDS` 의 서버들에 순서대로 넘기고, 받아준 서버의 answer 를
그대로 돌려줍니다. 영상(WebRTC)은 폰과 그 서버가 직접 주고받고, 이 서버는 시그널링만 중계합니다.

주소는 코드에 넣지 않습니다. `server/.env.example` 을 `server/.env` 로 복사해 채워 주세요
(커밋되지 않습니다). 서버가 뜰 때마다 읽으므로 재부팅 후에도 `python supervise.py` 만 하면 됩니다.
실제 환경변수가 있으면 파일보다 우선하며, 값을 바꾼 뒤에는 server.py 만 재시작하면 됩니다.

```bash
HEDDY_MAX_SESSIONS=3
HEDDY_REMOTE_BACKENDS=http://<원격 서버 Tailscale IP>:8080
```

- 주소는 MagicDNS 이름 대신 Tailscale IP 를 써 주세요. Windows 에서 이름 조회가
  ~2초 걸려 연결이 그만큼 늦어집니다(실측 2.1s → 0.1s).
- 학교 서버(주피터 도커, root 없음)는 `~/heddy/run.sh [GPU] [포트] [상한]` 으로 돌아갑니다.
  컨테이너에 GL 이 없어 `~/nvgl` 에 드라이버에서 꺼낸 EGL 을 씁니다(groom_renderer 는 리눅스에서
  EGL 백엔드를 씁니다). 컨테이너가 재시작되면 주피터 터미널에서 `~/ts/start.sh` 하나로
  Tailscale·SSH(dropbear:2222)·heddy 가 모두 뜹니다.
- 원격으로 넘긴 세션 수는 `/metrics` 의 `heddy_sessions_forwarded_total` 에서 볼 수 있습니다.

## 클라이언트 연동

```
POST /offer                  SDP 교환 (non-trickle ICE)
GET  /grooms                 헤어스타일 목록 (name/label/gender/type/tags/thumb/views/url)
GET  /grooms/{name}          GLB · 썸네일 파일
GET  /healthz, /readyz       헬스 체크
GET  /metrics                Prometheus 지표
DataChannel JSON             fit (groom, groom_color, 크기/위치) / mode / record / ping
```

서버는 DataChannel 로 `{"type": "groom", "status": "ok" | "error"}` 를 돌려줍니다.

## 헤어스타일 추가

`server/grooms/<이름>.glb` 와 `<이름>.json` 을 넣으면 목록에 바로 뜹니다(`<이름>.png`,
`<이름>_views.png` 가 있으면 썸네일로 쓰입니다). json 에는 `label`, `gender`, `type`, `tags`,
`user_fit` 을 적습니다.

GLB 는 오프라인에서 사진 한 장으로 만듭니다(WSL2, 장당 약 25분).

```
사진 → Im2Haircut 재구성 → 스트랜드 PLY → strands_to_cards.py → GLB(헤어카드 + head_occluder)
```

## 설정

모든 튜닝 값은 `server/config.py` 한 곳에 있고, `HEDDY_` 접두사 환경변수 또는 `server/.env` 로
덮어씁니다(키는 필드명을 대문자로 바꾼 것입니다). 우선순위는 명령행 > 환경변수 > `.env` > 기본값입니다.

```bash
HEDDY_MAX_SESSIONS=3 HEDDY_VIDEO_MAX_BITRATE=4000000 python server.py
```

## 저장소에 없는 것

| | 이유 |
|---|---|
| `server/grooms/*.glb` | 헤어 메시입니다. 용량이 커서 따로 공유합니다 |
| `server/models/*.pt` | 이마 인페인터 가중치(`forehead_net.pt`, 42MB) 등입니다 |
| `server/models/face_landmarker.task` | MediaPipe 모델입니다. 따로 받아 주세요 |
| `data/` | FFHQ 등 학습 데이터입니다. 100GB 규모입니다 |
| `external/` | (구) HairFastGAN 과 가중치입니다. 기본 경로에서는 쓰지 않습니다 |

SegFormer(`jonathandinu/face-parsing`)는 첫 실행 때 HF 허브에서 자동으로 내려받습니다.

## 테스트

```bash
cd server && python -m unittest discover -s tests
```

## 합성 품질 검증 (서버 없이)

```bash
cd server
python train/render_groom_offline.py train/frames/<세션> --groom <이름> --render 0,60,110
python train/render_groom_offline.py train/frames/<세션> --groom <이름> --debug   # 빨강=남은 원래 머리, 파랑=빈자리
```

녹화된 원본 프레임으로 실시간 합성 경로를 결정적으로 재현해 전/후 이미지를 저장합니다.
이마 인페인터 비교는 `train/compare_forehead.py` 를 사용해 주세요.

### 이마 인페인터 학습

```bash
python train/prep_forehead_data.py     # FFHQ → 맨이마 / 앞머리 데이터 (data/forehead)
python train/train_forehead.py         # L1 + LPIPS + PatchGAN, 30k step 약 3.5시간
```

서버와 GPU 를 같이 쓸 때는 VRAM 이 넘치지 않도록 bf16 과 배치 10 을 써 주세요.
