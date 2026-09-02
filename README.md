# AR-hairstyle

실시간 헤어 시뮬레이터 서버. 웹캠 영상을 WebRTC 로 받아 GPU 에서 헤어를 합성해
되돌려준다. 합성이 전부 서버에서 끝나므로 클라이언트는 받은 영상을 그대로
재생하기만 하면 되고, 영상과 헤어가 어긋날 수 없다.

## 구조

요구사항이 정반대인 두 모델을 한 GPU 에서 돌린다.

| | 역할 | 특성 |
|---|---|---|
| SegFormer 얼굴파싱 | 프레임마다 19클래스 분할 | 16ms, **지연**이 전부 |
| HairFastGAN | 각도별 헤어 생성 | 적재 90초 / 추론 9초, **처리량**만 중요 |

같은 프로세스에 두면 GAN 이 도는 동안 실시간 경로가 GIL 과 VRAM 을 빼앗긴다.
그래서 GAN 은 별도 프로세스에서 돌린다(`gan_process.py`). 실시간 경로가 GAN
적재에 막히지 않고, GAN 이 죽어도 서버는 살아 있으며, 자식을 죽이면 VRAM 5.5GB 가
통째로 돌아온다.

## 사용자 플로우

1. 헤어스타일 참고사진을 고른다
2. 고개를 좌우로 돌린다 — 목표 각도(±36°, 12° 간격)마다 프레임을 한 장씩 잡는다
3. 잡은 각도마다 GAN 이 헤어를 만든다 (칸당 약 9초)
4. 이후 고개를 돌리면 각도에 맞는 헤어가 실시간으로 따라붙는다

각도별로 미리 굽지 않는 이유: 미리 만든 헤어는 "그때 그 사람, 그때 그 조명" 에
맞춰진 물건이라 다른 사람이 앉으면 다시 만들어야 한다. 세션에서 만들면 언제나
지금 앉은 사람 기준이 된다.

## 실행

```bash
pip install -r server/requirements.txt

# HairFastGAN 을 external/ 에 두고 CUDA 확장 폴백 패치를 적용한다
python external/patch_stylegan_ops.py

cd server
python server.py --port 8080 --preload --preload-gan
```

죽어도 자동으로 되살아나게 하려면:

```bash
python supervise.py --port 8080 --preload --preload-gan
python supervise.py --status
python supervise.py --stop
```

## 클라이언트 연동

```
POST /offer                        SDP 교환 (non-trickle ICE)
GET  /references                   헤어스타일 목록 + 썸네일 URL
GET  /references/{id}/thumbnail    원형 미리보기
DataChannel JSON                   livebank / fit / capture / mode
```

자세한 내용은 `docs/` 참고.

## 설정

모든 튜닝 값은 `server/config.py` 한 곳에 있고 `HEDDY_` 접두사 환경변수로
덮어쓴다. 재시작 비용이 크기 때문에(SegFormer 수 초, HairFastGAN 90초) 값 하나
바꾸자고 코드를 고치지 않도록 했다.

```bash
HEDDY_MAX_SESSIONS=3 HEDDY_VIDEO_MAX_BITRATE=4000000 python server.py
```

## 저장소에 없는 것

| | 이유 |
|---|---|
| `external/` | HairFastGAN 과 가중치. 15GB 이고 자체 .git 을 갖는다 |
| `data/` | FFHQ 등 학습 데이터. 100GB 규모 |
| `server/references/*.png` | 식별 가능한 인물 사진. API 로 올린다 |
| `server/assets/*.png` | 그 얼굴에서 구운 헤어 에셋 |

참고사진은 `POST /references` 로 올리면 되고, 정적 에셋은 기본으로 쓰이지 않는다
(`serve_static_assets=False` — 프로덕션 플로우는 라이브 뱅크 하나뿐이다).

## 테스트

```bash
cd server && python -m unittest discover -s tests
```
