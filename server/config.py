"""중앙 설정.

왜 필요한가
-----------
튜닝 상수가 각 모듈의 모듈 전역에 흩어져 있으면 값을 하나 바꿀 때마다 코드를
고치고 프로세스를 재시작해야 한다. 이 서버는 재시작 비용이 크다(SegFormer 수 초,
HairFastGAN ~90초). 그래서 모든 튜닝 값을 한 곳에 모으고 환경변수로 덮어쓸 수
있게 한다.

사용법
------
    from config import CONFIG
    CONFIG.input_size

환경변수는 `HEDDY_` 접두사 + 대문자 필드명이다:

    HEDDY_MAX_SESSIONS=3  HEDDY_GAN_BACKEND=thread  python server.py

값 파싱은 필드의 타입 힌트를 따른다(bool 은 1/true/yes/on 을 참으로 본다).
tuple[float, ...] 필드는 쉼표로 구분한다: `HEDDY_LIVE_TARGETS=-30,0,30`

규칙
----
- 새 튜닝 상수는 여기에 넣는다. 모듈 전역 상수로 되돌리지 않는다.
- frozen 이다. 런타임에 바꾸지 않는다(스레드 사이에서 값이 갈리면 CUDA 그래프
  캡처 크기 같은 것이 어긋난다).
"""
from __future__ import annotations

import os
import typing
from dataclasses import dataclass, fields

ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Config:
    # ---------- 전송 / 세션 ----------
    host: str = "0.0.0.0"
    port: int = 8080
    #: HTTPS. 브라우저는 localhost 가 아닌 평문 HTTP 에서 getUserMedia(웹캠) 를 아예
    #: 안 준다 - 공인 IP 로 접속하면 [시작]이 "getUserMedia undefined" 로 죽는다.
    #: 인증서/키 파일이 둘 다 있으면 HTTP(port) 옆에 HTTPS(tls_port) 도 같이 연다.
    #: 자체서명이면 브라우저 경고를 한 번 넘기면 된다. 만들기: server/tls/make_cert.sh
    tls_cert: str = os.path.join(ROOT, "tls", "server.crt")
    tls_key: str = os.path.join(ROOT, "tls", "server.key")
    #: Tailscale 정식 인증서(Let's Encrypt). iOS 네이티브 앱(URLSession/WKWebView)은
    #: 자체서명을 "이동" 버튼 없이 거부하므로 앱은 이 도메인으로 붙는다.
    #: 있으면 SNI 로 갈라 쓴다: *.ts.net 으로 들어오면 이 인증서, IP 로 들어오면
    #: (SNI 없음) 위 자체서명. 받기/갱신: server/tls/renew_ts_cert.sh (90일 만료)
    tls_ts_cert: str = os.path.join(ROOT, "tls", "ts.crt")
    tls_ts_key: str = os.path.join(ROOT, "tls", "ts.key")
    #: 0 이면 HTTPS 를 안 연다(같은 PC 의 세션별 자식 서버 - 부모만 바깥을 받는다).
    tls_port: int = 8443
    #: 다른 출처에서 이 서버를 부를 수 있는 출처(포트 무시). 웹뷰 앱(Capacitor) 의 페이지 출처는
    #: capacitor://localhost 라 시그널링/참고사진 요청이 전부 교차 출처다 - 없으면 OPTIONS 사전
    #: 요청이 405 로 막힌다. 서버에 인증이 없으므로 "*" 로 열지 않는다.
    cors_origins: tuple[str, ...] = ("capacitor://localhost", "ionic://localhost",
                                     "http://localhost", "https://localhost")
    #: 동시에 받을 수 있는 피어 수. GPU 워커가 1개라 초과분은 거절하는 편이
    #: 전부 같이 느려지는 것보다 낫다.
    max_sessions: int = 2
    #: 이 서버가 꽉 찼을 때 /offer 를 넘겨 볼 다른 heddy 서버들(순서대로 시도). 쉼표 구분.
    #: 예: HEDDY_REMOTE_BACKENDS=http://<원격 서버 주소>:8080  (server/.env 에 둔다 - .env.example)
    #: 앱은 이 서버(Funnel 주소) 하나만 알고, 넘겨받은 서버의 answer 가 그대로 돌아가므로
    #: 영상(WebRTC)은 폰과 그 서버가 직접 주고받는다. 이 서버는 시그널링만 중계한다.
    #: 넘겨받은 쪽은 다시 넘기지 않는다(X-Heddy-Forwarded 헤더) - 서로를 적어도 루프가 안 생긴다.
    remote_backends: tuple[str, ...] = ()
    #: 켜면 꽉 찼을 때가 아니라 **매 세션마다** remote_backends 를 먼저 시도하고, 전부 닿지
    #: 않거나 꽉 찼을 때만 이 서버가 받는다. 학교 GPU 서버를 주력으로 쓰고 이 PC 는 예비로 둘 때.
    #: HEDDY_PREFER_REMOTE=1
    prefer_remote: bool = False
    #: 원격 /offer 한 번의 제한 시간(초). ICE 수집(STUN)까지 기다리므로 너무 짧으면 안 된다.
    remote_offer_timeout_s: float = 15.0
    #: answer 를 보낸 뒤 이 시간(초) 안에 연결(connected)되지 않은 세션은 닫아 자리를 비운다. 0 이면 끔.
    #: iOS 앱이 answer 를 받고도 ICE 체크를 안 보내고 버린 세션이 aioice 가 포기할 때까지(~30초)
    #: 상한 슬롯을 잡아, 11~21초 뒤 앱의 재시도가 원격으로 넘쳐 또 실패하거나 503 이 났다(2026-10-21).
    connect_deadline_s: float = 10.0
    #: WebRTC ICE 서버(쉼표 구분). 기본은 aiortc 기본값과 같은 구글 STUN - NAT 뒤 서버(학교 도커)는
    #: 이걸로 공인 주소를 알아내야 폰이 붙는다. 랜카드에 공인 IP 가 직접 붙은 서버는 STUN 결과가
    #: host 후보와 같아 쓸모가 없는데, 닿지 않는 인터페이스(Tailscale/Hyper-V/IPv6)마다 5초 타임아웃을
    #: 기다려 /offer 응답이 매번 7초 늦었다(실측, 이름 조회 ~2초 포함). 그런 서버는 .env 에서 비운다:
    #: HEDDY_ICE_SERVERS=
    ice_servers: tuple[str, ...] = ("stun:stun.l.google.com:19302",)
    #: ice_servers 의 turn:/turns: 항목에 붙일 자격(장기 자격). STUN 만으로는 학교 도커 + 학교 NAT
    #: 뒤 서버에 외부 브라우저가 못 붙어서(ICE failed), 이 PC 의 TURN(server/turn) 릴레이를 쓴다.
    #: aiortc 는 STUN/TURN 을 각각 **첫 번째 것만** 쓴다. 값은 server/.env 에(turn.exe 도 같은 키를 읽는다):
    #: HEDDY_ICE_SERVERS=stun:stun.l.google.com:19302,turn:<이 PC 공인 IP>:3478
    #: HEDDY_TURN_USERNAME=heddy  HEDDY_TURN_CREDENTIAL=<비밀번호>
    turn_username: str = ""
    turn_credential: str = ""
    #: 처리/송출 해상도 "가로x세로". 받은 프레임을 이 비율로 가운데를 잘라 이 크기로 맞춘 뒤 모든 처리를 한다.
    #: 플랫폼이 세로 화면(폰 카메라)이고 영상 칸이 400x649 라 기준이 세로다 - 기본은 그 2배(고해상도).
    #: 649 는 홀수라 못 쓴다(H.264 yuv420 은 짝수만): 648 / 1298. 폰이 정확히 이 크기를 안 줘도
    #: (iOS 는 720x1280 같은 고정 모드만 준다, 업링크가 나쁘면 브라우저가 낮춰 보낸다) 출력은 늘 이 크기다.
    #: 클라이언트가 /offer 본문에 out_size 를 실으면 out_sizes 안의 값일 때만 그걸 쓴다. 비우면 받은 그대로.
    out_size: str = "800x1298"
    out_sizes: tuple[str, ...] = ("400x648", "800x1298")
    #: 받은 프레임 비율이 out_size 와 다를 때 맞추는 방법.
    #: "contain": 원본 전체를 out_size 안에 들어가게 줄여서 처리하고, 송출 때 남는 칸을 같은 영상을 흐리고
    #:            어둡게 깐 배경으로 채운다 - 화각 손실 없음.
    #: "crop": 가운데를 잘라 채운다. 가로 웹캠(1298x800)이면 가운데 493px 만 남아 얼굴이 2.6배로 커진다(실측 로그).
    #: 비율 차이가 10% 이내면 어느 쪽이든 그냥 잘라 맞춘다 - iPhone 720x1280 은 9% 라 위아래 조금만 잘림(1.1배),
    #: contain 이면 좌우에 35px 짜리 흐린 띠가 생겨 더 어색하다.
    fit_mode: str = "contain"
    #: 이 시간 동안 프레임이 한 장도 안 들어오면 세션을 정리한다.
    #: 브라우저를 그냥 닫으면 ICE 가 failed 로 갈 때까지 리소스가 남는다.
    session_idle_timeout_s: float = 90.0
    #: 세션 정리 주기.
    session_reaper_interval_s: float = 15.0

    #: 내보내는 RTP 패킷의 최대 페이로드 크기(바이트). 0 이면 aiortc 기본값(1300).
    #:
    #: aiortc 기본 1300 은 RTP 헤더 12 + SRTP 태그 ~10 + UDP 8 + IP 20 을 더하면
    #: 회선상 약 1350 바이트가 된다. Chrome 은 1200(회선상 ~1250)을 쓴다.
    #: 경로 MTU 가 그 사이에 있으면 **Chrome 이 보낸 건 통과하는데 서버가 보낸
    #: 것만 버려진다.** 실측된 증상이 정확히 이랬다: 브라우저→서버는 30fps 로
    #: 멀쩡한데 서버→브라우저는 패킷 10개 중 4개가 사라지고 프레임이 한 장도
    #: 조립되지 않았다(코덱을 VP8 로 바꿔도 동일).
    #: 1100 은 터널/VPN 이 낀 경로까지 여유를 둔 값이다. 패킷 수가 조금 늘 뿐
    #: 화질/지연에는 사실상 영향이 없다.
    rtp_packet_max: int = 1100

    #: 서버가 **내보내는** 영상의 비트레이트(bps). 0 이면 aiortc 기본값.
    #:
    #: aiortc 는 인코더 비트레이트를 코덱 모듈의 전역 상수로 들고 있다:
    #:     DEFAULT_BITRATE  500,000   시작값
    #:     MAX_BITRATE    1,500,000   상한 (target_bitrate 세터가 여기서 자른다)
    #: 브라우저가 REMB 로 "더 보내도 된다"고 알려줘도 저 min() 에서 잘린다.
    #: 720p30 은 보통 2~4Mbps 가 필요해서 1.5Mbps 로는 얼굴 윤곽과 머리카락
    #: 경계가 뭉갠다.
    #:
    #: **업링크(클라이언트 -> 서버)와는 무관하다.** 저 상수는 서버 인코더 것이고,
    #: 올라오는 영상은 클라이언트 인코더와 aiortc 가 수신자로서 보내는 REMB
    #: (rtcrtpreceiver.py:461, 실측 기반이라 이 상수에 안 걸린다)가 정한다.
    #: 즉 이 값을 올려도 GAN 입력이나 얼굴 각도 추정 품질은 달라지지 않는다.
    #: 사용자가 **보는** 화면만 좋아진다.
    #:
    #: 상한 3 -> 6 Mbps. 720p30 에서 3 Mbps 는 픽셀당 0.11 bit 라 머리카락 같은
    #: 고주파가 인코더에서 먼저 뭉갠다 - 합성 품질을 아무리 올려도 화면에는
    #: 안 보인다. 실제 송출량은 브라우저 REMB 가 정하므로 회선이 느리면
    #: 알아서 내려간다. 상한은 "좋은 회선에서 얼마까지 허용하나"다.
    video_start_bitrate: int = 2_500_000
    video_max_bitrate: int = 6_000_000

    #: 서버가 내보낼 비디오 코덱. "vp8" | "h264" | "auto"
    #:
    #: aiortc 가 지원하는 건 VP8 과 H.264 둘뿐이다. 예전엔 "H.264 로 붙으면 Chrome 이 framesDecoded 0
    #: 인 채 검은 화면" 이라 VP8 로 고정했다고 적혀 있었는데, 2026-10-16 에 보니 고정(_prefer_codec)이
    #: 애초에 안 먹고 있었다 - 코덱은 브라우저 offer 순서(Chrome 은 VP8 먼저)로 정해졌을 뿐이다.
    #: 고정을 고친 뒤 H.264(x264 veryfast, level 지정 없음)는 Chrome 에서 정상 디코드됐다.
    #: H.264 가 기본인 이유: 1080p 인코드가 VP8 7.2ms -> x264 veryfast ~2ms 이고, 수신 디코드도 싸며,
    #: 나중에 NVENC(h264_nvenc)로 갈 때 코덱이 같다. 문제가 생기면 HEDDY_VIDEO_CODEC=vp8 로 되돌린다.
    #: "auto" 는 브라우저가 고르는 대로.
    video_codec: str = "h264"
    #: H.264 로 보낼 때 libx264 preset. aiortc 는 지정하지 않아 x264 기본 medium 이었다(1080p 인코드 9.7ms).
    #: veryfast 1080p ~2ms, 화질 차이는 PSNR 0.04dB. 비우면 aiortc 기본 그대로.
    video_x264_preset: str = "veryfast"
    #: 합성 결과를 GPU 에서 yuv420p 로 바꿔 인코더에 넘긴다(gpu_segmenter._download_i420). 끄면 BGR 을 넘겨
    #: 인코더가 CPU 에서 바꾼다(1080p ~3.5ms). VRAM 은 프레임 한 장 분량의 임시 텐서뿐.
    video_gpu_i420: bool = True
    #: 박스 블러(gpu_segmenter._box) 커널이 이 값의 2배 이상이면 k//box_ds_k 배 줄인 해상도에서 흐린다.
    #: 0 이면 항상 원래 해상도. 결과는 저주파라 거의 같고 1080p 에서 _box 비용이 크게 준다.
    box_ds_k: int = 8
    #: 3D 그룸 후처리를 머리 둘레 사각형(ROI)에서만 돌린다(gpu_segmenter._groom_roi). 끄면 전체 프레임.
    #: pad: 사각형 여유(눈 간격 배수) - 후처리 필터 반경보다 커야 가장자리 패딩이 결과에 안 닿는다.
    #: hair: 원래 머리 소프트 알파가 이 값을 넘는 곳까지 포함(그 밖은 지우지 않게 된다).
    groom_roi: bool = True
    groom_roi_pad: float = 0.75
    groom_roi_hair: float = 0.02
    #: 3D 그룸 ROI 후처리(지우기/채우기/연장/틈/속/그림자/라이트 랩)를 1/이 값 해상도로 돌린다(gpu_segmenter._lo_paste).
    #: 1 = 원래 해상도. 2 면 1080p 에서 픽셀 1/4 - 헤어 렌더·머리색 맞춤·가닥 디테일·그레인·원본 화면은 원래 해상도.
    groom_post_scale: int = 1

    # ---------- 세그멘테이션 ----------
    model_id: str = "jonathandinu/face-parsing"
    #: 허브 리비전 고정. 핀이 없으면 업스트림이 바뀐 날 결과가 조용히 달라진다.
    #: 로컬 HF 캐시에 실제로 받아둔 스냅샷 해시.
    model_revision: str = "758b82e15a0178c9db39c1ff666a8b56e3a550c8"
    #: CUDA 그래프 때문에 고정. 로짓은 이것의 1/4 해상도로 나온다.
    input_size: int = 512
    use_cuda_graph: bool = True
    #: 경계 알파의 급격함. 작을수록 부드럽게 번진다.
    soft_k: float = 1.5
    #: 조명 정합 배율의 상/하한. 세그멘테이션이 흔들려 피부 평균이 튀는 순간
    #: 헤어 색이 통째로 날아가는 걸 막는 안전장치.
    harmonize_min: float = 0.65
    harmonize_max: float = 1.55
    #: 헤어라인 그림자 띠의 폭(픽셀, 눈 간격 50px 기준 - 얼굴 크기에 비례해 조정된다).
    shadow_k: int = 21
    #: 합성 헤어가 눈을 덮지 못하게 하는 정도(0~1). 1 이면 눈은 항상 보인다.
    #: 앞머리가 눈썹을 덮는 건 자연스럽지만 눈까지 완전히 가리면 사람이 아니라
    #: 마네킹처럼 보인다. 파싱이 눈으로 분류한 픽셀에서만 새 헤어 알파를 깎는다.
    protect_eyes: float = 1.0
    #: 눈썹 보호 정도(0~1). 기본 0 - 실제 앞머리는 눈썹을 덮는 게 정상이다.
    #: 눈썹을 살리고 싶으면 올린다.
    protect_brows: float = 0.0
    #: 보호 마스크 경계를 부드럽게 하는 커널. 클래스 경계를 그대로 쓰면 눈 주위에
    #: 각진 구멍이 생긴다.
    protect_blur_k: int = 9
    #: 랜드마크 기준 눈 보호 타원의 반지름(눈 간격 배수, 가로/세로).
    #:
    #: 파싱 클래스만 믿으면 안 되는 이유: 앞머리 + 안경 + 뿌연 조명에서
    #: SegFormer 가 눈 부위까지 통째로 '머리'로 분류하는 프레임이 실제로 있다
    #: (train/frames 실측). 그러면 눈이 지워지고 GAN 얼굴 패치가 그 자리를
    #: 덮어 눈이 남의 것이 된다. MediaPipe 눈 위치는 그런 프레임에서도 잡히므로
    #: 기하학적 타원을 파싱 마스크에 합친다. 0 이면 끈다.
    #:
    #: 기본 0(끔): 3D 그룸 경로에선 지우기 단계에서 이 원 안의 원래 앞머리가 그대로 남아 눈에 동그란
    #: 보호막이 씌워진 것처럼 보였다(사용자 피드백). 이마 인페인터 v5 가 눈 윤곽을 보고 눈 위 앞머리를
    #: 지우고 눈을 다시 그리므로 필요 없다. 파싱이 '눈'으로 찍은 픽셀 보호(protect_eyes)는 그대로다.
    protect_eye_rx: float = 0.0
    protect_eye_ry: float = 0.0

    # ---------- 합성 정합 (이질감) ----------
    #: 워핑된 헤어 알파 경계를 프레임 스케일에서 부드럽게 하는 폭(눈 간격 배수).
    #: 에셋의 3px 페더는 1024 스케일 기준이라 프레임(눈 간격 50px)으로 4배 축소되면
    #: 1px 미만이 되어 오려붙인 경계가 된다. 0 이면 끈다.
    edge_feather: float = 0.05
    #: 장면 블랙레벨 정합. GAN 은 노출을 '정상'으로 되돌려 그리므로 플레어/역광
    #: 으로 뿌연 프레임에 얹으면 헤어만 새까맣게 튄다. harmonize(곱셈 배율)는
    #: 검은 것을 밝힐 수 없다. **머리 주변**(본인 머리/눈동자/콧구멍 - 항상
    #: 어두운 게 있다)의 어두운 분위수(black_pct)와 에셋 머리의 어두운 분위수를
    #: 맞추는 만큼 그림자를 들어올린다(lift). 프레임 전체를 쓰지 않는 이유는
    #: 흰 벽/밝은 옷뿐인 또렷한 장면에서도 "검은 게 없다" 고 오판하기 때문.
    #: 실측: 뿌연 프레임에서 본인 머리 휘도 ~65 = 이 값. 0 이면 끈다.
    black_match: float = 1.0
    black_pct: float = 0.01
    black_lift_max: float = 90.0
    #: 그레인 정합. 웹캠 프레임의 센서/압축 노이즈 σ 를 피부에서 재고 같은 σ 의
    #: 노이즈를 헤어/얼굴 패치에 얹는다. 합성물이 "너무 깨끗" 해서 튀는 걸 막는다.
    #: 1.0 = 프레임과 같은 σ. 0 이면 끈다.
    grain_match: float = 1.0
    #: 얼굴 패치의 저주파(색/음영)를 실제 피부에서 가져오는 정도(0~1). 패치는
    #: GAN 이 그린 이마라 색이 조금 다르고 앞머리 자리의 얼룩이 남을 수 있다.
    #: 큰 커널(눈 간격 x face_lf_k)로 흐린 성분만 실제 피부의 것으로 바꾸고
    #: 눈썹/음영 같은 세부는 패치 것을 남긴다.
    face_lf_match: float = 1.0
    face_lf_k: float = 0.5
    #: 얼굴 zone 안에서 erase 마스크를 단단하게 만드는 소프트 알파 구간.
    #: 원래 앞머리 경계 픽셀은 알파 0.3~0.7 라 절반만 지워져 이마에 어두운
    #: 테두리가 남는다. 이 구간을 0~1 로 다시 펴서 확실히 지운다.
    erase_hard_lo: float = 0.25
    erase_hard_hi: float = 0.60
    #: 그 단단한 erase 의 팽창 폭 (눈 간격 배수, 최소 3px). 640x480 의 3px ≈ 0.06d.
    erase_hard_grow: float = 0.06
    #: 배경 플레이트 기록에서 사람(배경 아닌 것) 둘레를 빼는 폭(프레임 높이 배수). 빠르게 움직이면 번진 머리
    #: 끝을 파서가 배경으로 찍어 플레이트에 머리색이 쌓였고, 지운 자리를 채울 때 원래 머리 잔상으로 나왔다.
    plate_margin: float = 0.012
    #: 플레이트 픽셀을 마지막으로 본 뒤 카메라가 이만큼(프레임 높이 배수) 움직이면 버린다 (0=끔). 폰을 손에 들면
    #: 배경이 밀려 예전 위치의 배경이 지운 자리에 어긋난 잔상으로 남았다. 0.006 ≈ 1298 세로에서 8px.
    plate_drift: float = 0.006
    #: 지운 머리 자리를 플레이트(지난 프레임에서 본 배경)로 채우나. 끄면 torso_net(크롭 안: 얼굴 옆~가슴) + 지금
    #: 프레임 배경/옷 확산(그 밖, body_fill 경로)만 쓴다 - 지난 프레임 정보가 없어 움직일 때 잔상이 원리상 없다.
    #: 대신 실제로 드러났던 배경 모서리(문틀 등)는 덜 정확하다. 2026-10-20 사용자 요청으로 끔.
    plate_fill: bool = False
    #: 그 카메라 이동을 재는 간격(프레임). 재는 데 작은 GPU 커널 ~40개라 매 프레임은 아깝다.
    plate_drift_every: int = 3
    #: 랜드마크 얼굴 타원 중 이 비율 넘게 '배경'으로 찍힌 프레임은 플레이트를 갱신하지 않는다(파서 실패 -
    #: 고개를 젖히면 얼굴/목이 통째로 배경으로 찍혀 얼굴 잔상이 됐다). 1 이면 끈다.
    plate_face_bg_max: float = 0.2
    #: 사람(배경 아닌 것) 면적이 최근 평균보다 이 비율 넘게 갑자기 줄어든 프레임은 플레이트를 갱신하지 않는다
    #: (파서가 얼굴/목 일부를 배경으로 찍은 프레임). 0 이면 끈다.
    plate_person_drop: float = 0.1
    #: 얼굴 zone 안에서 눈 위 이 높이(눈 간격 배수)보다 아래는 이마 패치가 덮는 만큼만 지운다(0=끔).
    erase_eye_keep: float = 0.35
    #: 이마 패치 둘레를 얼굴(지우는 자리)로 넓히는 폭 (눈 간격 배수, 0=끔). 패치가 오클루더 실루엣에서
    #: 끊긴 관자놀이의 앞머리 옆 끝을 피부톤 평면으로 이어 지운다.
    face_patch_grow: float = 0.25
    #: 이마 패치 밖 지운 자리(패치 둘레·관자놀이·정수리)를 피부톤 평면 대신 패치 속 + 실제 피부 색의 확산으로
    #: 채운다(gpu_segmenter). 평면은 패치와 색이 달라 이마 패치 둘레에 살색 띠가 둘러졌다.
    face_fill_extend: bool = True
    #: False 면 이마 패치가 있을 때 패치 자리만 얼굴(피부)로 채우고, 눈 타원 zone·패치 둘레 띠는 피부로 칠하지
    #: 않는다(배경 플레이트로). 패치 둘레 살색 띠를 없애 본 것(사용자 요청).
    face_fill_band: bool = False
    #: 지운 머리 자리를 채울 때 상수 살색 대신 **휘도 평면**(L=a+bx+cy)을 피팅해
    #: 그라디언트를 살린다. 실제 이마는 중앙이 밝고 관자놀이로 어두워지는데
    #: 상수로 덮으면 스티커처럼 보인다. 0 이면 예전 상수 살색.
    skin_plane: bool = True
    #: 합성 헤어에 거는 언샤프 마스크 세기(0~). 웹캠 입력 업스케일(눈 50px ->
    #: 1024)과 워핑 축소로 깎인 고주파를 되살린다. 헤어 rgb 에만 적용(얼굴은
    #: 실제 픽셀). 0 이면 끔. 과하면 'AI 텍스처'가 다시 도드라지니 온건하게.
    hair_sharpen: float = 0.6
    #: 통계용 GPU 동기화는 이 프레임 수마다 한 번만.
    stats_every: int = 5
    #: 이보다 작은 영역은 무게중심을 믿지 않는다.
    min_anchor_px: int = 20
    #: 단계별 시간 측정을 CUDA 이벤트 대신 torch.cuda.synchronize() 로 한다.
    #: 디바이스 전체를 멈추므로 기본은 끔. 이벤트로도 같은 숫자가 나온다.
    profile_blocking_sync: bool = False

    # ---------- 헤어 에셋 ----------
    #: server/assets/ 의 정적 에셋을 전 세션에 공유할지.
    #:
    #: 기본 꺼짐. 프로덕션 플로우는 "헤어 고르기 -> 고개 돌려 라이브 뱅크 생성 ->
    #: 씌우기" 하나뿐이고, 거기에 정적 에셋은 등장하지 않는다. 켜 두면 프로토타입
    #: 시절에 구워 둔 에셋과 절차적 기본 에셋이 클라이언트 목록에 그대로 나가서,
    #: 사용자가 자기 얼굴로 만들지 않은 헤어를 고를 수 있게 된다.
    #:
    #: 감추는 게 아니라 **읽지 않는다.** 목록에서만 빼면 /references 와
    #: stats.assets 중 한쪽에 남기 쉽다. 아예 안 읽으면 양쪽에서 동시에 사라지고
    #: fit {asset:...} 도 자연히 안 먹는다.
    #: 이 프레임의 랜드마커 결과를 기다리는 상한(초). 넘기면 직전 프레임 포즈로 그린다.
    #: 포즈 7.9ms(p95 10.7) 가 GPU 추론과 겹치므로 보통 수 ms 만 실제로 기다린다.
    pose_wait_s: float = 0.025
    #: 랜드마커가 얼굴을 놓쳐도 직전 포즈를 이 프레임 수까지 유지한다(고개 숙임/젖힘의 짧은 끊김).
    pose_hold_frames: int = 6
    #: 이 프레임 포즈가 없을 때(타임아웃/얼굴 놓침) 직전 포즈를 그대로 쓰지 않고, 분할 얼굴 위치가 그 포즈를
    #: 잰 프레임에서 지금까지 움직인 만큼 옮겨 쓴다(gpu_segmenter._reanchor_pose). 끄면 예전처럼 그대로.
    pose_reanchor: bool = True
    #: 랜드마커가 찾을 최대 얼굴 수. 2 이상이면 화면에 여럿일 때 가장 큰 얼굴을 착용자로 잡아 계속 따라간다
    #: (face_pose.pick_wearer). 1 이면 예전처럼 랜드마커가 잡은 아무 얼굴 하나 - 옆 사람에게 씌울 수 있다.
    max_faces: int = 2
    #: 착용자를 이 시간(ms) 넘게 못 보면 그때 보이는 가장 큰 얼굴로 넘긴다. 그 전엔 옆 사람에게 안 넘긴다.
    #: 1.5초였을 때 착용자가 옆모습으로 잠깐 안 잡히는 사이 옆 사람에게 넘어가, 돌아와도 점수 차가 작아
    #: 되찾지 못했다(실측, 서버 경로). 정말 자리를 떠났을 때만 넘어가게 길게.
    wearer_hold_ms: float = 5000.0
    #: 직전 착용자 얼굴 중심에서 (얼굴 크기 x 이 값) 안의 얼굴만 같은 사람으로 본다.
    wearer_max_jump: float = 0.6
    #: 처음 착용자를 고를 때 화면 가운데를 얼마나 우대할지. 얼굴 크기에 (1 - 이 값 x 가운데에서 벗어난 정도 0~1)
    #: 를 곱해 비교한다 - 크기가 비슷한 두 사람이면 가운데 사람.
    wearer_center_weight: float = 0.5
    #: 그 점수가 착용자의 이 배수를 넘는 얼굴이 wearer_switch_ms 동안 계속 보이면 착용자를 넘긴다.
    wearer_switch_ratio: float = 1.3
    wearer_switch_ms: float = 700.0
    #: 그 비교에서 착용자 크기는 최근 최대값을 이 시간(ms) 상수로 서서히 잊으며 쓴다 - 고개를 젖히면 얼굴 상자가
    #: 줄어 그 사이 옆 사람에게 넘어갔다(실측).
    wearer_size_mem_ms: float = 5000.0
    #: 착용자 머리만 지운다(gpu_segmenter._owner_mask). 끄면 화면 안 모든 머리가 지울 대상(예전 동작).
    owner_mask: bool = True
    #: 착용자 몸 덩어리를 이을 때 메우는 틈 (눈 간격 배수). 머리카락 사이/머리-어깨 사이 작은 배경 틈.
    owner_gap: float = 0.3
    #: 진단용으로 되살리려면 HEDDY_SERVE_STATIC_ASSETS=1.
    serve_static_assets: bool = False

    #: 서버 시작 때 GPU 세그멘터를 미리 올린다(server.py --preload 의 기본값). 끄면 첫 연결이
    #: 모델 적재를 기다려 그동안 화면이 검다.
    preload: bool = True

    # ---------- 3D 그룸 (groom_renderer.py) ----------
    #: 그룸 색 맞춤 - 사용자 머리 평균색 EMA 계수(stats_every 프레임마다 한 번).
    groom_color_alpha: float = 0.2
    #: 이보다 머리 픽셀이 적으면(모자/삭발) 색을 맞추지 않고 GLB 색을 쓴다.
    groom_color_min_px: int = 400
    #: 머리색 측정에 쓰는 휘도 분위(1=전체 평균). 0.5 면 어두운 절반만 - 윤기/반사광/조명 번짐 제외.
    groom_color_quantile: float = 0.5
    #: 리본 가장자리 페더 폭 (눈 간격 배수). v1 그룸(가는 결 + 8x MSAA 확률적 투명도)은 페더가
    #: 결을 뭉개므로 0. v0 굵은 리본에서만 0.04 가 유효했다.
    groom_feather: float = 0.0
    #: 머리색 샘플링을 허용하는 포즈 범위(도). 벗어나면 이전 값 유지(파서가 피부를 머리로 찍는다).
    groom_color_max_yaw: float = 25.0
    groom_color_max_pitch: float = 20.0
    #: 휘도 대비(표준편차) 맞춤 세기. 0=평균색만, 1=완전 일치. 배율은 [min,max] 로 제한.
    #: 기본 0: v1 그룸(가닥별 변주 + 이방성 하이라이트)은 자체 대비가 충분하고, 1.5배만 늘려도
    #: 어두운 결 뭉치가 근검정 얼룩이 됐다(실측). v0 굵은 리본에서만 유효했던 보정.
    #: 2026-10-15: 줄이기 전용으로 켬(max 1.0). v1 그룸은 이방성 하이라이트 때문에 대비가 실제 머리보다 커서
    #: 같은 평균색이어도 은색/가발처럼 떠 보였다(실측). 늘리기(>1)는 위 이유로 여전히 막는다.
    groom_contrast_match: float = 1.0
    groom_contrast_min: float = 0.5
    groom_contrast_max: float = 1.0
    #: 윤기(이방성 하이라이트)의 흰빛 바닥. 색 배율은 윤기에도 곱해져서 어두운 머리일수록 하이라이트가
    #: 사라져 검은 머리가 점토처럼 보였다(사용자 피드백). 색 배율의 휘도가 이 값보다 낮으면 모자란 만큼을
    #: 무채색으로 더해 준다 - 밝은 머리(배율 ~0.45 이상)는 예전과 같다. 0 이면 끔.
    groom_spec_floor: float = 0.45
    #: 머리숱 부피(1=재구성 원본). 두피에서 바깥으로 뜬 거리를 이 배율로 줄인다 - 헬멧처럼 붕 뜨는 것 방지.
    #: y0(정규 cm, 두상 중심 기준 +위) 아래로 늘어진 머리는 그대로(긴 머리가 목 쪽으로 빨려드는 것 방지).
    #: 3D 그룸에서 눈/안경 보호로 헤어를 깎는 것을 이 yaw(도) 구간에서 서서히 끈다(옆모습 렌즈 모양 구멍 방지).
    groom_protect_yaw_lo: float = 15.0
    groom_protect_yaw_hi: float = 30.0
    #: 3D 그룸 헤어를 눈 보호 마스크로 깎는 정도(0~1, protect_eyes 에 곱한다). 0 이면 앞머리가 눈을
    #: 실제처럼 덮는다(가는 결 사이로 비치는 건 렌더 알파). 1 이면 예전처럼 눈 둘레에 타원 구멍.
    #: 기존 머리를 지울 때 눈 위에 살색을 안 칠하는 보호(erase 쪽)는 이 값과 무관하게 그대로다.
    groom_protect_eyes: float = 0.0
    groom_volume: float = 0.6
    #: 앞머리 붙이기 (groom_renderer._hug_front, 그룸 로드 때 1회): 얼굴 앞쪽(방위 ±40°, ±70° 에서 0)
    #: 가닥 점이 두상 표면에서 groom_front_hug_cm 넘게 뜬 만큼을 이 배율로 줄인다(1=끔). 정면 사진 한 장
    #: 재구성이라 앞머리가 이마에서 2.5~4.7cm 떠 옆모습에서 차양처럼 길게 뻗었다(사용자 피드백 dandy-cut).
    #: y = (이 높이 아래 0 - 2cm 램프, 이 높이까지 전부, 이 높이에서 0) 정규 cm. 정수리 볼륨(포마드)은 그대로.
    groom_front_hug: float = 0.3
    groom_front_hug_cm: float = 0.6
    groom_front_hug_y: tuple[float, ...] = (-1.0, 9.0, 12.0)
    #: 그룸 자동 맞춤 (gpu_segmenter._update_autofit): 정면 포즈에서 얼굴 랜드마크와 그룸 두상의 대응점으로
    #: 가로/세로 배율과 이마 높이를 사용자에게 맞춘다. range 는 배율 한계, alpha 는 프레임당 EMA.
    #: 헤어 안쪽과 얼굴 사이에 끼인 틈(뒤 배경이 비침)을 헤어 그늘색으로 메우는 세기(0=끔). k 는 메울 수
    #: 있는 최대 틈 폭(눈 간격 배수, 닫힘 커널), down 은 눈 아래 이 높이까지만(목 옆 긴 머리 틈은 그대로).
    groom_gapfill: float = 1.0
    groom_gapfill_k: float = 0.35
    groom_gapfill_down: float = 0.6
    groom_gapfill_fh_w: float = 0.8
    groom_autofit: bool = True
    #: 그룸 렌더 결과를 머리 둘레 사각형만 GL 에서 읽는다(groom_renderer._read_rect). 끄면 전체 프레임.
    groom_read_crop: bool = True
    #: GL 렌더 결과(헤어 색/두상 좌표/얼굴 위치·노멀)를 CUDA-GL interop 로 GPU 안에서 바로 가져온다(groom_renderer._GLCuda).
    #: 끄거나 실패하면 예전처럼 CPU 로 읽어 다시 올린다 - 1080p 에서 프레임당 ~7ms(그리기 1.4ms 의 5배).
    groom_gl_interop: bool = True
    #: 그룸 후처리의 원소별 식 묶음을 torch.compile(Triton)로 합친다(fused.py). 기동 때 수 초 컴파일.
    #: 끄거나 컴파일이 안 되면(triton/컴파일러 없음) 같은 식을 eager 로 돈다. 기본 끔: 실측 이득 2~4ms 뿐이고
    #: 텐서 모양/메모리 배치가 바뀔 때 세션 중 재컴파일(수백 ms 멈춤)이 아직 남아 있다(2026-10-20).
    fused_compile: bool = False
    groom_autofit_max_yaw: float = 15.0
    groom_autofit_max_pitch: float = 15.0
    groom_autofit_range: tuple[float, ...] = (0.75, 1.35)
    groom_autofit_max_up_cm: float = 4.0
    groom_autofit_alpha: float = 0.1
    #: 깊이 맞춤(gpu_segmenter._update_autofit_depth): 이 yaw(도) 구간의 옆모습에서 두상 깊이 배율 sz 와 앞뒤
    #: 오프셋(cm)을 랜드마크에 맞춘다. 정면 맞춤은 깊이를 못 봐 옆모습에서 헤어가 얼굴 앞으로 떴다.
    #: sz/fz 는 (하한, 상한).
    groom_autofit_depth: bool = True
    groom_autofit_depth_yaw: tuple[float, ...] = (18.0, 55.0)
    groom_autofit_depth_sz: tuple[float, ...] = (0.8, 1.4)
    groom_autofit_depth_fz: tuple[float, ...] = (-3.0, 2.0)
    #: 두상 정규 좌표 y(cm, 두상 중심 기준 +위) 가 이보다 위면 두피 - 얼굴면/피부 채움에서 뺀다.
    groom_face_top_cm: float = 5.0
    groom_volume_y0: float = -1.0
    #: 긴 머리: 몸 뒤로 넘어간 뒷머리(어깨선 아래 + 상체 앞면보다 뒤)를 사용자 몸 영역(분할의 머리카락/배경
    #: 아닌 곳) 안에서 지운다(0=끔). 몸 오클루더가 없어 등 뒤 머리가 몸을 뚫고 가슴 위에 커튼처럼 떴다
    #: (사용자 피드백 glossy-straight). 깊이 판정은 그룸 정규 좌표(cm, 두상 중심 기준 +y 위 +z 앞)의 상체
    #: 근사: 목·가슴 단면 타원(중심 z=cz, 반폭 a, 앞쪽 반깊이 b) - 앞면보다 margin 넘게 뒤면 '몸 뒤'.
    groom_body_occlude: float = 1.0
    #: (턱 y - 이보다 위는 판정 안 함, 타원 중심 z, 목 단면이 끝나는 y, 가슴 단면이 시작되는 y)
    groom_body_y: tuple[float, ...] = (-9.0, -4.0, -15.0, -21.0)
    #: (목 반폭 a, 목 앞 반깊이 b, 가슴 반폭 a, 가슴 앞 반깊이 b, 앞면 여유 margin) cm
    groom_body_ab: tuple[float, ...] = (6.5, 6.0, 20.0, 8.0, 1.0)
    #: 두피 베이스 레이어 불투명도(0=끔). 그룸의 오클루더(FLAME 두상) 중 이 스타일 뿌리가 나는 자리를
    #: 헤어 아래에 헤어색(주변 가닥 색을 확산)으로 깐다. 없으면 가닥 사이/귀 위·뒤가 배경으로 뚫려
    #: 두상이 파여 보였다(실측: 원래 머리를 지운 자리에 플레이트 배경이 나옴). 실제 헤어카드 게임도
    #: 두피 텍스처를 깐다.
    groom_scalp: float = 0.92
    #: 3D 헤어 후처리(gpu_segmenter._groom_post). 몸통 채움 불투명도(0=끔): 헤어 실루엣 안의 성긴 자리
    #: 밑에 가닥색 베이스를 깐다 - 결 사이로 배경/이마가 비치던 것. body_k 는 밀도 창(눈 간격 배수),
    #: lohi 는 그 창 안 평균 알파가 lo 이하면 0(바깥 가는 결 유지), hi 이상이면 1.
    groom_body: float = 0.95
    #: 실루엣 안쪽 채우기(gpu_segmenter._groom_core): close(눈 간격 배수) 창 안 헤어 밀도가 lohi 이상인 곳을
    #: 속으로 보고 edge 만큼 깎아 이 불투명도로 채운다(0=끔). 가장자리 가는 머리끝은 원래 알파 그대로.
    groom_core: float = 0.97
    groom_core_close: float = 0.25
    groom_core_edge: float = 0.10
    groom_core_lohi: tuple[float, ...] = (0.55, 0.75)
    groom_body_k: float = 0.25
    groom_body_lohi: tuple[float, ...] = (0.30, 0.55)
    #: 카메라 선명도 맞춤 흐림 σ(px, 눈 간격 60px 기준). 가닥 디더 노이즈(뜨개실 느낌)를 누른다. 0=끔.
    #: 0.7 은 헤어 고주파가 실제 머리의 절반(실측 σ 2.0 vs 3.9~4.6, 피부 4~6)이라 화면에서 가장 매끈한 층이 돼
    #: "3D 모델 위에 뭔가 씌운 느낌"(사용자 피드백) + 윤곽 회색 번짐이 났다. 0.3 이 3.6 으로 실제 머리에 가깝다.
    groom_soften: float = 0.3
    #: 진단용: 3D 헤어 후처리 층(두피/몸통 채움·흐림·연장·틈 메우기·속 채움·페더·그림자·라이트 랩·그레인)을 전부
    #: 끄고 렌더+색 맞춤만 얹는다. 원래 머리 지우기와 이마 인페인팅은 그대로. HEDDY_GROOM_BARE=1
    groom_bare: bool = False
    #: 라이트 랩 세기(0=끔): 헤어 가장자리에 뒤 장면 색을 섞는다.
    groom_light_wrap: float = 0.35
    #: 두피/몸통 베이스 밝기 배율(뿌리/안쪽 결은 가닥 그늘에 있어 더 어둡다).
    groom_scalp_dark: float = 0.8
    #: 두상 정규 좌표 z(cm, +앞) 가 이 값 - 2 부터 이 값까지 두피 레이어를 뺀다(앞이마 제외).
    groom_scalp_front_cm: float = 3.0
    #: 두상(오클루더) 안의 두피 아닌 자리(귀/옆얼굴/목덜미)에서 지운 원래 머리를 배경 대신 피부톤으로.
    #: 플레이트는 배경만 모으므로 거기 채우면 머리 옆이 배경으로 파인다.
    groom_head_skin: bool = True
    #: 플레이트가 아직 못 본 자리(가만히 앉아 바로 씌움)의 원래 머리를 주변 배경 확산으로 채운다.
    #: 없으면 그 자리는 원래 머리가 그대로 남는다. 눈 아래 이 높이(눈 간격 배수)까지만 - 어깨/옷 위는
    #: 근거 없는 배경으로 칠하지 않는다.
    bg_diffuse_down: float = 1.2
    #: 그 아래(어깨/가슴까지 내려온 긴 머리)는 지금 프레임의 옷/목/배경을 확산해 채워 지운다(0=끔, 예전처럼
    #: 남김). 플레이트는 머리 밑의 옷을 영영 못 본다.
    body_fill: float = 1.0
    #: 옷 위로 내려온 긴 머리용 셀피 분할(MediaPipe, CPU ~100ms)을 백그라운드로 돌리는 빈도(Hz, 0=끔).
    #: 얼굴 파서는 그 머리를 '옷'으로 찍어 못 지운다. 3D 그룸 tryon 중에만 돈다.
    long_hair_seg_hz: float = 3.0
    #: 그 마스크가 이 시간(초)보다 오래되면 안 쓴다.
    long_hair_seg_max_age: float = 1.0
    #: 늦은 긴 머리 마스크/torso_net 결과를 두 눈 이동만큼 옮겨 쓴다(server._motion_warp). 고개 돌리기에서 역효과라 기본 끔.
    long_hair_motion_comp: bool = False
    #: 늦게 온 마스크를 움직임만큼 팽창하는 폭 (프레임 높이 배수).
    long_hair_grow: float = 0.015
    #: 그 마스크를 만든 프레임과 지금 프레임의 색 차이(1/8 해상도, 밝기 정규화, 0~255)가 (lo, hi) 사이에서 커지는
    #: 만큼 '배경' 자리에서는 마스크를 버린다. 머리가 떠나 벽이 드러난 자리를 묵은 마스크가 지우고 확산색으로 칠해
    #: 머리 뒤에 흐린 덩어리 잔상이 따라다녔다(짧은 머리에서도 - 셀피 분할은 머리 전체를 찍는다). 옷/목 위는 그대로.
    #: hi <= lo 면 끈다.
    long_hair_stale_diff: tuple[float, ...] = (10.0, 25.0)
    #: 긴 머리 지운 자리 중 플레이트가 못 본 곳을 학습 인페인터(torso_net, models/torso_net.pt)로 채운다. 긴 머리
    #: 분할과 같은 백그라운드 작업에서 같은 프레임으로 돈다(포즈 + 넷, CPU ~70ms + GPU ~10ms). 가중치가 없거나
    #: 실패하면 위 확산 그대로. HEDDY_TORSO_NET=0 으로 끈다.
    torso_net: bool = True
    #: torso_net 기억(머리가 비켜 보였던 픽셀을 쌓아 넷에 넣기). 기본 끔. 실제 영상(Slovo 4클립, 2026-10-20, 정답
    #: 대용에서 손 제외): 손 가리기(torso_net.hand_mask, 손 랜드마커) + 머리색 걸러내기(hair_color, 어두운 쪽 포함)
    #: 후 3Hz 에서 L1 0.085 -> 0.080, 채운 자리의 머리색 비율 15.3% -> 15.2% (걸러내기 전엔 18.9% - 분할이 놓친
    #: 검은 머리를 되살렸다). 이득이 작고 폰 실측 전이라 기본 끔. 끄면 넷은 기억 0 (학습의 40% '기억 없음' 경우)으로
    #: 돌고 손 랜드마커(CPU ~35ms)도 안 돈다. HEDDY_TORSO_MEM=1 로 켠다.
    torso_mem: bool = False
    #: torso_net 기억(머리가 비켜 보였던 픽셀)의 신선도 시간 상수(초)와, 이보다 오래 끊기면 버리는 시간(초).
    torso_mem_tau: float = 3.0
    torso_mem_reset: float = 5.0
    #: 3D 그룸 앞머리가 이마에 떨어뜨리는 아래쪽 그림자 세기(0=끔)와 이동/흐림 폭(눈 간격 배수).
    #: 원래 머리가 3D 헤어 실루엣 밖으로 삐져나온 자리를 헤어로 메우는 세기(0=끔, 예전처럼 배경으로
    #: 지움). 지우면 헤어 둘레에 뿌연 후광이 떴다(gpu_segmenter._groom_extend). k 는 그룸에서 이만큼
    #: (눈 간격 배수) 떨어진 곳까지만, eye 는 눈 위 이 높이부터(귀/옆얼굴 제외), fh_w 는 이마로 빼는
    #: 가로 반폭, detail 은 원래 머리 휘도 결을 빌리는 세기.
    #: k 는 0.45 였다 - 그룸이 원래 머리보다 작으면 그 띠(~3cm)만큼 원래 머리 결이 그룸 밖에 그대로 남아
    #: '실제 머리가 보인다'(사용자 피드백). 후광의 큰 원인이던 배경 위 그림자 고리는 groom_shadow_skin_only 로
    #: 뺐으므로 경계 틈만 메우는 폭으로 줄였다.
    groom_extend: float = 0.95
    groom_extend_k: float = 0.15
    groom_extend_eye: float = 0.2
    groom_extend_fh_w: float = 0.95
    groom_extend_detail: float = 0.8
    #: 3D 그룸 헤어라인/아래 그림자를 피부·얼굴(파서 1~12, 얼굴 zone, 두상 오클루더) 위에만 얹는다.
    #: 끄면 예전처럼 헤어 둘레 전체(배경 벽 포함) - 원래 머리보다 작은 그룸에서 뿌연 어두운 고리가 떴다.
    groom_shadow_skin_only: bool = True
    groom_drop_shadow: float = 0.45
    groom_drop_shadow_dy: float = 0.10
    groom_drop_shadow_k: float = 0.30
    #: 맨이마 패치를 그룸 두상 오클루더에 3D 투영해 붙인다(0 이면 예전 눈 앵커 닮음변환 + yaw 페이드).
    forehead_3d: bool = True
    #: 투영 경로: 패치를 찍은 순간 표면이 카메라를 향한 정도(코사인)가 lo 이하면 알파 0, hi 이상 1.
    forehead_3d_facing: tuple[float, ...] = (0.25, 0.5)
    #: 이마 패치 알파를 |yaw| 에 따라 빼는 구간(도): lo 이하 1, hi 이상 0.
    face_patch_yaw_lo: float = 25.0
    face_patch_yaw_hi: float = 50.0
    #: 맨이마 패치(forehead.py) 주기 갱신 간격(초). 0 이면 선택/수동 때만. 조명·표정 변화를 따라간다.
    #: 인페인트(LaMa ~45ms)는 별도 스레드/스트림이지만 GPU 경합+GIL 로 그 순간 프레임 하나가
    #: 40~50ms 로 튄다(실측). 2초에 한 프레임이면 체감이 안 되는 수준. 더 줄이려면 별도 프로세스.
    forehead_refresh_s: float = 3.0
    #: 이 각도(|yaw|, 도) 안일 때만 갱신 - 돌린 얼굴은 인페인트 문맥이 나쁘다.
    forehead_refresh_yaw: float = 15.0
    #: 새 패치로 바꿀 때 이전 패치와 섞는 시간(초). 툭 바뀌면 이마가 깜빡인다.
    forehead_fade_s: float = 0.4
    # 2차 운동 (hair_dynamics.py). 강체 헤어는 가발처럼 보인다.
    #: 스프링 상수(1/s²). 클수록 빨리 따라온다. 140 이면 0.2초 회전에 14° 뒤처졌다 0.6초에 정착.
    groom_dyn_stiffness: float = 140.0
    #: 감쇠비. 1 미만이면 한 번 넘쳤다 돌아온다(자연스러움), 1 이면 넘침 없이 정착.
    groom_dyn_damping: float = 0.55
    #: 중력: 고개를 기울였을 때 끝이 수직으로 처지는 비율 (0=두상 그대로, 1=완전 수직).
    groom_dyn_gravity: float = 0.5
    #: 이동 지연 상한(cm). 두상에서 너무 떨어지면 잘린 것처럼 보인다.
    groom_dyn_max_shift_cm: float = 2.5
    #: 정지해 있을 때의 미세 흔들림 진폭(cm, 끝 기준). 0 이면 완전 정지.
    groom_dyn_idle_cm: float = 0.12
    #: 흔들린 가닥이 두상(오클루더) 속으로 못 들어가게 표면으로 밀어낸다(groom_renderer 셰이더 u_collide).
    #: 끄면 얼굴 쪽으로 쏠린 가닥이 오클루더 뒤로 숨어 헤어 안쪽이 볼/턱 윤곽대로 잘려 보였다.
    #: margin: 표면에서 띄우는 거리(cm).
    groom_dyn_collide: bool = True
    groom_dyn_collide_margin_cm: float = 0.3
    #: 두상에 붙은 가닥은 덜 흔든다: (표면에서 이 cm 안이면 붙음, 이 cm 밖이면 자유, 붙은 가닥의 흔들림 비율).
    #: 볼을 감싼 옆머리가 통째로 바깥으로 떨어져 턱선이 드러났다(사용자 피드백). groom_dyn_collide 와 같이 켜진다.
    groom_dyn_hug: tuple[float, ...] = (0.8, 3.0, 0.3)
    #: 사용자 머리가 안 보일 때(모자/삭발) 쓰는 기본 머리색 (BGR).
    groom_default_bgr: tuple[float, ...] = (28.0, 32.0, 40.0)
    #: GPU 에 올려둘 에셋 텐서 상한(개). 512^2 RGBA float32 = 약 4MB/개.
    asset_cache_max: int = 48
    #: 세션 하나가 만들 수 있는 에셋 상한. 라이브 뱅크 1회가 7칸이다.
    session_asset_max: int = 32
    #: 세션에서 생성된 에셋을 여기에 저장해 재시작 후에도 살린다.
    generated_dir: str = os.path.join(ROOT, "assets_generated")
    #: 생성 에셋 저장소 상한(MB). 넘으면 오래된 것부터 지운다.
    generated_dir_max_mb: int = 512

    # ---------- 얼굴 포즈 ----------
    #: 이보다 정면이면 거리 캘리브레이션을 갱신한다.
    frontal_yaw_deg: float = 10.0
    cal_alpha: float = 0.08

    # ---------- GAN (HairFastGAN) ----------
    #: 2026-09-19 부터 헤어는 3D 그룸(groom_renderer.py) 이 기본이고 GAN 2D 경로는 쓰지 않는다.
    #: False 면 GAN 워커를 띄우지 않고(VRAM 5.5GB 절약) 촬영/라이브 뱅크 명령은 거절한다.
    #: 코드는 남겨 둔다 - HEDDY_GAN_ENABLED=1 로 되살릴 수 있다.
    gan_enabled: bool = False
    #: "process" = 별도 프로세스에서 실행(권장). 실시간 경로가 GAN 로딩/추론에
    #: 절대 막히지 않고, GAN 이 죽어도 서버가 안 죽는다.
    #: "thread"  = 기존처럼 같은 프로세스의 전용 스레드.
    #: 프로세스 기동에 실패하면 자동으로 thread 로 폴백한다.
    gan_backend: str = "process"
    #: 모델 적재 대기 상한(초). 실측 ~90초.
    gan_load_timeout_s: float = 420.0
    #: 합성 1회 대기 상한(초). 실측 ~9초.
    gan_call_timeout_s: float = 240.0
    #: FFHQ 정렬을 MediaPipe + GPU 워프로 할지. 끄면 예전 dlib + PIL 경로.
    #:
    #: HairFastGAN 의 align_face 가 파이프라인 최대 병목이었다(실측 1,068ms -
    #: GAN 네트워크보다 오래 걸린다). 68점 중 실제로 쓰는 건 눈 2점과 입꼬리
    #: 2점뿐이고, 사각형이 닮음변환이라 PIL 체인 전체가 grid_sample 한 번으로
    #: 대체된다. 실측: 2,083ms -> 5.6ms. 참고사진 2장 정렬이 6.10s -> 0.24s.
    #:
    #: GAN 최종 출력을 눈으로 비교해 얼굴/헤어가 동등함을 확인했다. 크롭이
    #: 2.3% 정도 달라지는데(랜드마크 출처가 다르므로) 품질 차이는 아니다.
    #: 문제가 생기면 0 으로 되돌린다 - dlib 경로가 그대로 남아 있다.
    gan_fast_align: bool = True
    #: 파인튜닝한 Rotate 체크포인트 경로. 빈 문자열이면 HairFastGAN 기본값
    #: (pretrained_models/Rotate/rotate_best.pth)을 쓴다.
    #:
    #: Rotate 는 참고사진의 머리를 사용자 얼굴 각도로 돌리는 모듈이라
    #: (models/Alignment.py:59) 큰 각도 품질을 좌우한다. 홀드아웃 600쌍 실측:
    #:     |dyaw| 30~40도  16.29 -> 15.42  (-5.3%)
    #:     |dyaw| 40도+    20.06 -> 19.15  (-4.5%)
    #: 회전량(rot_norm)과 정체성(id_cos)은 그대로라 트레이드오프가 아니다.
    #: 학습/평가 도구는 server/train/finetune/ 참고.
    gan_rotate_checkpoint: str = ""
    #: A/B 비교용 후보. GET/POST /model 이 이걸 "finetuned" 로 노출한다.
    #: 파일이 없으면 그냥 목록에서 빠진다(에러 아님).
    #:
    #: gan_rotate_checkpoint 와 나눈 이유: 저건 "기동 시 무엇을 올릴까"이고
    #: 이건 "런타임에 무엇과 비교할까"다. 하나로 합치면 기본 모델로 띄웠을 때
    #: 비교 대상이 사라져서 토글 자체가 불가능해진다.
    gan_rotate_finetuned: str = (
        "server/train/finetune/runs/HairFast-Rotate_control-random/step003000.pth")
    #: GAN 이 도는 동안에도 실시간 영상을 계속 내보낼지.
    #: 예전에는 검은 프레임을 흘려보냈다. GAN 이 같은 프로세스에 있어서 이벤트
    #: 루프와 GPU 를 통째로 잡아먹었기 때문인데, 별도 프로세스로 분리한 지금은
    #: 실시간 경로가 막히지 않는다. 하드웨어 경합으로 fps 는 떨어지지만, 검은
    #: 화면보다는 느린 화면이 낫다.
    #:
    #: 기본 False 로 되돌렸다. 같은 4070 에서 GAN 7칸(약 1분)을 도는 동안
    #: 실시간 경로가 GPU 를 나눠 쓰면 양쪽 다 느려진다. 그 1분 동안 사용자는
    #: 진행률만 보면 되고(클라이언트가 오버레이를 덮는다) 영상을 볼 이유가 없다.
    #: False 면 세그멘테이션 전에 빠져나가므로 GPU 를 아예 안 쓴다.
    #: 진행률은 어느 쪽이든 DataChannel 로 계속 나간다.
    stream_during_gan: bool = False
    #: 머리 위/양옆 여백 비율. 웹캠 클로즈업은 여백이 부족해 블렌딩이 배경을 침범한다.
    pad_ratio: float = 0.35
    #: 눈 간격 대비 크롭 반폭.
    crop_half_eyes: float = 2.6
    #: dlib 이 안정적으로 검출하는 눈 간격(px).
    target_eye_px: int = 130

    #: 업로드된 참고사진을 두는 곳. server/references/ 를 그대로 쓰지 않는 이유는
    #: 그 디렉터리가 git 에 추적되기 때문이다 - 업로드는 남의 얼굴 사진이라
    #: 실수로 커밋되면 안 된다. 여기는 .gitignore 로 막혀 있다.
    reference_upload_dir: str = os.path.join(ROOT, "references", "uploads")
    #: 업로드 1장의 최대 크기(MB).
    reference_max_mb: float = 20.0

    #: 클라이언트에 내려줄 이미지 URL 의 앞부분. 비우면 요청의 Host 로 만든다.
    #:
    #: 네이티브 앱은 상대경로를 해석할 기준이 없으므로 절대 URL 이 필요하다.
    #: 리버스 프록시나 도메인 뒤에 있으면 요청 Host 가 내부 주소일 수 있어
    #: 그때 여기에 공개 주소를 박는다. 예: https://heddy.example.com
    public_base_url: str = ""
    #: 헤어스타일 원형 미리보기의 한 변(px).
    reference_thumb_px: int = 320
    #: 썸네일 크롭 반폭(눈 간격 배수).
    #:
    #: 실측한 눈-정수리가 1.84, 눈-턱이 1.6 이지만 그 값으로 자르면 **머리가
    #: 잘린다.** 정수리는 두피 기준이고 헤어는 그 위로 더 솟기 때문이다.
    #: 헤어스타일 미리보기에서 헤어가 잘리면 아무 소용이 없어 여유를 크게 둔다.
    #: 원형으로 표시되면 모서리가 어차피 잘리므로 조금 넉넉한 편이 낫다.
    reference_thumb_half_eyes: float = 2.2
    #: 참고사진의 최소 눈 간격(px). 이보다 작으면 경고와 함께 거절하되,
    #: 업로드에 force 를 주면 통과한다.
    #:
    #: GAN 은 참고사진을 FFHQ 1024(눈 간격 ~256px)로 정렬해서 쓴다. 눈 간격이
    #: 작으면 그만큼 업스케일되어 헤어스타일 디테일이 뭉개진 채로 들어간다.
    #: 실측: 기존 korean-frontal.png 은 눈 간격이 50px 밖에 안 돼 5배 업스케일된다
    #: (korean-layered.png 은 320px). 그래서 결과가 흐릿했다.
    #:
    #: 처음엔 120 이었는데 상반신/전신 헤어 사진을 통째로 막아서 60 으로 내렸다.
    #: 이 가드의 목적은 **알려주는 것**이지 막는 것이 아니다. 판정에는 yaw 를
    #: 감안한 정면 등가값을 쓴다 - 눈 간격은 투영값이라 고개를 돌린 사진에서는
    #: 얼굴이 충분히 커도 cos(yaw) 만큼 짧게 잡힌다.
    reference_min_eye_px: int = 60
    #: 이보다 작으면 경고만 한다(거절하지는 않음).
    reference_warn_eye_px: int = 200

    #: GAN 이 만든 헤어에 곱하는 크기 보정.
    #:
    #: HairFastGAN 은 머리보다 큰 헤어를 만드는 경향이 있다. 실측으로 재봤다 -
    #: 정면 프레임 44장을 세그멘테이션해 본인 실제 머리와 비교하니
    #:     정수리 높이  본인 1.84 / 에셋 2.05  -> 0.901
    #:     폭          본인 3.10 / 에셋 3.52  -> 0.880
    #: 평균 0.891. 이건 사람/스타일마다 다를 수 있는 취향값이라 환경변수로 뺀다.
    #: 참고 스타일이 원래 풍성한 머리라면 1.0 에 가깝게 올리면 된다.
    gan_asset_scale: float = 0.891
    #: GAN 에 넣기 전에 앞머리(눈썹선 위 1.3D 이내의 머리)를 피부색으로 지울지.
    #:
    #: 앞머리 사용자에게 이마 노출 스타일을 입히면 GAN 이마에 앞머리 그림자
    #: 얼룩이 남는다(SEAN 인페인팅 -> e4e 재인코딩 경로의 한계). 미리 지우면
    #: 그 자리가 원본의 평평한 피부 FS 인코딩에서 오므로 깨끗해진다.
    #: 실측 4각도 모두 개선, 헤어라인은 참고 스타일대로 새로 그려진다.
    #: 근거와 파라미터는 gan_input.py.
    gan_prefill_forehead: bool = True

    # ---------- 라이브 뱅크 ----------
    #: 검출기(dlib CNN) 한계가 ±40도라 그 안쪽에서 12도 간격으로 잡는다.
    live_targets: tuple[float, ...] = (-36.0, -24.0, -12.0, 0.0, 12.0, 24.0, 36.0)
    #: 목표 각도 허용 오차(도).
    #: 7 -> 9. 칸 간격이 12도라 인접 범위가 조금 겹치는데, match() 가 가장
    #: 가까운 빈 칸을 고르므로 문제되지 않는다. 넓혀도 안전한 이유는 에셋에
    #: **측정된 yaw** 를 기록하기 때문이다(예전에는 목표 각도를 적어서, 실제와
    #: 벌어지면 런타임에 엉뚱한 칸이 붙었다).
    live_tol: float = 9.0
    #: 측정 yaw 가 EMA 에서 이만큼 벌어져 있으면 아직 움직이는 중.
    #: 3 -> 6. 3도는 거의 정지를 요구해서 "돌렸는데 안 잡힌다"의 주범이었다.
    #: 6도면 천천히 움직이는 중에도 잡히고, 그 정도 모션블러는 GAN 입력으로
    #: 쓸 만하다.
    live_steady: float = 6.0
    #: 칸마다 이만큼의 후보 프레임 중 **가장 선명한** 것을 GAN 입력으로 쓴다.
    #:
    #: 예전에는 허용 범위에 처음 들어온 프레임을 그대로 잡았다. 정지 판정
    #: (live_steady)은 yaw 변화만 보므로 상하 흔들림/초점 흔들림에 의한 모션
    #: 블러는 못 거른다. GAN 입력이 흐리면 e4e/FS 인버전이 그 흐림을 그대로
    #: 배워 결과 전체가 뭉갠다. 후보 N 장 중 얼굴 크롭의 라플라시안 분산이
    #: 최대인 프레임을 고른다(프레임당 ~0.3ms). 사용자가 N 장을 채우기 전에
    #: 그 각도를 벗어나면 그때까지의 최선으로 확정한다. 1 이면 예전 동작.
    live_pick_frames: int = 8
    #: 수집 단계에서 세그멘테이션을 몇 프레임에 한 번만 돌릴지.
    #:
    #: 수집 중에는 화면에 원본을 그대로 내보내므로 렌더용 분할이 필요 없다.
    #: GPU 를 쓰는 유일한 이유가 배경 플레이트 누적인데, 플레이트는 시간에
    #: 걸쳐 쌓는 것이라 매 프레임일 필요가 없다. 6프레임마다면 GPU 부하가
    #: 1/6 로 줄고 30fps 기준 초당 5장씩 쌓여 수집 시간(수십 초) 안에 충분히
    #: 찬다. 얼굴 각도는 CPU(MediaPipe)라 이 값과 무관하게 매 프레임 나간다.
    #:
    #: 0 이나 1 이면 매 프레임 돌린다(예전 동작).
    collect_seg_every: int = 6

    # ---------- 프레임 루프 ----------
    #: 소수. --infer-every-n 과 배수 관계가 생기면 로그 통계가 편향된다.
    log_every: int = 31
    #: 학습 데이터 수집 시 이 프레임 수마다 한 장씩 저장.
    rec_every: int = 5
    #: GPU 워커 대기열이 전 세션 합쳐 이만큼 밀려 있으면 새 프레임을 버린다(최신 우선).
    #: 세션 하나는 늘 1장까지만 - 버퍼는 지연을 줄이지 못한다(밀린 프레임을 계속 처리하면 지연만 누적).
    #: 예전 1 은 "GPU 에 다른 세션 프레임이 있으면 버림"이라, 2세션 1080p 에서 GPU 40%·CPU 32% 인데
    #: 세션당 새 프레임이 10.8/7.3fps(3세션 2.0/2.8/13.3)로 무너지고 차례를 못 받은 세션만 굶었다(실측).
    #: 세션 수 이상이면 각 세션이 1장씩 줄을 서서 gpu_executor 가 차례로 처리한다.
    max_inflight_frames: int = 4
    #: 파이썬 GIL 을 스레드 사이에 넘기는 주기(ms, sys.setswitchinterval). 0 이면 파이썬 기본(5ms).
    #: 합성(gpu_executor)은 프레임당 커널 ~2,400개를 파이썬에서 내는데, aiortc 루프·포즈·디코더 스레드가 GIL 을
    #: 잡으면 매번 최대 5ms 씩 기다린다. 1080p 그룸 합성 49ms 가 40% 바쁜 파이썬 스레드 1개 옆에서 58ms,
    #: 2개 옆에서 112ms 였고, 0.2ms 로 줄이면 51/54ms 였다(오프라인 실측). 서버 안 75ms vs 밖 52ms 의 주원인.
    gil_switch_ms: float = 0.2
    #: 이번 프레임을 인코딩하는 동안 다음 프레임을 미리 합성한다(server.SegmentedVideoTrack.recv).
    #: 끄면 예전처럼 받기 -> 합성 -> 인코딩이 한 줄로 이어진다.
    pipeline_frames: bool = True

    # ---------- 디스크 쿼터 ----------
    #: 클라이언트가 DataChannel 명령 하나로 디스크를 무한히 채울 수 있으면 안 된다.
    capture_dir_max_mb: int = 512
    record_dir_max_mb: int = 2048
    record_max_frames: int = 5000

    # ---------- 관측 ----------
    metrics_enabled: bool = True
    #: 얼굴 프레임을 디스크에 남기는 기능(학습 데이터 수집)을 켤지.
    #: 생체정보라 기본은 끄고, 명시적으로 켜야 쓰이게 한다.
    allow_frame_recording: bool = False


_TRUE = frozenset(("1", "true", "yes", "on", "y", "t"))
_FALSE = frozenset(("0", "false", "no", "off", "n", "f"))


def _coerce(raw: str, ftype):
    origin = typing.get_origin(ftype)
    if ftype is bool:
        # 모르는 값을 False 로 넘기지 않는다.
        # 예전엔 `raw in _TRUE` 한 줄이라 오타가 조용히 False 가 됐다. 그런데
        # 이 방식의 bool 은 대부분 **기본이 꺼짐인 안전장치**를 켜는 데 쓰인다
        # (allow_frame_recording 처럼). 오타 하나로 "켰다고 생각했는데 안 켜진
        # 채로 도는" 실패 모드가 생기고, 그건 로그에도 안 남는다.
        v = raw.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
        raise ValueError(f"bool 로 읽을 수 없는 값: {raw!r} "
                         f"(허용: {', '.join(sorted(_TRUE | _FALSE))})")
    if ftype is int:
        return int(raw)
    if ftype is float:
        return float(raw)
    if origin is tuple:
        (inner,) = {a for a in typing.get_args(ftype) if a is not Ellipsis} or {str}
        return tuple(_coerce(p, inner) for p in raw.split(",") if p.strip())
    return raw


#: 배포별 값(다른 서버 주소 등)을 두는 파일. 커밋하지 않는다(.gitignore) - 형식은 .env.example.
#: supervise.py 는 띄울 때의 환경변수를 자식에게 물려줄 뿐이라, 재부팅 후 환경변수를 빠뜨리면
#: 조용히 예전 설정으로 돈다. 파일로 두면 서버가 뜰 때마다 스스로 읽는다.
ENV_FILE = os.environ.get("HEDDY_ENV_FILE", os.path.join(ROOT, ".env"))


def read_env_file(path) -> dict:
    """KEY=VALUE 줄을 읽는다. 빈 줄/# 주석/앞의 'export ' 는 건너뛰고 값의 따옴표는 벗긴다.
    파일이 없으면 빈 dict."""
    out = {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return out
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, val = line.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"{path}:{n} KEY=VALUE 형식이 아닙니다: {line!r}")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[key.strip()] = val
    return out


def load_config(env=None) -> Config:
    """환경변수(HEDDY_*)로 덮어쓴 설정을 만든다.

    env 를 안 주면 ENV_FILE 위에 실제 환경변수를 덮어쓴 것을 쓴다(환경변수가 이긴다 -
    한 번만 바꿔 띄워 보는 `HEDDY_X=... python server.py` 가 파일보다 우선해야 한다).

    잘못된 값은 조용히 무시하지 않고 예외를 낸다 - 오타 난 튜닝 값으로 몇 시간
    실험하는 것보다 시작할 때 죽는 쪽이 낫다.
    """
    env = {**read_env_file(ENV_FILE), **os.environ} if env is None else env
    hints = typing.get_type_hints(Config)
    over = {}
    for f in fields(Config):
        raw = env.get("HEDDY_" + f.name.upper())
        if raw is None:
            continue
        try:
            over[f.name] = _coerce(raw, hints[f.name])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"환경변수 HEDDY_{f.name.upper()}={raw!r} 를 "
                f"{hints[f.name]} 로 읽을 수 없습니다") from e
    return Config(**over)


CONFIG = load_config()
