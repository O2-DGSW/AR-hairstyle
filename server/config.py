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
    #: 동시에 받을 수 있는 피어 수. GPU 워커가 1개라 초과분은 거절하는 편이
    #: 전부 같이 느려지는 것보다 낫다.
    max_sessions: int = 2
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

    #: 서버가 내보낼 비디오 코덱. "vp8" | "h264" | "auto"
    #:
    #: aiortc 가 지원하는 건 VP8 과 H.264 둘뿐이고, 코덱은 **브라우저 offer 의
    #: 순서**대로 정해진다. Chrome 이 H.264 를 앞에 두면 그쪽으로 붙는데, 그러면
    #: 패킷은 도착하는데 framesDecoded 가 0 인 채로 검은 화면이 된다(실측).
    #: aiortc 의 H.264 출력에서 Chrome 이 디코딩 가능한 키프레임을 못 얻는
    #: 것으로 보인다. VP8 로 고정하면 정상 재생된다.
    #: "auto" 는 예전 동작(브라우저가 고르는 대로)이다.
    video_codec: str = "vp8"

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
    #: 헤어라인 그림자 띠의 폭(픽셀).
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
    #: 진단용으로 되살리려면 HEDDY_SERVE_STATIC_ASSETS=1.
    serve_static_assets: bool = False
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
    #: "process" = 별도 프로세스에서 실행(권장). 실시간 경로가 GAN 로딩/추론에
    #: 절대 막히지 않고, GAN 이 죽어도 서버가 안 죽는다.
    #: "thread"  = 기존처럼 같은 프로세스의 전용 스레드.
    #: 프로세스 기동에 실패하면 자동으로 thread 로 폴백한다.
    gan_backend: str = "process"
    #: 모델 적재 대기 상한(초). 실측 ~90초.
    gan_load_timeout_s: float = 420.0
    #: 합성 1회 대기 상한(초). 실측 ~9초.
    gan_call_timeout_s: float = 240.0
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
    #: 화면보다는 느린 화면이 낫다. 경합이 심하면 False 로 되돌린다.
    stream_during_gan: bool = True
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

    # ---------- 라이브 뱅크 ----------
    #: 검출기(dlib CNN) 한계가 ±40도라 그 안쪽에서 12도 간격으로 잡는다.
    live_targets: tuple[float, ...] = (-36.0, -24.0, -12.0, 0.0, 12.0, 24.0, 36.0)
    #: 목표 각도 허용 오차(도).
    live_tol: float = 7.0
    #: 측정 yaw 가 EMA 에서 이만큼 벌어져 있으면 아직 움직이는 중.
    live_steady: float = 3.0

    # ---------- 프레임 루프 ----------
    #: 소수. --infer-every-n 과 배수 관계가 생기면 로그 통계가 편향된다.
    log_every: int = 31
    #: 학습 데이터 수집 시 이 프레임 수마다 한 장씩 저장.
    rec_every: int = 5
    #: GPU 워커 대기열이 이만큼 밀려 있으면 새 프레임을 버린다(최신 우선).
    #: 버퍼는 지연을 줄이지 못한다 - 밀린 프레임을 계속 처리하면 지연만 누적된다.
    max_inflight_frames: int = 1

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


def load_config(env=None) -> Config:
    """환경변수(HEDDY_*)로 덮어쓴 설정을 만든다.

    잘못된 값은 조용히 무시하지 않고 예외를 낸다 - 오타 난 튜닝 값으로 몇 시간
    실험하는 것보다 시작할 때 죽는 쪽이 낫다.
    """
    env = os.environ if env is None else env
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
