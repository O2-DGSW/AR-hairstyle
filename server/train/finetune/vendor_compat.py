"""vendored HairFastGAN 학습 코드를 현재 환경에서 import 가능하게 만든다.

external/ 은 .gitignore 로 통째로 제외돼 있어서 그 안을 직접 고치면 재현이
안 된다(다시 클론하면 사라진다). 그래서 수정 대신 **import 직전에 심을
주입**한다. patch_stylegan_ops.py 가 파일을 고치는 것과 달리 여기서는
프로세스 안에서만 손댄다.

반드시 `import scripts.*` **이전에** prepare() 를 부를 것.
"""
import os
import sys
import types


def install_wandb_stub():
    """학습 스크립트가 최상단에서 `import wandb` 를 한다.

    wandb 를 실제로 설치하지 않는 이유:
      1. wandb 0.29 가 protobuf>=5 를 끌어오는데 mediapipe 0.10.14 는
         protobuf<5 를 요구한다. 같은 venv 에 공존할 수 없다(실제로 설치했다가
         mediapipe 가 깨졌고 되돌렸다).
      2. WandbLogger.start_logging() 이 os.environ['WANDB_KEY'] 를 강제해
         계정 없이는 학습이 시작조차 안 된다.
    로깅은 locallog.LocalLogger 가 대신한다.
    """
    if "wandb" in sys.modules:
        return
    m = types.ModuleType("wandb")

    class _Image:                      # wandb.Image(pil) -> .image 로 원본 보관
        def __init__(self, image, *a, **k):
            self.image = image

    m.Image = _Image
    m.login = lambda *a, **k: None
    m.init = lambda *a, **k: None
    m.log = lambda *a, **k: None
    m.save = lambda *a, **k: None
    m.finish = lambda *a, **k: None
    m.run = types.SimpleNamespace(dir=".")
    sys.modules["wandb"] = m


def install_scipy_compat():
    """models/STAR/lib/metric/fr_and_auc.py 가 scipy.integrate.simps 를 쓴다.

    scipy 1.14 에서 제거되고 simpson 으로 이름이 바뀌었다. 별칭만 되살린다.
    (STAR 는 랜드마크 추출기이고, 이 함수는 NME/AUC 평가 지표 쪽이라
     우리 경로에서는 실제로 호출되지 않는다 - import 만 통과하면 된다.)
    """
    import scipy.integrate as si
    if not hasattr(si, "simps") and hasattr(si, "simpson"):
        si.simps = si.simpson


def install_star_landmarks_arg(hf_dir):
    """scripts/rotate_train.py 가 참조하는 utility.landmarks_arg 를 복원한다.

    업스트림에 이 정의가 **없다**. 레포 전체에서 rotate_train.py 두 줄
    (:77, :79)만 참조하고 어디에도 선언이 없다 - 저자가 로컬에만 두고
    커밋을 빠뜨린 것으로 보인다. 그래서 학습 스크립트가 그대로는 안 돈다.

    복원에 필요한 건 두 개뿐이다:
      config_name       get_config() 가 분기에 쓴다
      pretrained_weight rotate_train 이 직접 읽어 체크포인트를 연다
    나머지 설정은 conf/Alignment 기본값으로 채워진다(base.init_from_args 는
    이미 존재하는 속성만 덮어쓰므로 여분의 키를 넣어도 무해하다).

    동봉된 가중치가 WFLW 학습본이라 data_definition 기본값 'WFLW' 와 맞고,
    generate_key_points 가 98점 중 앞 76점을 쓰는 것과도 일치한다.

    반드시 `from scripts.rotate_train import ...` **이후**, Trainer() 생성
    **이전에** 부를 것 (속성은 import 시점이 아니라 생성 시점에 읽힌다).
    """
    from models.STAR.lib import utility
    if hasattr(utility, "landmarks_arg"):
        return
    utility.landmarks_arg = types.SimpleNamespace(
        config_name="alignment",
        pretrained_weight=os.path.join(
            hf_dir, "pretrained_models", "STAR",
            "WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl"),
    )


def prepare():
    install_wandb_stub()
    install_scipy_compat()
