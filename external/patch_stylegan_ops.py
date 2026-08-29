"""HairFastGAN의 StyleGAN2 커스텀 CUDA 연산을 '컴파일 없이도' 돌게 패치한다.

배경
----
HairFastGAN/models/stylegan2/op/{fused_act,upfirdn2d}.py 는 import 시점에
torch.utils.cpp_extension.load(...) 로 CUDA 확장을 **무조건 JIT 컴파일**한다.
try/except 가 없어서 툴체인이 없거나 nvcc-MSVC 버전이 안 맞으면 import 자체가
죽는다. (이 PC에서는 CUDA 12.4의 cudafe++ 가 MSVC 19.51 헤더에서
0xC0000409 로 죽었다. CUDA 12.4의 지원 상한은 MSVC 19.39.)

두 파일 모두 **순수 PyTorch 네이티브 구현**을 이미 갖고 있다
(upfirdn2d_native, fused_leaky_relu 의 cpu 분기). F.pad / F.leaky_relu /
reshape 만 쓰므로 CUDA 텐서에서도 그대로 동작한다. 다만 분기 조건이
`input.device.type == "cpu"` 로 되어 있어 GPU에서는 절대 안 탄다.

이 패치가 하는 일
-----------------
1. load() 를 try/except 로 감싸 실패 시 None 으로 두고 계속 진행
2. 분기 조건에 "컴파일된 확장이 없으면" 을 추가해 GPU에서도 네이티브를 타게 함
3. 네이티브 leaky_relu 가 negative_slope 인자를 무시하고 0.2 를 하드코딩한
   버그도 함께 고침

성능
----
융합 커널 대비 느리다(대략 1.5~3배). HairFastGAN 은 프레임당이 아니라
**오프라인 정지 이미지**(에셋 생성 / 고화질 캡처)에만 쓰므로 감수할 만하다.
나중에 툴체인이 갖춰지면 컴파일이 성공해 자동으로 빠른 경로를 탄다 -
이 패치는 폴백을 여는 것이지 컴파일을 막지 않는다.

사용: python patch_stylegan_ops.py [--revert]
"""
import argparse
import os
import shutil
import sys

OP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "HairFastGAN", "models", "stylegan2", "op")

PATCHES = [
    # (파일, 원본, 치환)
    ("fused_act.py",
     'fused = load(\n'
     '    "fused",\n'
     '    sources=[\n'
     '        os.path.join(module_path, "fused_bias_act.cpp"),\n'
     '        os.path.join(module_path, "fused_bias_act_kernel.cu"),\n'
     '    ],\n'
     ')',
     '# [heddy patch] 컴파일 실패해도 죽지 않고 네이티브 폴백으로 넘어간다.\n'
     'try:\n'
     '    fused = load(\n'
     '        "fused",\n'
     '        sources=[\n'
     '            os.path.join(module_path, "fused_bias_act.cpp"),\n'
     '            os.path.join(module_path, "fused_bias_act_kernel.cu"),\n'
     '        ],\n'
     '    )\n'
     'except Exception as _e:\n'
     '    fused = None\n'
     '    print(f"[heddy] fused CUDA 확장 컴파일 실패 -> 네이티브 PyTorch 사용: {_e}")'),

    ("fused_act.py",
     'def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):\n'
     '    if input.device.type == "cpu":\n'
     '        rest_dim = [1] * (input.ndim - bias.ndim - 1)\n'
     '        return (\n'
     '            F.leaky_relu(\n'
     '                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2\n'
     '            )\n'
     '            * scale\n'
     '        )',
     'def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):\n'
     '    # [heddy patch] 확장이 없으면 GPU 텐서라도 네이티브 경로를 쓴다.\n'
     '    if fused is None or input.device.type == "cpu":\n'
     '        rest_dim = [1] * (input.ndim - bias.ndim - 1)\n'
     '        return (\n'
     '            F.leaky_relu(\n'
     '                input + bias.view(1, bias.shape[0], *rest_dim),\n'
     '                negative_slope=negative_slope,\n'
     '            )\n'
     '            * scale\n'
     '        )'),

    ("upfirdn2d.py",
     'upfirdn2d_op = load(\n'
     '    "upfirdn2d",\n'
     '    sources=[\n'
     '        os.path.join(module_path, "upfirdn2d.cpp"),\n'
     '        os.path.join(module_path, "upfirdn2d_kernel.cu"),\n'
     '    ],\n'
     ')',
     '# [heddy patch] 컴파일 실패해도 죽지 않고 네이티브 폴백으로 넘어간다.\n'
     'try:\n'
     '    upfirdn2d_op = load(\n'
     '        "upfirdn2d",\n'
     '        sources=[\n'
     '            os.path.join(module_path, "upfirdn2d.cpp"),\n'
     '            os.path.join(module_path, "upfirdn2d_kernel.cu"),\n'
     '        ],\n'
     '    )\n'
     'except Exception as _e:\n'
     '    upfirdn2d_op = None\n'
     '    print(f"[heddy] upfirdn2d CUDA 확장 컴파일 실패 -> 네이티브 PyTorch 사용: {_e}")'),

    ("upfirdn2d.py",
     'def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):\n'
     '    if input.device.type == "cpu":',
     'def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):\n'
     '    # [heddy patch] 확장이 없으면 GPU 텐서라도 네이티브 경로를 쓴다.\n'
     '    if upfirdn2d_op is None or input.device.type == "cpu":'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true", help=".orig 백업으로 되돌린다")
    args = ap.parse_args()

    if not os.path.isdir(OP_DIR):
        print(f"경로를 찾을 수 없음: {OP_DIR}")
        return 1

    if args.revert:
        n = 0
        for fn in ("fused_act.py", "upfirdn2d.py"):
            bak = os.path.join(OP_DIR, fn + ".orig")
            if os.path.isfile(bak):
                shutil.copyfile(bak, os.path.join(OP_DIR, fn))
                n += 1
        print(f"되돌림: {n}개 파일")
        return 0

    applied = skipped = 0
    for fn, old, new in PATCHES:
        path = os.path.join(OP_DIR, fn)
        bak = path + ".orig"
        if not os.path.isfile(bak):
            shutil.copyfile(path, bak)

        with open(path, encoding="utf-8") as f:
            src = f.read()

        # 판정은 "원본 코드가 아직 남아있는가" 하나로만 한다.
        # (치환문의 첫 줄이 함수 시그니처처럼 변하지 않는 부분일 수 있어서,
        #  치환문 존재 여부로 판정하면 미적용을 적용됨으로 오판한다)
        if old not in src:
            skipped += 1
            continue

        with open(path, "w", encoding="utf-8") as f:
            f.write(src.replace(old, new, 1))
        applied += 1

    print(f"패치 적용 {applied}건, 건너뜀 {skipped}건")
    print(f"백업: {OP_DIR}\\*.py.orig  (되돌리려면 --revert)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
