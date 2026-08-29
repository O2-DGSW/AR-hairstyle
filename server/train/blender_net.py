"""실시간 블렌딩 네트워크 — 워핑 합성본을 GAN 품질로 끌어올리는 소형 UNet.

설계 제약이 구조를 결정한다
---------------------------
프레임 예산이 33ms인데 세그멘테이션이 이미 13ms를 쓴다. 여기에 크롭/역크롭과
여유까지 감안하면 **이 네트워크는 5ms 안에 끝나야 한다.** 그래서 화질보다
속도를 우선한 선택들:

  - 256x256 고정 입력 (눈 기준 정규화 크롭)
  - 채널 수를 작게(base=24), 다운샘플 3단계
  - 잔차 학습(residual): 출력 = 입력 + 보정량.
    합성본은 이미 대체로 맞으므로 '차이'만 배우면 된다. 처음부터 이미지를
    새로 그리게 하는 것보다 훨씬 적은 용량으로 수렴하고, 학습 초기에도
    입력을 그대로 통과시키므로 결과가 무너지지 않는다.
  - 출력에 tanh 대신 clamp: 잔차 범위를 굳이 포화시킬 이유가 없다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class BlenderUNet(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        b = base
        self.e1 = conv_block(3, b)
        self.e2 = conv_block(b, b * 2)
        self.e3 = conv_block(b * 2, b * 4)
        self.bott = conv_block(b * 4, b * 4)
        self.d3 = conv_block(b * 8, b * 2)
        self.d2 = conv_block(b * 4, b)
        self.d1 = conv_block(b * 2, b)
        self.out = nn.Conv2d(b, 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)   # 초기 잔차 0 -> 학습 시작 시 입력 그대로

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        bt = self.bott(F.max_pool2d(e3, 2))

        d3 = self.d3(torch.cat([F.interpolate(bt, scale_factor=2, mode="nearest"), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))
        return (x + self.out(d1)).clamp(0.0, 1.0)


if __name__ == "__main__":
    import time
    m = BlenderUNet().cuda().eval().half()
    x = torch.randn(1, 3, 256, 256, device="cuda", dtype=torch.half)
    n = sum(p.numel() for p in m.parameters())
    with torch.no_grad():
        for _ in range(10):
            m(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(50):
            m(x)
        torch.cuda.synchronize()
    print(f"파라미터 {n/1e6:.2f}M  |  추론 {(time.perf_counter()-t)/50*1000:.2f} ms @256")
