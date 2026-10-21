"""그룸 후처리의 원소별 연산 묶음을 torch.compile(Triton)로 커널 하나씩으로 합친다.

왜: 1080p 그룸 합성은 프레임당 CUDA 커널 ~2,400개라 해상도를 1/4 로 줄여도 32ms 밑으로 안 내려갔다
(540x960 32.4ms vs 1080p 52.3ms, 오프라인 실측) - 곱/더하기/clamp 하나하나가 따로 전체 해상도를 읽고
쓰고, 파이썬이 그걸 하나씩 낸다. 같은 식을 커널 하나로 합치면 읽기/쓰기와 런치가 같이 준다
(원소별 6연산 묶음 1080p: 1.88ms -> 0.18ms).

컴파일하는 건 **커널 크기 같은 정수 인자가 없는 원소별 식만**이다. 풀링 커널 크기는 눈 간격에 따라
바뀌어서, 함수 통째로 컴파일하면 세션 중에 재컴파일(수 초 멈춤)이 난다. 실수 인자와 텐서 크기는
동적(specialize_float=False, dynamic=True)이라 처음 몇 번만 컴파일된다 - warm() 으로 서버 기동 때 미리.

config.fused_compile 이 꺼져 있거나 컴파일이 실패하면(triton/컴파일러 없음 등) 같은 식을 그대로
eager 로 돈다. 결과는 같다(부동소수 오차 수준).
"""
from __future__ import annotations

import logging
import time

import torch

from config import CONFIG

logger = logging.getLogger("fused")


# ---- 식 (eager 로도 그대로 쓰인다) ----

def blend_under(rgb, a, pp, e):
    """헤어 레이어 (rgb, a) 밑(빈 자리)에 확산 색 pp 를 가중 e 만큼 깐다 -> (rgb, a). 프리멀티플라이드."""
    tot = a + e * (1.0 - a)
    return (rgb * a + pp * e * (1.0 - a)) / tot.clamp(min=1e-4), tot


def extend_weight(hair_a, a, near, below, ys, xs, ff, ahead, ey: float, cx: float, d: float,
                  eye_k: float, fh_w: float, gain: float):
    """_groom_extend 의 연장 가중 (h, w). ff/ahead 는 없으면 0 텐서."""
    above = ((ey - eye_k * d - ys) / (0.25 * d)).clamp(0.0, 1.0)
    inner = ((fh_w * d - (xs - cx).abs()) / (0.25 * d)).clamp(0.0, 1.0)
    forehead = torch.maximum(torch.maximum((below * inner).clamp(0.0, 1.0), ff), ahead)
    return hair_a * (1.0 - a) * near * above * (1.0 - forehead) * gain


def extend_final(new_rgb, new_a, pp, ext, lum, loc, detail: float):
    """_groom_extend 끝: 원래 머리 휘도 결을 확산 색에 입히고 밑에 깐다 -> (rgb 0~255, a)."""
    det = (lum / loc.clamp(min=1.0)).clamp(0.6, 1.5)
    pp = pp * (1.0 + (det - 1.0) * detail)
    rgb, tot = blend_under(new_rgb, new_a, pp, ext)
    return rgb.clamp(0, 255), tot


def pushpull_up(cur_up, c, w):
    """_pushpull 한 레벨 복원: 관측(가중 w)이 충분한 만큼 자기 값(c/w), 나머지는 위 레벨 값."""
    return torch.lerp(cur_up, c / w.clamp(min=1e-6), (w * 100.0).clamp(0.0, 1.0))


_EAGER = {"blend_under": blend_under, "extend_weight": extend_weight, "extend_final": extend_final,
          "pushpull_up": pushpull_up}
_fns = dict(_EAGER)
_state = {"tried": False, "ok": False}


def get(name: str):
    """이름의 함수 - 컴파일됐으면 그것, 아니면 eager."""
    return _fns[name]


def warm(device="cuda"):
    """서버 기동 때(gpu_executor 스레드) 미리 컴파일한다. 크기 두 벌로 두 번씩 불러 동적 그래프까지 만든다.
    실패하면 eager 로 남고 False."""
    if _state["tried"]:
        return _state["ok"]
    _state["tried"] = True
    if not CONFIG.fused_compile or not str(device).startswith("cuda"):
        return False
    t0 = time.perf_counter()
    try:
        import torch._dynamo
        torch._dynamo.config.specialize_float = False
        comp = {k: torch.compile(f, dynamic=True) for k, f in _EAGER.items()}
        for h, w in ((96, 80), (131, 77)):
            r = lambda *s: torch.rand(*s, device=device)
            rgb, a, pp, e = r(h, w, 3) * 255, r(h, w, 1), r(h, w, 3) * 255, r(h, w, 1)
            m = [r(h, w) for _ in range(8)]
            for k, args in (("blend_under", (rgb, a, pp, e)),
                            ("extend_weight", (*m, 40.5, 33.0, 21.0, 0.3, 0.8, 1.0)),
                            ("extend_final", (rgb, a, pp, e, a, a + 0.5, 0.7)),
                            ("pushpull_up", (rgb, rgb * a, a))):
                ref = _EAGER[k](*args)
                out = comp[k](*args)
                ref = ref if isinstance(ref, tuple) else (ref,)
                out = out if isinstance(out, tuple) else (out,)
                for x, y in zip(ref, out):
                    if not torch.allclose(x, y, rtol=1e-3, atol=1e-2):
                        raise RuntimeError(f"{k}: 컴파일 결과가 eager 와 다르다")
        torch.cuda.synchronize()
        _fns.update(comp)
        _state["ok"] = True
        logger.info("fused 커널 컴파일 %.1fs (%d개)", time.perf_counter() - t0, len(comp))
    except Exception as e:
        logger.warning("fused 컴파일 실패 - eager 로 돈다: %s", e)
    return _state["ok"]
