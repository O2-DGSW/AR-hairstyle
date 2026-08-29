"""Prometheus 텍스트 노출 형식을 손으로 만드는 최소 구현.

왜 prometheus_client 를 안 쓰는가
---------------------------------
이 서버가 노출할 지표는 20개 남짓이고 전부 프로세스 하나에 갇혀 있다. 그걸
위해 의존성을 하나 더 늘리면, 배포할 때마다 GPU 드라이버/torch 버전과 맞물린
휠 목록이 길어진다. 노출 형식 자체는 텍스트 몇 줄이라 직접 만드는 비용이
의존성 하나보다 싸다.

정직성 규칙
-----------
gauge 를 histogram 이라고 선언하지 않는다. Prometheus 는 타입 선언을 믿고
rate()/histogram_quantile() 를 계산하므로, 타입을 속이면 대시보드가 조용히
거짓말을 한다. 여기 Histogram 은 진짜 누적 버킷 + _sum + _count 를 낸다.

스레드 안전성
-------------
관측(observe/증가)은 **전부 aiohttp 이벤트 루프 스레드에서만** 일어난다
(프레임 루프도 코루틴이다). 그래서 락이 없다. GPU/GAN 워커 스레드에서
직접 이걸 건드리면 그 전제가 깨지므로, 워커 결과는 항상 루프로 돌아온 뒤에
기록한다.
"""
from __future__ import annotations

#: 프레임 처리 시간용 버킷(초). 30fps 예산이 33ms 라 그 근처를 촘촘하게 둔다.
#: 버킷은 한 번 정하면 바꾸기 어렵다 - 바꾸는 순간 과거 시계열과 이어지지 않는다.
SECONDS_BUCKETS = (0.005, 0.01, 0.02, 0.033, 0.05, 0.1, 0.25, 0.5, 1.0)


def escape_help(text: str) -> str:
    """HELP 문자열 이스케이프: 역슬래시와 개행만 (따옴표는 그대로 둔다)."""
    return str(text).replace("\\", r"\\").replace("\n", r"\n")


def escape_label(value) -> str:
    """라벨 값 이스케이프.

    라벨에는 예외 메시지나 디바이스 이름처럼 **우리가 만들지 않은 문자열**이
    들어간다. 따옴표 하나만 새어 들어가도 그 지점부터 노출 전체가 파싱 불가가
    되어 지표가 통째로 사라진다(스크레이프 실패는 '값이 이상함'이 아니라
    '아무 값도 없음'으로 보인다).
    """
    return (str(value).replace("\\", r"\\")
                      .replace('"', r'\"')
                      .replace("\n", r"\n"))


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    v = float(value)
    if v != v:
        return "NaN"
    if v == float("inf"):
        return "+Inf"
    if v == float("-inf"):
        return "-Inf"
    return repr(v)


def _labels(labels) -> str:
    if not labels:
        return ""
    parts = []
    for k, v in labels.items():
        if v is None:
            continue
        parts.append('%s="%s"' % (k, escape_label(v)))
    return "{" + ",".join(parts) + "}" if parts else ""


class Histogram:
    """고정 버킷 히스토그램. 관측값은 초 단위."""

    __slots__ = ("buckets", "_counts", "count", "sum", "last")

    def __init__(self, buckets=SECONDS_BUCKETS):
        self.buckets = tuple(buckets)
        self._counts = [0] * (len(self.buckets) + 1)   # 마지막 칸은 +Inf
        self.count = 0
        self.sum = 0.0
        self.last = 0.0        # 최근값. 히스토그램으로는 "지금" 을 못 본다.

    def observe(self, seconds: float) -> None:
        v = float(seconds)
        self.count += 1
        self.sum += v
        self.last = v
        for i, b in enumerate(self.buckets):
            if v <= b:
                self._counts[i] += 1
                return
        self._counts[-1] += 1

    def cumulative(self):
        """(le, 누적개수) 목록. Prometheus 버킷은 누적이라 여기서 더한다."""
        out, acc = [], 0
        for b, c in zip(self.buckets, self._counts):
            acc += c
            out.append((b, acc))
        out.append((float("inf"), acc + self._counts[-1]))
        return out


class Exposition:
    """노출 텍스트 조립기. 같은 이름의 HELP/TYPE 은 한 번만 낸다."""

    def __init__(self):
        self._lines = []
        self._declared = set()

    def _declare(self, name, help_text, typ):
        if name in self._declared:
            return
        self._declared.add(name)
        self._lines.append("# HELP %s %s" % (name, escape_help(help_text)))
        self._lines.append("# TYPE %s %s" % (name, typ))

    def gauge(self, name, help_text, value, labels=None):
        if value is None:
            return
        self._declare(name, help_text, "gauge")
        self._lines.append("%s%s %s" % (name, _labels(labels), _fmt(value)))

    def counter(self, name, help_text, value, labels=None):
        # 이름이 _total 로 끝나야 rate() 를 쓰는 쪽에서 헷갈리지 않는다.
        if value is None:
            return
        self._declare(name, help_text, "counter")
        self._lines.append("%s%s %s" % (name, _labels(labels), _fmt(value)))

    def histogram(self, name, help_text, hist: Histogram, labels=None):
        self._declare(name, help_text, "histogram")
        base = dict(labels or {})
        for le, acc in hist.cumulative():
            lb = dict(base)
            lb["le"] = "+Inf" if le == float("inf") else _fmt(le)
            self._lines.append("%s_bucket%s %d" % (name, _labels(lb), acc))
        self._lines.append("%s_sum%s %s" % (name, _labels(base), _fmt(hist.sum)))
        self._lines.append("%s_count%s %d" % (name, _labels(base), hist.count))

    def text(self) -> str:
        # 노출은 개행으로 끝나야 한다. 없으면 일부 파서가 마지막 줄을 버린다.
        return "\n".join(self._lines) + "\n"


class Metrics:
    """서버 전체 카운터/히스토그램. 앱 하나에 하나."""

    def __init__(self):
        self.sessions_total = 0
        self.sessions_rejected_total = 0
        self.frames_total = 0
        self.frames_dropped_total = 0
        self.frame_errors_total = 0
        self.gan_swaps_total = 0
        self.gan_errors_total = 0
        self.assets_generated_total = 0
        self.infer = Histogram()
        self.process = Histogram()
        self.gan = Histogram(buckets=(1.0, 2.5, 5.0, 10.0, 20.0, 60.0, 120.0))
