/* Phase 0 - 헤어 워핑 추적 테스트
 *
 * 서버로 프레임을 보내지 않는다. MediaPipe Face Landmarker를 WebGL GPU
 * 델리게이트로 이 브라우저에서 돌리고, 얼굴 랜드마크 3점(이마중앙/좌우광대)에
 * 헤어 에셋의 앵커 3점을 맞추는 어파인 워핑으로 캔버스에 합성한다.
 *
 * 목적은 "예쁜 합성"이 아니라 추적 품질 측정이다:
 *  - jitter: 가만히 있을 때 앵커점이 프레임마다 얼마나 떨리는지 (px RMS)
 *  - yaw/pitch: 어느 각도부터 2D 어파인이 깨지는지 (Phase 2 근거)
 */

import { FaceLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// MediaPipe face mesh 인덱스
const LM_FOREHEAD = 10;   // 이마 중앙 위
const LM_CHEEK_L  = 234;  // 화면 기준 왼쪽 얼굴 윤곽
const LM_CHEEK_R  = 454;  // 화면 기준 오른쪽 얼굴 윤곽

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");

const el = (id) => document.getElementById(id);
const startBtn = el("start"), stopBtn = el("stop"), statusEl = el("status");

const ui = {
  sync: el("c-sync"),
  hair: el("c-hair"), marks: el("c-marks"), mesh: el("c-mesh"), filter: el("c-filter"),
  scale: el("c-scale"), offset: el("c-offset"), alpha: el("c-alpha"), ema: el("c-ema"),
  file: el("c-file"),
};
const stat = {
  delegate: el("s-delegate"), src: el("s-src"), fps: el("s-fps"), detect: el("s-detect"),
  age: el("s-age"), dup: el("s-dup"), late: el("s-late"), horizon: el("s-horizon"),
  face: el("s-face"),
  jitter: el("s-jitter"), yaw: el("s-yaw"), pitch: el("s-pitch"), roll: el("s-roll"),
};

["scale", "offset", "alpha", "ema"].forEach((k) => {
  const sync = () => { el("v-" + k).textContent = ui[k].value; };
  ui[k].addEventListener("input", sync);
  sync();
});

let landmarker = null;
let stream = null;
let running = false;
let rafId = null;

function setStatus(msg) { statusEl.textContent = msg; console.log("[warp]", msg); }

/* ---------- 헤어 에셋 (기본은 절차적 생성) ----------
 * 추적 테스트에는 사진보다 이쪽이 낫다: 앵커 대비 어긋남이 바로 보인다.
 * 에셋 좌표계 512x512, 앵커 3점은 아래 ASSET_ANCHOR 고정.            */
const ASSET_SIZE = 512;
const ASSET_ANCHOR = {
  forehead: { x: 256, y: 190 },
  cheekL:   { x: 86,  y: 300 },
  cheekR:   { x: 426, y: 300 },
};

function buildProceduralHair() {
  const c = document.createElement("canvas");
  c.width = c.height = ASSET_SIZE;
  const g = c.getContext("2d");

  const grad = g.createLinearGradient(0, 80, 0, 430);
  grad.addColorStop(0, "#5b3a2e");
  grad.addColorStop(0.55, "#3a241c");
  grad.addColorStop(1, "#241511");

  g.beginPath();
  // 바깥 실루엣: 왼쪽 아래 -> 정수리 -> 오른쪽 아래
  g.moveTo(44, 412);
  g.bezierCurveTo(20, 250, 78, 92, 256, 92);
  g.bezierCurveTo(434, 92, 492, 250, 468, 412);
  // 안쪽 헤어라인: 오른쪽 아래 -> 이마 -> 왼쪽 아래
  g.lineTo(410, 412);
  g.bezierCurveTo(430, 296, 398, 194, 256, 174);
  g.bezierCurveTo(114, 194, 82, 296, 102, 412);
  g.closePath();
  g.fillStyle = grad;
  g.fill();

  // 결 느낌만 살짝 (추적 시 회전 확인용 시각 단서)
  g.strokeStyle = "rgba(255,255,255,0.07)";
  g.lineWidth = 2;
  for (let i = -5; i <= 5; i++) {
    g.beginPath();
    g.moveTo(256 + i * 26, 104);
    g.quadraticCurveTo(256 + i * 44, 260, 256 + i * 52, 400);
    g.stroke();
  }
  return c;
}

let hairAsset = buildProceduralHair();

ui.file.addEventListener("change", (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  const img = new Image();
  img.onload = () => {
    // 업로드 PNG는 에셋 좌표계에 맞춰 정사각으로 눕힌다. 앵커는 기본값을
    // 그대로 쓰므로 [크기]/[상하] 슬라이더로 맞추면 된다.
    const c = document.createElement("canvas");
    c.width = c.height = ASSET_SIZE;
    c.getContext("2d").drawImage(img, 0, 0, ASSET_SIZE, ASSET_SIZE);
    hairAsset = c;
    setStatus("헤어 에셋 교체됨 — [크기]/[상하] 슬라이더로 맞추세요.");
  };
  img.onerror = () => setStatus("이미지를 읽지 못했습니다.");
  img.src = URL.createObjectURL(f);
});

/* ---------- 어파인 ----------
 * [a,b,c,d,e,f] 는 canvas setTransform 규약:
 *   x' = a*x + c*y + e ,  y' = b*x + d*y + f
 */
function solveAffine(p0, p1, p2, q0, q1, q2) {
  const ux = p1.x - p0.x, uy = p1.y - p0.y;
  const vx = p2.x - p0.x, vy = p2.y - p0.y;
  const det = ux * vy - uy * vx;
  if (Math.abs(det) < 1e-8) return null;

  const i00 = vy / det,  i01 = -vx / det;
  const i10 = -uy / det, i11 = ux / det;

  const upx = q1.x - q0.x, upy = q1.y - q0.y;
  const vpx = q2.x - q0.x, vpy = q2.y - q0.y;

  const a = upx * i00 + vpx * i10;
  const c = upx * i01 + vpx * i11;
  const b = upy * i00 + vpy * i10;
  const d = upy * i01 + vpy * i11;
  const e = q0.x - (a * p0.x + c * p0.y);
  const f = q0.y - (b * p0.x + d * p0.y);
  return [a, b, c, d, e, f];
}

// m1 을 적용한 뒤 m2 를 적용하는 합성 (= m2 ∘ m1)
function compose(m2, m1) {
  const [a2, b2, c2, d2, e2, f2] = m2;
  const [a1, b1, c1, d1, e1, f1] = m1;
  return [
    a2 * a1 + c2 * b1,
    b2 * a1 + d2 * b1,
    a2 * c1 + c2 * d1,
    b2 * c1 + d2 * d1,
    a2 * e1 + c2 * f1 + e2,
    b2 * e1 + d2 * f1 + f2,
  ];
}

/* ---------- 포즈 / 스무딩 ---------- */
function eulerFromMatrix(data) {
  // data 는 column-major 4x4
  const r00 = data[0], r10 = data[1], r20 = data[2];
  const r21 = data[6], r22 = data[10];
  const deg = 180 / Math.PI;
  return {
    pitch: Math.atan2(r21, r22) * deg,
    yaw: Math.atan2(-r20, Math.hypot(r21, r22)) * deg,
    roll: Math.atan2(r10, r00) * deg,
  };
}

/* ---------- 랜드마크 필터 ----------
 * EMA(1차 IIR)를 기본으로 쓰던 게 "천천히 따라오는" 증상의 주원인이었다.
 * k=0.55면 90% 따라잡는 데 3.85 샘플 = 20fps에서 193ms. 게다가 비디오
 * 프레임당이 아니라 루프 반복당 적용돼서, 추론이 느려지면 시정수가 같이
 * 늘어나는 양의 되먹임까지 있었다.
 *
 * 대체: 알파-베타 필터(등속도 모델의 정상상태 칼만과 수학적으로 동일하나
 * 연산은 훨씬 쌈). 위치와 속도를 같이 추정하므로
 *   - ALPHA 가 jitter를 잡고
 *   - horizon 이 지연을 지운다
 * 두 노브가 직교한다. 고정 EMA는 이게 불가능했다(빠른 움직임에도 같은 벌점).
 *
 * horizon 주의: 절대 지연(카메라 포함) 전체를 쓰면 안 된다. 그러면 헤어가
 * "실제 현재 위치"로 가는데 화면의 얼굴은 수십 ms 전이라 헤어가 얼굴보다
 * 앞서 뜬다. 화면에 그려진 얼굴 대비 오버레이가 늦은 양(differential)만 쓴다.
 * [카메라 동기화]가 켜져 있으면 얼굴과 랜드마크가 같은 프레임이므로
 * differential = 0 이고 예측은 아예 불필요하다.
 */
const V_DEAD = 40;       // px/s 미만 속도는 감쇠 (jitter가 가짜 속도를 만드는 것 차단)
const MAX_PRED_PX = 28;  // 예측이 측정값에서 벗어날 수 있는 최대 거리 (최후 안전망)

const makeTracker = () => ({ init: false, x: 0, y: 0, vx: 0, vy: 0, rAvg: 0 });
let trackers = [makeTracker(), makeTracker(), makeTracker()];
let smoothed = null;

function abStep(s, zx, zy, dt, horizon, A, B) {
  if (!s.init || dt <= 0 || dt > 0.25) {   // 첫 프레임 / 얼굴 소실 / 탭 복귀
    s.x = zx; s.y = zy; s.vx = s.vy = 0; s.rAvg = 0; s.init = true;
    return { x: zx, y: zy };
  }
  const px = s.x + s.vx * dt, py = s.y + s.vy * dt;   // 측정 시각으로 예측
  const rx = zx - px, ry = zy - py;                   // 잔차
  s.x = px + A * rx;  s.y = py + A * ry;              // 보정
  s.vx += (B / dt) * rx;  s.vy += (B / dt) * ry;

  const sp = Math.hypot(s.vx, s.vy);                  // 속도 데드존
  if (sp < V_DEAD) { const g = sp / V_DEAD; s.vx *= g; s.vy *= g; }

  const rm = Math.hypot(rx, ry);                      // 잔차 기반 신뢰도:
  s.rAvg = s.rAvg * 0.9 + rm * 0.1;                   // 등속 모델이 안 맞으면
  const conf = 1 / (1 + (s.rAvg / 6) ** 2);           // 예측을 줄여 오버슛 억제

  const h = horizon * conf;
  let ox = s.x + s.vx * h, oy = s.y + s.vy * h;

  const dx = ox - zx, dy = oy - zy, d = Math.hypot(dx, dy);
  if (d > MAX_PRED_PX) { ox = zx + dx * MAX_PRED_PX / d; oy = zy + dy * MAX_PRED_PX / d; }
  return { x: ox, y: oy };
}

function filterPoints(pts, dt, horizon) {
  const mode = ui.filter.value;

  if (mode === "none") { smoothed = null; return pts; }

  if (mode === "ema") {
    const k = Number(ui.ema.value) / 100;
    if (!smoothed || smoothed.length !== pts.length) {
      smoothed = pts.map((p) => ({ ...p }));
      return smoothed;
    }
    for (let i = 0; i < pts.length; i++) {
      smoothed[i].x = smoothed[i].x * k + pts[i].x * (1 - k);
      smoothed[i].y = smoothed[i].y * k + pts[i].y * (1 - k);
    }
    return smoothed;
  }

  const A = Number(ui.ema.value) / 100;
  const B = (A * A) / (2 - A);   // 임계감쇠 이론값
  return pts.map((p, i) => abStep(trackers[i], p.x, p.y, dt, horizon, A, B));
}

/* jitter: 원본(스무딩 전) 앵커의 프레임간 이동량 RMS */
let prevRaw = null;
const jitterBuf = [];
function measureJitter(raw) {
  if (prevRaw) {
    let sum = 0;
    for (let i = 0; i < raw.length; i++) {
      sum += (raw[i].x - prevRaw[i].x) ** 2 + (raw[i].y - prevRaw[i].y) ** 2;
    }
    jitterBuf.push(Math.sqrt(sum / raw.length));
    if (jitterBuf.length > 60) jitterBuf.shift();
  }
  prevRaw = raw.map((p) => ({ ...p }));
  if (!jitterBuf.length) return null;
  return Math.sqrt(jitterBuf.reduce((s, v) => s + v * v, 0) / jitterBuf.length);
}

/* ---------- 렌더 루프 ----------
 * requestVideoFrameCallback(rVFC)이 있으면 그걸 쓴다. rAF는 디스플레이
 * 주사율(보통 60Hz)에 맞춰 도는데 웹캠은 30fps라, rAF로 돌리면 같은 비디오
 * 프레임을 두 번씩 추론하게 된다 - GPU를 두 배로 쓰면서 얻는 건 없고,
 * 그만큼 다음 프레임 처리가 밀린다. rVFC는 새 프레임이 실제로 도착할 때만
 * 불리고, 덤으로 캡처 시각 메타데이터를 줘서 진짜 지연을 잴 수 있다.
 */
const HAS_RVFC = typeof HTMLVideoElement !== "undefined" &&
                 "requestVideoFrameCallback" in HTMLVideoElement.prototype;

const fpsBuf = [];
let lastT = 0;
let lastVideoTime = -1;
let dupSkips = 0, frameTries = 0;
let ageBuf = [];
let lastCaptureSec = null;   // dt 계산용 (루프 시각이 아니라 캡처 시각 기준)
let lateBuf = [];            // 오버레이가 비디오 표시 시각보다 늦은 정도

function scheduleNext() {
  if (!running) return;
  if (HAS_RVFC) {
    rafId = video.requestVideoFrameCallback((now, meta) => processFrame(now, meta));
  } else {
    rafId = requestAnimationFrame((now) => processFrame(now, null));
  }
}

function cancelNext() {
  if (rafId === null) return;
  if (HAS_RVFC && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(rafId);
  else cancelAnimationFrame(rafId);
  rafId = null;
}

function processFrame(now, meta) {
  if (!running) return;
  scheduleNext();

  if (video.readyState < 2) return;

  frameTries++;
  // 같은 비디오 프레임이면 추론을 건너뛴다.
  const vt = meta ? meta.mediaTime : video.currentTime;
  if (vt === lastVideoTime) {
    dupSkips++;
    stat.dup.textContent = ((dupSkips / frameTries) * 100).toFixed(0) + "%";
    stat.dup.classList.toggle("warn", dupSkips / frameTries > 0.2);
    return;
  }
  lastVideoTime = vt;
  stat.dup.textContent = ((dupSkips / frameTries) * 100).toFixed(0) + "%";

  // 캡처 시각을 주는 브라우저면 "카메라가 찍은 순간 -> 우리가 처리 시작"
  // 까지의 실제 경과를 잰다. 이게 우리 코드가 손대기 전에 이미 까먹은 시간.
  if (meta) {
    const capture = meta.captureTime ?? meta.presentationTime;
    if (capture != null) {
      ageBuf.push(now - capture);
      if (ageBuf.length > 30) ageBuf.shift();
      const avgAge = ageBuf.reduce((s, v) => s + v, 0) / ageBuf.length;
      stat.age.textContent = avgAge.toFixed(0) + " ms" +
        (meta.captureTime == null ? " (근사)" : "");
      stat.age.classList.toggle("warn", avgAge > 60);
    }
  } else {
    stat.age.textContent = "n/a (rAF)";
  }

  if (lastT) {
    fpsBuf.push(now - lastT);
    if (fpsBuf.length > 30) fpsBuf.shift();
    const avg = fpsBuf.reduce((s, v) => s + v, 0) / fpsBuf.length;
    stat.fps.textContent = (1000 / avg).toFixed(1);
  }
  lastT = now;

  ctx.clearRect(0, 0, overlay.width, overlay.height);

  // [카메라 동기화] 핵심.
  // 끄면: 밑의 <video>가 실시간으로 흐르고 오버레이만 캔버스에 얹힌다. 그런데
  //   오버레이는 이 프레임을 "추론한 뒤"에 그려지므로 항상 비디오보다 늦다.
  //   둘이 서로 다른 시점을 보여주니 헤어가 머리에서 떨어져 따로 논다.
  // 켜면: 지금 이 프레임을 캔버스에 직접 그리고 그 위에 오버레이를 얹는다.
  //   화면 전체가 추론 시간만큼 늦어지는 대신, 비디오와 오버레이가 항상
  //   같은 시점이라 헤어가 머리에서 절대 떨어지지 않는다.
  // 반드시 추론 "전에" 그려야 같은 프레임이 보장된다.
  if (ui.sync.checked) {
    ctx.drawImage(video, 0, 0, overlay.width, overlay.height);
  }

  const t0 = performance.now();
  const res = landmarker.detectForVideo(video, now);
  const detectMs = performance.now() - t0;
  stat.detect.textContent = detectMs.toFixed(1) + " ms";
  stat.detect.classList.toggle("warn", detectMs > 25);
  stat.detect.classList.toggle("good", detectMs <= 25);

  const faces = res.faceLandmarks || [];
  if (!faces.length) {
    stat.face.textContent = "없음";
    stat.face.classList.add("warn");
    prevRaw = null;
    // 얼굴이 다시 잡히면 엉뚱한 속도로 튀지 않도록 추정기를 리셋한다.
    trackers = [makeTracker(), makeTracker(), makeTracker()];
    smoothed = null;
    lastCaptureSec = null;
    return;
  }
  stat.face.textContent = "검출됨";
  stat.face.classList.remove("warn");

  const lm = faces[0];
  const W = overlay.width, H = overlay.height;
  const toPx = (i) => ({ x: lm[i].x * W, y: lm[i].y * H });

  const raw = [toPx(LM_FOREHEAD), toPx(LM_CHEEK_L), toPx(LM_CHEEK_R)];

  // dt: 캡처 시각 차분. 루프 시각을 쓰면 프레임을 건너뛴 구간에서 dt가
  // 틀어져 속도 추정이 망가진다.
  const capSec = meta ? (meta.mediaTime ?? now / 1000) : now / 1000;
  const dt = lastCaptureSec === null ? 0 : capSec - lastCaptureSec;
  lastCaptureSec = capSec;

  // horizon: 화면에 그려진 얼굴 대비 오버레이가 늦은 양(초).
  // 동기화가 켜져 있으면 얼굴과 랜드마크가 같은 프레임 -> 0.
  let horizon = 0;
  if (!ui.sync.checked && lateBuf.length) {
    const avgLate = lateBuf.reduce((s, v) => s + v, 0) / lateBuf.length;
    horizon = Math.min(Math.max(avgLate, 0), 120) / 1000;
  }
  stat.horizon.textContent = ui.sync.checked
    ? "0 ms (동기화)"
    : (horizon * 1000).toFixed(0) + " ms";

  const j = measureJitter(raw);
  if (j !== null) {
    stat.jitter.textContent = j.toFixed(2) + " px";
    stat.jitter.classList.toggle("warn", j > 1.5);
    stat.jitter.classList.toggle("good", j <= 1.5);
  }

  const pts = filterPoints(raw, dt, horizon);

  if (res.facialTransformationMatrixes && res.facialTransformationMatrixes.length) {
    const e = eulerFromMatrix(res.facialTransformationMatrixes[0].data);
    stat.yaw.textContent = e.yaw.toFixed(1) + "°";
    stat.pitch.textContent = e.pitch.toFixed(1) + "°";
    stat.roll.textContent = e.roll.toFixed(1) + "°";
    stat.yaw.classList.toggle("warn", Math.abs(e.yaw) > 25);
  }

  if (ui.mesh.checked) {
    ctx.fillStyle = "rgba(80,220,255,0.5)";
    for (let i = 0; i < lm.length; i += 2) {
      ctx.fillRect(lm[i].x * W - 0.5, lm[i].y * H - 0.5, 1.5, 1.5);
    }
  }

  if (ui.hair.checked) {
    let M = solveAffine(
      ASSET_ANCHOR.forehead, ASSET_ANCHOR.cheekL, ASSET_ANCHOR.cheekR,
      pts[0], pts[1], pts[2]
    );
    if (M) {
      // 목적지 공간에서 크기/상하 보정
      const s = Number(ui.scale.value) / 100;
      const cx = (pts[0].x + pts[1].x + pts[2].x) / 3;
      const cy = (pts[0].y + pts[1].y + pts[2].y) / 3;
      M = compose([s, 0, 0, s, cx - s * cx, cy - s * cy], M);

      // 얼굴 기준 "위" 방향으로 오프셋 (에셋 -y 를 변환한 방향)
      const upx = -M[2], upy = -M[3];
      const ulen = Math.hypot(upx, upy) || 1;
      const off = Number(ui.offset.value);
      M = compose([1, 0, 0, 1, (upx / ulen) * off, (upy / ulen) * off], M);

      ctx.save();
      ctx.globalAlpha = Number(ui.alpha.value) / 100;
      ctx.setTransform(M[0], M[1], M[2], M[3], M[4], M[5]);
      ctx.drawImage(hairAsset, 0, 0);
      ctx.restore();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
    }
  }

  if (ui.marks.checked) {
    const colors = ["#ffd400", "#ff2fc8", "#22d3ee"];
    pts.forEach((p, i) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = colors[i];
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }

  // 오버레이를 다 그린 지금이, 이 프레임이 화면에 뜰 것으로 예상된 시각보다
  // 얼마나 늦었는가. 이게 사용자가 실제로 보는 "헤어가 얼굴보다 늦은 양"이고,
  // 동기화를 껐을 때 예측 horizon으로 쓸 값이다.
  if (meta && meta.expectedDisplayTime != null) {
    lateBuf.push(performance.now() - meta.expectedDisplayTime);
    if (lateBuf.length > 30) lateBuf.shift();
    const avgLate = lateBuf.reduce((s, v) => s + v, 0) / lateBuf.length;
    stat.late.textContent = avgLate.toFixed(0) + " ms";
    stat.late.classList.toggle("warn", avgLate > 40);
    stat.late.classList.toggle("good", avgLate <= 40);
  }
}

/* ---------- 시작/정지 ---------- */
async function start() {
  startBtn.disabled = true;
  try {
    setStatus("MediaPipe WASM 로딩 중... (CDN)");
    const vision = await FilesetResolver.forVisionTasks(CDN + "/wasm");

    let delegate = "GPU";
    try {
      landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFacialTransformationMatrixes: true,
      });
    } catch (gpuErr) {
      console.warn("GPU 델리게이트 실패, CPU로 폴백:", gpuErr);
      delegate = "CPU (폴백)";
      landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL, delegate: "CPU" },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFacialTransformationMatrixes: true,
      });
    }
    stat.delegate.textContent = delegate;
    stat.delegate.classList.toggle("good", delegate === "GPU");
    stat.delegate.classList.toggle("warn", delegate !== "GPU");

    setStatus("웹캠 요청 중...");
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, frameRate: { ideal: 30 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    running = true;
    stopBtn.disabled = false;
    stat.src.textContent = HAS_RVFC ? "rVFC" : "rAF (폴백)";
    stat.src.classList.toggle("good", HAS_RVFC);
    stat.src.classList.toggle("warn", !HAS_RVFC);
    setStatus("동작 중 — 고개를 천천히 좌우/위아래로 돌려보세요. 가만히 있을 때의 jitter 값도 확인해 보세요.");
    scheduleNext();
  } catch (err) {
    setStatus("시작 실패: " + err.message +
      " (다른 기기에서 http로 접속했다면 브라우저가 보안 컨텍스트가 아니라며 웹캠을 막았을 수 있습니다)");
    startBtn.disabled = false;
  }
}

function stop() {
  running = false;
  cancelNext();
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  video.srcObject = null;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  smoothed = null; prevRaw = null; jitterBuf.length = 0; fpsBuf.length = 0; lastT = 0;
  lastVideoTime = -1; dupSkips = 0; frameTries = 0; ageBuf = [];
  lastCaptureSec = null; lateBuf = [];
  trackers = [makeTracker(), makeTracker(), makeTracker()];
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("정지됨");
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

/* preflight.js 가 "모듈이 끝까지 살아서 올라왔는지" 판정하는 신호 */
window.__pageReady = true;
