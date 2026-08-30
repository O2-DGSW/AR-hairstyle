/* 서버 합성 경로.
 * 웹캠 -> WebRTC 업로드 -> 서버가 GPU 얼굴파싱 + 오버레이 합성 ->
 * 완성된 영상을 WebRTC로 되받아 그대로 재생.
 * 클라이언트는 레이어를 하나만 그리므로 오버레이가 영상과 어긋날 수 없다.
 * DataChannel은 통계와 ping RTT 용도로만 쓴다.
 */
const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const statusEl = document.getElementById("status");
const localVideo = document.getElementById("local");
const remoteVideo = document.getElementById("remote");

const el = (id) => document.getElementById(id);
const stat = {
  device: el("s-device"), graph: el("s-graph"), infer: el("s-infer"),
  prepost: el("s-prepost"), proc: el("s-proc"), wait: el("s-wait"),
  fps: el("s-fps"), hair: el("s-hair"), frame: el("s-frame"), rtt: el("s-rtt"),
  cov: el("s-cov"), plateFrames: el("s-plateframes"),
};
const modeSel = el("mode");
const assetSel = el("asset");
const bankSel = el("bank");
const fScale = el("f-scale"), fOffset = el("f-offset");
const fHarmonize = el("f-harmonize"), fShadow = el("f-shadow");
const fBlend = el("f-blend");
const fSmooth = el("f-smooth");
const anchorEl = el("s-anchor");
let assetsLoaded = false;

function send(obj) {
  if (channel && channel.readyState === "open") channel.send(JSON.stringify(obj));
}

modeSel.addEventListener("change", () => send({ type: "mode", mode: modeSel.value }));

/* 프로덕션 플로우는 "헤어 고르기 -> 각도 수집 -> 생성 -> 씌우기" 하나뿐이라
 * 진단용 뷰는 감춘다. 서버에는 그대로 남아 있으므로 ?debug=1 로 되살린다. */
if (!new URLSearchParams(location.search).has("debug")) {
  modeSel.querySelectorAll("option[data-debug]").forEach((o) => o.remove());
}

function sendFit() {
  send({
    type: "fit",
    asset: assetSel.value || undefined,
    scale: Number(fScale.value) / 100,
    offset: Number(fOffset.value),
    harmonize: fHarmonize.checked,
    blend: fBlend.checked ? 1.0 : 0.0,
    shadow: Number(fShadow.value) / 100,
    smooth: Number(fSmooth.value) / 100,
  });
}
fHarmonize.addEventListener("change", sendFit);
fBlend.addEventListener("change", sendFit);
[fScale, fOffset, fShadow, fSmooth].forEach((s) => {
  s.addEventListener("input", () => {
    el("v-" + s.id).textContent = s.value;
    sendFit();
  });
});
assetSel.addEventListener("change", () => {
  bankSel.value = "";                 // 개별 에셋을 고르면 뱅크 해제
  send({ type: "fit", asset: assetSel.value, bank: "" });
  sendFit();
});
bankSel.addEventListener("change", () => send({ type: "fit", bank: bankSel.value }));

/* ---------- 사진 찍기 (GAN) ---------- */
const shootBtn = el("shoot"), refSel = el("reference");
const shootStatus = el("shoot-status");
let refsLoaded = false;

function fillSelect(sel, names) {
  sel.innerHTML = "";
  names.forEach((name) => {
    const o = document.createElement("option");
    o.value = name; o.textContent = name;
    sel.appendChild(o);
  });
}

// 페이지 로드 즉시 목록을 채운다. 연결을 기다리지 않는다.
/* ---- 참고 헤어스타일 등록/삭제 ----
 * GAN 의 입력은 "각도별 에셋" 이 아니라 참고사진 한 장이다. 그래서 다른
 * 헤어스타일을 시험하려면 여기서 사진을 갈아끼우면 된다. 서버가 등록 시점에
 * 눈 간격을 재서 확대 배율을 알려준다 - 얼굴이 작은 사진은 1024 로 늘려 쓰느라
 * 헤어 디테일이 뭉개지기 때문이다. */
const refFile = el("ref-file"), refUpBtn = el("ref-upload"),
      refDelBtn = el("ref-delete"), refStatus = el("ref-status");

function refSay(msg, color) {
  refStatus.textContent = msg;
  refStatus.style.color = color || "#888";
}

/* ---------- Rotate 모델 A/B ----------
 * Rotate 는 참고사진의 머리를 사용자 얼굴 각도로 돌리는 모듈이라 큰 각도
 * 품질을 좌우한다. 원본과 파인튜닝본을 번갈아 보려면 예전에는 서버를 다시
 * 띄워야 했고 매번 모델 적재에 20~90초가 들었다. 25MB 짜리 Rotate 만
 * 갈아끼우면 되므로 서버가 /model 로 런타임 교체를 받는다. */
const rotSel = el("rotate-model"), rotStatus = el("rotate-status");
const ROT_LABEL = { base: "원본 (HairFastGAN 배포본)", finetuned: "파인튜닝본",
                    startup: "기동 시 지정본", custom: "직접 지정" };

function rotSay(msg, color) {
  rotStatus.textContent = msg;
  rotStatus.style.color = color || "#888";
}

async function loadRotateModel() {
  try {
    const r = await fetch("/model");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    rotSel.innerHTML = "";
    for (const name of d.available) {
      const o = document.createElement("option");
      o.value = name;
      o.textContent = ROT_LABEL[name] || name;
      rotSel.appendChild(o);
    }
    if (d.variant) rotSel.value = d.variant;
    rotSay(d.loaded ? "적재됨" : "첫 촬영 때 적재됩니다 (약 20~90초)");
  } catch (e) {
    rotSay(`목록을 못 읽었습니다: ${e.message}`, "#ff5f56");
  }
}

rotSel.addEventListener("change", async () => {
  const want = rotSel.value;
  rotSel.disabled = true;
  rotSay("교체 중... (모델이 아직이면 적재까지 최대 90초)", "#ffd400");
  try {
    const r = await fetch("/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ variant: want }),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.message || `HTTP ${r.status}`);
    rotSay(`${ROT_LABEL[d.variant] || d.variant} 적용됨. 새로 촬영해 보세요.`, "#4ade80");
  } catch (e) {
    rotSay(`교체 실패: ${e.message}`, "#ff5f56");
    await loadRotateModel();          // 서버의 실제 상태로 되돌린다
  } finally {
    rotSel.disabled = false;
  }
});

loadRotateModel();

refUpBtn.addEventListener("click", async () => {
  const f = refFile.files && refFile.files[0];
  if (!f) return refSay("사진 파일을 먼저 고르세요.", "#ffd400");
  const fd = new FormData();
  fd.append("file", f);
  refUpBtn.disabled = true;
  refSay(`업로드 중... (${(f.size / 1048576).toFixed(1)}MB)`);
  try {
    const r = await fetch("/references", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) {
      refSay(d.message || `등록 실패 (HTTP ${r.status})`, "#ff5f56");
    } else {
      fillSelect(refSel, d.references);
      refSel.value = d.name;
      refsLoaded = true;
      const base = `등록됨: ${d.name} · ${d.width}x${d.height} · ` +
                   `눈 간격 ${d.eye_px}px · GAN 입력으로 ${d.upscale}배 확대`;
      refSay(d.warning ? `${base}\n${d.warning}` : base,
             d.warning ? "#ffd400" : "#4ade80");
    }
  } catch (e) {
    refSay(`등록 실패: ${e.message}`, "#ff5f56");
  } finally {
    refUpBtn.disabled = false;
  }
});

refDelBtn.addEventListener("click", async () => {
  const name = refSel.value;
  if (!name) return;
  refDelBtn.disabled = true;
  try {
    const r = await fetch(`/references/${encodeURIComponent(name)}`,
                          { method: "DELETE" });
    const d = await r.json();
    if (!r.ok) refSay(d.message || `삭제 실패 (HTTP ${r.status})`, "#ff5f56");
    else { fillSelect(refSel, d.references); refSay(`삭제됨: ${d.deleted}`); }
  } catch (e) {
    refSay(`삭제 실패: ${e.message}`, "#ff5f56");
  } finally {
    refDelBtn.disabled = false;
  }
});

(async function loadLists() {
  try {
    const r = await fetch("/references");
    const d = await r.json();
    if (d.references && d.references.length) {
      fillSelect(refSel, d.references);
      refsLoaded = true;
      shootStatus.textContent =
        "실시간 미리보기로 각도를 맞춘 뒤 [사진 찍기]를 누르세요. (연결 후 활성화)";
    } else {
      shootStatus.textContent = "server/references/ 에 헤어스타일 원본 사진이 없습니다.";
      shootStatus.style.color = "#ff5f56";
    }
    if (d.assets && d.assets.length) {
      fillSelect(assetSel, d.assets);
      assetsLoaded = true;
    }
    fillSelect(bankSel, ["", ...(d.banks || [])]);
    bankSel.options[0].textContent = "(사용 안 함)";
  } catch (e) {
    // 조용히 실패하면 "드롭다운이 빈" 증상만 남고 원인을 알 수 없다.
    shootStatus.textContent = "목록을 불러오지 못했습니다: " + e.message;
    shootStatus.style.color = "#ff5f56";
    console.warn("목록 로드 실패", e);
  }
})();

/* ---------- 라이브 뱅크 (세션 중 각도별 GAN 생성) ---------- */
const lbBtn = el("livebank"), lbStatus = el("lb-status"), lbBuckets = el("lb-buckets");
let lbOn = false;

const LB_COLOR = {
  pending: ["#333", "#888"], queued: ["#3b3357", "#c4b5fd"],
  running: ["#4c3d00", "#ffd400"], done: ["#14432a", "#4ade80"],
  failed: ["#4a1d1d", "#ff5f56"],
};
const LB_WORD = { pending: "대기", captured: "찍힘", running: "생성중", done: "완료", failed: "실패" };

const lbOv = el("lb-overlay"), lbOvTitle = el("lb-ov-title"),
      lbOvSub = el("lb-ov-sub"), lbOvBar = el("lb-ov-bar");

function lbOverlay(on, title, sub, pct) {
  lbOv.style.display = on ? "flex" : "none";
  if (!on) return;
  if (title !== undefined) lbOvTitle.textContent = title;
  if (sub !== undefined) lbOvSub.textContent = sub;
  if (pct !== undefined) lbOvBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

function lbSetOn(on) {
  lbOn = on;
  lbBtn.textContent = on ? "⏹ 라이브 뱅크 중지" : "🎞 라이브 뱅크 만들기";
  lbBtn.style.background = on ? "#ef4444" : "#8b5cf6";
}

// 칸 수 프리셋. 검출기 한계가 ±40도라 그 안쪽에서만 나눈다.
const LB_PRESETS = {
  3: [-30, 0, 30],
  5: [-32, -16, 0, 16, 32],
  7: [-36, -24, -12, 0, 12, 24, 36],
};

lbBtn.addEventListener("click", () => {
  lbSetOn(!lbOn);
  // 클릭이 났는지 / 채널로 나갔는지를 화면에서 구분할 수 있게 남긴다.
  // (버튼이 비활성이면 이 핸들러 자체가 안 불린다)
  const open = !!channel && channel.readyState === "open";
  lbStatus.style.color = open ? "#ffd400" : "#ff5f56";
  lbStatus.textContent = open
    ? (lbOn ? "요청 보냄 — 서버 응답 대기 중..." : "중지 요청 보냄...")
    : `DataChannel 이 열려 있지 않습니다 (${channel ? channel.readyState : "없음"})`;
  console.log("[livebank] click on=%s channel=%s", lbOn, channel && channel.readyState);
  send({ type: "livebank", on: lbOn, reference: refSel.value,
         targets: LB_PRESETS[el("lb-count").value] || LB_PRESETS[7] });
});

/** 어느 쪽으로 고개를 돌려야 하는지. 서버 yaw 부호와 화면 좌우를 맞춘다. */
function lbGuide(next) {
  if (next === null || next === undefined) return "각도 수집 완료.";
  if (Math.abs(next) < 5) return "정면을 보고 잠깐 멈춰주세요.";
  return `고개를 ${next < 0 ? "왼" : "오른"}쪽으로 ${Math.abs(next).toFixed(0)}° 돌리고 잠깐 멈춰주세요.`;
}

function onLiveBank(d) {
  if (Array.isArray(d.buckets)) {
    lbBuckets.innerHTML = "";
    for (const b of d.buckets) {
      const [bg, fg] = LB_COLOR[b.status] || LB_COLOR.pending;
      const chip = document.createElement("div");
      chip.style.cssText = `flex:1 1 60px;text-align:center;padding:6px 4px;border-radius:6px;
                            background:${bg};color:${fg};font-size:11.5px;line-height:1.4`;
      chip.innerHTML = `<b>${b.yaw > 0 ? "+" : ""}${b.yaw.toFixed(0)}°</b><br>${LB_WORD[b.status] || b.status}`;
      lbBuckets.appendChild(chip);
    }
  }

  if (d.status === "error") {
    lbStatus.style.color = "#ff5f56";
    lbStatus.textContent = d.message || "실패";
    return;
  }
  if (d.status === "stopped") {
    lbSetOn(false);
    lbStatus.style.color = "#888";
    lbStatus.textContent = "중지됨.";
    return;
  }

  if (d.status === "started") {
    lbOverlay(false);
    lbStatus.style.color = "#ffd400";
    lbStatus.textContent = "얼굴을 양옆으로 천천히 돌려보세요 — 각도마다 잠깐 멈추면 찍힙니다.";
    return;
  }

  if (d.status === "generating") {
    // 수집이 끝난 시점. 서버가 영상 전송을 멈추므로 오버레이로 덮는다.
    const pct = ((d.index - 1) / d.total) * 100;
    lbOverlay(true, `헤어를 생성하는 중입니다 (${d.index}/${d.total})`,
              `${d.current_yaw > 0 ? "+" : ""}${d.current_yaw.toFixed(0)}° · 칸당 약 9초`, pct);
    lbStatus.style.color = "#ffd400";
    lbStatus.textContent = `생성 중 ${d.index}/${d.total}`;
    return;
  }

  if (d.status === "complete") {
    lbOverlay(false);
    lbSetOn(false);
    modeSel.value = "tryon";
    if (Array.isArray(d.banks)) {
      fillSelect(bankSel, ["", ...d.banks]);
      bankSel.options[0].textContent = "(사용 안 함)";
    }
    if (d.bank) bankSel.value = d.bank;
    lbStatus.style.color = d.done === d.total ? "#4ade80" : "#ffd400";
    lbStatus.textContent = `${d.done}/${d.total}칸 완성 (${d.seconds}초) — 고개를 돌려보세요.`;
    return;
  }

  if (d.status === "filled") {
    // 첫 칸이 들어오면 서버가 이미 tryon + 뱅크로 전환해 둔 상태다. UI 를 맞춘다.
    modeSel.value = "tryon";
    if (Array.isArray(d.banks)) {
      fillSelect(bankSel, ["", ...d.banks]);
      bankSel.options[0].textContent = "(사용 안 함)";
    }
    if (d.bank) bankSel.value = d.bank;
  }

  const head = `${d.done}/${d.total} 완료`;
  const tail = d.status === "running" ? (d.message || "생성 중...")
             : d.status === "captured" ? "찍었습니다 — 대기열에 넣었습니다."
             : lbGuide(d.next);
  lbStatus.style.color = d.done === d.total ? "#4ade80" : "#ffd400";
  lbStatus.textContent = `${head} · ${tail}`;
}

/* ---------- 학습 프레임 수집 ---------- */
const recBtn = el("record"), recInfo = el("rec-info");
let recording = false;

recBtn.addEventListener("click", () => {
  recording = !recording;
  send({ type: "record", on: recording });
  recBtn.textContent = recording ? "⏹ 수집 중지" : "⏺ 학습 프레임 수집";
  recBtn.style.background = recording ? "#ef4444" : "#444";
});

shootBtn.addEventListener("click", () => {
  if (!channel || channel.readyState !== "open") return;
  shootBtn.disabled = true;
  shootStatus.style.color = "#ffd400";
  shootStatus.textContent = "요청 전송 중...";
  send({ type: "capture", reference: refSel.value });
});

function onCapture(d) {
  if (d.status === "loading") {
    shootStatus.style.color = "#ffd400";
    shootStatus.textContent = d.message;
  } else if (d.status === "running") {
    shootStatus.style.color = "#ffd400";
    shootStatus.textContent = "합성 중... (약 9초)";
  } else if (d.status === "done") {
    el("result-before").src = d.before + "?t=" + Date.now();
    el("result-after").src = d.url + "?t=" + Date.now();
    el("result-meta").textContent = `(GAN ${d.gan_seconds}초 / 전체 ${d.total_seconds}초)`;
    el("result-wrap").style.display = "block";

    if (d.asset) {
      // GAN 결과에서 뽑은 헤어를 실시간 워핑에 바로 물린다.
      if (Array.isArray(d.assets)) fillSelect(assetSel, d.assets);
      assetSel.value = d.asset;
      modeSel.value = "tryon";
      shootStatus.style.color = "#4ade80";
      shootStatus.textContent =
        `실시간 적용됨 — 이제 이 헤어가 30fps로 따라옵니다 (GAN ${d.gan_seconds}초)`;
    } else {
      shootStatus.style.color = "#ffd400";
      shootStatus.textContent =
        `합성은 됐지만 실시간 에셋 추출에 실패했습니다 (GAN ${d.gan_seconds}초)`;
    }
    shootBtn.disabled = false;
  } else if (d.status === "error") {
    shootStatus.style.color = "#ff5f56";
    shootStatus.textContent = "실패: " + d.message;
    shootBtn.disabled = false;
  }
}

let pc = null, channel = null, localStream = null, pingTimer = null, statsTimer = null;

/* ---------- 미리보기 지연 버퍼 ----------
 * 서버 합성본은 왕복(인코딩 -> 네트워크 -> 처리 -> 디코딩)만큼 늦게 도착한다.
 * 그 옆의 로컬 미리보기는 지연이 없으니 둘이 서로 다른 시점을 보여준다.
 * 버퍼로 '빠른 쪽'을 늦춰서 맞춘다 - 버퍼는 늦은 쪽을 당길 수 없고, 이게
 * 버퍼가 동기화에 기여할 수 있는 유일한 방향이다.
 */
const PV_W = 200, PV_H = 150, PV_POOL = 24;
const pvCanvas = el("local-canvas"), pvCtx = pvCanvas.getContext("2d");
const syncPreview = el("sync-preview");
const pvPool = [];      // {canvas, ctx}
const pvQueue = [];     // {slot, t}
let pvNext = 0, pvDelayMs = 0, pvRaf = null;

for (let i = 0; i < PV_POOL; i++) {
  const c = document.createElement("canvas");
  c.width = PV_W; c.height = PV_H;
  pvPool.push({ canvas: c, ctx: c.getContext("2d") });
}

function estimateDelay(d) {
  // 왕복 지연 추정: ping RTT(네트워크 왕복) + 서버 처리 + 지터버퍼 + 코덱 대략치.
  // 정확한 값은 아니지만, 미리보기를 맞추는 기준으로는 충분하다.
  const rtt = lastRtt || 0;
  const proc = (d && d.proc_ms) || 0;
  return Math.max(0, Math.min(600, rtt + proc + lastJitterBuffer + 40));
}

function capturePreview() {
  const v = document.getElementById("local");
  if (!localStream || !v.srcObject) return;
  const grab = (now) => {
    if (!localStream) return;
    const slot = pvPool[pvNext % PV_POOL];
    pvNext++;
    slot.ctx.drawImage(v, 0, 0, PV_W, PV_H);
    pvQueue.push({ slot, t: performance.now() });
    // 풀보다 하나 적게 유지한다. 꽉 채우면 방금 덮어쓴 슬롯이 큐의 가장
    // 오래된 항목과 같은 슬롯이 되어 그 프레임이 오염된다.
    while (pvQueue.length > PV_POOL - 1) pvQueue.shift();
    schedule();
  };
  const schedule = () => {
    if (!localStream) return;
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
      v.requestVideoFrameCallback((n) => grab(n));
    } else {
      requestAnimationFrame((n) => grab(n));
    }
  };
  schedule();
}

function renderPreview() {
  pvRaf = requestAnimationFrame(renderPreview);
  if (!pvQueue.length) return;

  const delay = syncPreview.checked ? pvDelayMs : 0;
  const target = performance.now() - delay;

  // target 이하 중 가장 최신 프레임을 고른다 (없으면 가장 오래된 것)
  let pick = pvQueue[0];
  for (const f of pvQueue) {
    if (f.t <= target) pick = f; else break;
  }
  // 큐에 담기는 건 {slot, t} 이고 캔버스는 slot 안에 있다(pvPool 은 {canvas, ctx}).
  // pick.canvas 로 읽으면 undefined 가 drawImage 에 들어가 매 프레임 TypeError 가
  // 났다. rAF 콜백 안이라 예외가 화면에 안 드러나고 미리보기만 조용히 비어 있었다.
  pvCtx.drawImage(pick.slot.canvas, 0, 0, PV_W, PV_H);
  el("delay-label").textContent = delay > 0 ? `(−${delay.toFixed(0)}ms)` : "";
  el("s-pvdelay").textContent = delay.toFixed(0) + " ms";
}

let lastRtt = 0, lastJitterBuffer = 0;

/* 서버 메시지나 예외 문구를 innerHTML 로 넣기 전에 거른다. */
function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function setStatus(msg, isError) {
  // 오류는 눈에 띄어야 하고 <b> 강조를 살려야 하므로 innerHTML 로 넣는다.
  // 평상시 메시지는 textContent 로 둔다 - 서버가 준 문자열이 섞여 들어올 수
  // 있는데(거절 사유 등) 그걸 마크업으로 해석하면 안 된다.
  if (isError) {
    statusEl.innerHTML = msg;
    statusEl.style.color = "#ff5f56";
    console.error("[status]", statusEl.textContent);
  } else {
    statusEl.textContent = msg;
    statusEl.style.color = "";
    console.log("[status]", msg);
  }
}

function warnIf(node, v, threshold) { node.classList.toggle("warn", v > threshold); }

async function waitForIceGatheringComplete(peerConnection) {
  if (peerConnection.iceGatheringState === "complete") return;
  await new Promise((resolve) => {
    function check() {
      if (peerConnection.iceGatheringState === "complete") {
        peerConnection.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    }
    peerConnection.addEventListener("icegatheringstatechange", check);
  });
}

function onMessage(event) {
  let d;
  try { d = JSON.parse(event.data); } catch (e) { return; }

  if (d.type === "pong") {
    const rtt = performance.now() - d.t_client;
    lastRtt = rtt;
    stat.rtt.textContent = rtt.toFixed(0) + " ms";
    warnIf(stat.rtt, rtt, 150);
    return;
  }

  if (d.type === "capture") { onCapture(d); return; }
  if (d.type === "livebank") { onLiveBank(d); return; }
  if (d.type === "record") {
    recInfo.textContent = d.on ? `수집 중... ${d.count}장` : (d.count ? `${d.count}장 저장됨` : "");
    return;
  }
  if (d.type !== "stats") return;

  // 프레임 통계가 오고 있다 = 연결이 살아있고 찍을 프레임이 있다.
  // 수집은 참고사진/에셋과 무관하게 원본 프레임만 있으면 되므로 여기서 연다.
  if (recBtn.disabled) recBtn.disabled = false;
  if (lbBtn.disabled) {
    lbBtn.disabled = false;
    lbStatus.style.color = "#888";
    lbStatus.textContent = "준비됨 — [라이브 뱅크 만들기]를 누르세요.";
  }

  stat.device.textContent = d.device;
  stat.device.classList.toggle("good", d.device === "cuda");
  stat.device.classList.toggle("warn", d.device !== "cuda");
  stat.graph.textContent = d.cuda_graph ? "사용" : "미사용";
  stat.graph.classList.toggle("good", !!d.cuda_graph);

  stat.infer.textContent = d.infer_ms.toFixed(1) + " ms";
  stat.infer.classList.toggle("good", d.infer_ms <= 20);
  warnIf(stat.infer, d.infer_ms, 20);

  stat.prepost.textContent = d.pre_ms.toFixed(1) + " / " + d.post_ms.toFixed(1) + " ms";
  stat.proc.textContent = d.proc_ms.toFixed(1) + " ms";
  warnIf(stat.proc, d.proc_ms, 33);
  stat.wait.textContent = d.wait_ms.toFixed(0) + " ms";
  stat.fps.textContent = d.server_fps.toFixed(1);
  pvDelayMs = estimateDelay(d);
  stat.hair.textContent = d.hair_px.toLocaleString();
  stat.frame.textContent = d.frame;
  stat.plateFrames.textContent = d.plate_frames.toLocaleString();

  // 연결이 살아있고 참고사진이 있으면 촬영 버튼을 연다
  if (Array.isArray(d.references) && d.references.length && shootBtn.disabled) {
    if (!refsLoaded) { fillSelect(refSel, d.references); refsLoaded = true; }
    shootBtn.disabled = false;
  }

  if (!assetsLoaded && Array.isArray(d.assets) && d.assets.length) {
    fillSelect(assetSel, d.assets);
    assetsLoaded = true;
  }

  const label = { eye: "세그(눈)", brow: "세그(눈썹)", pose: "3D 포즈" };
  anchorEl.textContent = d.anchor ? (label[d.anchor] || d.anchor) : "-";
  anchorEl.classList.toggle("good", d.anchor === "pose");
  anchorEl.classList.toggle("warn", d.anchor === "brow");

  if (d.rec_count !== null && d.rec_count !== undefined) {
    recInfo.textContent = `수집 중... ${d.rec_count}장`;
  }
  if (d.asset_used) el("s-used").textContent = d.asset_used;
  el("s-blend").textContent = d.blended ? "적용됨"
    : (!d.blender_ready ? "모델 없음" : (fBlend.checked ? "대기(새 헤어 씌우기 모드)" : "꺼짐"));
  el("s-harm").textContent = d.harmonized ? "적용됨" : (fHarmonize.checked ? "기준색 없음" : "꺼짐");
  el("s-yaw2").textContent = d.yaw === null || d.yaw === undefined ? "-" : d.yaw.toFixed(1) + "°";
  el("s-tz").textContent = d.tz === null || d.tz === undefined ? "-" : d.tz.toFixed(1);
  el("s-dd").textContent = (d.d_measured === null || d.d_measured === undefined)
    ? "-" : d.d_measured.toFixed(0) + " / " + d.d_corrected.toFixed(0) + " px";

  if (d.coverage === null || d.coverage === undefined) {
    stat.cov.textContent = "-";
    stat.cov.className = "";
  } else {
    const pct = d.coverage * 100;
    stat.cov.textContent = pct.toFixed(0) + "%";
    stat.cov.classList.toggle("good", pct >= 80);
    stat.cov.classList.toggle("warn", pct < 40);
  }
}

let jbPrev = null;

async function pollRtcStats() {
  if (!pc) return;
  const report = await pc.getStats();
  let sawInbound = false;
  report.forEach((r) => {
    if (r.type !== "inbound-rtp" || r.kind !== "video") return;
    sawInbound = true;

    // jitterBufferDelay 는 누적값이라 증분으로 평균을 낸다.
    if (r.jitterBufferDelay != null && r.jitterBufferEmittedCount) {
      if (jbPrev && r.jitterBufferEmittedCount > jbPrev.count) {
        const dd = r.jitterBufferDelay - jbPrev.delay;
        const dc = r.jitterBufferEmittedCount - jbPrev.count;
        const ms = (dd / dc) * 1000;
        lastJitterBuffer = ms;
        el("s-jb").textContent = ms.toFixed(0) + " ms";
        el("s-jb").classList.toggle("warn", ms > 120);
        el("s-jb").classList.toggle("good", ms <= 60);
      }
      jbPrev = { delay: r.jitterBufferDelay, count: r.jitterBufferEmittedCount };
    }
    if (r.framesPerSecond != null) el("s-infps").textContent = r.framesPerSecond.toFixed(1);
    const dropped = (r.framesDropped || 0);
    el("s-drop").textContent = dropped.toLocaleString();
    el("s-drop").classList.toggle("warn", dropped > 30);

    diagnoseRemote(r, report);
  });

  // inbound-rtp 자체가 없으면 위 콜백이 한 번도 안 돈다. 수신 트랙이 협상되지
  // 않았거나 아직 아무것도 안 온 상태이고, 이것도 화면에서는 똑같이 검다.
  if (!sawInbound) diagnoseRemote({}, report);
}

/* 검은 화면의 원인을 스스로 말하게 한다.
 *
 * "영상이 안 나온다"는 증상 하나에 원인이 최소 세 가지다: 패킷이 아예 안 오거나,
 * 오는데 디코딩이 안 되거나(코덱 불일치가 대표적), 디코딩은 되는데 <video> 에
 * 안 붙어 있거나. 세 경우가 화면에서 똑같이 검게 보여서, 구분하려면 매번
 * chrome://webrtc-internals 를 열어야 했다. 그 판정을 여기서 대신한다. */
let remoteDiag = null;
function diagnoseRemote(r, report) {
  const bytes = r.bytesReceived || 0;
  const received = r.framesReceived || 0;     // 패킷이 '프레임'으로 조립된 수
  const decoded = r.framesDecoded || 0;       // 디코더가 실제로 푼 수
  const keyframes = r.keyFramesDecoded || 0;
  const pkts = r.packetsReceived || 0;
  const lost = r.packetsLost || 0;
  let codec = "";
  if (r.codecId) {
    const c = report.get(r.codecId);
    if (c && c.mimeType) codec = c.mimeType;
  }

  let verdict;
  if (!remoteVideo.srcObject) {
    verdict = ["no-srcobject",
      "서버 트랙이 <video> 에 연결되지 않았습니다 (track 이벤트에 스트림이 없음)"];
  } else if (bytes === 0) {
    verdict = ["no-bytes",
      "서버 영상 패킷이 도착하지 않습니다 — 전송 경로 문제입니다 (방화벽/ICE)"];
  } else if (received === 0) {
    // 패킷은 오는데 한 프레임도 조립이 안 된다. 조각이 다 안 모이는 것이므로
    // 코덱 문제가 아니라 손실/MTU 쪽이다. 720p 키프레임은 수십 조각으로 쪼개져서
    // 경로가 큰 패킷을 흘리면 영원히 완성되지 않는다.
    verdict = ["no-assembly",
      `패킷 ${pkts}개(${(bytes / 1024).toFixed(0)}KB)가 도착했지만 한 프레임도 ` +
      `조립되지 않았습니다 (손실 ${lost}) — 코덱이 아니라 패킷 손실/MTU 문제입니다. ` +
      `캡처 해상도를 640x480 으로 낮춰 보세요`];
  } else if (decoded === 0) {
    verdict = ["no-decode",
      `프레임 ${received}개가 조립됐지만 디코딩이 0입니다` +
      (codec ? ` — 협상된 코덱 ${codec} 을 브라우저가 못 풉니다` : "")];
  } else {
    verdict = ["ok", ""];
  }

  console.log("[remote]", { codec, bytes, pkts, lost, received, decoded, keyframes,
                            size: `${r.frameWidth || "?"}x${r.frameHeight || "?"}`,
                            verdict: verdict[0], srcObject: !!remoteVideo.srcObject });
  if (verdict[0] === remoteDiag) return;      // 상태가 바뀔 때만 알린다
  remoteDiag = verdict[0];
  if (verdict[0] !== "ok") setStatus("<b>검은 화면 원인</b>: " + escapeHtml(verdict[1]), true);
  else setStatus("연결됨 — 서버 영상 수신 중 (" + escapeHtml(codec || "?") + ")");
}

async function start() {
  startBtn.disabled = true;
  setStatus("웹캠 요청 중...");

  try {
    const [capW, capH] = (el("capres").value || "1280x720").split("x").map(Number);
    localStream = await navigator.mediaDevices.getUserMedia({
      // GAN 입력 화질은 여기서 결정된다. 640x480 이면 얼굴 크롭이 ~312px 인데
      // HairFastGAN 은 1024px 로 정렬하므로 3배 넘게 늘려 넣게 되어 뭉갠다.
      // 720p 면 실제 디테일이 2배가 된다. 대신 업링크 대역폭이 늘어나므로
      // 연결이 불안정하면 640x480 으로 내리면 된다.
      video: { width: capW, height: capH, frameRate: { ideal: 30 } },
      audio: false,
    });
    const t = localStream.getVideoTracks()[0].getSettings();
    console.log("[capture] 요청 %dx%d -> 실제 %dx%d", capW, capH, t.width, t.height);
  } catch (err) {
    setStatus("웹캠 접근 실패: " + err.message +
      " (다른 기기에서 http로 접속했다면 브라우저가 보안 컨텍스트가 아니라며 막았을 수 있습니다)");
    startBtn.disabled = false;
    return;
  }
  localVideo.srcObject = localStream;
  await localVideo.play().catch(() => {});
  capturePreview();
  if (!pvRaf) renderPreview();

  pc = new RTCPeerConnection();
  statsTimer = setInterval(pollRtcStats, 1000);

  pc.addEventListener("track", (event) => {
    if (event.track.kind === "video") remoteVideo.srcObject = event.streams[0];
  });
  pc.addEventListener("connectionstatechange", () => {
    setStatus("연결 상태: " + pc.connectionState);
  });

  channel = pc.createDataChannel("stats");
  channel.addEventListener("open", () => {
    pingTimer = setInterval(() => {
      if (channel.readyState === "open") {
        channel.send(JSON.stringify({ type: "ping", t_client: performance.now() }));
      }
    }, 1000);
  });
  channel.addEventListener("message", onMessage);

  localStream.getTracks().forEach((t) => pc.addTrack(t, localStream));

  const offerDesc = await pc.createOffer();
  await pc.setLocalDescription(offerDesc);
  await waitForIceGatheringComplete(pc);

  setStatus("서버로 offer 전송 중... (첫 연결은 GPU 모델 로딩 때문에 몇 초 걸립니다)");

  // 여기부터는 반드시 예외를 붙잡는다.
  // 예전에는 fetch 와 setRemoteDescription 이 보호 없이 있어서, 실패하면
  // 처리되지 않은 rejection 으로 조용히 사라졌다. 화면은 "offer 전송 중..."
  // 에 멈춘 채였고, 콘솔을 열기 전에는 서버가 죽었는지 거절했는지 알 수 없었다.
  try {
    const res = await fetch("/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
    });

    // 상태 코드를 확인하지 않고 곧장 json 을 answer 로 넘기면, 서버가 보낸
    // 오류 본문이 sdp 없는 객체라 setRemoteDescription 에서 엉뚱한 예외가 난다.
    // 서버는 동시 접속 상한에 걸리면 503 + {error, message} 를 준다.
    if (!res.ok) {
      let detail = "";
      try {
        const body = await res.json();
        detail = body.message || body.error || "";
      } catch (_) {
        detail = (await res.text().catch(() => "")).slice(0, 200);
      }
      throw new Error(`서버가 연결을 거절했습니다 (HTTP ${res.status})` +
                      (detail ? ` — ${detail}` : ""));
    }

    await pc.setRemoteDescription(await res.json());
  } catch (err) {
    console.error("[heddy] 연결 실패:", err);
    setStatus("<b>연결 실패</b>: " + escapeHtml((err && err.message) || err), true);
    stop();
    return;
  }

  stopBtn.disabled = false;
}

function stop() {
  if (pingTimer) { clearInterval(pingTimer); pingTimer = null; }
  if (statsTimer) { clearInterval(statsTimer); statsTimer = null; }
  if (pvRaf) { cancelAnimationFrame(pvRaf); pvRaf = null; }
  pvQueue.length = 0; jbPrev = null; lastRtt = 0; lastJitterBuffer = 0;
  pvCtx.clearRect(0, 0, PV_W, PV_H);
  if (channel) { channel.close(); channel = null; }
  if (pc) { pc.close(); pc = null; }
  if (localStream) { localStream.getTracks().forEach((t) => t.stop()); localStream = null; }
  localVideo.srcObject = null;
  remoteVideo.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  shootBtn.disabled = true;
  recBtn.disabled = true;
  recording = false;
  recBtn.textContent = "⏺ 학습 프레임 수집";
  recBtn.style.background = "#444";
  refsLoaded = false;
  setStatus("종료됨");
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

/* 마지막 안전망.
 *
 * 이 페이지에서 실패는 대부분 "아무 일도 안 일어난 것처럼" 보였다. 비동기
 * 경로에서 던져진 예외는 콘솔에만 남고 화면은 직전 문구에 멈춰 있어서,
 * 서버가 거절했는지 스크립트가 죽었는지 구분할 수가 없었다.
 * 개별 try/catch 로 다 못 잡는 것들을 여기서 화면까지 끌어올린다. */
/* 같은 오류를 접어서 보여준다.
 * 이 페이지의 오류는 rAF 콜백(초당 60회)이나 프레임 루프에서 나는 일이 잦다.
 * 그대로 흘리면 상태줄이 초당 60번 덮어써져서 정작 읽을 수가 없고, 콘솔도
 * 같은 줄로 가득 차 첫 번째 원인을 찾기 어려워진다. */
let lastErrKey = null, errRepeat = 0;
function reportError(label, msg, where) {
  const key = label + msg + (where || "");
  if (key === lastErrKey) {
    errRepeat++;
    // 접힌 횟수만 갱신한다. 새 오류가 아니므로 콘솔에는 다시 안 찍는다.
    statusEl.innerHTML = `<b>${label}</b>: ${escapeHtml(msg)}` +
      (where ? ` <span style="opacity:.7">(${escapeHtml(where)})</span>` : "") +
      ` <span style="opacity:.7">×${errRepeat + 1}</span>`;
    return;
  }
  lastErrKey = key;
  errRepeat = 0;
  setStatus(`<b>${label}</b>: ${escapeHtml(msg)}` +
            (where ? ` <span style="opacity:.7">(${escapeHtml(where)})</span>` : ""), true);
}

window.addEventListener("unhandledrejection", (e) => {
  const r = e.reason;
  reportError("처리되지 않은 오류", String((r && r.message) || r));
});
window.addEventListener("error", (e) => {
  reportError("스크립트 오류", e.message || "",
              e.filename ? `${e.filename}:${e.lineno}` : "");
});

/* 새로고침/탭 닫기에서 세션을 즉시 놓아준다.
 *
 * 안 놓으면 서버는 ICE 가 실패로 떨어질 때까지(또는 idle 타임아웃까지) 그
 * 세션을 활성으로 센다. 동시 접속 상한이 있으므로, 새로고침을 몇 번 하면
 * 자기 자신이 자리를 다 차지해 다음 접속이 503 으로 거절당한다. */
window.addEventListener("pagehide", () => {
  try { if (pc) pc.close(); } catch (_) {}
});

/* preflight.js 가 "모듈이 끝까지 살아서 올라왔는지" 판정하는 신호 */
window.__pageReady = true;
