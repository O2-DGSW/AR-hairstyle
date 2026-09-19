/* 3D 헤어 렌더링 — GLB + 오클루더 + 스프링본 + 라이팅 매칭
 *
 * 왜 3D인가
 * ---------
 * 2D 닮음변환은 자유도가 4개(이동/평면내 회전/크기)라 **평면 밖 회전을 표현할 수
 * 없다.** 그래서 고개를 돌리면 헤어가 정면을 향한 채 남았고, 각도별 에셋 뱅크로
 * 근사하려니 전환이 끊겼다. 메시를 실제 3D 포즈로 렌더하면 이 문제가 원리적으로
 * 사라진다 - 뱅크 자체가 필요 없어진다.
 *
 * 왜 온디바이스인가
 * -----------------
 * 3D 렌더링은 GPU 래스터라이즈다. 영상을 서버로 보내 렌더해서 되돌리면 왕복
 * 지연만 늘고 얻는 게 없다. 브라우저가 제일 잘하는 일이므로 여기서 한다.
 *
 * 좌표계
 * ------
 * MediaPipe 의 facialTransformationMatrix 는 정규 얼굴 모델(센티미터 단위)을
 * 카메라 공간으로 보내는 4x4 행렬이다. 이걸 그대로 Object3D 에 물리면 머리의
 * 회전/거리가 한 번에 들어온다. 별도 보정이 필요 없다는 게 이 방식의 핵심 이점.
 */
// 버전은 hair3d.html 의 import map 에서 고정한다. GLTFLoader 같은
// examples/jsm 파일들이 내부적으로 맨 이름 'three' 를 import 하기 때문에,
// 여기서 CDN 절대 URL 을 직접 쓰면 그쪽 해석이 실패한다.
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const W = 640, H = 480;
const el = (id) => document.getElementById(id);
const video = el("video"), glCanvas = el("gl");
const startBtn = el("start"), stopBtn = el("stop"), statusEl = el("status");

const ui = {
  size: el("c-size"), fwd: el("c-fwd"), up: el("c-up"),
  yaw: el("c-yaw"), pitch: el("c-pitch"),
  occ: el("c-occ"), occShow: el("c-occ-show"), fixed: el("c-fixed"),
  spring: el("c-spring"), stiff: el("c-stiff"), damp: el("c-damp"), grav: el("c-grav"),
  light: el("c-light"), lint: el("c-lint"),
};
["size", "fwd", "up", "yaw", "pitch", "stiff", "damp", "grav", "lint"].forEach((k) => {
  const sync = () => { el("v-c-" + k).textContent = ui[k].value; };
  ui[k].addEventListener("input", sync); sync();
});

let landmarker = null, stream = null, running = false, rafId = null;

function setStatus(m) { statusEl.textContent = m; console.log("[3d]", m); }

/* ---------------- three.js 기본 구성 ---------------- */
glCanvas.width = W; glCanvas.height = H;
const renderer = new THREE.WebGLRenderer({ canvas: glCanvas, alpha: true, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(W, H, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();

// MediaPipe 변환행렬은 정규 얼굴 모델을 '카메라 공간'으로 보낸다. 그 공간을
// 그대로 쓰려면 카메라를 원점에 두고 -Z 를 보게 한 뒤, 화각만 웹캠과 맞추면 된다.
const camera = new THREE.PerspectiveCamera(63, W / H, 1, 5000);
camera.position.set(0, 0, 0);
camera.lookAt(0, 0, -1);

const ambient = new THREE.AmbientLight(0xffffff, 0.75);
const key = new THREE.DirectionalLight(0xffffff, 1.1);
key.position.set(0.4, 0.6, 1);
scene.add(ambient, key);

// 머리에 붙는 루트. 변환행렬을 여기에 직접 물린다.
const headRoot = new THREE.Object3D();
headRoot.matrixAutoUpdate = false;
scene.add(headRoot);

// 헤어를 담는 노드 (크기/오프셋 조절용)
const hairPivot = new THREE.Object3D();
headRoot.add(hairPivot);

/* ---------------- 오클루더 ----------------
 * 색을 쓰지 않고 깊이만 기록하는 머리 모양 메시. 뒤통수 쪽 머리카락이 얼굴을
 * 뚫고 나오는 걸 막는다. 영상(배경)은 캔버스 밖에 있으므로 오클루더가 칠해도
 * 영상이 가려지지 않고, 그 자리의 헤어만 깊이 테스트에서 탈락한다.
 */
const occluderMat = new THREE.MeshBasicMaterial({
  colorWrite: false, depthWrite: true, side: THREE.FrontSide,
});
const occluderDebugMat = new THREE.MeshNormalMaterial({ wireframe: true });
const occluder = new THREE.Mesh(new THREE.SphereGeometry(8.2, 32, 24), occluderMat);
occluder.scale.set(1.0, 1.22, 1.12);      // 사람 두상 비율에 가깝게
occluder.position.set(0, 1.5, -1.0);
occluder.renderOrder = -1;                 // 헤어보다 먼저 그려 깊이를 채운다
headRoot.add(occluder);

/* ---------------- 절차적 헤어 (GLB 없을 때) ----------------
 * 파이프라인 검증용. 뼈대가 도는지 먼저 확인하고 에셋은 나중에 끼운다.
 * 스프링본을 붙일 수 있도록 아래로 늘어지는 가닥을 별도 오브젝트로 만든다.
 */
const hairMat = new THREE.MeshStandardMaterial({
  color: 0x2a1c14, roughness: 0.62, metalness: 0.0,
});

function buildProceduralHair() {
  const g = new THREE.Group();

  // 두피를 덮는 셸
  const cap = new THREE.Mesh(
    new THREE.SphereGeometry(8.9, 40, 28, 0, Math.PI * 2, 0, Math.PI * 0.62), hairMat);
  cap.scale.set(1.03, 1.2, 1.14);
  cap.position.set(0, 1.9, -1.0);
  g.add(cap);

  // 옆/뒤로 늘어지는 가닥 - 스프링본 대상
  const strands = [];
  const N = 14;
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2;
    const r = 7.6;
    const strand = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.85, 5.2, 4, 8), hairMat);
    strand.position.set(Math.cos(a) * r, 1.0, Math.sin(a) * r - 1.0);
    // 뒤통수 쪽을 더 길게
    const back = (Math.sin(a) + 1) / 2;
    strand.scale.set(1, 0.75 + back * 0.9, 1);
    g.add(strand);
    strands.push({
      obj: strand,
      rest: strand.position.clone(),
      vel: new THREE.Vector3(),
      cur: strand.position.clone(),
    });
  }
  g.userData.strands = strands;
  return g;
}

let hairObj = buildProceduralHair();
hairPivot.add(hairObj);

/* ---------------- GLB 로딩 ---------------- */
const loader = new GLTFLoader();

function setFit(fit) {
  // 그룸 json 의 user_fit -> 슬라이더. 없으면 기본값으로 되돌린다(이전 스타일 보정이 남지 않게).
  const f = fit || {};
  ui.size.value = Math.round((f.scale_mul ?? 1) * 100);
  ui.up.value = Math.round((f.offset_up_cm ?? 0) * 10);
  ui.fwd.value = Math.round((f.offset_fwd_cm ?? 0) * 10);
  ui.yaw.value = 0; ui.pitch.value = 0;
  ["size", "up", "fwd", "yaw", "pitch"].forEach((k) => ui[k].dispatchEvent(new Event("input")));
}

let groomOccluder = null;   // GLB 안의 head_occluder 메시 (없으면 내장 타원체 사용)

function useHair(obj) {
  hairPivot.remove(hairObj);
  hairObj = obj;
  hairPivot.add(hairObj);
  if (!obj.getObjectByName || !obj.getObjectByName("head_occluder")) groomOccluder = null;
}

/** url 의 GLB 를 헤어로 교체. label 은 상태 표시용, fit 은 그룸 json 의 user_fit.
 *  onDone 은 성공/실패 뒤 한 번 호출(로컬 파일의 blob URL 해제용). */
function loadGroom(url, label, fit, onDone) {
  setStatus(`${label} 불러오는 중…`);
  loader.load(url, (gltf) => {
    // 아바타 통짜 GLB(Ready Player Me 등)는 바운딩박스가 몸 전체라, 그대로 쓰면
    // 어깨 폭이 19cm 로 정규화되어 헤어가 얼굴에 파묻힌다. 이름에 hair 가 들어간
    // 노드가 있으면 그것만 떼어 쓴다. (스킨드 메시면 본 계층에서 분리되므로
    // 바인드 포즈로 고정된다 — 정적 헤어라 문제 없다)
    // 우리 그룸 GLB 는 heddy_hair + head_occluder 두 노드가 같은 프레임에 있으므로 통째로 쓴다.
    let occ = null;
    gltf.scene.traverse((o) => { if (!occ && o.isMesh && o.name === "head_occluder") occ = o; });
    let picked = null;
    if (!occ) {
      gltf.scene.traverse((o) => {
        if (!picked && o !== gltf.scene && /hair|헤어|髪/i.test(o.name)) picked = o;
      });
      if (picked) picked.removeFromParent();
    }
    const obj = picked || gltf.scene;
    if (occ) {
      // FLAME 두상을 깊이 전용 오클루더로: 헤어 뿌리가 이 표면 위에 정확히 놓여 있어
      // 내장 타원체처럼 정수리를 삼키지 않는다. 내장 타원체는 이 그룸 동안 끈다.
      occ.material = occluderMat;
      occ.renderOrder = -1;
      occ.userData.isGroomOccluder = true;
    }
    groomOccluder = occ;

    // 에셋 크기가 제각각이므로 두상 기준으로 정규화한다.
    // (에셋마다 단위가 미터/센티미터/임의로 달라서 이 단계가 없으면 안 보이거나
    //  화면을 뒤덮는다)
    // 단, 우리 변환기(strands_to_cards.py)가 만든 GLB 는 이미 MediaPipe 정규 프레임
    // (cm, 두상 중심 원점) 이라 바운딩박스 정규화가 오히려 망친다 — 그대로 쓴다.
    const ud = gltf.scene.userData || {};
    const canonical = !!(ud.heddy_canonical || (ud.extras && ud.extras.heddy_canonical));
    let k = 1;
    if (!canonical) {
      const box = new THREE.Box3().setFromObject(obj);
      const size = new THREE.Vector3(); box.getSize(size);
      const center = new THREE.Vector3(); box.getCenter(center);
      const target = 19.0;                     // 정규 얼굴 모델 기준 머리 폭(cm 근사)
      k = target / Math.max(size.x, 1e-6);
      obj.scale.setScalar(k);
      obj.position.sub(center.multiplyScalar(k));
      obj.position.y += 1.5;
    }

    // 본이 있으면 스프링본 대상으로 잡는다
    const bones = [];
    obj.traverse((o) => { if (o.isBone) bones.push(o); });
    obj.userData.bones = bones;

    useHair(obj);
    setFit(fit);
    setStatus(`${label} 로드됨 — 노드 "${obj.name || "(루트 전체)"}", 본 ${bones.length}개, ` +
              (canonical ? "정규 프레임(자동 스케일 없음). " : `자동 스케일 ×${k.toFixed(3)}. `) +
              `방향이 틀어졌으면 [좌우돌림]/[앞뒤기울기] 로 맞추세요.`);
    if (onDone) onDone();
  }, undefined, (err) => {
    setStatus(`${label} 로드 실패: ` + err.message);
    if (onDone) onDone();
  });
}

// 로컬 파일
el("glb").addEventListener("change", (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  el("groom").value = "";
  const url = URL.createObjectURL(f);
  loadGroom(url, f.name, null, () => URL.revokeObjectURL(url));
});

// 서버 그룸 드롭다운 (server/grooms/*.glb — /grooms 가 목록을 준다)
const groomSel = el("groom");
let groomIndex = {};
const groomsReady = fetch("/grooms").then((r) => r.json()).then((j) => {
  for (const g of j.grooms || []) {
    groomIndex[g.name] = g;
    const o = document.createElement("option");
    o.value = g.name;
    o.textContent = g.n_strands ? `${g.name} (${g.n_strands} 가닥)` : g.name;
    groomSel.appendChild(o);
  }
}).catch((e) => console.warn("[3d] /grooms 실패", e));
function selectGroom(name) {
  el("glb").value = "";
  if (!name) { useHair(buildProceduralHair()); setFit(null); setStatus("절차적 헤어로 전환"); return; }
  const g = groomIndex[name];
  if (!g) { setStatus(`스타일 없음: ${name}`); return; }
  loadGroom(g.url, name, g.user_fit);
}
groomSel.addEventListener("change", () => selectGroom(groomSel.value));

// 딥링크: ?groom=<이름>&autostart=1  (헤드리스 재현/공유용)
const qs = new URLSearchParams(location.search);
groomsReady.then(() => {
  const want = qs.get("groom");
  if (want) { groomSel.value = want; selectGroom(want); }
  if (qs.get("fixed") === "1") ui.fixed.checked = true;
  if (qs.get("autostart") === "1") start();
});

/* ---------------- 스프링본 ----------------
 * 각 가닥을 질량-스프링-감쇠로 푼다. 머리가 움직이면 관성 때문에 가닥이 뒤따라
 * 흔들린다. 본이 있는 GLB 라면 본 체인에, 없으면 절차적 가닥에 적용한다.
 */
const prevHeadPos = new THREE.Vector3();
let hasPrevHead = false;

function stepSprings(dt) {
  const strands = hairObj.userData && hairObj.userData.strands;
  if (!ui.spring.checked || !strands) return;

  const stiff = Number(ui.stiff.value);
  const damp = Number(ui.damp.value) / 100;
  const grav = Number(ui.grav.value) / 100;

  // 머리의 월드 가속을 관성력으로 쓴다
  const hp = new THREE.Vector3().setFromMatrixPosition(headRoot.matrix);
  const inertia = new THREE.Vector3();
  if (hasPrevHead) inertia.subVectors(prevHeadPos, hp).multiplyScalar(60 * dt);
  prevHeadPos.copy(hp); hasPrevHead = true;

  for (const s of strands) {
    // 복원력(정지 위치로) + 중력 + 머리 관성
    const f = new THREE.Vector3().subVectors(s.rest, s.cur).multiplyScalar(stiff);
    f.y -= grav * 9.8;
    f.add(inertia.clone().multiplyScalar(6));

    s.vel.addScaledVector(f, dt);
    s.vel.multiplyScalar(Math.pow(damp, dt * 60));
    s.cur.addScaledVector(s.vel, dt);

    // 너무 멀리 날아가지 않게 제한
    const d = s.cur.distanceTo(s.rest);
    if (d > 3.2) s.cur.copy(s.rest).addScaledVector(
      s.cur.clone().sub(s.rest).normalize(), 3.2);

    s.obj.position.copy(s.cur);
    s.obj.lookAt(s.rest.clone().setY(s.rest.y + 4));
  }
}

/* ---------------- 라이팅 매칭 ----------------
 * 영상 프레임에서 밝기/색을 뽑아 조명에 반영한다. 헤어만 원래 조명으로 렌더하면
 * 색온도가 어긋나 붙여넣은 티가 난다.
 */
const lightCanvas = document.createElement("canvas");
lightCanvas.width = 32; lightCanvas.height = 24;
const lightCtx = lightCanvas.getContext("2d", { willReadFrequently: true });

function matchLighting() {
  if (!ui.light.checked) {
    ambient.color.setRGB(1, 1, 1); key.color.setRGB(1, 1, 1);
    return;
  }
  lightCtx.drawImage(video, 0, 0, 32, 24);
  const d = lightCtx.getImageData(0, 0, 32, 24).data;

  let r = 0, g = 0, b = 0, n = 0;
  let lx = 0, ly = 0, lw = 0;
  for (let y = 0; y < 24; y++) {
    for (let x = 0; x < 32; x++) {
      const i = (y * 32 + x) * 4;
      r += d[i]; g += d[i + 1]; b += d[i + 2]; n++;
      const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      // 밝은 쪽으로 무게를 실어 광원 방향을 추정
      const wgt = Math.pow(lum / 255, 3);
      lx += (x / 31 - 0.5) * wgt; ly += (0.5 - y / 23) * wgt; lw += wgt;
    }
  }
  r /= n * 255; g /= n * 255; b /= n * 255;
  const gain = Number(ui.lint.value) / 100;

  // 평균색을 정규화해 색온도만 취한다 (밝기는 세기 슬라이더로)
  const mx = Math.max(r, g, b) || 1;
  ambient.color.setRGB(r / mx, g / mx, b / mx);
  key.color.copy(ambient.color);
  ambient.intensity = 0.55 * gain;
  key.intensity = 1.15 * gain;

  if (lw > 1e-4) {
    const dx = lx / lw, dy = ly / lw;
    key.position.set(dx * 2, dy * 2 + 0.35, 1).normalize();
    el("s-ldir").textContent = `${dx >= 0 ? "→" : "←"} ${Math.abs(dx).toFixed(2)}, ${dy >= 0 ? "↑" : "↓"} ${Math.abs(dy).toFixed(2)}`;
  }
  el("s-lcolor").textContent =
    `#${[r, g, b].map((v) => Math.round(v * 255).toString(16).padStart(2, "0")).join("")}`;
}

/* ---------------- 메인 루프 ---------------- */
const fpsBuf = [];
let lastT = 0, lastVideoTime = -1, lastDetectTs = -1;
const mpMatrix = new THREE.Matrix4();

// MediaPipe(tasks-vision 0.10.14) 의 facialTransformationMatrix 는 이미 three.js 와 같은
// 규약이다: x 오른쪽, y 위, 카메라가 -z 를 보고 얼굴은 z≈-50 에 온다(실측 (4.8, -3.8, -49.4),
// 정면 회전 = 단위행렬). 예전에 y/z 를 뒤집던 FLIP 은 얼굴을 카메라 **뒤**(z=+50)로 보내고
// 거꾸로 세워서 아무것도 안 보이게 했다. 그래서 항등행렬로 둔다 — 규약이 다른 버전을 만나면
// 여기만 바꾸면 된다.
const FLIP = new THREE.Matrix4();

function loop(now) {
  if (!running) return;
  rafId = requestAnimationFrame(loop);
  const fixed = ui.fixed.checked;
  if (!fixed) {
    if (video.readyState < 2) return;
    if (video.currentTime === lastVideoTime) return;   // 같은 프레임 재처리 방지
  }
  const dt = Math.min(0.05, lastT ? (now - lastT) / 1000 : 0.016);
  if (lastT) {
    fpsBuf.push(now - lastT);
    if (fpsBuf.length > 30) fpsBuf.shift();
    const avg = fpsBuf.reduce((a, b) => a + b, 0) / fpsBuf.length;
    el("s-fps").textContent = (1000 / avg).toFixed(1);
  }
  lastT = now;
  lastVideoTime = video.currentTime;

  let mats = null;
  if (fixed) {
    // 추적 없이 정면 45cm (MediaPipe 와 같은 규약: 회전 없음, z=-45)
    mats = [{ data: new THREE.Matrix4().makeTranslation(0, 0, -45).toArray() }];
    el("s-detect").textContent = "고정";
  } else if (landmarker) {
    const t0 = performance.now();
    const ts = Math.max(Math.floor(now), lastDetectTs + 1);   // 단조증가 보장
    lastDetectTs = ts;
    const res = landmarker.detectForVideo(video, ts);
    el("s-detect").textContent = (performance.now() - t0).toFixed(1) + " ms";
    mats = res.facialTransformationMatrixes;
  }
  if (mats && mats.length) {
    // column-major 16개 -> three.js Matrix4
    mpMatrix.fromArray(mats[0].data);
    const raw = mpMatrix.elements;
    el("s-raw").textContent = `${raw[12].toFixed(1)}, ${raw[13].toFixed(1)}, ${raw[14].toFixed(1)}`;
    mpMatrix.premultiply(FLIP);
    headRoot.matrix.copy(mpMatrix);
    headRoot.matrixWorldNeedsUpdate = true;

    const e = new THREE.Euler().setFromRotationMatrix(mpMatrix, "YXZ");
    const deg = (v) => (v * 180 / Math.PI).toFixed(0);
    el("s-ypr").textContent = `${deg(e.y)}° / ${deg(e.x)}° / ${deg(e.z)}°`;
    const tz = Math.abs(mpMatrix.elements[14]);
    el("s-tz").textContent = tz.toFixed(1);

    // 자동 스케일: 변환행렬이 이미 거리를 담고 있으므로 추가 보정이 거의 필요
    // 없다. 사용자 슬라이더만 곱한다. (2D 때는 여기에 K/tz 캘리브레이션이
    // 필요했는데, 3D 에서는 행렬이 그 일을 대신한다)
    const s = Number(ui.size.value) / 100;
    hairPivot.scale.setScalar(s);
    hairPivot.position.set(0, Number(ui.up.value) / 10, -Number(ui.fwd.value) / 10);
    // 에셋마다 축 방향이 달라서(뒤통수가 앞으로 오는 경우가 흔하다) 수동 보정을 둔다
    const d2r = Math.PI / 180;
    hairPivot.rotation.set(Number(ui.pitch.value) * d2r, Number(ui.yaw.value) * d2r, 0);
    el("s-scale").textContent = `×${s.toFixed(2)} (행렬 자동)`;

    headRoot.visible = true;
  } else {
    headRoot.visible = false;
    el("s-ypr").textContent = "-";
  }

  if (groomOccluder) {
    occluder.visible = false;
    groomOccluder.visible = ui.occ.checked || ui.occShow.checked;
    groomOccluder.material = ui.occShow.checked ? occluderDebugMat : occluderMat;
  } else {
    occluder.visible = ui.occ.checked || ui.occShow.checked;
    occluder.material = ui.occShow.checked ? occluderDebugMat : occluderMat;
  }

  if (fixed) { ambient.color.setRGB(1, 1, 1); key.color.setRGB(1, 1, 1); ambient.intensity = 0.75; key.intensity = 1.1; }
  else matchLighting();
  stepSprings(dt);
  renderer.render(scene, camera);
}

/* ---------------- 시작/정지 ---------------- */
async function start() {
  startBtn.disabled = true;
  if (ui.fixed.checked) {
    // 디버그: 웹캠/MediaPipe 없이 렌더 루프만
    running = true; stopBtn.disabled = false;
    setStatus("포즈 고정 모드 — 정면 45cm 에 헤어만 렌더 (웹캠 없음)");
    rafId = requestAnimationFrame(loop);
    return;
  }
  try {
    setStatus("MediaPipe 로딩 중...");
    const vision = await FilesetResolver.forVisionTasks(CDN + "/wasm");
    let delegate = "GPU";
    try {
      landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
        runningMode: "VIDEO", numFaces: 1,
        outputFacialTransformationMatrixes: true,
      });
    } catch (e) {
      delegate = "CPU (폴백)";
      landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL, delegate: "CPU" },
        runningMode: "VIDEO", numFaces: 1,
        outputFacialTransformationMatrixes: true,
      });
    }
    el("s-delegate").textContent = delegate;
    el("s-delegate").className = delegate === "GPU" ? "good" : "warn";

    setStatus("웹캠 요청 중...");
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: W, height: H, frameRate: { ideal: 30 } }, audio: false,
    });
    video.srcObject = stream;
    await video.play();

    running = true;
    stopBtn.disabled = false;
    setStatus("동작 중 — 고개를 돌려보세요. 3D 포즈로 렌더하므로 각도 뱅크 없이 따라 돌아야 합니다.");
    // 첫 호출도 rAF 로: performance.now() 를 직접 넘기면 다음 rAF 타임스탬프가 그보다 작을 수
    // 있고, MediaPipe VIDEO 모드는 타임스탬프가 줄면 그래프가 영구히 죽는다.
    rafId = requestAnimationFrame(loop);
  } catch (err) {
    setStatus("시작 실패: " + err.message);
    startBtn.disabled = false;
  }
}

function stop() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  video.srcObject = null;
  renderer.clear();
  fpsBuf.length = 0; lastT = 0; lastVideoTime = -1; lastDetectTs = -1; hasPrevHead = false;
  startBtn.disabled = false; stopBtn.disabled = true;
  setStatus("정지됨");
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

/* preflight.js 가 "모듈이 끝까지 살아서 올라왔는지" 판정하는 신호 */
window.__pageReady = true;
