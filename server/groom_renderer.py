"""3D 헤어 그룸(헤어카드 GLB) 을 얼굴 포즈로 렌더해 (rgb, alpha) GPU 텐서로 돌려준다.

왜 이게 있나
------------
2D 에셋 워핑(gpu_segmenter._warp_asset)은 닮음변환이라 평면 밖 회전을 못 만든다 - 고개를
돌리면 헤어가 정면인 채 남는다. 각도별 뱅크로 근사하면 칸 전환이 끊긴다. 3D 메시를
**그 프레임의 포즈 행렬로** 직접 래스터라이즈하면 이 문제가 원리적으로 사라진다.
반환 계약은 _warp_asset 과 같아서(rgb (h,w,3) BGR 0~255, a (h,w,1) 0~1, 둘 다 GPU) 뒤의
지우기/플레이트/이마/그림자/그레인 합성 코드는 한 줄도 안 바뀐다.

좌표계
------
MediaPipe facialTransformationMatrix 는 정규 얼굴 모델(cm) -> 카메라 공간. 카메라는 -z 를
보고 y 가 위, 얼굴은 z≈-50 에 온다(face_pose.py 실측). PnP 가 **수직 화각 63°** 의 가상
카메라를 가정하므로 같은 카메라로 투영해야 랜드마크와 정확히 겹친다 - 실제 웹캠 화각은
몰라도 된다. 그룸 GLB 는 strands_to_cards.py 가 같은 정규 프레임(cm, 두상 중심 원점,
+y 위, +z 정면)으로 만든 것이라 별도 보정 없이 행렬 하나로 붙는다.

렌더
----
헤어카드는 알파 리본이라 순서 의존 투명도 문제가 있다. 정렬 없이 두 패스로 충분히
가린다: (1) 알파 ≥ 0.5 는 알파 테스트 + 깊이 쓰기 (2) 나머지 반투명 가장자리는 블렌드 +
깊이 읽기만. head_occluder 노드(FLAME 두상)는 색 없이 깊이만 써서 뒤통수 쪽 가닥이
얼굴을 뚫지 않게 한다. MSAA 4x 로 리본 가장자리 계단을 누른다.
결과는 검은 투명 배경 위 블렌드라 색이 프리멀티플라이드 -> 알파로 나눠 되돌린다.

스레드
------
GL 컨텍스트는 만든 스레드에서만 쓸 수 있다. gpu_segmenter.process 가 도는 gpu_executor
단일 스레드에서 처음 부를 때 만들고 그 스레드에서만 쓴다 (CUDA 그래프와 같은 제약).
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time

import numpy as np
import torch

logger = logging.getLogger("groom")

#: MediaPipe face geometry 의 기본 가상 카메라. 이 값으로 PnP 를 풀었으니 이 값으로 투영한다.
MP_VERTICAL_FOV_DEG = 63.0

_VS = """
#version 400
uniform mat4 u_proj;        // 카메라 공간 -> 클립
uniform mat4 u_model;       // 두상(정규, cm) -> 카메라 공간
uniform mat3 u_nrm;
// 2차 운동 (hair_dynamics.py): 뿌리 기준 상대좌표를 [현재 -> 지연 자세] 로 끝쪽만 섞는다
uniform mat3 u_lag;         // 두상 공간 회전 (현재 -> 지연)
uniform vec3 u_shift;       // 두상 공간 이동 지연 (cm)
uniform float u_tip;        // 끝 가중치 배율 (0 이면 강체)
uniform float u_idle;       // 미세 흔들림 진폭 (cm)
uniform float u_time;
uniform int u_billboard;    // 1: _center/_tangent/_side/_halfw 로 카메라를 향하는 리본을 만든다 (v1 GLB)
uniform float u_scale;      // 두상 스케일(폭에 곱함)
in vec3 in_pos; in vec2 in_uv; in vec3 in_nrm; in vec3 in_root;
in vec3 in_center; in vec3 in_tangent; in float in_side; in float in_rand; in float in_halfw;
out vec2 v_uv; out vec3 v_nrm; out vec3 v_tan; out vec3 v_pos; flat out float v_rand;
void main() {
    float s = in_uv.y;                      // 0 = 뿌리, 1 = 끝 (strands_to_cards 의 v 를 로더가 뒤집음)
    vec3 base = (u_billboard == 1) ? in_center : in_pos;
    vec3 rel = base - in_root;
    float w = clamp(u_tip * s * s, 0.0, 1.0);
    vec3 rel2 = u_lag * rel + u_shift * s;
    vec3 d = mix(rel, rel2, w);
    float ph = dot(in_root, vec3(12.9898, 78.233, 37.719));
    d += vec3(sin(u_time * 1.7 + ph), 0.0, cos(u_time * 1.3 + ph * 0.7)) * (u_idle * s * s);
    float L = length(rel);
    float Ld = length(d);
    if (L > 1e-4 && Ld > 1e-4) d *= L / Ld;
    vec3 p = in_root + d;                   // 두상 공간
    vec4 pc = u_model * vec4(p, 1.0);       // 카메라 공간
    vec3 tc = normalize(mat3(u_model) * ((u_billboard == 1) ? in_tangent : in_nrm));
    if (u_billboard == 1) {
        // 리본 폭 방향 = 접선 x 시선. 매 프레임 카메라를 향하므로 꼬임/계단이 없고 폭이 일정하다.
        vec3 view = normalize(-pc.xyz);
        vec3 side = cross(tc, view);
        float sl = length(side);
        side = (sl > 1e-4) ? side / sl : vec3(1.0, 0.0, 0.0);
        pc.xyz += side * (in_side * in_halfw * u_scale);
    }
    gl_Position = u_proj * pc;
    v_uv = in_uv;
    v_nrm = normalize(u_nrm * in_nrm);
    v_tan = tc;
    v_pos = pc.xyz;
    v_rand = in_rand;
}
"""

_FS = """
#version 400
uniform sampler2D u_tex;
uniform vec3 u_base;        // 베이스 색 (0~1, RGB)
uniform int u_pass;         // 0: 컷아웃(a>=0.5), 1: 반투명 가장자리(a<0.5), 2: 깊이만, 3: 알파-투-커버리지 단일 패스
uniform vec3 u_light;       // 카메라 공간 광원 방향
uniform int u_billboard;
uniform int u_samples;      // MSAA 샘플 수 (패스 3)
in vec2 v_uv; in vec3 v_nrm; in vec3 v_tan; in vec3 v_pos; flat in float v_rand;
out vec4 f_color;
float hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
void main() {
    // gl_SampleMask 를 한 곳에서라도 쓰면 안 쓴 경로에선 값이 미정의다(GLSL 규칙) -> 항상 전부 덮음으로 시작
    gl_SampleMask[0] = 0xFFFFFFFF;
    if (u_pass == 2) { f_color = vec4(0.0); return; }
    vec4 t = texture(u_tex, v_uv);
    float a = t.a;
    if (u_pass == 0 && a < 0.5) discard;
    if (u_pass == 1 && a >= 0.5) discard;
    if (u_pass == 3) {
        // 확률적 투명도: 알파 비율만큼의 샘플을 덮되, 어느 샘플부터 덮을지를 가닥+픽셀 해시로 돌린다.
        // 하드웨어 알파-투-커버리지는 알파값마다 고정 패턴이라 겹치는 가닥이 같은 샘플만 덮어
        // 밀도가 누적되지 않았다(실측: 알파 합이 1/7). 마스크를 돌리면 겹칠수록 덮인다 = 블렌딩과 같아진다.
        if (a < 0.02) discard;
        float n = float(u_samples);
        float jitter = hash12(gl_FragCoord.xy + v_rand * 97.0);
        int k = int(clamp(floor(a * n + jitter), 0.0, n));   // 디더로 소수 알파도 평균적으로 맞춘다
        if (k <= 0) discard;
        int off = int(hash12(gl_FragCoord.yx * 1.7 + v_rand * 31.0) * n);
        int mask = 0;
        for (int i = 0; i < k; i++) mask |= 1 << ((off + i) % u_samples);
        gl_SampleMask[0] = mask;
    }
    vec3 Ld = normalize(u_light);
    float shade;
    if (u_billboard == 1) {
        // Kajiya-Kay: 머리카락은 접선 방향으로 늘어진 원기둥. 확산은 sin(T,L), 반사는 결을 따라 흐른다.
        vec3 V = normalize(-v_pos);
        vec3 H = normalize(Ld + V);
        float tl = dot(v_tan, Ld);
        float diff = sqrt(max(0.0, 1.0 - tl * tl));
        float th = dot(v_tan, H);
        float spec = pow(sqrt(max(0.0, 1.0 - th * th)), 48.0);
        shade = 0.45 + 0.55 * diff + 0.35 * spec;
        shade *= 0.82 + 0.36 * v_rand;      // 가닥별 밝기 변주 (뭉침 방지)
    } else {
        float nl = abs(dot(v_nrm, Ld));
        shade = 0.55 + 0.45 * nl;
    }
    vec3 c = u_base * t.rgb * shade;
    // 패스 0 은 컷아웃이라 완전 불투명으로 쓴다(블렌드 없음). 패스 1 은 SRC_ALPHA 블렌드가
    // 프리멀티플라이드로 누적하므로 스트레이트 색을 낸다. 패스 3(A2C)은 커버리지가 알파를 나른다.
    // 패스 3 은 샘플 마스크가 알파를 나르므로 알파 1 로 쓴다(리졸브 = 커버리지).
    f_color = (u_pass == 0 || u_pass == 3) ? vec4(c, 1.0) : vec4(c, a);
}
"""


# 두상 좌표 맵: 오클루더(FLAME 두상) 앞면의 정규 좌표(cm) 를 8비트로 인코딩해 그린다.
# 합성 쪽이 픽셀마다 "두피/옆머리/목" 을 가려 지운 원래 머리 자리를 무엇으로 채울지 정한다.
_HEAD_VS = """
#version 400
uniform mat4 u_mvp;
in vec3 in_pos; in float in_scalp;
out vec3 v_c; out float v_scalp;
void main() { v_c = in_pos; v_scalp = in_scalp; gl_Position = u_mvp * vec4(in_pos, 1.0); }
"""
_HEAD_FS = """
#version 400
in vec3 v_c; in float v_scalp;
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_pos;
layout(location = 2) out vec4 f_nrm;
void main() {
    // R: 두피 가중(이 스타일의 뿌리가 나는 자리 1), G: y -> (y+25)/45, B: z -> (z+20)/30, A: 1 = 두상 위
    f_color = vec4(v_scalp, (v_c.y + 25.0) / 45.0, (v_c.z + 20.0) / 30.0, 1.0);
    // 정규 좌표(cm, float16) 와 면 노멀 - 다른 순간에 찍은 사진을 이 표면에 투영 텍스처로 붙이는 데 쓴다
    f_pos = vec4(v_c, 1.0);
    vec3 n = normalize(cross(dFdx(v_c), dFdy(v_c)));
    f_nrm = vec4(n * 0.5 + 0.5, 1.0);
}
"""
HEAD_ENC_Y = (25.0, 45.0)   # (offset, range) - 위 셰이더와 맞출 것
HEAD_ENC_Z = (20.0, 30.0)
#: 두피 가중: 오클루더 정점에서 가장 가까운 뿌리까지 이 거리(cm) 안이면 1, 밖으로 0 까지 선형.
SCALP_NEAR_CM, SCALP_FAR_CM = 0.6, 1.6


def _scalp_weight(occ_v: np.ndarray, roots: np.ndarray) -> np.ndarray:
    """오클루더 정점마다 '이 스타일 머리가 나는 자리인가' 0~1. 뿌리 점과의 최근접 거리로."""
    if roots is None or len(roots) == 0:
        return np.zeros(len(occ_v), np.float32)
    r = np.unique(np.round(roots, 1), axis=0).astype(np.float32)
    best = np.full(len(occ_v), np.inf, np.float32)
    for i in range(0, len(r), 2048):
        d = ((occ_v[:, None, :] - r[None, i:i + 2048, :]) ** 2).sum(-1).min(1)
        best = np.minimum(best, d)
    d = np.sqrt(best)
    return np.clip((SCALP_FAR_CM - d) / (SCALP_FAR_CM - SCALP_NEAR_CM), 0.0, 1.0).astype(np.float32)


def _perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    t = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = t / aspect
    m[1, 1] = t
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


class Groom:
    """GLB 한 벌의 GPU 자원. GroomRenderer.load() 가 만든다."""

    def __init__(self, name: str, path: str, meta: dict):
        self.name = name
        self.path = path
        self.meta = meta or {}
        fit = self.meta.get("user_fit", {})
        self.scale_mul = float(fit.get("scale_mul", 1.0))
        self.offset_up_cm = float(fit.get("offset_up_cm", 0.0))
        self.offset_fwd_cm = float(fit.get("offset_fwd_cm", 0.0))
        self.hair_vao = None
        self.occ_vao = None
        self.head_vao = None     # 오클루더 + 정점별 두피 가중, 두상 좌표 맵 프로그램용
        self.scalp_cov = 0.0
        self.tex = None
        self.base = (0.23, 0.14, 0.09)
        self.n_tris = 0
        self.billboard = False   # v1 GLB (_center/_tangent/_side/_halfw 속성) 이면 True


class GroomRenderer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._ctx = None
        self._prog = None
        self._thread = None
        self._fbo_ms = None      # (w,h) 별 MSAA 프레임버퍼
        self._fbo = None         # 리졸브 대상
        self._fbo_size = None
        self._samples = 8
        self._head_prog = None
        self._fbo_head = None    # 두상 좌표 맵 (MSAA 없음)
        self._grooms: dict[str, Groom] = {}
        self.last_ms = 0.0
        self.last_head = None
        self.last_geom = None    # (pos (h,w,3) cm, nrm (h,w,3)) - render(geom=True) 때만
        self.last_fit = None
        self.last_proj_m = None

    # ---- 컨텍스트 ----
    def _ensure_ctx(self):
        if self._ctx is not None:
            if threading.get_ident() != self._thread:
                raise RuntimeError("GroomRenderer 는 만든 스레드(gpu_executor)에서만 쓸 수 있다")
            return
        import moderngl
        t0 = time.perf_counter()
        self._ctx = moderngl.create_standalone_context(require=400)   # gl_SampleMask
        self._thread = threading.get_ident()
        self._prog = self._ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        self._head_prog = self._ctx.program(vertex_shader=_HEAD_VS, fragment_shader=_HEAD_FS)
        logger.info("moderngl 컨텍스트: %s (%.0fms)", self._ctx.info.get("GL_RENDERER"),
                    (time.perf_counter() - t0) * 1000)

    def _ensure_fbo(self, w: int, h: int):
        if self._fbo_size == (w, h):
            return
        ctx = self._ctx
        for f in (self._fbo_ms, self._fbo, self._fbo_head):
            if f is not None:
                f.release()
        self._fbo_head = None
        # 8x: 알파-투-커버리지가 알파를 8단계로 디더링한다 (4x 면 4단계라 결이 거칠다)
        ns = min(8, ctx.max_samples)
        self._samples = ns
        self._fbo_ms = ctx.framebuffer(
            color_attachments=[ctx.renderbuffer((w, h), components=4, samples=ns)],
            depth_attachment=ctx.depth_renderbuffer((w, h), samples=ns))
        self._fbo = ctx.framebuffer(color_attachments=[ctx.texture((w, h), components=4)])
        if self._fbo_head is not None:
            self._fbo_head.release()
        self._fbo_head = ctx.framebuffer(color_attachments=[ctx.texture((w, h), components=4),
                                                            ctx.texture((w, h), components=4, dtype="f2"),
                                                            ctx.texture((w, h), components=4)],
                                         depth_attachment=ctx.depth_renderbuffer((w, h)))
        self._fbo_size = (w, h)

    # ---- 로딩 ----
    def load(self, name: str, path: str, meta: dict | None = None) -> Groom:
        """GLB -> GPU. 같은 이름은 캐시. gpu_executor 스레드에서 부를 것."""
        g = self._grooms.get(name)
        if g is not None:
            return g
        self._ensure_ctx()
        import trimesh
        ctx = self._ctx
        t0 = time.perf_counter()
        scene = trimesh.load(path, process=False)
        g = Groom(name, path, meta or {})
        occ = None          # (v, faces) - 두상 좌표 맵은 뿌리를 다 모은 뒤에 만든다
        all_roots = []
        for gname, geom in scene.geometry.items():
            T = scene.graph.get(gname)[0] if gname in scene.graph.nodes_geometry else np.eye(4)
            v = trimesh.transform_points(geom.vertices, T).astype(np.float32)
            faces = np.asarray(geom.faces, dtype=np.int32)
            # vertex_normals 는 100만 정점에서 매우 느리다 - 면 노멀을 정점에 뿌린다
            fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
            n = np.zeros_like(v)
            np.add.at(n, faces.ravel(), np.repeat(fn, 3, 0))
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
            uv = getattr(geom.visual, "uv", None)
            if uv is None or len(uv) != len(v):
                uv = np.zeros((len(v), 2), np.float32)
            uv = np.asarray(uv, np.float32).copy()
            uv[:, 1] = 1.0 - uv[:, 1]           # glTF 는 v 가 위->아래, GL 텍스처는 아래->위
            # 뿌리 위치(2차 운동의 회전 중심). strands_to_cards 는 정점을 [strand][pt][좌/우] 순으로
            # 쓰므로 pts_per_strand 로 되감을 수 있다. 맞지 않으면(다른 GLB) 자기 자신 = 강체.
            root = v.copy()
            P = int((meta or {}).get("pts_per_strand", 0))
            if gname != "head_occluder" and P > 0 and len(v) % (P * 2) == 0:
                strands = v.reshape(-1, P, 2, 3)
                r0 = strands[:, 0, :, :].mean(axis=1)                       # (N,3) 뿌리(좌우 평균)
                root = np.repeat(r0, P * 2, axis=0).astype(np.float32)
            va = getattr(geom, "vertex_attributes", {}) or {}
            has_bb = all(k in va for k in ("_center", "_tangent", "_side", "_halfw")) and gname != "head_occluder"
            if has_bb:
                center = trimesh.transform_points(np.asarray(va["_center"], np.float32), T).astype(np.float32)
                tangent = (np.asarray(va["_tangent"], np.float32) @ T[:3, :3].T).astype(np.float32)
                side = np.asarray(va["_side"], np.float32).reshape(-1, 1)
                halfw = np.asarray(va["_halfw"], np.float32).reshape(-1, 1)
                rand = np.asarray(va.get("_rand", np.zeros(len(v))), np.float32).reshape(-1, 1)
                if P > 0 and len(v) % (P * 2) == 0:
                    # 뿌리는 중심선 기준으로 (구운 POSITION 은 폭 오프셋이 들어 있다)
                    r0 = center.reshape(-1, P, 2, 3)[:, 0, :, :].mean(axis=1)
                    root = np.repeat(r0, P * 2, axis=0).astype(np.float32)
                g.billboard = True
            else:
                center = v; tangent = n
                side = np.zeros((len(v), 1), np.float32); halfw = side; rand = side
            vbo = ctx.buffer(np.hstack([v, uv, n, root, center, tangent, side, rand, halfw]).astype(np.float32).tobytes())
            ibo = ctx.buffer(faces.tobytes())
            vao = ctx.vertex_array(self._prog, [(vbo, "3f 2f 3f 3f 3f 3f 1f 1f 1f",
                                                 "in_pos", "in_uv", "in_nrm", "in_root",
                                                 "in_center", "in_tangent", "in_side", "in_rand", "in_halfw")], ibo)
            if gname == "head_occluder":
                g.occ_vao = vao
                occ = (v, faces)
                continue
            if P > 0 and len(v) % (P * 2) == 0:
                all_roots.append(root[::P * 2])
            g.hair_vao = vao
            g.n_tris += len(faces)
            mat = getattr(geom.visual, "material", None)
            img = getattr(mat, "baseColorTexture", None) if mat is not None else None
            if img is not None:
                img = img.convert("RGBA")
                g.tex = ctx.texture(img.size, 4, img.tobytes())
                g.tex.build_mipmaps()
                g.tex.repeat_x = g.tex.repeat_y = False
            bcf = getattr(mat, "baseColorFactor", None) if mat is not None else None
            if bcf is not None:
                g.base = tuple(float(c) / 255.0 for c in np.asarray(bcf)[:3])
        if g.hair_vao is None:
            raise ValueError(f"{path}: 헤어 메시가 없다")
        if occ is not None:
            ov, of = occ
            sw = _scalp_weight(ov, np.concatenate(all_roots) if all_roots else None)
            g.scalp_cov = float(sw.mean())
            hb = ctx.buffer(np.hstack([ov, sw[:, None]]).astype(np.float32).tobytes())
            g.head_vao = ctx.vertex_array(self._head_prog, [(hb, "3f 1f", "in_pos", "in_scalp")],
                                          ctx.buffer(of.tobytes()))
        if g.tex is None:
            g.tex = ctx.texture((1, 1), 4, bytes([255, 255, 255, 255]))
        self._grooms[name] = g
        logger.info("그룸 로드 %s: %d tris, 오클루더 %s (%.0fms)", name, g.n_tris,
                    "있음" if g.occ_vao else "없음", (time.perf_counter() - t0) * 1000)
        return g

    def evict(self, name: str) -> None:
        g = self._grooms.pop(name, None)
        if g is None:
            return
        for r in (g.hair_vao, g.occ_vao, g.head_vao, g.tex):
            if r is not None:
                r.release()

    # ---- 렌더 ----
    def render(self, groom: Groom, matrix: np.ndarray, w: int, h: int,
               scale_mul: float = 1.0, offset_up_cm: float = 0.0, offset_fwd_cm: float = 0.0,
               base_rgb=None, dyn: dict | None = None, head: bool = False, geom=None):
        """-> (rgb (h,w,3) BGR float 0~255, a (h,w,1) float 0~1) GPU 텐서, 또는 (None, None).

        matrix: MediaPipe 4x4 row-major (정규 얼굴 -> 카메라 공간, cm).
        scale/offset: 그룸 json 의 user_fit 위에 세션 슬라이더를 곱/더한 최종값.
        base_rgb: 베이스 색 (0~1 RGB). None 이면 GLB 의 색.
        head: True 면 self.last_head 에 두상 좌표 맵 (h,w,4) GPU uint8 을 남긴다(오클루더 없으면 None).
        geom: (head 일 때) 이미지 사각형 (x0, y0, x1, y1). 주어지면 그 안의 두상 표면 정규 좌표 cm 와
            노멀을 self.last_geom 에 남긴다(fit 은 self.last_fit) - project_uv() 로 다른 순간 사진을
            투영 텍스처로 붙인다. 전체 프레임을 읽으면 GL 읽기가 ~2ms 라 얼굴 둘레만 읽는다.
        """
        import moderngl
        self._ensure_ctx()
        self._ensure_fbo(w, h)
        t0 = time.perf_counter()
        ctx, prog = self._ctx, self._prog

        M = np.asarray(matrix, np.float32).reshape(4, 4)
        # 정규 프레임 안에서의 사용자 보정: 두상 중심 기준 스케일 + 위/앞 이동 (hair3d.js 와 동일)
        fit = np.eye(4, dtype=np.float32)
        fit[0, 0] = fit[1, 1] = fit[2, 2] = float(scale_mul)
        fit[1, 3] = float(offset_up_cm)
        fit[2, 3] = float(offset_fwd_cm)
        model = M @ fit
        # 셰이더의 상대좌표 변형은 정규 공간(fit 적용 전)에서 일어나므로 u_shift 는 스케일과 무관
        proj = _perspective(MP_VERTICAL_FOV_DEG, w / float(h), 1.0, 1000.0)
        # 이미지 좌표는 y 가 아래로 자란다. 투영의 y 를 뒤집어 프레임버퍼 0행이 이미지 0행이 되게 한다.
        proj[1] *= -1.0
        mvp = proj @ model
        nrm = model[:3, :3].copy()
        # 스케일이 들어 있으니 노멀은 역전치로 (균일 스케일이라 정규화만 하면 되지만 안전하게)
        try:
            nrm = np.linalg.inv(nrm).T
        except np.linalg.LinAlgError:
            pass

        prog["u_proj"].write(np.ascontiguousarray(proj.T).tobytes())     # column-major
        prog["u_model"].write(np.ascontiguousarray(model.T).tobytes())
        prog["u_nrm"].write(np.ascontiguousarray(nrm.T.astype(np.float32)).tobytes())
        prog["u_billboard"].value = 1 if groom.billboard else 0
        prog["u_scale"].value = float(scale_mul)
        prog["u_light"].value = (0.3, 0.8, 1.0)
        # 2차 운동 (hair_dynamics.HairDynamics.step 의 결과). 없으면 강체.
        if dyn is not None:
            prog["u_lag"].write(np.ascontiguousarray(np.asarray(dyn["lag"], np.float32).T).tobytes())
            prog["u_shift"].value = tuple(float(x) for x in dyn["shift"])
            prog["u_tip"].value = float(dyn["tip"])
            prog["u_idle"].value = float(dyn["idle"])
            prog["u_time"].value = float(dyn["time"])
        else:
            prog["u_lag"].write(np.eye(3, dtype=np.float32).tobytes())
            prog["u_shift"].value = (0.0, 0.0, 0.0)
            prog["u_tip"].value = 0.0
            prog["u_idle"].value = 0.0
            prog["u_time"].value = 0.0
        base = groom.base if base_rgb is None else tuple(float(c) for c in base_rgb)
        prog["u_base"].value = base
        prog["u_tex"].value = 0
        groom.tex.use(0)

        fbo = self._fbo_ms
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)

        # 0) 두상 오클루더: 깊이만
        if groom.occ_vao is not None:
            ctx.disable(moderngl.BLEND)
            ctx.depth_mask = True
            fbo.color_mask = (False, False, False, False)
            prog["u_pass"].value = 2
            groom.occ_vao.render()
            fbo.color_mask = (True, True, True, True)
        if groom.billboard:
            # v1: 알파-투-커버리지 단일 패스. 알파가 MSAA 샘플 커버리지로 바뀌어 깊이 쓰기와
            # 함께 순서 무관하게 반투명 가닥이 겹친다(정렬 불필요, 2패스 불필요). 리졸브 결과는
            # 색이 커버리지로 프리멀티플라이드된 것 -> 아래에서 알파로 나눈다.
            ctx.disable(moderngl.BLEND)
            ctx.depth_mask = True
            prog["u_pass"].value = 3
            prog["u_samples"].value = int(self._samples)
            groom.hair_vao.render()
        else:
            # v0: 1) 불투명 컷아웃
            ctx.disable(moderngl.BLEND)
            ctx.depth_mask = True
            prog["u_pass"].value = 0
            groom.hair_vao.render()
            # 2) 반투명 가장자리 (깊이 읽기만). 색은 프리멀티플라이드로 누적, 알파는 커버리지.
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
                              moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            ctx.depth_mask = False
            prog["u_pass"].value = 1
            groom.hair_vao.render()
            ctx.depth_mask = True
            ctx.disable(moderngl.BLEND)

        ctx.copy_framebuffer(self._fbo, fbo)                  # MSAA 리졸브
        self.last_head = None
        self.last_geom = None
        if head and groom.head_vao is not None:
            fh = self._fbo_head
            fh.use()
            fh.clear(0.0, 0.0, 0.0, 0.0)
            hp = self._head_prog
            hp["u_mvp"].write(np.ascontiguousarray(mvp.T.astype(np.float32)).tobytes())
            groom.head_vao.render()
            hraw = fh.read(components=4, dtype="f1")
            self.last_head = torch.frombuffer(bytearray(hraw), dtype=torch.uint8).view(h, w, 4).to(
                self.device, non_blocking=True)
            if geom is not None:
                x0, y0, x1, y1 = (int(v) for v in geom)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                if x1 - x0 >= 4 and y1 - y0 >= 4:
                    gw, gh = x1 - x0, y1 - y0
                    vp = (x0, y0, gw, gh)      # 프레임버퍼 행 = 이미지 행 (투영 y 반전)
                    praw = fh.read(viewport=vp, components=4, dtype="f2", attachment=1)
                    nraw = fh.read(viewport=vp, components=4, dtype="f1", attachment=2)
                    pos = torch.frombuffer(bytearray(praw), dtype=torch.float16).view(gh, gw, 4).to(
                        self.device, non_blocking=True).float()
                    nrm = torch.frombuffer(bytearray(nraw), dtype=torch.uint8).view(gh, gw, 4).to(
                        self.device, non_blocking=True).float()[:, :, :3] / 127.5 - 1.0
                    self.last_geom = (pos[:, :, :3], nrm, (x0, y0, x1, y1))
                self.last_fit = fit
                self.last_proj_m = proj
        raw = self._fbo.read(components=4, dtype="f1")         # RGBA8, 0행 = 이미지 0행 (투영 y 반전)
        t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(h, w, 4).to(
            self.device, non_blocking=True).float()
        # 컷아웃/A2C 는 커버리지, 가장자리는 SRC_ALPHA 누적 -> 어느 쪽이든 프리멀티플라이드라 알파로 나눈다
        a = t[:, :, 3:4] / 255.0
        rgb = t[:, :, :3] / a.clamp(min=1e-3)
        rgb = rgb[:, :, [2, 1, 0]].clamp(0.0, 255.0)          # 합성 코드는 BGR
        self.last_ms = (time.perf_counter() - t0) * 1000
        return rgb, a

    def project_uv(self, matrix_b):
        """직전 render(geom=True) 의 두상 표면을, 다른 순간의 포즈 matrix_b 로 찍었을 때의 이미지 좌표.

        -> (grid (gh,gw,2) grid_sample 좌표 [-1,1], facing (gh,gw) 그 순간 표면이 카메라를 향한 정도 0~1,
        rect (x0,y0,x1,y1) 이 결과가 덮는 지금 프레임의 사각형) 또는 None.
        투영 텍스처: 그 순간 프레임을 grid 로 샘플하면 지금 두상 위에 그대로 붙는다.
        """
        if self.last_geom is None:
            return None
        pos, nrm, rect = self.last_geom
        mv = torch.as_tensor(np.asarray(matrix_b, np.float32).reshape(4, 4) @ self.last_fit,
                             device=pos.device)
        P = torch.as_tensor(self.last_proj_m, device=pos.device)
        pc = pos @ mv[:3, :3].T + mv[:3, 3]                     # 그 순간 카메라 공간
        clip = pc @ P[:, :3].T + P[:, 3]                         # (h,w,4)
        grid = clip[:, :, :2] / clip[:, :, 3:4].clamp(min=1e-4)
        nc = nrm @ mv[:3, :3].T
        facing = (nc * (-pc)).sum(-1).abs() / (nc.norm(dim=-1) * pc.norm(dim=-1)).clamp(min=1e-4)
        return grid, facing, rect

    def close(self):
        for name in list(self._grooms):
            self.evict(name)
        for f in (self._fbo_ms, self._fbo, self._fbo_head):
            if f is not None:
                f.release()
        self._fbo_ms = self._fbo = self._fbo_head = None
        if self._ctx is not None:
            self._ctx.release()
            self._ctx = None
