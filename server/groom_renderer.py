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
#version 330
uniform mat4 u_mvp;
uniform mat3 u_nrm;
// 2차 운동 (hair_dynamics.py): 뿌리 기준 상대좌표를 [현재 -> 지연 자세] 로 끝쪽만 섞는다
uniform mat3 u_lag;         // 두상 공간 회전 (현재 -> 지연)
uniform vec3 u_shift;       // 두상 공간 이동 지연 (cm)
uniform float u_tip;        // 끝 가중치 배율 (0 이면 강체)
uniform float u_idle;       // 미세 흔들림 진폭 (cm)
uniform float u_time;
in vec3 in_pos; in vec2 in_uv; in vec3 in_nrm; in vec3 in_root;
out vec2 v_uv; out vec3 v_nrm;
void main() {
    float s = in_uv.y;                      // 0 = 뿌리, 1 = 끝 (strands_to_cards 의 v 를 로더가 뒤집음)
    vec3 rel = in_pos - in_root;
    float w = clamp(u_tip * s * s, 0.0, 1.0);
    vec3 rel2 = u_lag * rel + u_shift * s;
    vec3 d = mix(rel, rel2, w);
    // 살아 있는 느낌: 뿌리 위치로 위상을 흩뜨린 느린 흔들림 (끝일수록 크게)
    float ph = dot(in_root, vec3(12.9898, 78.233, 37.719));
    d += vec3(sin(u_time * 1.7 + ph), 0.0, cos(u_time * 1.3 + ph * 0.7)) * (u_idle * s * s);
    // 가닥 길이 보존 (섞으면 짧아진다)
    float L = length(rel);
    float Ld = length(d);
    if (L > 1e-4 && Ld > 1e-4) d *= L / Ld;
    vec3 pos = in_root + d;
    gl_Position = u_mvp * vec4(pos, 1.0);
    v_uv = in_uv;
    v_nrm = normalize(u_nrm * in_nrm);
}
"""

_FS = """
#version 330
uniform sampler2D u_tex;
uniform vec3 u_base;        // 베이스 색 (0~1, RGB)
uniform int u_pass;         // 0: 불투명 컷아웃(a>=0.5), 1: 반투명 가장자리(a<0.5), 2: 깊이만
uniform vec3 u_light;       // 카메라 공간 광원 방향
in vec2 v_uv; in vec3 v_nrm;
out vec4 f_color;
void main() {
    if (u_pass == 2) { f_color = vec4(0.0); return; }
    vec4 t = texture(u_tex, v_uv);
    float a = t.a;
    if (u_pass == 0 && a < 0.5) discard;
    if (u_pass == 1 && a >= 0.5) discard;
    // 양면 조명. 리본은 두께가 없어 뒷면도 그대로 보인다.
    float nl = abs(dot(v_nrm, normalize(u_light)));
    float shade = 0.55 + 0.45 * nl;
    vec3 c = u_base * t.rgb * shade;
    // 패스 0 은 컷아웃이라 완전 불투명으로 쓴다(블렌드 없음). 패스 1 은 SRC_ALPHA 블렌드가
    // 프리멀티플라이드로 누적하므로 스트레이트 색을 낸다. 어느 쪽이든 최종 색/알파 관계는
    // 프리멀티플라이드라 렌더 뒤 알파로 나누면 된다.
    f_color = (u_pass == 0) ? vec4(c, 1.0) : vec4(c, a);
}
"""


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
        self.tex = None
        self.base = (0.23, 0.14, 0.09)
        self.n_tris = 0


class GroomRenderer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._ctx = None
        self._prog = None
        self._thread = None
        self._fbo_ms = None      # (w,h) 별 MSAA 프레임버퍼
        self._fbo = None         # 리졸브 대상
        self._fbo_size = None
        self._grooms: dict[str, Groom] = {}
        self.last_ms = 0.0

    # ---- 컨텍스트 ----
    def _ensure_ctx(self):
        if self._ctx is not None:
            if threading.get_ident() != self._thread:
                raise RuntimeError("GroomRenderer 는 만든 스레드(gpu_executor)에서만 쓸 수 있다")
            return
        import moderngl
        t0 = time.perf_counter()
        self._ctx = moderngl.create_standalone_context()
        self._thread = threading.get_ident()
        self._prog = self._ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        logger.info("moderngl 컨텍스트: %s (%.0fms)", self._ctx.info.get("GL_RENDERER"),
                    (time.perf_counter() - t0) * 1000)

    def _ensure_fbo(self, w: int, h: int):
        if self._fbo_size == (w, h):
            return
        ctx = self._ctx
        for f in (self._fbo_ms, self._fbo):
            if f is not None:
                f.release()
        self._fbo_ms = ctx.framebuffer(
            color_attachments=[ctx.renderbuffer((w, h), components=4, samples=4)],
            depth_attachment=ctx.depth_renderbuffer((w, h), samples=4))
        self._fbo = ctx.framebuffer(color_attachments=[ctx.texture((w, h), components=4)])
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
            vbo = ctx.buffer(np.hstack([v, uv, n, root]).astype(np.float32).tobytes())
            ibo = ctx.buffer(faces.tobytes())
            vao = ctx.vertex_array(self._prog, [(vbo, "3f 2f 3f 3f", "in_pos", "in_uv", "in_nrm", "in_root")], ibo)
            if gname == "head_occluder":
                g.occ_vao = vao
                continue
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
        for r in (g.hair_vao, g.occ_vao, g.tex):
            if r is not None:
                r.release()

    # ---- 렌더 ----
    def render(self, groom: Groom, matrix: np.ndarray, w: int, h: int,
               scale_mul: float = 1.0, offset_up_cm: float = 0.0, offset_fwd_cm: float = 0.0,
               base_rgb=None, dyn: dict | None = None):
        """-> (rgb (h,w,3) BGR float 0~255, a (h,w,1) float 0~1) GPU 텐서, 또는 (None, None).

        matrix: MediaPipe 4x4 row-major (정규 얼굴 -> 카메라 공간, cm).
        scale/offset: 그룸 json 의 user_fit 위에 세션 슬라이더를 곱/더한 최종값.
        base_rgb: 베이스 색 (0~1 RGB). None 이면 GLB 의 색.
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

        prog["u_mvp"].write(np.ascontiguousarray(mvp.T).tobytes())   # column-major
        prog["u_nrm"].write(np.ascontiguousarray(nrm.T.astype(np.float32)).tobytes())
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
        # 1) 불투명 컷아웃
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
        raw = self._fbo.read(components=4, dtype="f1")         # RGBA8, 0행 = 이미지 0행 (투영 y 반전)
        t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(h, w, 4).to(
            self.device, non_blocking=True).float()
        # 컷아웃은 알파 1, 가장자리는 SRC_ALPHA 로 프리멀티플라이드 누적 -> 알파로 나눠 되돌린다
        a = t[:, :, 3:4] / 255.0
        rgb = t[:, :, :3] / a.clamp(min=1e-3)
        rgb = rgb[:, :, [2, 1, 0]].clamp(0.0, 255.0)          # 합성 코드는 BGR
        self.last_ms = (time.perf_counter() - t0) * 1000
        return rgb, a

    def close(self):
        for name in list(self._grooms):
            self.evict(name)
        for f in (self._fbo_ms, self._fbo):
            if f is not None:
                f.release()
        self._fbo_ms = self._fbo = None
        if self._ctx is not None:
            self._ctx.release()
            self._ctx = None
