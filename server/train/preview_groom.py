"""헤어 그룸 GLB -> 프론트용 미리보기 PNG (헤어만, 투명 배경. --with-head 면 마네킹 두상 포함).


    python server/train/preview_groom.py server/grooms        # 폴더 안 모든 *.glb
    python preview_groom.py x.glb --out-dir out/previews

출력 (GLB 옆 또는 --out-dir):
  <name>.png        정면에서 살짝 돌린 3/4 뷰 정사각 (카드 썸네일)
  <name>_views.png  정면 / 3/4 / 옆 / 뒤 4컷 가로 띠 (상세 화면용)

render_glb.py(게이트 판정용)와 달리 보여주기용: 8x MSAA, 두상은 GLB 의 FLAME 오클루더 노드,
헤어는 자연 흑갈색 + 결 방향(Kajiya-Kay) 하이라이트, 밝은 회색 배경.
"""
import argparse, math, os
import numpy as np, moderngl, trimesh
from PIL import Image

VS = """#version 400
uniform mat4 mvp; uniform mat4 view;
in vec3 in_pos; in vec2 in_uv; in vec3 in_nrm; in vec3 in_tan;
out vec2 v_uv; out vec3 v_nrm; out vec3 v_tan; out vec3 v_vpos;
void main(){ gl_Position = mvp * vec4(in_pos,1.0); v_uv = in_uv;
  v_nrm = mat3(view) * in_nrm; v_tan = mat3(view) * in_tan; v_vpos = (view * vec4(in_pos,1.0)).xyz; }"""
FS = """#version 400
uniform sampler2D tex; uniform vec4 base; uniform int use_tex; uniform int mode;   // 0 피부, 1 헤어
uniform vec3 hair2; uniform int depth_only;   // 1: 투명 배경과 같은 (0,0,0,0) 을 써서 깊이만 남김   // 두피 색 (두상 정점 중 뿌리 근처 = v_uv.x 가중)
in vec2 v_uv; in vec3 v_nrm; in vec3 v_tan; in vec3 v_vpos; out vec4 f;
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
void main(){
  gl_SampleMask[0] = 0xFF;
  vec4 c = base; if(use_tex==1) c.a *= texture(tex, v_uv).a;
  if (mode == 0) c.rgb = mix(c.rgb, hair2, v_uv.x);
  if (mode == 1) {
    // 확률적 투명도(서버 groom_renderer 와 같은 방식): 알파만큼 MSAA 샘플을 덮어 정렬 없이 겹친다.
    // 알파 블렌드는 그리는 순서에 따라 뒤쪽 가닥이 반투명 회색 줄로 남았다.
    if (c.a < 0.02) discard;
    int k = int(clamp(floor(c.a * 8.0 + hash12(gl_FragCoord.xy + v_uv * 977.0)), 0.0, 8.0));
    if (k <= 0) discard;
    int off = int(hash12(gl_FragCoord.yx * 1.7 + v_uv * 311.0) * 8.0);
    int mask = 0; for (int i = 0; i < k; i++) mask |= 1 << ((off + i) % 8);
    gl_SampleMask[0] = mask; c.a = 1.0;
  }
  vec3 L1 = normalize(vec3(0.35, 0.75, 1.0));      // 키 라이트 (앞 위)
  vec3 L2 = normalize(vec3(-0.6, 0.2, 0.6));       // 보조
  vec3 V = normalize(-v_vpos);
  vec3 col;
  if (mode == 0) {
    vec3 N = normalize(v_nrm); if (dot(N, V) < 0.0) N = -N;
    float d = max(dot(N, L1), 0.0) * 0.75 + max(dot(N, L2), 0.0) * 0.25;
    col = c.rgb * (0.38 + 0.7 * d);
  } else {
    vec3 T = normalize(v_tan);
    float tl = dot(T, L1); float diff = sqrt(max(0.0, 1.0 - tl*tl));
    vec3 H = normalize(L1 + V); float th = dot(T, H);
    float spec = pow(sqrt(max(0.0, 1.0 - th*th)), 60.0);
    col = c.rgb * (0.35 + 0.75 * diff) + vec3(0.30, 0.24, 0.20) * spec * 0.55;
  }
  f = (depth_only == 1) ? vec4(0.0) : vec4(col, 1.0); }"""


def persp(fov, aspect, n, fr):
    t = 1 / math.tan(math.radians(fov) / 2)
    return np.array([[t / aspect, 0, 0, 0], [0, t, 0, 0], [0, 0, (fr + n) / (n - fr), 2 * fr * n / (n - fr)],
                     [0, 0, -1, 0]], "f4")


def look_at(eye, tgt, up=(0, 1, 0)):
    f = np.array(tgt, "f4") - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s); u = np.cross(s, f)
    m = np.eye(4, dtype="f4"); m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


BG = (0.925, 0.918, 0.902)


def fit_crop(imgs, S, pad=0.08):
    """투명 배경 이미지들을 공통 알파 bbox 로 잘라 정사각 S 로 (헤어만 그릴 때 화면 가운데 꽉 차게).
    여러 장(4컷)은 같은 bbox 를 써서 컷끼리 크기가 같다."""
    boxes = [im.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox() for im in imgs]
    boxes = [b for b in boxes if b]
    if not boxes:
        return imgs
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    side = int(max(x1 - x0, y1 - y0) * (1 + 2 * pad))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    box = (cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)
    return [im.crop(box).resize((S, S), Image.LANCZOS) for im in imgs]


def hexrgb(h):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def load(ctx, prog, path):
    scene = trimesh.load(path, process=False)
    draws, pts = [], []
    for name, geom in scene.geometry.items():
        T = scene.graph.get(name)[0] if name in scene.graph.nodes_geometry else np.eye(4)
        v = trimesh.transform_points(geom.vertices, T).astype("f4")
        F = np.asarray(geom.faces, "i4")
        fn = np.cross(v[F[:, 1]] - v[F[:, 0]], v[F[:, 2]] - v[F[:, 0]])
        n = np.zeros_like(v); np.add.at(n, F.ravel(), np.repeat(fn, 3, 0))
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        va = getattr(geom, "vertex_attributes", {}) or {}
        tan = (np.asarray(va["_tangent"], "f4") @ T[:3, :3].T).astype("f4") if "_tangent" in va else n
        uv = getattr(geom.visual, "uv", None)
        uv = np.zeros((len(v), 2), "f4") if uv is None or len(uv) != len(v) else np.asarray(uv, "f4").copy()
        uv[:, 1] = 1 - uv[:, 1]
        bb = None
        if all(k in va for k in ("_center", "_side", "_halfw")):
            ctr = trimesh.transform_points(np.asarray(va["_center"], "f4"), T).astype("f4")
            bb = (ctr, tan, np.asarray(va["_side"], "f4").reshape(-1, 1), np.asarray(va["_halfw"], "f4").reshape(-1, 1), uv)
        buf = ctx.buffer(np.hstack([v, uv, n, tan]).astype("f4").tobytes(), dynamic=bb is not None)
        vao = ctx.vertex_array(prog, [(buf, "3f 2f 3f 3f", "in_pos", "in_uv", "in_nrm", "in_tan")],
                               ctx.buffer(F.tobytes()))
        mat = getattr(geom.visual, "material", None)
        tex = None
        if mat is not None and getattr(mat, "baseColorTexture", None) is not None:
            img = mat.baseColorTexture.convert("RGBA")
            tex = ctx.texture(img.size, 4, img.tobytes()); tex.build_mipmaps()
        is_hair = name not in ("head", "head_occluder")
        draws.append([name, vao, tex, is_hair, buf, bb, v, n])
        pts.append(v)
    roots = None
    for d in draws:
        if d[5] is not None:
            ctr = d[5][0]
            P = 24 if len(ctr) % 48 == 0 else 0
            if P:
                roots = ctr.reshape(-1, P, 2, 3)[:, 0].mean(1)
    for d in draws:
        if not d[3] and roots is not None:
            from scipy.spatial import cKDTree
            dist, _ = cKDTree(roots).query(d[6])
            w = np.clip((1.6 - dist) / 1.0, 0.0, 1.0).astype("f4")     # 뿌리 1.6cm 이내 = 두피
            uvh = np.c_[w, np.zeros_like(w)].astype("f4")
            d[4].write(np.hstack([d[6], uvh, d[7], d[7]]).astype("f4").tobytes())
    return [tuple(d[:6]) for d in draws], np.concatenate(pts)


def render(ctx, prog, draws, center, dist, yaw, pitch, S, hair_rgb, skin_rgb, show_head=False):
    fbo_ms = ctx.framebuffer(color_attachments=[ctx.renderbuffer((S, S), 4, samples=8)],
                             depth_attachment=ctx.depth_renderbuffer((S, S), samples=8))
    fbo = ctx.simple_framebuffer((S, S), components=4)
    y, p = math.radians(yaw), math.radians(pitch)
    eye = center + np.array([dist * math.sin(y) * math.cos(p), dist * math.sin(p),
                             dist * math.cos(y) * math.cos(p)], "f4")
    view = look_at(eye, center)
    mvp = (persp(28, 1.0, 1.0, 500.0) @ view)
    prog["mvp"].write(mvp.T.copy().astype("f4").tobytes())
    prog["view"].write(view.T.copy().astype("f4").tobytes())
    fbo_ms.use()
    if show_head:
        fbo_ms.clear(*BG, 1.0)
    else:
        fbo_ms.clear(0.0, 0.0, 0.0, 0.0)          # 투명 배경 (헤어만)
    ctx.enable(moderngl.DEPTH_TEST); ctx.disable(moderngl.CULL_FACE)
    prog["depth_only"].value = 0
    for pass_hair in (False, True):
        ctx.disable(moderngl.BLEND); ctx.depth_mask = True
        for name, vao, tex, is_hair, buf, bb in draws:
            if is_hair != pass_hair:
                continue
            if bb is not None:
                # v1 그룸은 런타임에 카메라를 향해 펼치는 리본(빌보드) - 이 시점 기준으로 폭 방향을 다시 계산
                ctr, tg, sd, hw, uvb = bb
                vd = eye[None, :] - ctr
                side = np.cross(tg, vd); side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-9
                pos = ctr + side * sd * hw
                nrm = vd / (np.linalg.norm(vd, axis=1, keepdims=True) + 1e-9)
                buf.write(np.hstack([pos, uvb, nrm, tg]).astype("f4").tobytes())
            if tex is not None:
                tex.use(0)
            prog["tex"].value = 0
            prog["use_tex"].value = 1 if (tex is not None and is_hair) else 0
            prog["mode"].value = 1 if is_hair else 0
            prog["base"].value = (*(hair_rgb if is_hair else skin_rgb), 1.0)
            prog["hair2"].value = tuple(c * 0.7 for c in hair_rgb)
            if not is_hair and not show_head:
                # 두상은 깊이만: 뒤통수/반대편 머리카락이 얼굴 자리로 비치지 않게 가리되 화면엔 안 그린다
                # (Framebuffer.color_mask 는 여기서 적용되지 않아 셰이더가 배경과 같은 투명색을 쓴다)
                prog["depth_only"].value = 1
                vao.render()
                prog["depth_only"].value = 0
                continue
            vao.render()
    ctx.copy_framebuffer(fbo, fbo_ms)
    img = Image.frombytes("RGBA", (S, S), fbo.read(components=4)).transpose(Image.FLIP_TOP_BOTTOM)
    fbo_ms.release(); fbo.release()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="GLB 파일 또는 폴더")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--hair", default="#2b1d16", help="헤어 색 (자연 흑갈색)")
    ap.add_argument("--skin", default="#d9b8a0")
    ap.add_argument("--with-head", action="store_true", help="마네킹 두상을 같이 그림 (기본: 헤어만, 투명 배경)")
    a = ap.parse_args()
    files = [a.src] if a.src.endswith(".glb") else sorted(
        os.path.join(a.src, f) for f in os.listdir(a.src) if f.endswith(".glb"))
    ctx = moderngl.create_standalone_context(require=400)
    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        od = a.out_dir or os.path.dirname(path); os.makedirs(od, exist_ok=True)
        draws, pts = load(ctx, prog, path)
        pts = np.concatenate([d[5][0] for d in draws if d[5] is not None]) if any(d[5] is not None for d in draws) else pts
        # 두상 기준 고정 구도: 스타일마다 헤어 bbox 로 맞추면 단발은 꽉 차고 장발은 작게 나와 목록에서 크기가
        # 들쭉날쭉했다. 두상(오클루더) 크기/위치로 정하면 모든 썸네일의 얼굴 크기가 같다.
        hv = next((d for d in draws if not d[3]), None)
        if hv is not None:
            hp = np.frombuffer(hv[4].read(), "f4").reshape(-1, 11)[:, :3]
            hlo, hhi = hp.min(0), hp.max(0); H = float(hhi[1] - hlo[1])
            center = ((hlo + hhi) / 2).astype("f4"); center[1] -= 0.12 * H
            dist = H * 3.6
        else:
            lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
            center = ((lo + hi) / 2).astype("f4"); dist = float(np.max(hi - lo)) * 2.8
        S = a.size
        thumb = render(ctx, prog, draws, center, dist, 30, 6, S, hexrgb(a.hair), hexrgb(a.skin), a.with_head)
        if not a.with_head:
            thumb = fit_crop([thumb], S)[0]
        thumb.save(os.path.join(od, f"{name}.png"))
        views = [render(ctx, prog, draws, center, dist, yw, 6, S, hexrgb(a.hair), hexrgb(a.skin), a.with_head)
                 for yw in (0, 35, 90, 180)]
        if not a.with_head:
            views = fit_crop(views, S)
        strip = Image.new("RGBA", (S * len(views), S))
        for i, v in enumerate(views):
            strip.paste(v, (i * S, 0))
        strip.save(os.path.join(od, f"{name}_views.png"))
        for _, vao, tex, _, _, _ in draws:
            vao.release()
            if tex is not None:
                tex.release()
        print("->", name)


if __name__ == "__main__":
    main()
