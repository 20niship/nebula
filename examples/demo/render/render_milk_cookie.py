"""demo_milk_cookie の .nbp を Blender (Cycles) でレンダリングする。クッキーが混ざる様子は、合成メッシュの頂点属性 (クッキー混合率) で色を変えて表す。"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import bpy
import bmesh  # noqa: E402
import common as C  # noqa: E402
import numpy as np  # noqa: E402
from nbp import density_grid, load_nbp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out", default="out/milk_cookie_png")
ap.add_argument("--frames", type=int, nargs=2, default=[0, 0], metavar=("START", "END"))
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
ap.add_argument("--samples", type=int, default=128)
ap.add_argument("--close", action="store_true", help="液面のクローズアップ (既定はクッキーの落下も入る引き)")
ap.add_argument("--jar", action="store_true", help="ガラスのジャーも描く (既定は液体と刃だけ)")
ap.add_argument("--engine", choices=["cycles", "eevee"], default="cycles", help="eevee は高速 (確認・全コマ用)")
ap.add_argument("--voxel", type=float, default=0.0, help="surface reconstruction voxel [m] (0 = 1.4 x particle spacing, auto)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# demo_milk_cookie.cpp と同じジャー寸法 [m]
CX, CZ, JAR_R, FLOOR_Y, RIM_Y, GLASS = 0.5, 0.5, 0.25, 0.05, 0.62, 0.012
MILK_VOLUME = math.pi * JAR_R**2 * (0.33 - FLOOR_Y)  # 初期ミルクの体積から粒子間隔を逆算する

sc = C.reset_scene(tuple(a.res), a.samples)
if a.engine == "eevee":
    sc.render.engine = "BLENDER_EEVEE"
    sc.eevee.taa_render_samples = a.samples
sc.view_settings.exposure = 0.0
w = bpy.data.worlds.new("w")
sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.012, 0.012, 0.016, 1)
w.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0

if a.close:
    cam = C.camera(C.sim_to_blender((CX + 0.4, 1.45, CZ + 1.05)), C.sim_to_blender((CX, 0.25, CZ)), lens=50)
else:
    cam = C.camera(C.sim_to_blender((CX + 0.5, 1.6, CZ + 1.8)), C.sim_to_blender((CX, 0.4, CZ)), lens=40)
cam.data.dof.use_dof = True
cam.data.dof.focus_distance = 2.2
cam.data.dof.aperture_fstop = 8.0

# 天板 (暗い光沢面) と、ジャーを縁取る照明
C.box("table", (CX, -0.01, CZ), (60, 0.01, 60), C.material("table", (0.03, 0.025, 0.022, 1), 0.55, Specular_IOR_Level=0.3))


def area(name, loc, target, size, energy, color=(1, 1, 1)):
    ob = bpy.data.objects.new(name, bpy.data.lights.new(name, "AREA"))
    ob.data.shape = "RECTANGLE"
    ob.data.size, ob.data.size_y = size
    ob.data.energy = energy
    ob.data.color = color
    bpy.context.collection.objects.link(ob)
    ob.location = C.sim_to_blender(loc)
    d = np.array(C.sim_to_blender(target)) - np.array(ob.location)
    ob.rotation_euler = (__import__("mathutils").Vector(d)).to_track_quat("-Z", "Y").to_euler()


area("key", (CX - 0.9, 0.95, CZ + 0.7), (CX, 0.3, CZ), (3.0, 3.0), 140, (1.0, 0.93, 0.85))
area("rim_l", (CX - 0.7, 0.55, CZ - 0.9), (CX, 0.3, CZ), (1.0, 3.0), 160, (0.8, 0.9, 1.0))
area("rim_r", (CX + 0.8, 0.55, CZ - 0.7), (CX, 0.3, CZ), (1.0, 3.0), 120, (1.0, 0.85, 0.7))
area("top", (CX, 1.4, CZ + 0.1), (CX, 0.3, CZ), (1.2, 1.2), 30)


def revolve(profile, segs=128, cx=CX, cz=CZ):
    """(r, y) の折れ線を Y 軸まわりに回転した面。"""
    P = np.asarray(profile, np.float32)
    th = np.linspace(0, 2 * math.pi, segs, endpoint=False)
    V = np.stack([cx + P[None, :, 0] * np.cos(th)[:, None], np.broadcast_to(P[None, :, 1], (segs, len(P))), cz + P[None, :, 0] * np.sin(th)[:, None]], -1).reshape(-1, 3)
    idx = np.arange(segs * len(P)).reshape(segs, len(P))
    F = []
    for j in range(len(P) - 1):
        a_, b_ = idx[:, j], idx[:, j + 1]
        a2, b2 = np.roll(a_, -1), np.roll(b_, -1)
        F.append(np.stack([a_, b_, b2], -1))
        F.append(np.stack([a_, b2, a2], -1))
    return V, np.concatenate(F)


glass = C.material("glass", (0.97, 0.99, 1, 1), 0.03, Transmission_Weight=1.0, IOR=1.47)
jar_v, jar_f = revolve([(0, 0), (JAR_R + GLASS, 0), (JAR_R + GLASS, RIM_Y), (JAR_R, RIM_Y), (JAR_R, FLOOR_Y), (0, FLOOR_Y)])
jar = C.mesh_object("jar", jar_v, jar_f, glass)
bm = bmesh.new()
bm.from_mesh(jar.data)
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
bm.to_mesh(jar.data)
bm.free()

jar.hide_render = not a.jar
steel = C.material("steel", (0.75, 0.75, 0.78, 1), 0.18, Metallic=1.0)
MIX = {"slat": []}
for line in open(f"{a.src}/mixer.txt"):
    k, *v = line.split()
    if k == "slat":
        MIX["slat"].append(np.array(v, float))
    else:
        MIX[k] = np.array(v, float)


def rotY(t):
    c, s_ = math.cos(t), math.sin(t)
    return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])


def rotZ(t):
    c, s_ = math.cos(t), math.sin(t)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])


def mixer_mesh(angle):
    """sim 座標のミキサー (傾いた軸 + 板の籠) の頂点・面。demo_milk_cookie.cpp が書いた mixer.txt の形状を同じ剛体変換で動かす。"""
    O, tilt = MIX["origin"], MIX["tilt"][0]
    r_sh, y0, y1 = MIX["shaft"]
    R = rotZ(tilt) @ rotY(angle)
    V, F = [], []
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    cf = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    for sl in MIX["slat"]:
        c, h, yaw = sl[0:3], sl[3:6], sl[6]
        V.append(O + (R @ (rotY(yaw) @ (cube * h).T + c[:, None])).T)
        F.append(cf + 8 * len(F))
    n = 24
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.stack([r_sh * np.cos(th), np.zeros(n), r_sh * np.sin(th)], -1)
    pts = np.concatenate([ring + [0, y0, 0], ring + [0, y1, 0]])
    V.append(O + (R @ pts.T).T)
    base = 8 * len(MIX["slat"])
    for k in range(n):
        k2 = (k + 1) % n
        F.append(np.array([[base + k, base + k2, base + n + k2], [base + k, base + n + k2, base + n + k]]))
    return np.concatenate(V).astype(np.float32), np.concatenate(F).astype(np.int32)


# 液体: 混合率で 白いミルク -> コーヒー牛乳 -> クッキー に塗り分ける 1 つのマテリアル
mat = bpy.data.materials.new("mix")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
attr = nt.nodes.new("ShaderNodeVertexColor")
attr.layer_name = "cf"
sep = nt.nodes.new("ShaderNodeSeparateColor")
nt.links.new(attr.outputs["Color"], sep.inputs[0])
ramp = nt.nodes.new("ShaderNodeValToRGB")  # 混合率 -> 色
ramp.color_ramp.interpolation = "EASE"
ramp.color_ramp.elements[0].position = 0.02
ramp.color_ramp.elements[0].color = (0.93, 0.9, 0.82, 1)
ramp.color_ramp.elements[1].position = 0.75
ramp.color_ramp.elements[1].color = (0.05, 0.022, 0.01, 1)
mid = ramp.color_ramp.elements.new(0.15)
mid.color = (0.42, 0.26, 0.15, 1)
nt.links.new(sep.outputs[0], ramp.inputs[0])
nt.links.new(ramp.outputs[0], bsdf.inputs["Base Color"])
rough = nt.nodes.new("ShaderNodeMapRange")  # 液体は艶あり、固いクッキーはマット
rough.inputs["From Min"].default_value = 0.3
rough.inputs["From Max"].default_value = 0.8
rough.inputs["To Min"].default_value = 0.18
rough.inputs["To Max"].default_value = 0.85
nt.links.new(sep.outputs[0], rough.inputs["Value"])
nt.links.new(rough.outputs[0], bsdf.inputs["Roughness"])
bsdf.inputs["Subsurface Weight"].default_value = 0.35
bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.8, 0.6)
bsdf.inputs["Subsurface Scale"].default_value = 0.01
bsdf.inputs["Coat Weight"].default_value = 0.3
bsdf.inputs["Coat Roughness"].default_value = 0.05


def surface(pm, pc, voxel):
    """ミルクとクッキーの合成密度の等値面と、頂点ごとのクッキー混合率 (float[n]) を返す。"""
    from scipy.ndimage import map_coordinates
    from skimage.measure import marching_cubes

    allp = np.concatenate([pm, pc]) if len(pc) else pm
    lo = np.maximum(allp.min(0) - 4 * voxel, 0)
    hi = np.minimum(allp.max(0) + 4 * voxel, 1.0)
    g = density_grid(allp, lo, hi, voxel, 1.5)
    gc = density_grid(pc, lo, hi, voxel, 2.5) if len(pc) else np.zeros_like(g)
    gt = density_grid(allp, lo, hi, voxel, 2.5)
    inner = np.percentile(g[g > 1e-3], 90)
    v, f, _, _ = marching_cubes(np.pad(g, 1), 0.5 * inner, spacing=(voxel,) * 3)
    gi = (v / voxel - 1.0).T  # パディング分を戻したグリッド座標
    cf = map_coordinates(gc, gi, order=1, mode="nearest") / np.maximum(map_coordinates(gt, gi, order=1, mode="nearest"), 1e-3)
    return (v + lo - voxel).astype(np.float32), f.astype(np.int32), np.clip(cf, 0, 1).astype(np.float32)


angles = {}
for line in open(f"{a.src}/blade.txt"):
    k, v = line.split()
    angles[int(k)] = float(v)

for f in range(a.frames[0], a.frames[1] + 1, a.step):
    if C.done(f"{a.out}/frame_{f:05d}.png"):
        continue
    t0 = time.time()
    pm, _, lo, hi = load_nbp(f"{a.src}/milk_{f:05d}.nbp")
    pc, _, _, _ = load_nbp(f"{a.src}/cookie_{f:05d}.nbp")
    voxel = a.voxel if a.voxel > 0 else 1.4 * (MILK_VOLUME / max(len(pm), 1)) ** (1 / 3)
    v, fa, cf = surface(pm, pc, voxel)
    ob = C.mesh_object("liquid", v, fa, mat)
    ca = ob.data.color_attributes.new("cf", "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", np.stack([cf, cf, cf, np.ones_like(cf)], -1).ravel())
    ang = angles.get(f, 0.0)
    mv, mf = mixer_mesh(ang)
    mixer = C.mesh_object("mixer", mv, mf, steel, smooth=False)
    t1 = time.time()
    C.render_still(f"{a.out}/frame_{f:05d}.png")
    bpy.data.objects.remove(ob)
    bpy.data.objects.remove(mixer)
    print(f"frame {f}: {len(pm)}+{len(pc)} particles -> {len(fa)} tris, mesh {t1 - t0:.1f}s render {time.time() - t1:.1f}s", flush=True)
