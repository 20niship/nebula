"""demo_city_wind の .pvox を Blender (Cycles) でレンダリングする。

ビル群 (city.txt) + トレーサ粒子の軌跡 (tracers.bin。風速で色付けした発光カーブ)。
軌跡は時間方向に連続して流れるので、瞬間速度場から毎フレーム引き直す流線と違ってちらつかない。
uv run python render/render_wind.py out/wind --frames 0 0 --out out/wind_png
"""
import argparse
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import bpy  # noqa: E402
import common as C  # noqa: E402
import numpy as np  # noqa: E402


def load_tracers(path):
    """(pos[steps,M,3] float32, gen[steps,M] uint32, dt, domain) を返す。軌跡は pool の添字 i ごとに時間方向へ連続し、湧き直すと gen が変わる。"""
    with open(path, "rb") as f:
        assert f.read(4) == b"TRC1", path
        steps, m = struct.unpack("<2I", f.read(8))
        dt, *dom = struct.unpack("<4f", f.read(16))
        raw = np.frombuffer(f.read(steps * m * 16), np.dtype([("p", "<f4", 3), ("g", "<u4")])).reshape(steps, m)
    return raw["p"].copy(), raw["g"].copy(), dt, np.array(dom, np.float32)


def curves(P, idx, radius, name):
    """軌跡 P[:, idx] ([n, m, 3] のうち有効な先頭 n 点) を 1 つの curve オブジェクトにまとめる。"""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 1
    for i in idx:
        pts, n = P[i]
        sp = cu.splines.new("POLY")
        sp.points.add(n - 1)
        co = np.zeros((n, 4), np.float32)
        co[:, :3] = C.to_blender(pts[:n])
        co[:, 3] = 1
        sp.points.foreach_set("co", co.ravel())
    ob = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(ob)
    return ob


ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out", default="out/wind_png")
ap.add_argument("--frames", type=int, nargs=2, default=[0, 0], metavar=("START", "END"))
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
ap.add_argument("--samples", type=int, default=96)
ap.add_argument("--trail", type=int, default=36, help="軌跡の長さ [ステップ] (1 ステップ = dt)")
ap.add_argument("--radius", type=float, default=0.05)
ap.add_argument("--max-streams", type=int, default=3500)
ap.add_argument("--vmax", type=float, default=17.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

city = np.loadtxt(f"{a.src}/city.txt").reshape(-1, 6)
TP, TG, DT, DOM = load_tracers(f"{a.src}/tracers.bin")
sc = C.reset_scene(tuple(a.res), a.samples)
sc.view_settings.exposure = 0.0
C.sky_world(25, 200, 0.18)
sc.view_settings.view_transform = "Standard"
C.camera(C.sim_to_blender(DOM * (0.02, 1.55, -0.55)), C.sim_to_blender(DOM * (0.52, 0.0, 0.48)), lens=28)  # 風上斜め上から街区を見下ろす (シミュ座標)

ground = C.material("ground", (0.08, 0.085, 0.09, 1), 0.8)
bld = C.material("building", (0.16, 0.17, 0.2, 1), 0.3)
C.box("ground", (DOM[0] / 2, -0.5, DOM[2] / 2), (300, 0.5, 300), ground)
for i, b in enumerate(city):
    C.box(f"b{i}", (b[:3] + b[3:]) / 2, (b[3:] - b[:3]) / 2, bld)
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 2.0
sun.rotation_euler = (0.8, 0.0, 0.9)
bpy.context.collection.objects.link(sun)

# 流線は平均風速で 8 段階に分け、段ごとに色の違う発光マテリアルを使う (青 -> 緑 -> 赤)
NB = 8
COLORS = [(0.05, 0.25, 1.0), (0.0, 0.6, 1.0), (0.0, 0.9, 0.8), (0.1, 1.0, 0.4), (0.7, 1.0, 0.1), (1.0, 0.8, 0.0), (1.0, 0.45, 0.05), (1.0, 0.1, 0.05)]
GLOWS = []
for i, col in enumerate(COLORS):
    m = bpy.data.materials.new(f"glow{i}")
    m.use_nodes = True
    n = m.node_tree
    n.nodes.clear()
    em = n.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*col, 1)
    em.inputs["Strength"].default_value = 2.2
    out = n.nodes.new("ShaderNodeOutputMaterial")
    n.links.new(em.outputs[0], out.inputs[0])
    GLOWS.append(m)

for f in range(a.frames[0], a.frames[1] + 1, a.step):
    if C.done(f"{a.out}/frame_{f:05d}.png"):
        continue
    t0 = time.time()
    f = min(f, len(TP) - 1)
    lo = max(0, f - a.trail + 1)
    gen = TG[lo : f + 1]
    # 各トレーサの軌跡 = 現在の世代と同じ gen が続いている末尾の区間
    same = (gen == gen[-1]).astype(np.int8)[::-1].cumprod(axis=0)[::-1].astype(bool)
    P = TP[lo : f + 1]
    n = same.sum(0)
    sel = np.nonzero(n >= 6)[0]
    sel = sel[np.linspace(0, len(sel) - 1, min(len(sel), a.max_streams)).astype(int)] if len(sel) else sel
    items, spd = {}, np.zeros(len(P[0]))
    for i in sel:
        pts = P[len(P) - n[i] :, i]
        items[i] = (pts, len(pts))
        spd[i] = np.linalg.norm(np.diff(pts, axis=0), axis=1).mean() / DT
    bins = np.clip((spd / a.vmax * NB).astype(int), 0, NB - 1)
    obs = []
    for k in range(NB):
        idx = [i for i in sel if bins[i] == k]
        if idx:
            o = curves(items, idx, a.radius, f"streams{k}")
            o.data.materials.append(GLOWS[k])
            obs.append(o)
    t1 = time.time()
    C.render_still(f"{a.out}/frame_{f:05d}.png")
    for o in obs:
        bpy.data.objects.remove(o)
    print(f"frame {f}: {len(sel)} trails, build {t1 - t0:.1f}s render {time.time() - t1:.1f}s", flush=True)
