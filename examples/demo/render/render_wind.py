"""demo_city_wind の .pvox を Blender (Cycles) でレンダリングする。

ビル群 (city.txt) + 速度場から追跡した流線 (風速で色付けした発光カーブ) + 煙ボリューム。
uv run python render/render_wind.py out/wind --frames 60 60 --out out/wind_png
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
from scipy.ndimage import map_coordinates  # noqa: E402


def load_pvox(path):
    with open(path, "rb") as f:
        d = f.read()
    nx, ny, nz = struct.unpack_from("<3I", d, 4)
    cell, _ = struct.unpack_from("<2f", d, 16)
    (nc,) = struct.unpack_from("<I", d, 24)
    off, chans = 28, []
    for _ in range(nc):
        chans.append((d[off:off + 16].split(b"\0")[0].decode(), struct.unpack_from("<I", d, off + 16)[0]))
        off += 20
    out = {}
    for name, comp in chans:
        n = nx * ny * nz * comp
        a = np.frombuffer(d, "<f4", n, off)
        off += n * 4
        a = a.reshape(nz, ny, nx, comp).transpose(2, 1, 0, 3) if comp > 1 else a.reshape(nz, ny, nx).transpose(2, 1, 0)
        if name in ("velocity", "density"):
            out[name] = a[..., :3] if comp > 1 else a
    return out, cell


def trace(vel, cell, seeds, steps=500, h=0.6):
    """速度場 (nx,ny,nz,3) を RK2 で追跡した流線 [(steps+1), n, 3] [m] と速さを返す。"""
    p = seeds.copy() / cell - 0.5
    P, S = [p.copy()], []

    def samp(q):
        return np.stack([map_coordinates(vel[..., k], q.T, order=1, mode="nearest") for k in range(3)], 1)

    for _ in range(steps):
        v1 = samp(p)
        v2 = samp(p + 0.5 * h * v1 / cell)
        S.append(np.linalg.norm(v2, axis=1))
        p = p + h * v2 / cell
        P.append(p.copy())
    return (np.array(P) + 0.5) * cell, np.array(S)


def curves(P, idx, radius, name):
    """流線 P[:, idx] を 1 つの curve オブジェクトにまとめる。"""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 1
    for i in idx:
        pts = P[:, i]
        keep = (pts[:, 0] < 191) & (pts[:, 1] > 0.3) & (pts[:, 1] < 71) & (pts[:, 2] > 1) & (pts[:, 2] < 95)
        n = int(np.argmin(keep)) if not keep.all() else len(pts)
        if n < 8:
            continue
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
ap.add_argument("--streams", type=int, default=380)
ap.add_argument("--vmax", type=float, default=17.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

city = np.loadtxt(f"{a.src}/city.txt").reshape(-1, 6)
sc = C.reset_scene(tuple(a.res), a.samples)
sc.view_settings.exposure = 0.0
C.sky_world(25, 200, 0.18)
sc.view_settings.view_transform = "Standard"
C.camera(C.sim_to_blender((-30, 78, 6)), C.sim_to_blender((105, 8, 50)), lens=32)  # 風上斜め上から街区を見下ろす (シミュ座標)

ground = C.material("ground", (0.08, 0.085, 0.09, 1), 0.8)
bld = C.material("building", (0.16, 0.17, 0.2, 1), 0.3)
C.box("ground", (96, -0.5, 48), (300, 0.5, 300), ground)
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

rng = np.random.default_rng(3)
for f in range(a.frames[0], a.frames[1] + 1, a.step):
    if C.done(f"{a.out}/frame_{f:05d}.png"):
        continue
    t0 = time.time()
    vox, cell = load_pvox(f"{a.src}/frame_{f:05d}.pvox")
    vel = vox["velocity"]
    seeds = np.stack([np.full(a.streams, 3.0), rng.uniform(1.5, 26, a.streams), rng.uniform(3, 93, a.streams)], 1)
    P, S = trace(vel, cell, seeds)
    mean = np.percentile(S, 85, axis=0)
    bins = np.clip((mean / a.vmax * NB).astype(int), 0, NB - 1)
    obs = []
    for k in range(NB):
        idx = np.nonzero(bins == k)[0]
        if len(idx):
            o = curves(P, idx, 0.07, f"streams{k}")
            o.data.materials.append(GLOWS[k])
            obs.append(o)
    t1 = time.time()
    C.render_still(f"{a.out}/frame_{f:05d}.png")
    for o in obs:
        bpy.data.objects.remove(o)
    print(f"frame {f}: trace {t1 - t0:.1f}s render {time.time() - t1:.1f}s", flush=True)
