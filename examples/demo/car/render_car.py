# /// script
# requires-python = ">=3.10"
# dependencies = ["pyvista", "numpy", "scipy", "pillow"]
# ///
"""demo_car_wind の出力を PyVista (VTK) で描画する。フォトリアルは不要なので Blender は使わない。

  q           Q 基準 (速度勾配テンソルの渦度 - ひずみ、Q>0 が渦) の等値面を速度で色付け。FluidX3D の渦構造の可視化と同じ。車体に被らない (既定)
  streamlines トレーサ粒子の軌跡 (tracers.bin)。実行が終わらないとヘッダのステップ数が確定しない
  both        両方
車は暗い半透明で、自転角 omega*dt*(sim step) で回す (エンジンの回転コライダーと同じ向き)。黒背景に領域の枠線 (FluidX3D 風)。

  uv run car/render_car.py out/car --frames 0 150 --step 4 --out out/car_png
  ffmpeg -framerate 15 -i out/car_png/frame_%05d.png -c:v libx264 -pix_fmt yuv420p car.mp4
"""
import argparse
import os
import struct
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter

pv.OFF_SCREEN = True
ASSETS = Path(__file__).resolve().parent.parent / "assets_local" / "car"


def load_qvox(path, boost):
    """(present, speed01, lo, cell)。AABB (+余白) に切り詰めた密配列。q は書き出し閾値の 1..8 倍を対数で 0..255。"""
    with open(path, "rb") as f:
        assert f.read(4) == b"QVX1", path
        nx, ny, nz = struct.unpack("<3I", f.read(12))
        cell, vmax, n = struct.unpack("<ffI", f.read(12))
        d = np.frombuffer(f.read(n * 8), np.dtype([("p", "<u2", 3), ("s", "u1"), ("q", "u1")]))
    d = d[d["q"] >= int(np.log2(max(boost, 1.0)) / 3.0 * 255)]
    p = d["p"].astype(np.int64)
    lo = np.maximum(p.min(0) - 4, 0)
    hi = np.minimum(p.max(0) + 5, [nx, ny, nz])
    present = np.zeros(tuple(hi - lo), np.float32)
    speed = np.zeros(tuple(hi - lo), np.float32)
    q = p - lo
    present[q[:, 0], q[:, 1], q[:, 2]] = 1.0
    speed[q[:, 0], q[:, 1], q[:, 2]] = d["s"] / 255.0
    return present, speed, lo, cell


def vortex_mesh(path, smooth, boost):
    present, speed, lo, cell = load_qvox(path, boost)
    pf = gaussian_filter(present, smooth)
    sp = gaussian_filter(speed, smooth) / np.maximum(pf, 1e-3)  # 存在するセルの速度の加重平均
    g = pv.ImageData(dimensions=present.shape, spacing=(cell,) * 3, origin=(lo + 0.5) * cell)
    g.point_data["present"] = pf.ravel(order="F")
    g.point_data["speed"] = np.clip(sp, 0, 1).ravel(order="F")
    m = g.contour([0.35], scalars="present", method="flying_edges")
    if not m.n_points:
        return m
    m = m.smooth(n_iter=8, relaxation_factor=0.1)
    return m.sample(g)  # 等値面の頂点に速度を補間 (contour は等値面のスカラーしか残さない)


def load_tracers(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"TRC1", path
        steps, m = struct.unpack("<2I", f.read(8))
        dt, *dom = struct.unpack("<4f", f.read(16))
        raw = np.frombuffer(f.read(steps * m * 16), np.dtype([("p", "<f4", 3), ("g", "<u4")])).reshape(steps, m)
    return raw["p"], raw["g"], dt


def streamlines(TP, TG, f, trail, cap):
    lo = max(0, f - trail + 1)
    gen = TG[lo : f + 1]
    same = (gen == gen[-1]).astype(np.int8)[::-1].cumprod(axis=0)[::-1].astype(bool)
    n = same.sum(0)
    sel = np.nonzero(n >= 6)[0]
    sel = sel[np.linspace(0, len(sel) - 1, min(len(sel), cap)).astype(int)] if len(sel) else sel
    pts, lines, spd = [], [], []
    for i in sel:
        q = TP[lo : f + 1][len(gen) - n[i] :, i]
        base = len(pts)
        pts.extend(q)
        lines.extend([len(q), *range(base, base + len(q))])
        spd.extend(np.r_[0.0, np.linalg.norm(np.diff(q, axis=0), axis=1)])
    if not pts:
        return None
    pd = pv.PolyData(np.array(pts, np.float32), lines=np.array(lines))
    pd["speed"] = np.array(spd, np.float32)
    return pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="out/car_png")
    ap.add_argument("--vis", choices=["q", "streamlines", "both"], default="q")
    ap.add_argument("--frames", type=int, nargs=2, default=[0, 0], metavar=("START", "END"))
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
    ap.add_argument("--q-boost", type=float, default=4.0, help="書き出し時の Q 閾値の何倍以上を面にするか (1..8)")
    ap.add_argument("--smooth", type=float, default=0.8, help="Q 等値面のガウス平滑 [セル]")
    ap.add_argument("--trail-sec", type=float, default=0.12)
    ap.add_argument("--max-streams", type=int, default=3000)
    ap.add_argument("--cam", type=float, nargs=3, default=[3.5, 2.4, -5.5], help="車の自転軸からの相対位置 [m]")
    ap.add_argument("--target", type=float, nargs=3, default=[2.0, 0.4, 0.0])
    ap.add_argument("--fov", type=float, default=45.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sc = np.loadtxt(f"{a.src}/scene.txt")
    dom, pivot, omega, dt, warm = sc[:3], sc[4:6], float(sc[6]), float(sc[7]), int(sc[8])
    P3 = np.array([pivot[0], 0.0, pivot[1]])
    TP = TG = None
    if a.vis != "q":
        TP, TG, _ = load_tracers(f"{a.src}/tracers.bin")

    npz = np.load(ASSETS / "car.npz")
    car0 = pv.PolyData(npz["verts"].astype(np.float32), np.c_[np.full(len(npz["faces"]), 3), npz["faces"]].ravel())
    # 車体の部位ごとに色を付ける (ローカル座標で判定): リアウイング=オレンジ (スロットが見える)、車輪=濃いグレー、車体=明るい青みがかった白
    lp = car0.points
    col = np.tile(np.array([205, 214, 230], np.uint8), (len(lp), 1))
    wheel = np.zeros(len(lp), bool)
    for wx in (1.35, -1.25):
        wheel |= (np.hypot(lp[:, 0] - wx, lp[:, 1] - 0.34) < 0.37) & (np.abs(np.abs(lp[:, 2]) - 0.80) < 0.18)
    col[wheel] = (45, 48, 56)
    col[(lp[:, 0] < -1.55) & (lp[:, 1] > 0.88)] = (255, 140, 20)  # ウイング (2 枚 + 端板 + 支柱の上部)
    col[(lp[:, 0] > 1.75) & (lp[:, 1] < 0.14)] = (255, 90, 90)    # フロントスプリッター
    car0["rgb"] = col
    qs = sorted(int(n[2:7]) for n in os.listdir(a.src) if n.startswith("q_") and n.endswith(".qvox"))

    pl = pv.Plotter(off_screen=True, window_size=tuple(a.res))
    pl.set_background("black")
    pl.add_mesh(pv.Box(bounds=(0, dom[0], 0, dom[1], 0, dom[2])), style="wireframe", color="#3a6a3a", line_width=1)
    pl.add_mesh(pv.Plane(center=(dom[0] / 2, 0, dom[2] / 2), direction=(0, 1, 0), i_size=dom[0], j_size=dom[2]), color="#10131a")
    pl.camera.position = tuple(P3 + a.cam)
    pl.camera.focal_point = tuple(P3 + a.target)
    pl.camera.up = (0, 1, 0)
    pl.camera.view_angle = a.fov

    qcache = {}
    for f in range(a.frames[0], a.frames[1] + 1, a.step):
        out = f"{a.out}/frame_{f:05d}.png"
        if os.path.exists(out):
            continue
        t0 = time.time()
        theta = omega * dt * (warm + f + 1)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # y 軸まわり (エンジンの Ry と同じ向き)
        car = car0.copy()
        car.points = (car0.points @ R.T + P3).astype(np.float32)
        actors = [pl.add_mesh(car, scalars="rgb", rgb=True, opacity=0.92, smooth_shading=True, specular=0.4)]
        if a.vis in ("q", "both"):
            qf = min(qs, key=lambda k: abs(k - f))
            if qcache.get("qf") != qf:  # 同じ Q ファイルのフレーム (q-every=4 なら 4 枚) は等値面を作り直さない
                qcache.update(qf=qf, mesh=vortex_mesh(f"{a.src}/q_{qf:05d}.qvox", a.smooth, a.q_boost))
            m = qcache["mesh"]
            if m.n_points:
                actors.append(pl.add_mesh(m, scalars="speed", cmap="turbo", clim=(0.0, 1.0), smooth_shading=True, show_scalar_bar=False, ambient=0.35))
        if a.vis in ("streamlines", "both"):
            tr = streamlines(TP, TG, min(f, len(TG) - 1), max(8, round(a.trail_sec / dt)), a.max_streams if a.vis == "streamlines" else 600)
            if tr is not None:
                actors.append(pl.add_mesh(tr, scalars="speed", cmap="turbo", line_width=1.5, show_scalar_bar=False, opacity=0.8))
        pl.screenshot(out)
        for ac in actors:
            pl.remove_actor(ac)
        print(f"frame {f}: theta {np.degrees(theta):.0f}deg  {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
