# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "scikit-image", "pillow"]
# ///
"""手続き生成のスポーツカー (全長 4.4m, 後ろに 2 枚構成のリアウイング) の SDF とメッシュを作る。アセットは git 管理外 (assets_local/car/)。

座標はローカル (x: 前, y: 上, z: 横)。地面は y=0、車輪の底が接地する。回転の中心は原点 (車体のほぼ中央)。
  assets_local/car/car.sdf : int nx,ny,nz | float cell | float min[3] | float data[nx*ny*nz] (x 最速。節点値, 節点 (i,j,k) = min + (i,j,k)*cell)
  assets_local/car/car.npz : verts/faces (marching cubes, Blender 用)
  assets_local/car/side.png / top.png / front.png : 形状確認用の断面 (SDF<0 の占有)
  uv run car/make_car.py [--cell 0.02] [--gap 0.08]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from skimage.measure import marching_cubes

OUT = Path(__file__).resolve().parent.parent / "assets_local/car"
LO = np.array([-2.9, 0.0, -1.25])
HI = np.array([2.7, 1.7, 1.25])


def smin(a, b, k):
    """多項式 smooth min (k: ブレンド幅 [m])。"""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def profile(xs, ys, x):
    """制御点 (xs, ys) を滑らかに補間した値。"""
    g = np.linspace(-3.2, 3.0, 1300)
    v = np.interp(g, xs, ys)
    v = gaussian_filter1d(v, 28, mode="nearest")
    return np.interp(x, g, v)


def ellipse_sdf(px, py, cx, cy, a, b, ang):
    """(x,y) 断面の楕円 (半軸 a,b, 傾き ang) の近似 SDF。"""
    c, s = np.cos(ang), np.sin(ang)
    dx, dy = px - cx, py - cy
    u, v = c * dx + s * dy, -s * dx + c * dy
    return (np.sqrt((u / a) ** 2 + (v / b) ** 2) - 1.0) * min(a, b)


def box_sdf(P, c, h):
    q = np.abs(P - np.array(c)) - np.array(h)
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(-1), 0.0)


def car_sdf(P, gap=0.08):
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    az = np.abs(z)

    # ── 車体: x ごとに断面 (超楕円) が変わるロフト ──
    X0, X1 = -2.25, 2.2
    top = profile([-2.4, -2.25, -1.5, -0.6, 0.3, 0.9, 1.5, 2.0, 2.25, 2.4], [0.70, 0.76, 0.80, 0.76, 0.70, 0.62, 0.45, 0.33, 0.26, 0.2], x)
    bot = profile([-2.4, -2.1, -1.0, 1.0, 2.0, 2.4], [0.30, 0.20, 0.13, 0.13, 0.12, 0.17], x)
    half_w = profile([-2.4, -2.25, -1.6, -0.6, 0.2, 1.0, 1.7, 2.1, 2.4], [0.40, 0.52, 0.84, 0.95, 0.94, 0.88, 0.76, 0.6, 0.38], x)
    yc, hh = 0.5 * (top + bot), np.maximum(0.5 * (top - bot), 1e-3)
    f = (az / half_w) ** 3 + (np.abs(y - yc) / hh) ** 3
    sd_cross = (np.cbrt(f) - 1.0) * np.minimum(half_w, hh)
    dx = np.maximum(np.maximum(x - X1, X0 - x), 0.0)
    body = np.hypot(np.maximum(sd_cross, 0.0), dx) + np.minimum(sd_cross, 0.0) * (dx == 0)

    # ── キャビン (風防とルーフ) ──
    cab = (np.sqrt(((x + 0.15) / 1.10) ** 2 + ((y - 0.74) / 0.42) ** 2 + (az / 0.66) ** 2) - 1.0) * 0.42
    body = smin(body, cab, 0.18)

    # ── 車輪 (z 軸の円柱)。前後 2 軸 ──
    wheels = np.full(x.shape, 1e3)
    for wx in (1.35, -1.25):
        r = np.hypot(x - wx, y - 0.34) - 0.34
        w = np.abs(az - 0.80) - 0.15
        wheels = np.minimum(wheels, np.hypot(np.maximum(r, 0), np.maximum(w, 0)) + np.minimum(np.maximum(r, w), 0))
    sd = np.minimum(body, wheels)

    # ── フロントスプリッター ──
    sd = np.minimum(sd, box_sdf(P, (2.05, 0.075, 0), (0.30, 0.02, 0.82)))

    # ── リアウイング: 2 枚の翼型 (間にスロット) + 端板 + 支柱 ──
    span = np.abs(z) - 0.86
    def extrude(d2):
        return np.hypot(np.maximum(d2, 0), np.maximum(span, 0)) + np.minimum(np.maximum(d2, span), 0)
    main = extrude(ellipse_sdf(x, y, -2.00, 1.12, 0.30, 0.030, np.radians(-8)))
    flap = extrude(ellipse_sdf(x, y, -2.27 - gap * 0.4, 1.12 + 0.12 + gap, 0.19, 0.022, np.radians(-30)))
    plates = box_sdf(P, (-2.12, 1.20, 0), (0.42, 0.24, 0.9))
    plates = np.maximum(plates, -(box_sdf(P, (-2.12, 1.20, 0), (0.6, 0.3, 0.87))))  # 中をくり抜いて左右の端板だけ残す
    posts = np.minimum(box_sdf(P, (-1.98, 0.97, 0.38), (0.045, 0.17, 0.02)), box_sdf(P, (-1.98, 0.97, -0.38), (0.045, 0.17, 0.02)))
    sd = np.minimum(sd, np.minimum(np.minimum(main, flap), np.minimum(plates, posts)))
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=float, default=0.01, help="シミュレーション用 SDF の節点間隔 [m]")
    ap.add_argument("--mesh-cell", type=float, default=0.012, help="メッシュ (marching cubes) の間隔 [m]")
    ap.add_argument("--gap", type=float, default=0.08, help="リアウイングの 2 枚の間のスロット幅 [m]")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    def grid(cell):
        ax = [np.arange(l, h + cell * 0.5, cell, dtype=np.float32) for l, h in zip(LO, HI)]
        P = np.stack(np.meshgrid(*ax, indexing="ij"), -1)  # [x,y,z]
        return P, [len(t) for t in ax]

    P, (nx, ny, nz) = grid(a.cell)
    sd = car_sdf(P, a.gap).astype(np.float32)
    del P
    with open(OUT / "car.sdf", "wb") as f:
        f.write(np.array([nx, ny, nz], np.int32).tobytes() + np.array([a.cell], np.float32).tobytes() + LO.astype(np.float32).tobytes())
        f.write(np.ascontiguousarray(sd.transpose(2, 1, 0)).tobytes())  # x 最速

    # 形状確認用の断面
    def save(name, mask):
        Image.fromarray((mask * 255).astype(np.uint8)).resize((mask.shape[1] * 3, mask.shape[0] * 3), Image.NEAREST).save(OUT / name)
    inside = sd < 0
    save("side.png", inside.any(2).T[::-1])                         # x-y (z 方向に投影)
    save("top.png", inside.any(1).T[::-1])                          # x-z
    save("front.png", inside.any(0).T[::-1])                        # z-y
    save("side_mid.png", inside[:, :, nz // 2 - 1].T[::-1])          # z=0 の断面 (スロットの確認)

    Pm, _ = grid(a.mesh_cell)
    v, fc, _, _ = marching_cubes(car_sdf(Pm, a.gap).astype(np.float32), 0.0, spacing=(a.mesh_cell,) * 3)
    del Pm
    np.savez(OUT / "car.npz", verts=(v + LO).astype(np.float32), faces=fc.astype(np.int32))
    print(f"sdf {nx}x{ny}x{nz} cell {a.cell}  mesh {len(v)} verts {len(fc)} tris  slot {a.gap} m = {a.gap / a.cell:.1f} cells")


if __name__ == "__main__":
    main()
