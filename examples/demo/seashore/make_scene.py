# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""海岸シーン (砂浜の斜面 + 岩) を作り、nebula 用の SDF と Blender/プレビュー用のシーン記述を書き出す。
先に seashore/fetch_rocks.py で岩を取得しておく。出力は assets_local/seashore/ (git 管理外):
  scene.sdf   : int res, float cell, 続けて float[res^3] (ix + iy*res + iz*res^2, 原点=ドメイン原点)。demo_seashore が読む
  scene.json  : ドメイン・水位・パドル・岩の配置
  terrain.npz : 砂浜ハイトフィールド h[nz,nx] と step

座標は Y-up (x: 沖→岸, y: 高さ, z: 幅)。岩 i の頂点は  pos + Ry(yaw) @ (scale * (v - pivot))  (pivot = AABB の x,z 中心 / y 最小)。
  uv run seashore/make_scene.py --seed 3
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt

ASSETS = Path(__file__).resolve().parent.parent / "assets_local/seashore"
DOMAIN = np.array([44.0, 12.0, 16.0])
WATER = 3.0  # 静水面の高さ [m]


def beach_height(x, z):
    """x=12 から岸に向かって上がる斜面 (約 1:5)。水際は x≈26。"""
    t = np.clip((x - 12.0) / 28.0, 0, 1)
    return 6.5 * (0.35 * t + 0.65 * t * t) + 0.12 * np.sin(0.9 * x + 1.3 * z) * np.sin(0.5 * z + 0.4) * t


def sample_surface(v, f, spacing, rng):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    k = np.maximum(1, np.ceil(area / (0.25 * spacing**2)).astype(int))
    tri = np.repeat(np.arange(len(f)), k)
    u, w = rng.random((2, len(tri)))
    flip = u + w > 1
    u, w = np.where(flip, 1 - u, u), np.where(flip, 1 - w, w)
    return a[tri] + u[:, None] * (b - a)[tri] + w[:, None] * (c - a)[tri]


def rock_sdf(pts, lo_idx, shape, cell):
    """表面点群を体素化 -> 内部を塗りつぶし -> 符号付き距離 (外側 +)。"""
    occ = np.zeros(shape, bool)
    ij = np.floor(pts / cell).astype(int) - lo_idx
    occ[tuple(ij.T)] = True
    inside = binary_fill_holes(occ)
    return (distance_transform_edt(~inside) - distance_transform_edt(inside)) * cell - 0.5 * cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--rocks", type=int, default=12)
    ap.add_argument("--sdf-cell", type=float, default=0.15, help="SDF 解像度 [m] (シミュの格子とは独立)")
    ap.add_argument("--kinds", default="seashore,granite", help="使う岩種 (名前に含む文字列, カンマ区切り)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    files = sorted(p for p in (ASSETS / "rocks").glob("*.npz") if any(k in p.name for k in a.kinds.split(",")))
    if not files:
        raise SystemExit("岩が無い: uv run seashore/fetch_rocks.py")
    meshes = {p.stem: np.load(p) for p in files}

    # 岩の配置: 水際〜砕波帯 (x 22..38) に間隔を空けて置く。下 40% は砂に埋める
    placed = []
    for _ in range(4000):
        if len(placed) >= a.rocks:
            break
        name = rng.choice(list(meshes))
        x, z = rng.uniform(22, 38), rng.uniform(1.5, DOMAIN[2] - 1.5)
        s = rng.uniform(0.8, 1.3)
        r = 2.1 * s
        if any(math.hypot(p["pos"][0] - x, p["pos"][2] - z) < p["r"] + r - 0.6 for p in placed):
            continue
        v = meshes[name]["verts"]
        h = np.ptp(v[:, 1]) * s
        placed.append(dict(name=str(name), pos=[float(x), float(beach_height(x, z) - 0.4 * h), float(z)], yaw=float(rng.uniform(0, 2 * math.pi)), scale=float(s), r=r))

    cell = a.sdf_cell
    res = int(math.ceil(DOMAIN.max() / cell))
    ix = (np.arange(res) + 0.5) * cell
    X, Y, Z = np.meshgrid(ix, ix, ix, indexing="ij")  # [x,y,z]
    h = beach_height(X[:, 0, :], Z[:, 0, :])
    gx, gz = np.gradient(h, cell, cell)
    n = 1.0 / np.sqrt(1 + gx**2 + gz**2)
    sdf = (Y - h[:, None, :]) * n[:, None, :]
    for p in placed:
        v, f = meshes[p["name"]]["verts"], meshes[p["name"]]["faces"]
        pivot = np.array([(v[:, 0].min() + v[:, 0].max()) / 2, v[:, 1].min(), (v[:, 2].min() + v[:, 2].max()) / 2])
        c, s_ = math.cos(p["yaw"]), math.sin(p["yaw"])
        R = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
        w = (p["scale"] * (v - pivot)) @ R.T + p["pos"]
        p["pivot"] = pivot.tolist()
        lo = np.floor(w.min(0) / cell).astype(int) - 8
        hi = np.ceil(w.max(0) / cell).astype(int) + 8
        lo, hi = np.maximum(lo, 0), np.minimum(hi, res)
        d = rock_sdf(sample_surface(w, f, cell, rng), lo, tuple(hi - lo), cell)
        sl = tuple(slice(l, h_) for l, h_ in zip(lo, hi))
        sdf[sl] = np.minimum(sdf[sl], d)
    with open(ASSETS / "scene.sdf", "wb") as fh:
        fh.write(np.array([res], np.int32).tobytes() + np.array([cell], np.float32).tobytes())
        # [x,y,z] -> 平坦化 ix + iy*res + iz*res^2 (x が最速)
        fh.write(np.ascontiguousarray(sdf.transpose(2, 1, 0), np.float32).tobytes())

    step = 0.2
    gxs, gzs = np.arange(0, DOMAIN[0] + step, step), np.arange(0, DOMAIN[2] + step, step)
    np.savez(ASSETS / "terrain.npz", h=beach_height(gxs[None, :], gzs[:, None]).astype(np.float32), step=step)
    json.dump(dict(domain=DOMAIN.tolist(), water=WATER, paddle_x=2.0, rocks=[{k: v for k, v in p.items() if k != "r"} for p in placed]), open(ASSETS / "scene.json", "w"), indent=1)
    print(f"{len(placed)} rocks, sdf {res}^3 cell {cell} -> {ASSETS}")


if __name__ == "__main__":
    main()
