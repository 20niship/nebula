# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""demo_seashore の出力 (.nbp/.nbf) を点群のまま斜め俯瞰でプレビュー PNG にする。Blender に渡す前の確認用。
水粒子=速さで青->白、泡=白 (spray/foam/bubble で明るさ違い)、地形=砂浜ハイトフィールド、岩=SDF<0 の点群。

  uv run seashore/preview_points.py out/seashore --frames 0 360 --step 12 --out out/seashore_preview
  ffmpeg -framerate 12 -i out/seashore_preview/p_%05d.png -pix_fmt yuv420p preview.mp4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "render"))
from nbp import load_nbf, load_nbp  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets_local/seashore"


def splat(img, depth, P, col, view, size, r=1):
    """P: (n,3) ワールド座標。view: (M(3x3), scale, offset)。z-buffer で点を r px 四方に描く。"""
    M, s, off = view
    Q = (P - np.array([22.0, 3.0, 8.0])) @ M.T
    x, y = (Q[:, 0] * s + off[0]).astype(int), (off[1] - Q[:, 1] * s).astype(int)
    z = Q[:, 2]
    ok = (x >= r) & (x < size[0] - r) & (y >= r) & (y < size[1] - r)
    x, y, z, col = x[ok], y[ok], z[ok], col[ok]
    o = np.argsort(-z)  # 奥から手前へ書く (同じ画素は手前が勝つ)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            img[y[o] + dy, x[o] + dx] = col[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--frames", type=int, nargs=2, default=[0, 0])
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out", default="out/seashore_preview")
    ap.add_argument("--size", type=int, nargs=2, default=[1600, 720])
    ap.add_argument("--yaw", type=float, default=-28.0)
    ap.add_argument("--pitch", type=float, default=24.0)
    ap.add_argument("--zoom", type=float, default=34.0, help="px/m")
    ap.add_argument("--max-points", type=int, default=2_000_000, help="水粒子の描画上限 (間引く)")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    size = tuple(a.size)
    yaw, pit = np.radians(a.yaw), np.radians(a.pitch)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pit), -np.sin(pit)], [0, np.sin(pit), np.cos(pit)]])
    view = (Rx @ Ry, a.zoom, (size[0] / 2, size[1] * 0.62))

    # 静的な背景: 砂浜と岩 (SDF から内部点を抽出)
    T = np.load(ASSETS / "terrain.npz")
    h, st = T["h"], float(T["step"])
    gz, gx = np.meshgrid(np.arange(h.shape[0]) * st, np.arange(h.shape[1]) * st, indexing="ij")
    ter = np.stack([gx, h, gz], -1).reshape(-1, 3)
    with open(ASSETS / "scene.sdf", "rb") as f:
        res = int(np.frombuffer(f.read(4), np.int32)[0])
        cell = float(np.frombuffer(f.read(4), np.float32)[0])
        sdf = np.frombuffer(f.read(), np.float32).reshape(res, res, res)  # [z,y,x]
    iz, iy, ix = np.nonzero((sdf < 0.0) & (sdf > -1.6 * cell) & (np.indices(sdf.shape)[1] * cell > 0.6) & (np.arange(res)[None, None, :] * cell > 18))
    rock = (np.stack([ix, iy, iz], -1) + 0.5) * cell
    rock = rock[rock[:, 1] > np.interp(rock[:, 0], np.arange(h.shape[1]) * st, h.mean(0)) + 0.25]  # 砂面より上に出ている部分だけ

    for f in range(a.frames[0], a.frames[1] + 1, a.step):
        img = np.zeros((size[1], size[0], 3), np.uint8)
        img[:] = (14, 22, 34)
        splat(img, None, ter, np.tile(np.array([[150, 130, 95]], np.uint8), (len(ter), 1)), view, size)
        splat(img, None, rock, np.tile(np.array([[95, 92, 88]], np.uint8), (len(rock), 1)), view, size, 1)
        pos, spd, _, _ = load_nbp(f"{a.src}/frame_{f:05d}.nbp")
        if len(pos) > a.max_points:
            sel = np.random.default_rng(0).choice(len(pos), a.max_points, replace=False)
            pos, spd = pos[sel], spd[sel]
        t = np.clip(spd / 8.0, 0, 1)[:, None]
        splat(img, None, pos, (np.array([[20, 90, 200]]) * (1 - t) + np.array([[200, 235, 255]]) * t).astype(np.uint8), view, size)
        fp = Path(f"{a.src}/foam_{f:05d}.nbf")
        nf = 0
        if fp.exists():
            fpos, kind = load_nbf(str(fp))
            nf = len(fpos)
            splat(img, None, fpos, np.array([[0, 0, 0], [255, 255, 255], [235, 240, 245], [150, 190, 230]], np.uint8)[kind], view, size)
        Image.fromarray(img).save(f"{a.out}/p_{f:05d}.png")
        print(f"frame {f}: water {len(pos)} foam {nf}")


if __name__ == "__main__":
    main()
