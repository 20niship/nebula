#!/usr/bin/env python3
"""
山岳地形 STL 生成スクリプト

mpm_avalanche の地形関数と同一の terrainHeight(x, z, W) を使用し、
三角形メッシュを STL ファイルとして出力する。

出力: assets/terrain_mpm.stl

使い方:
  python3 tools/gen_terrain_stl.py [--world-size W] [--res N] [--out PATH]
"""

import argparse
import math
import struct
from pathlib import Path


def terrain_height(x: float, z: float, W: float) -> float:
    slope   = W * 0.55 * (1.0 - z / W)
    ridgeL  = W * 0.10 * math.exp(-((x - W*0.28) / (W*0.06))**2)
    ridgeR  = W * 0.10 * math.exp(-((x - W*0.72) / (W*0.06))**2)
    couloir = -W * 0.12 * math.exp(-((x - W*0.50) / (W*0.10 + z*0.006))**2)
    rough   = W * 0.020 * math.sin(x * 4.0) * math.cos(z * 3.0)
    texture = W * 0.010 * math.sin(x * 9.0 + z * 7.0)
    return max(0.2, slope + ridgeL + ridgeR + couloir + rough + texture)


def triangle_normal(p0, p1, p2):
    ax, ay, az = p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]
    bx, by, bz = p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]
    nx = ay*bz - az*by
    ny = az*bx - ax*bz
    nz = ax*by - ay*bx
    length = math.sqrt(nx*nx + ny*ny + nz*nz) or 1.0
    return nx/length, ny/length, nz/length


def write_binary_stl(path: Path, triangles):
    """triangles: list of (n, p0, p1, p2) where each is (x,y,z)"""
    header = b"MPM Mountain Terrain" + b" " * 60
    header = header[:80]
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for n, p0, p1, p2 in triangles:
            f.write(struct.pack("<3f", *n))
            f.write(struct.pack("<3f", *p0))
            f.write(struct.pack("<3f", *p1))
            f.write(struct.pack("<3f", *p2))
            f.write(struct.pack("<H", 0))  # attribute byte count


def build_terrain_stl(W: float, res: int) -> list:
    """
    res×res グリッドで地形サーフェスを三角形分割し、
    底面・側面を加えて完全閉合メッシュにする。
    """
    dx = W / res
    # 頂点グリッド生成 (res+1 × res+1)
    verts = []
    for iz in range(res + 1):
        row = []
        for ix in range(res + 1):
            x = ix * dx
            z = iz * dx
            y = terrain_height(x, z, W)
            row.append((x, y, z))
        verts.append(row)

    triangles = []

    # ── サーフェス三角形 ────────────────────────────────────────────────
    for iz in range(res):
        for ix in range(res):
            p00 = verts[iz  ][ix  ]
            p10 = verts[iz  ][ix+1]
            p01 = verts[iz+1][ix  ]
            p11 = verts[iz+1][ix+1]
            # 左下三角
            n = triangle_normal(p00, p10, p01)
            triangles.append((n, p00, p10, p01))
            # 右上三角
            n = triangle_normal(p10, p11, p01)
            triangles.append((n, p10, p11, p01))

    # ── 底面 (y=0, 法線 -Y) ──────────────────────────────────────────
    b_nrm = (0.0, -1.0, 0.0)
    for iz in range(res):
        for ix in range(res):
            b00 = (ix     * dx, 0.0, iz     * dx)
            b10 = ((ix+1) * dx, 0.0, iz     * dx)
            b01 = (ix     * dx, 0.0, (iz+1) * dx)
            b11 = ((ix+1) * dx, 0.0, (iz+1) * dx)
            triangles.append((b_nrm, b00, b01, b10))
            triangles.append((b_nrm, b10, b01, b11))

    # ── 4 側面 ────────────────────────────────────────────────────────
    # -Z 面 (iz=0)
    for ix in range(res):
        t = verts[0][ix  ]
        t2 = verts[0][ix+1]
        b  = (t [0], 0.0, t [2])
        b2 = (t2[0], 0.0, t2[2])
        n  = (0.0, 0.0, -1.0)
        triangles.append((n, t,  b,  t2))
        triangles.append((n, b,  b2, t2))

    # +Z 面 (iz=res)
    for ix in range(res):
        t  = verts[res][ix  ]
        t2 = verts[res][ix+1]
        b  = (t [0], 0.0, t [2])
        b2 = (t2[0], 0.0, t2[2])
        n  = (0.0, 0.0, 1.0)
        triangles.append((n, t2, b2, t ))
        triangles.append((n, b2, b,  t ))

    # -X 面 (ix=0)
    for iz in range(res):
        t  = verts[iz  ][0]
        t2 = verts[iz+1][0]
        b  = (t [0], 0.0, t [2])
        b2 = (t2[0], 0.0, t2[2])
        n  = (-1.0, 0.0, 0.0)
        triangles.append((n, t2, b2, t ))
        triangles.append((n, b2, b,  t ))

    # +X 面 (ix=res)
    for iz in range(res):
        t  = verts[iz  ][res]
        t2 = verts[iz+1][res]
        b  = (t [0], 0.0, t [2])
        b2 = (t2[0], 0.0, t2[2])
        n  = (1.0, 0.0, 0.0)
        triangles.append((n, t,  b,  t2))
        triangles.append((n, b,  b2, t2))

    return triangles


def main():
    root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description="山岳地形 STL 生成")
    parser.add_argument("--world-size", type=float, default=10.0,
                        help="地形の一辺サイズ [m] (デフォルト: 10)")
    parser.add_argument("--res",        type=int,   default=256,
                        help="グリッド解像度 (デフォルト: 256)")
    parser.add_argument("--out",        type=str,
                        default=str(root / "assets" / "terrain_mpm.stl"),
                        help="出力 STL ファイルパス")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"地形生成中: W={args.world_size}m, res={args.res}×{args.res}")
    triangles = build_terrain_stl(args.world_size, args.res)
    print(f"  三角形数: {len(triangles):,}")

    write_binary_stl(out_path, triangles)
    size_kb = out_path.stat().st_size / 1024
    print(f"  出力完了: {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
