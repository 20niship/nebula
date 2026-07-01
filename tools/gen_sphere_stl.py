#!/usr/bin/env python3
"""
球体 (icosphere) STL 生成スクリプト

mpm_stl_drop の障害物として使用する球体 STL を生成する。
icosahedron を指定回数再分割して球に投影し、binary STL で出力する。

使い方:
  python3 tools/gen_sphere_stl.py [--cx X] [--cy Y] [--cz Z] \
                                   [--radius R] [--subdivisions N] \
                                   [--out PATH]
"""

import argparse
import math
import struct
from pathlib import Path


def normalize(v):
    x, y, z = v
    n = math.sqrt(x*x + y*y + z*z)
    return (x/n, y/n, z/n)


def midpoint(v0, v1):
    return normalize(((v0[0]+v1[0])/2, (v0[1]+v1[1])/2, (v0[2]+v1[2])/2))


def build_icosphere(subdivisions: int):
    """単位球 icosphere を構築 (頂点・面リスト)"""
    phi = (1 + math.sqrt(5)) / 2
    base_verts = [
        normalize((-1,  phi, 0)), normalize(( 1,  phi, 0)),
        normalize((-1, -phi, 0)), normalize(( 1, -phi, 0)),
        normalize(( 0, -1,  phi)), normalize(( 0,  1,  phi)),
        normalize(( 0, -1, -phi)), normalize(( 0,  1, -phi)),
        normalize(( phi, 0, -1)), normalize(( phi, 0,  1)),
        normalize((-phi, 0, -1)), normalize((-phi, 0,  1)),
    ]
    base_faces = [
        (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
        (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
        (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
    ]

    verts = list(base_verts)
    faces = list(base_faces)
    mid_cache = {}

    def get_mid(i, j):
        key = (min(i,j), max(i,j))
        if key in mid_cache:
            return mid_cache[key]
        m = midpoint(verts[i], verts[j])
        verts.append(m)
        idx = len(verts) - 1
        mid_cache[key] = idx
        return idx

    for _ in range(subdivisions):
        new_faces = []
        for f in faces:
            a, b, c = f
            ab = get_mid(a, b)
            bc = get_mid(b, c)
            ca = get_mid(c, a)
            new_faces += [(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)]
        faces = new_faces

    return verts, faces


def face_normal(v0, v1, v2):
    ax, ay, az = v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]
    bx, by, bz = v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]
    nx = ay*bz - az*by
    ny = az*bx - ax*bz
    nz = ax*by - ay*bx
    length = math.sqrt(nx*nx + ny*ny + nz*nz) or 1.0
    return nx/length, ny/length, nz/length


def write_binary_stl(path: Path, triangles):
    header = b"Icosphere obstacle STL" + b" " * 58
    header = header[:80]
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for n, p0, p1, p2 in triangles:
            f.write(struct.pack("<3f", *n))
            f.write(struct.pack("<3f", *p0))
            f.write(struct.pack("<3f", *p1))
            f.write(struct.pack("<3f", *p2))
            f.write(struct.pack("<H", 0))


def main():
    root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description="球体 (icosphere) STL 生成")
    parser.add_argument("--cx",           type=float, default=5.0,  help="中心 X [m]")
    parser.add_argument("--cy",           type=float, default=2.5,  help="中心 Y [m]")
    parser.add_argument("--cz",           type=float, default=5.0,  help="中心 Z [m]")
    parser.add_argument("--radius",       type=float, default=1.5,  help="半径 [m]")
    parser.add_argument("--subdivisions", type=int,   default=3,    help="再分割回数 (デフォルト: 3 → 1280 面)")
    parser.add_argument("--out",          type=str,
                        default=str(root / "assets" / "sphere_obstacle.stl"),
                        help="出力 STL ファイルパス")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Icosphere 生成: center=({args.cx},{args.cy},{args.cz}) radius={args.radius} subdivisions={args.subdivisions}")
    verts, faces = build_icosphere(args.subdivisions)
    print(f"  頂点数: {len(verts)}  面数: {len(faces)}")

    cx, cy, cz, r = args.cx, args.cy, args.cz, args.radius
    triangles = []
    for f in faces:
        v0 = (verts[f[0]][0]*r+cx, verts[f[0]][1]*r+cy, verts[f[0]][2]*r+cz)
        v1 = (verts[f[1]][0]*r+cx, verts[f[1]][1]*r+cy, verts[f[1]][2]*r+cz)
        v2 = (verts[f[2]][0]*r+cx, verts[f[2]][1]*r+cy, verts[f[2]][2]*r+cz)
        # 外向き法線 = 球の中心から面重心への方向
        gx = (v0[0]+v1[0]+v2[0])/3 - cx
        gy = (v0[1]+v1[1]+v2[1])/3 - cy
        gz = (v0[2]+v1[2]+v2[2])/3 - cz
        nl = math.sqrt(gx*gx + gy*gy + gz*gz) or 1.0
        n = (gx/nl, gy/nl, gz/nl)
        triangles.append((n, v0, v1, v2))

    write_binary_stl(out_path, triangles)
    size_kb = out_path.stat().st_size / 1024
    print(f"  出力完了: {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
