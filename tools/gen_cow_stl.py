#!/usr/bin/env python3
"""
低ポリ「牛」STL 生成スクリプト (ブロック体型: 胴体+頭+脚4本+角2本+尻尾)

pyro_cow_blast の障害物として使用する。三角形数を抑えるため各パーツは
軸平行ボックス (12三角形/箱) の組み合わせで構成する (buildMeshSDF は
O(gridRes³ × 三角形数) のブルートフォースのため低ポリが必須)。

使い方:
  python3 tools/gen_cow_stl.py [--cx X] [--cy Y] [--cz Z] \
                                 [--scale S] [--out PATH]
  (cx,cy,cz) は牛が地面に立つ足元中心のワールド座標。
"""

import argparse
import struct
from pathlib import Path


def box_triangles(center, half_extents):
    """軸平行ボックス→12三角形 (外向き法線, CCW winding)"""
    cx, cy, cz = center
    hx, hy, hz = half_extents

    # 8頂点
    def v(sx, sy, sz):
        return (cx + sx * hx, cy + sy * hy, cz + sz * hz)

    tris = []
    # +X面
    tris += [((1, 0, 0), v(1, -1, -1), v(1, 1, -1), v(1, 1, 1)),
             ((1, 0, 0), v(1, -1, -1), v(1, 1, 1), v(1, -1, 1))]
    # -X面
    tris += [((-1, 0, 0), v(-1, -1, 1), v(-1, 1, 1), v(-1, 1, -1)),
             ((-1, 0, 0), v(-1, -1, 1), v(-1, 1, -1), v(-1, -1, -1))]
    # +Y面
    tris += [((0, 1, 0), v(-1, 1, -1), v(-1, 1, 1), v(1, 1, 1)),
             ((0, 1, 0), v(-1, 1, -1), v(1, 1, 1), v(1, 1, -1))]
    # -Y面
    tris += [((0, -1, 0), v(-1, -1, 1), v(-1, -1, -1), v(1, -1, -1)),
             ((0, -1, 0), v(-1, -1, 1), v(1, -1, -1), v(1, -1, 1))]
    # +Z面
    tris += [((0, 0, 1), v(1, -1, 1), v(1, 1, 1), v(-1, 1, 1)),
             ((0, 0, 1), v(1, -1, 1), v(-1, 1, 1), v(-1, -1, 1))]
    # -Z面
    tris += [((0, 0, -1), v(-1, -1, -1), v(-1, 1, -1), v(1, 1, -1)),
             ((0, 0, -1), v(-1, -1, -1), v(1, 1, -1), v(1, -1, -1))]
    return tris


def build_cow(base, scale):
    """base = 足元中心 (ワールド座標)。牛は +X 方向を向く。"""
    bx, by, bz = base
    s = scale

    leg_h  = 0.5 * s
    leg_he = (0.08 * s, 0.25 * s, 0.08 * s)  # half extents
    body_center_y  = by + leg_h + 0.28 * s
    body_he = (0.55 * s, 0.28 * s, 0.28 * s)
    head_he = (0.18 * s, 0.16 * s, 0.15 * s)
    horn_he = (0.03 * s, 0.08 * s, 0.03 * s)
    tail_he = (0.05 * s, 0.05 * s, 0.28 * s)

    tris = []

    # 胴体
    tris += box_triangles((bx, body_center_y, bz), body_he)

    # 脚4本 (前後左右)
    leg_x_off = body_he[0] * 0.6
    leg_z_off = body_he[2] * 0.7
    for sx in (-1, 1):
        for sz in (-1, 1):
            leg_center = (bx + sx * leg_x_off, by + leg_h * 0.5, bz + sz * leg_z_off)
            tris += box_triangles(leg_center, leg_he)

    # 頭 (胴体前方・やや高め)
    head_center = (bx + body_he[0] + head_he[0] * 0.8, body_center_y + 0.05 * s, bz)
    tris += box_triangles(head_center, head_he)

    # 角2本 (頭の上)
    for sz in (-1, 1):
        horn_center = (head_center[0] - head_he[0] * 0.3,
                       head_center[1] + head_he[1] + horn_he[1] * 0.8,
                       bz + sz * head_he[2] * 0.6)
        tris += box_triangles(horn_center, horn_he)

    # 尻尾 (胴体後方)
    tail_center = (bx - body_he[0] - tail_he[0] * 0.5, body_center_y + 0.1 * s, bz)
    tris += box_triangles(tail_center, tail_he)

    return tris


def write_binary_stl(path: Path, triangles):
    header = (b"Low-poly cow obstacle STL" + b" " * 80)[:80]
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

    parser = argparse.ArgumentParser(description="低ポリ牛 STL 生成 (ブロック体型)")
    parser.add_argument("--cx", type=float, default=5.0, help="足元中心 X [m]")
    parser.add_argument("--cy", type=float, default=0.0, help="足元中心 Y [m] (地面の高さ)")
    parser.add_argument("--cz", type=float, default=5.0, help="足元中心 Z [m]")
    parser.add_argument("--scale", type=float, default=2.5, help="全体スケール")
    parser.add_argument("--out", type=str,
                        default=str(root / "assets" / "cow_obstacle.stl"),
                        help="出力 STL ファイルパス")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"牛 STL 生成: base=({args.cx},{args.cy},{args.cz}) scale={args.scale}")
    tris = build_cow((args.cx, args.cy, args.cz), args.scale)
    print(f"  三角形数: {len(tris)}")

    write_binary_stl(out_path, tris)
    size_kb = out_path.stat().st_size / 1024
    print(f"  出力完了: {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
