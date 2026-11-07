#!/usr/bin/env python3
"""examples が使うアセット (sphere.obj / sphere_obstacle.stl / cow_obstacle.stl / cube_sb.sb) を生成する。
リポジトリにはバイナリやメッシュを置かず、CMake の configure 時にビルドディレクトリへ出力する。
使い方: python tools/gen_assets.py <出力ディレクトリ>"""
import math
import subprocess
import sys
from pathlib import Path

T = Path(__file__).parent


def uv_sphere_obj(path: Path, seg=32, rings=16):
    """半径 1 の UV 球 (Blender 既定相当: 482 頂点 / 512 面)。"""
    v = [(0.0, 1.0, 0.0)]
    for r in range(1, rings):
        t = math.pi * r / rings
        for s in range(seg):
            p = 2 * math.pi * s / seg
            v.append((math.sin(t) * math.sin(p), math.cos(t), -math.sin(t) * math.cos(p)))
    v.append((0.0, -1.0, 0.0))
    f = [(1, 2 + (s + 1) % seg, 2 + s) for s in range(seg)]
    for r in range(rings - 2):
        a, b = 2 + r * seg, 2 + (r + 1) * seg
        f += [(a + s, a + (s + 1) % seg, b + (s + 1) % seg, b + s) for s in range(seg)]
    last = len(v)
    a = 2 + (rings - 2) * seg
    f += [(last, a + s, a + (s + 1) % seg) for s in range(seg)]
    path.write_text("o Sphere\n" + "".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in v) + "".join("f " + " ".join(map(str, i)) + "\n" for i in f))


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    uv_sphere_obj(out / "sphere.obj")
    for tool, args in (("gen_sphere_stl.py", ["--out", out / "sphere_obstacle.stl"]),
                       ("gen_cow_stl.py", ["--out", out / "cow_obstacle.stl"]),
                       ("gen_softbody.py", ["--cube", "--grid", "5", "--size", "1.5", "-o", out / "cube_sb.sb"])):
        subprocess.run([sys.executable, str(T / tool), *map(str, args)], check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
