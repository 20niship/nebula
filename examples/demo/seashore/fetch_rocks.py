# /// script
# requires-python = ">=3.10"
# dependencies = ["usd-core", "numpy"]
# ///
"""海岸デモ用の岩メッシュを virtual-tokyo PR #6 (terra の超高ポリ岩 _hi) から取得し、npz (verts/faces, Y-up, m) に変換する。
アセットは git 管理外 (assets_local/seashore/rocks/)。gh CLI のログインが必要 (リポジトリが private のため)。

  uv run seashore/fetch_rocks.py            # 全 12 種
  uv run seashore/fetch_rocks.py --only seashore
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

REPO = "20niship/virtual-tokyo"
SHA = "c218854292a5465ca4c72accfb50a81513e8d6d4"  # PR #6 の head (ブランチは削除済みのため sha 固定)
NAMES = [f"rock_{k}_{f}_hi" for k in ("seashore", "granite", "andesite") for f in ("boulder_l_0", "boulder_l_2", "boulder_l_4", "outcrop_2")]
OUT = Path(__file__).resolve().parent.parent / "assets_local/seashore/rocks"


def usd_to_mesh(path):
    st = Usd.Stage.Open(str(path))
    up, mpu = UsdGeom.GetStageUpAxis(st), UsdGeom.GetStageMetersPerUnit(st) or 1.0
    cache = UsdGeom.XformCache()
    vs, fs, off = [], [], 0
    for prim in st.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        p = np.array(m.GetPointsAttr().Get(), np.float64)
        cnt = np.array(m.GetFaceVertexCountsAttr().Get())
        idx = np.array(m.GetFaceVertexIndicesAttr().Get())
        M = np.array(cache.GetLocalToWorldTransform(prim))  # 行ベクトル規約
        p = (np.c_[p, np.ones(len(p))] @ M)[:, :3]
        start = np.cumsum(cnt) - cnt  # 多角形を扇状に三角形化
        tri = [np.stack([idx[start[cnt == c]], idx[start[cnt == c] + k], idx[start[cnt == c] + k + 1]], 1) for c in np.unique(cnt) for k in range(1, c - 1)]
        vs.append(p)
        fs.append(np.concatenate(tri) + off)
        off += len(p)
    v = np.concatenate(vs) * mpu
    if up == "Z":
        v = v[:, [0, 2, 1]] * np.array([1, 1, -1])
    return v.astype(np.float32), np.concatenate(fs).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="名前に含まれる文字列で絞る")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for n in (n for n in NAMES if a.only in n):
        usdc, npz = OUT / f"{n}.usdc", OUT / f"{n}.npz"
        if not usdc.exists():
            data = subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github.raw", f"repos/{REPO}/contents/assets/terra/modules/{n}.usdc?ref={SHA}"], check=True, capture_output=True).stdout
            usdc.write_bytes(data)
        if not npz.exists():
            v, f = usd_to_mesh(usdc)
            np.savez(npz, verts=v, faces=f)
            print(f"{n}: {len(v)} verts {len(f)} tris size {np.ptp(v, 0).round(2)}")


if __name__ == "__main__":
    main()
