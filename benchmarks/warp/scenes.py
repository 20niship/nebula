"""比較シーン定義 (nebula / Warp 共通の単一ソース)。

両実装とも同じ初期粒子配置・dt・substeps・材料パラメータ・重力 (-Y) で動かす。
nebula 側へは nebula_args() で CLI 引数として渡し、Warp 側は warp_bench.py が直接参照する。
"""

import math
from pathlib import Path

DOMAIN = 10.0  # 立方体ドメイン [m] (原点固定)
CELL = DOMAIN / 64.0  # グリッドセル [m]
DT = 1.0 / 60.0  # フレーム dt [s]
GRAVITY = 9.8  # -Y 方向 [m/s^2]
WARMUP = 30  # 計測しないウォームアップフレーム数 (GPU クロック上昇待ちも兼ねる)
FRAMES = 100  # 計測フレーム数

# MLS-MPM の粒子間隔 = セルの半分 (セルあたり 8 粒子)
SPACING = CELL / 2.0

# XPBD softbody 用 .sb メッシュ (tools/gen_softbody.py --cube で生成する。リポジトリには含めない)
SB_GRID = 16  # 16^3 = 4096 頂点
SB_SIZE = 3.0  # 立方体の一辺 [m]
SB_FILENAME = f"cube_{SB_GRID}.sb"

SCENES = {
    # 流体: 角に置いた水柱 (dam break)。MPM + FLIP(0.95)、弱圧縮 EOS: τ = K (J - 1) I
    "fluid": dict(
        kind="mpm",
        title="流体 (MPM FLIP, water)",
        substeps=20,
        flip_ratio=0.95,
        material=dict(model="fluid", rho0=1000.0, bulk_k=5e3),
        block_min=(0.5, 0.5, 0.5),
        block_n=(38, 38, 38),
    ),
    # 粉体: 砂柱の崩壊。MPM + APIC、Drucker-Prager (Hencky 弾性 + SVD 射影)
    "sand": dict(
        kind="mpm",
        title="粉体 (MPM APIC, sand)",
        substeps=20,
        flip_ratio=-1.0,
        material=dict(model="dp", rho0=1600.0, E=1e5, nu=0.3, M=0.577, qc=0.0),
        block_min=(3.8, 0.5, 3.8),
        block_n=(30, 50, 30),
    ),
    # Softbody: 四面体ジェリーキューブの落下。XPBD (edge 距離 + tet 体積拘束、グラフ彩色 Gauss-Seidel)
    "softbody": dict(
        kind="xpbd",
        title="Softbody (XPBD tet cube)",
        substeps=15,
        iters=5,
        offset=(3.5, 5.0, 3.5),
        stretch_compliance=1e-5,
        vol_compliance=5e-4,
        linear_damping=0.03,
        particle_radius=0.02,
    ),
}


def mpm_particle_count(s):
    return math.prod(s["block_n"])


def _v3(t):
    return ",".join(repr(float(x)) for x in t)


def nebula_args(name, sb_dir, warmup=WARMUP, frames=FRAMES):
    """nebula_bench に渡す CLI 引数。"""
    s = SCENES[name]
    a = ["--scene", name, "--dt", repr(DT), "--substeps", str(s["substeps"]),
         "--warmup", str(warmup), "--frames", str(frames),
         "--domain", repr(DOMAIN), "--cell-size", repr(CELL), "--gravity", repr(GRAVITY)]
    if s["kind"] == "mpm":
        a += ["--block-min", _v3(s["block_min"]), "--block-n", _v3(s["block_n"]),
              "--spacing", repr(SPACING), "--flip-ratio", repr(s["flip_ratio"])]
    else:
        a += ["--sb", str(Path(sb_dir) / SB_FILENAME), "--sb-offset", _v3(s["offset"]),
              "--iters", str(s["iters"])]
    return a
