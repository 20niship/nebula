"""比較シーン定義 (nebula / Warp 共通の単一ソース)。

両実装とも同じ初期粒子配置・フレーム dt・粒子数・グリッドセル・重力で動かす。
nebula 側へは nebula_args() で CLI 引数として渡し、Warp 側 (NVIDIA Newton) は warp_bench.py が直接参照する。

ソルバの種類が違う (nebula: 陽的 MLS-MPM / XPBD、Newton: implicit MPM / XPBD) ため、
substeps や材料モデルのパラメータはソルバごとに適切な値を持つ ("newton" キー)。
"""

import copy
import math
from pathlib import Path

DOMAIN = 10.0  # 立方体ドメイン [m] (原点固定)
CELL = DOMAIN / 64.0  # グリッドセル [m]
DT = 1.0 / 60.0  # フレーム dt [s]
GRAVITY = 9.8  # -Y 方向 [m/s^2]
WARMUP = 30  # 計測しないウォームアップフレーム数 (GPU クロック上昇待ちも兼ねる)
FRAMES = 100  # 計測フレーム数

# nebula の MPM は粒子を壁から 1.5 セル内側に閉じ込める (mpm_g2p.comp の margin)。Newton 側の壁もこの位置に置く
WALL_MARGIN = 1.5 * CELL

# MLS-MPM の粒子間隔 = セルの半分 (セルあたり 8 粒子)
SPACING = DOMAIN / 128.0  # セルを変えても粒子間隔は固定 (グリッドスイープで粒子数を一定に保つため)

# XPBD softbody 用 .sb メッシュ (tools/gen_softbody.py --cube で生成する。リポジトリには含めない)
SB_GRID = 16  # 16^3 = 4096 頂点
SB_SIZE = 3.0  # 立方体の一辺 [m]
SB_FILENAME = f"cube_{SB_GRID}.sb"

PARTICLE_SCALE = 1.0  # MPM 粒子数の倍率 (configure() で変更)

SCENES = {
    # 流体: 角に置いた水柱 (dam break)。MPM + FLIP(0.95)、弱圧縮 EOS: τ = K (J - 1) I
    "fluid": dict(
        kind="mpm",
        title="流体 (MPM FLIP, water)",
        title_en="Fluid (MPM, water dam break)",
        substeps=20,
        flip_ratio=0.95,
        # presetWater の K=5e3 は 3 m の水柱の静水圧 (~29 kPa) より小さく、水柱が崩れず縦に潰れるため大きくする
        material=dict(model="fluid", rho0=1000.0, bulk_k=2e5),
        block_min=(0.5, 0.5, 0.5),
        block_n=(38, 38, 38),
        newton=dict(substeps=2, density=1000.0, friction=0.0, tensile_yield_ratio=1.0, viscosity=0.0,
                    max_iterations=50, tolerance=1e-3),
    ),
    # 粉体: 砂柱の崩壊。MPM + APIC、Drucker-Prager (Hencky 弾性 + SVD 射影)
    "sand": dict(
        kind="mpm",
        title="粉体 (MPM APIC, sand)",
        title_en="Granular (MPM, sand column)",
        substeps=20,
        flip_ratio=-1.0,
        material=dict(model="dp", rho0=1600.0, E=1e5, nu=0.3, M=0.577, qc=0.0),
        block_min=(3.8, 0.5, 3.8),
        block_n=(30, 50, 30),
        # 砂は Newton の granular example 既定 (剛に近い弾性 + Drucker-Prager 降伏) に摩擦係数 tan(30°) だけ合わせる
        newton=dict(substeps=2, density=1600.0, friction=0.577, max_iterations=250, tolerance=1e-4),
    ),
    # Softbody: 四面体ジェリーキューブの落下。XPBD (edge 距離 + tet 体積拘束、グラフ彩色 Gauss-Seidel)
    "softbody": dict(
        kind="xpbd",
        title="Softbody (XPBD tet cube)",
        title_en="Soft body (XPBD, tet cube)",
        substeps=15,
        iters=5,
        offset=(3.5, 5.0, 3.5),
        stretch_compliance=1e-5,
        vol_compliance=5e-4,
        linear_damping=0.03,
        particle_radius=0.02,
        # 同じ 16^3 頂点の立方体を add_soft_grid で作る (tet 分割は nebula と一致しない)。
        # 密度は nebula の 1 頂点 = 1 kg (4096 kg / 27 m^3 ≒ 150) に合わせる
        newton=dict(substeps=15, iters=5, density=150.0, k_mu=1e5, k_lambda=1e5, k_damp=0.0),
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
        if s["material"]["model"] == "fluid":
            a += ["--bulk-k", repr(s["material"]["bulk_k"])]
    else:
        a += ["--sb", str(Path(sb_dir) / SB_FILENAME), "--sb-offset", _v3(s["offset"]),
              "--iters", str(s["iters"])]
    return a


_BASE = copy.deepcopy(SCENES)


def configure(cell=None, particle_scale=None, sb_grid=None):
    """スイープ用にモジュール定数を書き換える (呼び出し元プロセスのみ有効)。

    cell: グリッドセル [m] (粒子間隔は固定なので粒子数は変わらない)
    particle_scale: MPM 粒子数の倍率 (各軸 scale^(1/3) 倍)
    sb_grid: softbody 立方体の一辺の頂点数
    """
    global CELL, WALL_MARGIN, SB_GRID, SB_FILENAME, PARTICLE_SCALE
    if cell is not None:
        CELL = cell
        WALL_MARGIN = 1.5 * CELL
    if particle_scale is not None:
        PARTICLE_SCALE = particle_scale
        f = particle_scale ** (1.0 / 3.0)
        for name, s in SCENES.items():
            if s["kind"] == "mpm":
                s["block_n"] = tuple(max(1, round(n * f)) for n in _BASE[name]["block_n"])
    if sb_grid is not None:
        SB_GRID = sb_grid
        SB_FILENAME = f"cube_{SB_GRID}.sb"


def particle_count(name):
    s = SCENES[name]
    return math.prod(s["block_n"]) if s["kind"] == "mpm" else SB_GRID ** 3
