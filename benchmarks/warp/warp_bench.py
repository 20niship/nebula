"""NVIDIA Warp 側のベンチマーク: Warp 製の物理エンジン Newton のソルバで nebula と同等のシーンを組む。

  流体 / 粉体 : newton.solvers.SolverImplicitMPM (implicit MPM)
  Softbody    : newton.solvers.SolverXPBD (四面体 FEM 拘束)

nebula と同じフレーム dt・粒子数・グリッドセル・初期配置・重力にそろえる。ただしソルバのアルゴリズムが違う
(nebula は陽的 MLS-MPM / 辺+体積拘束 XPBD) ため substeps 等は scenes.py の "newton" キーで別に持つ。
Newton は Z-up なので nebula の (x, y, z) を (x, z, y) に入れ替え、壁は nebula の粒子クランプ位置
(WALL_MARGIN) に合わせて原点をずらしている。

stdout に `BENCH_JSON {...}` を engine ごとに 1 行出力する:
  warp        : 通常の Python からのソルバ呼び出し
  warp_graph  : 1 フレーム分を CUDA graph にキャプチャして再生 (キャプチャできるシーンのみ)
"""

import argparse
import json
import time

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM, SolverXPBD

import scenes as S

wp.config.log_level = wp.LOG_WARNING

M = S.WALL_MARGIN
L = S.DOMAIN - 2.0 * M  # 壁の内側の一辺


def to_newton(p):
    """nebula 座標 (Y-up) → Newton 座標 (Z-up, 壁内側が原点)。"""
    x, y, z = p
    return (x - M, z - M, y - M)


def to_nebula(q):
    """Newton 座標の (N, 3) 配列 → nebula 座標系に戻した (N, 3)。"""
    return np.stack([q[:, 0] + M, q[:, 2] + M, q[:, 1] + M], axis=1)


def add_tank(b):
    """床 (z=0) と 4 枚の側壁。内側の面が [0, L] に来る。"""
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, density=0.0)
    t = 1.0  # 壁の厚み (半分)
    b.add_ground_plane(cfg=cfg)
    h = L / 2.0 + t
    for sx, sy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        c = wp.vec3(L / 2.0 + sx * (L / 2.0 + t), L / 2.0 + sy * (L / 2.0 + t), L / 2.0)
        b.add_shape_box(body=-1, xform=wp.transform(c, wp.quat_identity()),
                        hx=t if sx else h, hy=t if sy else h, hz=L, cfg=cfg)


# ── MPM (流体 / 粉体) ────────────────────────────────────────────────────────


class MPMScene:
    def __init__(self, name, device):
        s = S.SCENES[name]
        nw = s["newton"]
        nx, ny, nz = s["block_n"]
        sp = S.SPACING
        b = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(b)
        lo = to_newton(s["block_min"])
        b.add_particle_grid(
            pos=wp.vec3(lo[0] + sp / 2, lo[1] + sp / 2, lo[2] + sp / 2), rot=wp.quat_identity(), vel=wp.vec3(0.0),
            dim_x=nx, dim_y=nz, dim_z=ny,  # nebula の y (高さ) が Newton の z
            cell_x=sp, cell_y=sp, cell_z=sp, mass=nw["density"] * sp ** 3, jitter=0.0, radius_mean=sp / 2,
            flags=newton.ParticleFlags.ACTIVE)
        add_tank(b)
        self.model = b.finalize(device=device)
        self.model.set_gravity((0.0, 0.0, -S.GRAVITY))
        mpm = self.model.mpm
        mpm.friction.fill_(nw["friction"])
        if "tensile_yield_ratio" in nw:
            mpm.tensile_yield_ratio.fill_(nw["tensile_yield_ratio"])
            mpm.viscosity.fill_(nw["viscosity"])
        cfg = SolverImplicitMPM.Config()
        cfg.voxel_size = S.CELL
        cfg.max_iterations = nw["max_iterations"]
        cfg.tolerance = nw["tolerance"]
        self.solver = SolverImplicitMPM(self.model, config=cfg)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.state_0.mpm.particle_Jp.fill_(1.0)
        self.n = self.model.particle_count
        self.substeps = nw["substeps"]
        self.dt = S.DT / self.substeps
        self.capturable = self.solver.grid_type == "fixed"

    def step(self):
        for _ in range(self.substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.dt)
            self.solver.project_outside(self.state_1, self.state_1, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def centroid(self):
        return to_nebula(self.state_0.particle_q.numpy()).mean(axis=0)


# ── XPBD softbody ────────────────────────────────────────────────────────────


class SoftBodyScene:
    def __init__(self, name, device):
        s = S.SCENES[name]
        nw = s["newton"]
        n_cells = S.SB_GRID - 1
        cell = S.SB_SIZE / n_cells
        b = newton.ModelBuilder()
        b.add_ground_plane()
        o = to_newton(s["offset"])
        b.add_soft_grid(pos=wp.vec3(*o), rot=wp.quat_identity(), vel=wp.vec3(0.0),
                        dim_x=n_cells, dim_y=n_cells, dim_z=n_cells, cell_x=cell, cell_y=cell, cell_z=cell,
                        density=nw["density"], k_mu=nw["k_mu"], k_lambda=nw["k_lambda"], k_damp=nw["k_damp"],
                        particle_radius=s["particle_radius"])
        self.model = b.finalize(device=device)
        self.model.set_gravity((0.0, 0.0, -S.GRAVITY))
        self.solver = SolverXPBD(self.model, iterations=nw["iters"])
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.n = self.model.particle_count
        self.substeps = nw["substeps"]
        self.dt = S.DT / self.substeps
        self.capturable = True

    def step(self):
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def centroid(self):
        return to_nebula(self.state_0.particle_q.numpy()).mean(axis=0)


# ── 計測 ─────────────────────────────────────────────────────────────────────


def make_scene(name, device):
    return MPMScene(name, device) if S.SCENES[name]["kind"] == "mpm" else SoftBodyScene(name, device)


def run(name, device, warmup, frames, graph):
    with wp.ScopedDevice(device):
        sc = make_scene(name, device)
        if graph and not sc.capturable:
            return None
        if graph:
            sc.step()  # モジュールロード (キャプチャ前に必要)
            with wp.ScopedCapture(device=device) as cap:
                sc.step()
            fn = lambda: wp.capture_launch(cap.graph)  # noqa: E731
            done = 2
        else:
            fn, done = sc.step, 0
        for _ in range(warmup - done):
            fn()
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(frames):
            fn()
        wp.synchronize_device(device)
        ms = (time.perf_counter() - t0) * 1000.0 / frames
        return dict(engine="warp_graph" if graph else "warp", scene=name, n=sc.n, frames=frames,
                    substeps=sc.substeps, ms_per_frame=ms, centroid=[float(c) for c in sc.centroid()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=list(S.SCENES))
    ap.add_argument("--warmup", type=int, default=S.WARMUP)
    ap.add_argument("--frames", type=int, default=S.FRAMES)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    for graph in (False, True):
        r = run(a.scene, a.device, a.warmup, a.frames, graph)
        if r:
            print("BENCH_JSON " + json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
