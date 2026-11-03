"""NVIDIA Warp 側のベンチマーク (nebula_bench と同条件)。

nebula の MPM (mpm_p2g / mpm_grid_update / mpm_g2p.comp) と XPBD softbody
(predict / solve_stretch / sb_solve_volume / sdf_collision_velocity.comp) を Warp カーネルへ
ほぼそのまま移植している。Warp は wp.sim 等の既製ソルバを使わず、同一アルゴリズムを自前実装することで
「同じ計算を CUDA と Vulkan compute のどちらが速く回すか」を比較する。

stdout に `BENCH_JSON {...}` を engine ごとに 1 行出力する:
  warp        : 通常の wp.launch (Python からのカーネル起動)
  warp_graph  : 1 フレーム分を CUDA graph にキャプチャして再生 (起動オーバーヘッド除去)
"""

import argparse
import json
import struct
import time

import numpy as np
import warp as wp

import scenes as S

wp.config.quiet = True

# ── MLS-MPM ──────────────────────────────────────────────────────────────────


@wp.func
def bspline2(d: float):
    ad = wp.abs(d)
    if ad < 0.5:
        return 0.75 - ad * ad
    if ad < 1.5:
        t = 1.5 - ad
        return 0.5 * t * t
    return 0.0


@wp.func
def bspline2g(d: float):
    ad = wp.abs(d)
    if ad < 0.5:
        return -2.0 * d
    if ad < 1.5:
        return -wp.sign(d) * (1.5 - ad)
    return 0.0


@wp.kernel
def zero_grid(gm: wp.array(dtype=float), gv: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    gm[i] = 0.0
    gv[i] = wp.vec3(0.0)


@wp.kernel
def p2g(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), B: wp.array(dtype=wp.mat33),
        tau: wp.array(dtype=wp.mat33), Vp: float, rho0: float, apic: int, dt: float, h: float, res: int,
        gm: wp.array(dtype=float), gv: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    xp = x[p]
    mp = rho0 * Vp
    Bp = wp.mat33(0.0)
    if apic == 1:
        Bp = B[p]
    d_inv = 4.0 / (h * h)
    bx = int(wp.floor(xp[0] / h)) - 1
    by = int(wp.floor(xp[1] / h)) - 1
    bz = int(wp.floor(xp[2] / h)) - 1
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                nx = bx + ix
                ny = by + iy
                nz = bz + iz
                if nx < 0 or ny < 0 or nz < 0 or nx >= res or ny >= res or nz >= res:
                    continue
                xi = wp.vec3((float(nx) + 0.5) * h, (float(ny) + 0.5) * h, (float(nz) + 0.5) * h)
                d = (xp - xi) / h
                wx = bspline2(d[0])
                wy = bspline2(d[1])
                wz = bspline2(d[2])
                w = wx * wy * wz
                if w < 1.0e-12:
                    continue
                dw = wp.vec3(bspline2g(d[0]) * wy * wz, wx * bspline2g(d[1]) * wz, wx * wy * bspline2g(d[2])) / h
                v_eff = v[p] + d_inv * (Bp * (xi - xp))
                idx = (nx * res + ny) * res + nz
                wp.atomic_add(gm, idx, w * mp)
                wp.atomic_add(gv, idx, w * mp * v_eff - dt * Vp * (tau[p] * dw))


@wp.kernel
def grid_update(gm: wp.array(dtype=float), gv: wp.array(dtype=wp.vec3), gv_old: wp.array(dtype=wp.vec3),
                g: wp.vec3, dt: float, res: int):
    i = wp.tid()
    m = gm[i]
    if m < 1.0e-10:
        return
    v = gv[i] / m
    speed = wp.length(v)
    if speed > 15.0:  # nebula と同じグリッド速度キャップ [m/s]
        v = v * (15.0 / speed)
    gv_old[i] = v
    v = v + g * dt
    ix = i // (res * res)
    iy = (i // res) % res
    iz = i % res
    n = res - 1
    if ix <= 1 and v[0] < 0.0:
        v[0] = 0.0
    if ix >= n - 1 and v[0] > 0.0:
        v[0] = 0.0
    if iy <= 1 and v[1] < 0.0:
        v[1] = 0.0
    if iy >= n - 1 and v[1] > 0.0:
        v[1] = 0.0
    if iz <= 1 and v[2] < 0.0:
        v[2] = 0.0
    if iz >= n - 1 and v[2] > 0.0:
        v[2] = 0.0
    gv[i] = v


@wp.kernel
def g2p(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), F: wp.array(dtype=wp.mat33),
        B: wp.array(dtype=wp.mat33), tau: wp.array(dtype=wp.mat33), gm: wp.array(dtype=float),
        gv: wp.array(dtype=wp.vec3), gv_old: wp.array(dtype=wp.vec3), model: int, mu: float, lam: float,
        bulk_k: float, fric_m: float, q_c: float, flip_ratio: float, dt: float, h: float, res: int,
        world: float):
    p = wp.tid()
    xp = x[p]
    bx = int(wp.floor(xp[0] / h)) - 1
    by = int(wp.floor(xp[1] / h)) - 1
    bz = int(wp.floor(xp[2] / h)) - 1
    vpic = wp.vec3(0.0)
    vpic_old = wp.vec3(0.0)
    vgrad = wp.mat33(0.0)
    b_new = wp.mat33(0.0)
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                nx = bx + ix
                ny = by + iy
                nz = bz + iz
                if nx < 0 or ny < 0 or nz < 0 or nx >= res or ny >= res or nz >= res:
                    continue
                idx = (nx * res + ny) * res + nz
                if gm[idx] < 1.0e-10:
                    continue
                xi = wp.vec3((float(nx) + 0.5) * h, (float(ny) + 0.5) * h, (float(nz) + 0.5) * h)
                d = (xp - xi) / h
                wx = bspline2(d[0])
                wy = bspline2(d[1])
                wz = bspline2(d[2])
                w = wx * wy * wz
                if w < 1.0e-12:
                    continue
                dw = wp.vec3(bspline2g(d[0]) * wy * wz, wx * bspline2g(d[1]) * wz, wx * wy * bspline2g(d[2])) / h
                vi = gv[idx]
                vpic += w * vi
                vgrad += wp.outer(vi, dw)
                b_new += w * wp.outer(vi, xi - xp)
                if flip_ratio > 0.01:
                    vpic_old += w * gv_old[idx]

    f_new = (wp.identity(3, float) + dt * vgrad) * F[p]
    tau_new = wp.mat33(0.0)

    if model == 0:
        # 弱圧縮流体: det(F) のみ保存し F を等方にリセット
        J = wp.max(wp.determinant(f_new), 1.0e-4)
        f_new = wp.pow(J, 1.0 / 3.0) * wp.identity(3, float)
        tau_new = bulk_k * (J - 1.0) * wp.identity(3, float)
    else:
        # Drucker-Prager 射影 + Hencky 弾性
        U = wp.mat33(0.0)
        sig = wp.vec3(0.0)
        V = wp.mat33(0.0)
        wp.svd3(f_new, U, sig, V)
        sig = wp.vec3(wp.max(wp.abs(sig[0]), 1.0e-6), wp.max(wp.abs(sig[1]), 1.0e-6), wp.max(wp.abs(sig[2]), 1.0e-6))
        eps = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
        tr = eps[0] + eps[1] + eps[2]
        dev = eps - (tr / 3.0) * wp.vec3(1.0)
        nd = wp.length(dev)
        K = lam + 2.0 * mu / 3.0
        q_yield = fric_m * (-K * tr) + q_c
        if q_yield <= 1.0e-8:
            p_proj = -q_c / wp.max(fric_m, 1.0e-8)
            tr = -p_proj / K
            eps = (tr / 3.0) * wp.vec3(1.0)
        else:
            delta = nd - q_yield / (2.0 * mu)
            if delta > 0.0 and nd > 1.0e-8:
                dev = dev - (delta / nd) * dev
                eps = dev + (tr / 3.0) * wp.vec3(1.0)
        sig2 = wp.vec3(wp.exp(eps[0]), wp.exp(eps[1]), wp.exp(eps[2]))
        f_new = U * wp.diag(sig2) * wp.transpose(V)
        tr2 = eps[0] + eps[1] + eps[2]
        kp = lam * tr2 * wp.vec3(1.0) + 2.0 * mu * eps
        tau_new = U * wp.diag(kp) * wp.transpose(U)

    j_reset = bool(False)
    j_det = wp.abs(wp.determinant(f_new))
    if j_det < 0.15 or j_det > 10.0:
        f_new = wp.identity(3, float)
        tau_new = wp.mat33(0.0)
        j_reset = True

    if flip_ratio > -0.5:
        b_new = wp.mat33(0.0)

    v_new = vpic
    if flip_ratio > 0.01:
        v_flip = v[p] + (vpic - vpic_old)
        v_new = vpic + flip_ratio * (v_flip - vpic)
    if j_reset:
        v_new = v_new * 0.1

    x_new = xp + dt * v_new

    margin = h * 1.5
    lo = margin
    hi = world - margin
    for a in range(3):
        if x_new[a] < lo:
            x_new[a] = lo
            v_new[a] = wp.max(v_new[a], 0.0)
        if x_new[a] > hi:
            x_new[a] = hi
            v_new[a] = wp.min(v_new[a], 0.0)

    x[p] = x_new
    v[p] = v_new
    F[p] = f_new
    tau[p] = tau_new
    B[p] = b_new


class MPMScene:
    def __init__(self, name, device):
        s = S.SCENES[name]
        self.s, self.device = s, device
        nx, ny, nz = s["block_n"]
        ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        # nebula_bench.cpp と同じ並び (i が最速) ではなく集合として同一であれば良い
        pos = (np.stack([ii, jj, kk], -1).reshape(-1, 3) + 0.5) * S.SPACING + np.array(s["block_min"])
        self.n = len(pos)
        self.Vp = S.SPACING ** 3
        self.res = int(round(S.DOMAIN / S.CELL))
        m = s["material"]
        self.rho0 = m["rho0"]
        if m["model"] == "fluid":
            self.model, self.mu, self.lam, self.bulk = 0, 0.0, 0.0, m["bulk_k"]
            self.fric, self.qc = 0.0, 0.0
        else:
            E, nu = m["E"], m["nu"]
            self.model, self.bulk = 1, 0.0
            self.mu = E / (2.0 * (1.0 + nu))
            self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            self.fric, self.qc = m["M"], m["qc"]
        d = device
        self.x = wp.array(pos.astype(np.float32), dtype=wp.vec3, device=d)
        self.v = wp.zeros(self.n, dtype=wp.vec3, device=d)
        eye = np.tile(np.eye(3, dtype=np.float32), (self.n, 1, 1))
        self.F = wp.array(eye, dtype=wp.mat33, device=d)
        self.B = wp.zeros(self.n, dtype=wp.mat33, device=d)
        self.tau = wp.zeros(self.n, dtype=wp.mat33, device=d)
        nc = self.res ** 3
        self.gm = wp.zeros(nc, dtype=float, device=d)
        self.gv = wp.zeros(nc, dtype=wp.vec3, device=d)
        self.gv_old = wp.zeros(nc, dtype=wp.vec3, device=d)
        self.nc = nc

    def step(self):
        s, d = self.s, self.device
        sub_dt = S.DT / s["substeps"]
        apic = 1 if s["flip_ratio"] < -0.5 else 0
        g = wp.vec3(0.0, -S.GRAVITY, 0.0)
        for _ in range(s["substeps"]):
            wp.launch(zero_grid, self.nc, [self.gm, self.gv], device=d)
            wp.launch(p2g, self.n, [self.x, self.v, self.B, self.tau, self.Vp, self.rho0, apic, sub_dt, S.CELL,
                                    self.res, self.gm, self.gv], device=d)
            wp.launch(grid_update, self.nc, [self.gm, self.gv, self.gv_old, g, sub_dt, self.res], device=d)
            wp.launch(g2p, self.n, [self.x, self.v, self.F, self.B, self.tau, self.gm, self.gv, self.gv_old,
                                    self.model, self.mu, self.lam, self.bulk, self.fric, self.qc,
                                    s["flip_ratio"], sub_dt, S.CELL, self.res, S.DOMAIN], device=d)

    def centroid(self):
        return self.x.numpy().mean(axis=0)


# ── XPBD softbody ────────────────────────────────────────────────────────────


@wp.kernel
def sb_predict(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), pred: wp.array(dtype=wp.vec3),
               inv_m: wp.array(dtype=float), g: wp.vec3, max_speed: float, dt: float):
    i = wp.tid()
    if inv_m[i] == 0.0:
        pred[i] = x[i]
        return
    vi = v[i] + g * dt
    sp = wp.length(vi)
    if sp > max_speed:
        vi = vi * (max_speed / sp)
    v[i] = vi
    pred[i] = x[i] + vi * dt


@wp.kernel
def sb_solve_edge(pred: wp.array(dtype=wp.vec3), inv_m: wp.array(dtype=float), edges: wp.array(dtype=int),
                  rest: wp.array(dtype=float), lam: wp.array(dtype=float), start: int, alpha: float):
    e = start + wp.tid()
    p = edges[e * 2]
    q = edges[e * 2 + 1]
    wp_ = inv_m[p]
    wq = inv_m[q]
    d = pred[p] - pred[q]
    dist = wp.length(d)
    if dist < 1.0e-6 or (wp_ == 0.0 and wq == 0.0):
        return
    c = dist - rest[e]
    dl = (-c - alpha * lam[e]) / (wp_ + wq + alpha)
    lam[e] = lam[e] + dl
    n = d / dist
    if wp_ > 0.0:
        pred[p] = pred[p] + dl * wp_ * n
    if wq > 0.0:
        pred[q] = pred[q] - dl * wq * n


@wp.kernel
def sb_solve_volume(pred: wp.array(dtype=wp.vec3), inv_m: wp.array(dtype=float), tets: wp.array(dtype=int),
                    rest_vol: wp.array(dtype=float), lam: wp.array(dtype=float), start: int, alpha: float):
    t = start + wp.tid()
    i0 = tets[t * 4]
    i1 = tets[t * 4 + 1]
    i2 = tets[t * 4 + 2]
    i3 = tets[t * 4 + 3]
    w0 = inv_m[i0]
    w1 = inv_m[i1]
    w2 = inv_m[i2]
    w3 = inv_m[i3]
    if w0 == 0.0 and w1 == 0.0 and w2 == 0.0 and w3 == 0.0:
        return
    p0 = pred[i0]
    e1 = pred[i1] - p0
    e2 = pred[i2] - p0
    e3 = pred[i3] - p0
    c = wp.dot(e1, wp.cross(e2, e3)) / 6.0 - rest_vol[t]
    g1 = wp.cross(e2, e3) / 6.0
    g2 = wp.cross(e3, e1) / 6.0
    g3 = wp.cross(e1, e2) / 6.0
    g0 = -(g1 + g2 + g3)
    denom = w0 * wp.dot(g0, g0) + w1 * wp.dot(g1, g1) + w2 * wp.dot(g2, g2) + w3 * wp.dot(g3, g3) + alpha
    if denom < 1.0e-12:
        return
    dl = (-c - alpha * lam[t]) / denom
    lam[t] = lam[t] + dl
    if w0 > 0.0:
        pred[i0] = pred[i0] + dl * w0 * g0
    if w1 > 0.0:
        pred[i1] = pred[i1] + dl * w1 * g1
    if w2 > 0.0:
        pred[i2] = pred[i2] + dl * w2 * g2
    if w3 > 0.0:
        pred[i3] = pred[i3] + dl * w3 * g3


@wp.kernel
def sb_particle_collision(pred: wp.array(dtype=wp.vec3), inv_m: wp.array(dtype=float), n: int, r: float):
    # nebula の sb_particle_collision.comp と同じ O(N^2) ループ (r=0 では補正量は常に 0 だがループ自体は回る)
    i = wp.tid()
    wi = inv_m[i]
    if wi <= 0.0:
        return
    pi = pred[i]
    r2 = r * r
    corr = wp.vec3(0.0)
    for j in range(n):
        if j == i:
            continue
        wj = inv_m[j]
        if wj <= 0.0 and wi <= 0.0:
            continue
        d = pi - pred[j]
        dist2 = wp.dot(d, d)
        if dist2 >= r2 or dist2 < 1.0e-10:
            continue
        dist = wp.sqrt(dist2)
        denom = wi + wj
        if denom < 1.0e-10:
            continue
        corr -= (wi / denom) * (dist - r) * (d / dist)
    pred[i] = pi + corr


@wp.kernel
def sb_update_velocity(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), pred: wp.array(dtype=wp.vec3),
                       lo: float, hi: float, damping: float, dt: float):
    i = wp.tid()
    p = pred[i]
    p = wp.vec3(wp.clamp(p[0], lo, hi), wp.clamp(p[1], lo, hi), wp.clamp(p[2], lo, hi))
    vi = (p - x[i]) / dt
    vi = vi * wp.exp(-damping * dt)
    x[i] = p
    v[i] = vi


def load_sb(path):
    """tools/gen_softbody.py の .sb フォーマットを読む。"""
    with open(path, "rb") as f:
        buf = f.read()
    assert buf[:4] == b"SFTB", "bad .sb magic"
    n, ne, nt, nec, ntc = struct.unpack_from("<5I", buf, 4)
    off = 24

    def take(dtype, count):
        nonlocal off
        a = np.frombuffer(buf, dtype=dtype, count=count, offset=off)
        off += a.nbytes
        return a

    pos = take("<f4", n * 3).reshape(n, 3)
    inv_m = take("<f4", n)
    ecb = take("<u4", nec + 1)
    ed = take("<u4", ne * 3).reshape(ne, 3)
    tcb = take("<u4", ntc + 1)
    td = take("<u4", nt * 4).reshape(nt, 4)
    rv = take("<f4", nt)
    return dict(pos=pos, inv_m=inv_m, ecb=ecb.astype(int), edges=ed[:, :2].astype(np.int32),
                rest=ed[:, 2].view(np.float32), tcb=tcb.astype(int), tets=td.astype(np.int32), rest_vol=rv)


class SoftBodyScene:
    def __init__(self, name, device, sb_path):
        s = S.SCENES[name]
        self.s, self.device = s, device
        m = load_sb(sb_path)
        self.m = m
        d = device
        self.n = len(m["pos"])
        pos = m["pos"] + np.array(s["offset"], dtype=np.float32)
        self.x = wp.array(pos, dtype=wp.vec3, device=d)
        self.v = wp.zeros(self.n, dtype=wp.vec3, device=d)
        self.pred = wp.zeros(self.n, dtype=wp.vec3, device=d)
        self.inv_m = wp.array(m["inv_m"], dtype=float, device=d)
        self.edges = wp.array(m["edges"].reshape(-1), dtype=int, device=d)
        self.rest = wp.array(m["rest"], dtype=float, device=d)
        self.tets = wp.array(m["tets"].reshape(-1), dtype=int, device=d)
        self.rest_vol = wp.array(m["rest_vol"], dtype=float, device=d)
        self.e_lam = wp.zeros(len(m["edges"]), dtype=float, device=d)
        self.t_lam = wp.zeros(len(m["tets"]), dtype=float, device=d)

    def step(self):
        s, d, m = self.s, self.device, self.m
        sub_dt = S.DT / s["substeps"]
        g = wp.vec3(0.0, -S.GRAVITY, 0.0)
        a_edge = s["stretch_compliance"] / (sub_dt * sub_dt)
        a_vol = s["vol_compliance"] / (sub_dt * sub_dt)
        max_speed = S.CELL * 0.4 / sub_dt
        r = s["particle_radius"]
        for _ in range(s["substeps"]):
            wp.launch(sb_predict, self.n, [self.x, self.v, self.pred, self.inv_m, g, max_speed, sub_dt], device=d)
            self.e_lam.zero_()
            self.t_lam.zero_()
            for _ in range(s["iters"]):
                for c in range(len(m["ecb"]) - 1):
                    a, b = int(m["ecb"][c]), int(m["ecb"][c + 1])
                    if b > a:
                        wp.launch(sb_solve_edge, b - a, [self.pred, self.inv_m, self.edges, self.rest, self.e_lam, a, a_edge], device=d)
                for c in range(len(m["tcb"]) - 1):
                    a, b = int(m["tcb"][c]), int(m["tcb"][c + 1])
                    if b > a:
                        wp.launch(sb_solve_volume, b - a, [self.pred, self.inv_m, self.tets, self.rest_vol, self.t_lam, a, a_vol], device=d)
            wp.launch(sb_particle_collision, self.n, [self.pred, self.inv_m, self.n, 0.0], device=d)
            wp.launch(sb_update_velocity, self.n, [self.x, self.v, self.pred, r, S.DOMAIN - r, s["linear_damping"], sub_dt], device=d)

    def centroid(self):
        return self.x.numpy().mean(axis=0)


# ── 計測 ─────────────────────────────────────────────────────────────────────


def make_scene(name, device, sb_path):
    if S.SCENES[name]["kind"] == "mpm":
        return MPMScene(name, device)
    return SoftBodyScene(name, device, sb_path)


def run(name, device, sb_path, warmup, frames, graph):
    sc = make_scene(name, device, sb_path)
    if graph:
        with wp.ScopedCapture(device=device) as cap:
            sc.step()
        fn = lambda: wp.capture_launch(cap.graph)  # noqa: E731
    else:
        fn = sc.step
    for _ in range(warmup):
        fn()
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(frames):
        fn()
    wp.synchronize_device(device)
    ms = (time.perf_counter() - t0) * 1000.0 / frames
    return dict(engine="warp_graph" if graph else "warp", scene=name, n=sc.n, frames=frames,
                ms_per_frame=ms, centroid=[float(c) for c in sc.centroid()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=list(S.SCENES))
    ap.add_argument("--sb-dir", default=".", help=".sb メッシュのあるディレクトリ")
    ap.add_argument("--warmup", type=int, default=S.WARMUP)
    ap.add_argument("--frames", type=int, default=S.FRAMES)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    sb_path = f"{a.sb_dir}/{S.SB_FILENAME}"
    # カーネルコンパイルを計測外に出すため、先に 1 フレーム回しておく
    make_scene(a.scene, a.device, sb_path).step()
    wp.synchronize_device(a.device)
    for graph in (False, True):
        print("BENCH_JSON " + json.dumps(run(a.scene, a.device, sb_path, a.warmup, a.frames, graph)), flush=True)


if __name__ == "__main__":
    main()
