#!/usr/bin/env python3
"""
XPBD (現行 SoftBodyEngine の edge-distance 拘束) vs VBD (Vertex Block Descent) の
収束速度を比較する小規模な数値プロトタイプ。

シナリオ: 5粒子のチェーン (粒子0を固定端としてピン留め)、4本のエッジ距離拘束。
初期状態を静止長から引き伸ばした状態にし、1タイムステップぶんの反復回数を揃えて
どちらがより速く (少ない反復で) 拘束残差を減らせるかを比較する。

XPBD 側は SoftBodyEngine (src/engine/SoftBodyEngine.cpp) の
shaders/solve_stretch.comp と同じ Δλ = -(C + α̃λ)/(w_sum + α̃) 更新則を、
グラフ彩色バッチ (色内 Jacobi、色間 Gauss-Seidel) で再現する。

VBD 側は Chen et al. 2024 "Vertex Block Descent" の変分定式化に従い、
各頂点で (慣性項 + 接続エッジのバネエネルギー) の合計エネルギーに対する
Newton ステップ (勾配 g, ヘシアン H, Δp = -H^-1 g) を頂点ごとに解く。
"""

import numpy as np

# ── シナリオ設定 ────────────────────────────────────────────────────────────
N = 5
rest_len = 1.0
dt = 1.0 / 60.0
mass = 1.0
inv_mass = np.array([0.0, 1.0, 1.0, 1.0, 1.0])  # 粒子0はピン留め (invMass=0)

# 静止長から引き伸ばした初期配置 (1.3倍に伸長)
stretch_factor = 1.3
p0 = np.array([[i * rest_len * stretch_factor, 0.0, 0.0] for i in range(N)])
p_prev = p0.copy()  # 前フレーム位置 (静止長のチェーンだったと仮定)
for i in range(N):
    p_prev[i] = np.array([i * rest_len, 0.0, 0.0])
v0 = (p0 - p_prev) / dt  # 引き伸ばしから生じる見かけの速度 (実際は単に伸びた初期条件)
v0[:] = 0.0  # 速度はゼロから開始 (静的引き伸ばし)

edges = [(i, i + 1) for i in range(N - 1)]  # 4エッジ

# XPBD compliance (SoftBodyEngine のデフォルト stretchCompliance=1e-5 相当)
compliance = 1e-5
alpha_tilde = compliance / (dt * dt)

# VBD 側の等価剛性 k = 1/compliance (XPBDのcomplianceは 1/剛性 のディメンション)
k_edge = 1.0 / compliance


def predicted_positions():
    """XPBD の predict ステップ: p_tilde = p + v*dt (重力なし、引き伸ばし静解析のため)"""
    return p0 + v0 * dt


def greedy_edge_coloring(edges, n):
    """エッジの貪欲彩色 (同色内は頂点を共有しない)。GPU側のedgeColorBatchと同じ考え方。"""
    colors = [-1] * len(edges)
    vertex_color_used = [set() for _ in range(n)]
    for ei, (a, b) in enumerate(edges):
        used = vertex_color_used[a] | vertex_color_used[b]
        c = 0
        while c in used:
            c += 1
        colors[ei] = c
        vertex_color_used[a].add(c)
        vertex_color_used[b].add(c)
    return colors


def greedy_vertex_coloring(edges, n):
    """頂点の貪欲彩色 (エッジで直接つながる頂点同士は異なる色)。VBD の並列バッチ用。"""
    adjacency = [set() for _ in range(n)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    colors = [-1] * n
    for v in range(n):
        used = {colors[u] for u in adjacency[v] if colors[u] != -1}
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    return colors


def constraint_residual(p):
    """全エッジの |C| = |距離 - restLen| の合計 (拘束残差の指標)"""
    total = 0.0
    for a, b in edges:
        d = np.linalg.norm(p[a] - p[b])
        total += abs(d - rest_len)
    return total


def run_xpbd(iters):
    p = predicted_positions().copy()
    lam = {e: 0.0 for e in edges}
    edge_colors = greedy_edge_coloring(edges, N)
    n_colors = max(edge_colors) + 1
    residuals = []
    for _ in range(iters):
        for c in range(n_colors):
            # 色内は Jacobi (同時読み取り): 更新はこのループの最後にまとめて適用
            updates = []
            for ei, (a, b) in enumerate(edges):
                if edge_colors[ei] != c:
                    continue
                n_vec = p[a] - p[b]
                d = np.linalg.norm(n_vec)
                if d < 1e-9:
                    continue
                n_hat = n_vec / d
                C = d - rest_len
                w_sum = inv_mass[a] + inv_mass[b]
                dlam = -(C + alpha_tilde * lam[(a, b)]) / (w_sum + alpha_tilde)
                lam[(a, b)] += dlam
                updates.append((a, inv_mass[a] * dlam * n_hat))
                updates.append((b, -inv_mass[b] * dlam * n_hat))
            for idx, delta in updates:
                p[idx] += delta
        residuals.append(constraint_residual(p))
    return residuals


def run_vbd(iters):
    p = predicted_positions().copy()
    p_tilde = predicted_positions()  # 慣性項の目標位置 (このシナリオでは predict と同じ)
    vertex_colors = greedy_vertex_coloring(edges, N)
    n_colors = max(vertex_colors) + 1
    adjacency = [[] for _ in range(N)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    residuals = []
    for _ in range(iters):
        for c in range(n_colors):
            updates = {}
            for v in range(N):
                if vertex_colors[v] != c or inv_mass[v] == 0.0:
                    continue
                m = 1.0 / inv_mass[v]
                # 慣性項: g_inertia = m/dt^2 * (p_v - p_tilde_v), H_inertia = m/dt^2 * I
                g = (m / (dt * dt)) * (p[v] - p_tilde[v])
                H = (m / (dt * dt)) * np.eye(3)
                for u in adjacency[v]:
                    n_vec = p[v] - p[u]
                    d = np.linalg.norm(n_vec)
                    if d < 1e-9:
                        continue
                    n_hat = n_vec / d
                    C = d - rest_len
                    # スプリングエネルギー E=0.5*k*C^2 の勾配・ヘシアン (標準的なmass-spring Hessian)
                    g += k_edge * C * n_hat
                    H += k_edge * (np.outer(n_hat, n_hat) +
                                   (C / d) * (np.eye(3) - np.outer(n_hat, n_hat)))
                try:
                    dp = -np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    dp = np.zeros(3)
                updates[v] = dp
            for v, dp in updates.items():
                p[v] += dp
        residuals.append(constraint_residual(p))
    return residuals


def run_scenario(iters, label):
    res_xpbd = run_xpbd(iters)
    res_vbd = run_vbd(iters)
    print(f"=== {label} ===")
    print(f"{'iter':>4} | {'XPBD residual':>15} | {'VBD residual':>15} | {'VBD/XPBD':>9}")
    print("-" * 54)
    for i in range(iters):
        ratio = res_vbd[i] / res_xpbd[i] if res_xpbd[i] > 1e-12 else float("nan")
        print(f"{i+1:>4} | {res_xpbd[i]:>15.6f} | {res_vbd[i]:>15.6f} | {ratio:>9.3f}")
    print()


if __name__ == "__main__":
    run_scenario(15, "シナリオA: 5粒子チェーン (各頂点の次数<=2、疎な連結)")

    # ── シナリオB: 高連結度 (ハブ&スポーク) ─────────────────────────────────
    # 中心の自由頂点1個 + 周囲6個の固定頂点。中心頂点は6本のエッジを同時に持つ
    # (四面体メッシュの内部頂点に近い連結度)。VBDは全接続エッジのエネルギーを
    # 一括してNewtonステップで解くため、この設定で優位性が出やすいはず。
    N = 7
    rest_len = 1.0
    inv_mass = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 頂点0のみ自由

    # 頂点0を中心、頂点1-6を正八面体的に配置 (静止長1.0の位置から少しずらして開始)
    directions = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ], dtype=float)
    p0 = np.zeros((N, 3))
    p0[0] = np.array([0.3, 0.2, 0.1])  # 中心を静止位置からずらして開始 (歪み初期条件)
    for i in range(6):
        p0[i + 1] = directions[i] * rest_len
    p_prev = p0.copy()
    p_prev[0] = np.array([0.0, 0.0, 0.0])  # 前フレームは中心が原点にあったと仮定
    v0 = np.zeros((N, 3))

    edges = [(0, i + 1) for i in range(6)]

    run_scenario(15, "シナリオB: ハブ&スポーク (中心頂点の次数=6、密な連結)")
