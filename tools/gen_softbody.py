"""
gen_softbody.py  —  XPBD ソフトボディ用 .sb バイナリを生成する。

.sb フォーマット:
  magic       : 4 bytes  "SFTB"
  n_particles : uint32
  n_edges     : uint32
  n_tets      : uint32
  n_edge_colors: uint32
  n_tet_colors : uint32
  positions   : n_particles * 3 * float32  (xyz)
  inv_masses  : n_particles * float32
  edge_color_batch : (n_edge_colors+1) * uint32  (累積エッジ数)
  edge_data   : n_edges * 3 * uint32  [p0, q0, floatBitsToUint(restLen)]  色順ソート
  tet_color_batch  : (n_tet_colors+1) * uint32  (累積四面体数)
  tet_data    : n_tets * 4 * uint32  [i0,i1,i2,i3]  色順ソート
  tet_rest_vols: n_tets * float32                    色順ソートに対応

使い方:
  # プロシージャルジェリーキューブ (追加 pip 依存なし)
  python tools/gen_softbody.py --cube --grid 5 --size 1.5 -o assets/cube_sb.sb

  # Stanford Bunny (要: pip install tetgen trimesh numpy networkx)
  python tools/gen_softbody.py --bunny --input assets/bunny.obj \\
         --scale 1.5 --max-vol 0.003 -o assets/bunny_sb.sb
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

SB_MAGIC = b"SFTB"

# ─── ユーティリティ ────────────────────────────────────────────────────────────

def tet_signed_vol(p0, p1, p2, p3):
    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0
    return np.dot(e1, np.cross(e2, e3)) / 6.0


def fix_tet_winding(nodes, tets):
    """全四面体の符号付き体積を正にする。"""
    fixed = tets.copy()
    for i, t in enumerate(tets):
        v = tet_signed_vol(nodes[t[0]], nodes[t[1]], nodes[t[2]], nodes[t[3]])
        if v < 0:
            fixed[i, 2], fixed[i, 3] = fixed[i, 3], fixed[i, 2]
    return fixed


def extract_unique_edges(tets):
    """四面体リストから重複なしエッジ集合を返す。(a < b) で正規化。"""
    edge_set = set()
    for t in tets:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[0], t[3]),
                 (t[1], t[2]), (t[1], t[3]), (t[2], t[3])]
        for a, b in pairs:
            edge_set.add((min(a, b), max(a, b)))
    return sorted(edge_set)


def color_constraints(constraint_list, n_verts, is_tet=False):
    """
    貪欲グラフ彩色。
    constraint_list: エッジなら [(a,b), ...]、四面体なら [(a,b,c,d), ...]
    共有頂点を持つ制約を同一バッチに入れない。
    戻り値: constraints と同じ長さの色インデックスリスト
    """
    n = len(constraint_list)
    colors = [-1] * n

    # 各頂点がどの制約に属するかを記録
    vert_to_constraints = [[] for _ in range(n_verts)]
    for ci, c in enumerate(constraint_list):
        for v in c:
            vert_to_constraints[v].append(ci)

    for ci in range(n):
        # 隣接する制約の色を収集
        forbidden = set()
        for v in constraint_list[ci]:
            for other in vert_to_constraints[v]:
                if colors[other] >= 0:
                    forbidden.add(colors[other])
        # 使われていない最小の色を割り当て
        c = 0
        while c in forbidden:
            c += 1
        colors[ci] = c

    return colors


def sort_by_color(items, colors):
    """items を colors でソートし、(sorted_items, color_batch) を返す。
       color_batch[c] = 色 c の開始インデックス、color_batch[-1] = 総数。
    """
    n_colors = max(colors) + 1
    buckets = [[] for _ in range(n_colors)]
    for item, color in zip(items, colors):
        buckets[color].append(item)

    sorted_items = []
    color_batch = [0]
    for bucket in buckets:
        sorted_items.extend(bucket)
        color_batch.append(len(sorted_items))

    return sorted_items, color_batch, n_colors


# ─── プロシージャル Cube ──────────────────────────────────────────────────────

def make_cube(grid_n, size, mass_per_particle=1.0):
    """
    grid_n × grid_n × grid_n の均一格子を生成し、5-tet 分解で四面体化する。
    追加ライブラリ不要。
    """
    n = grid_n
    step = size / float(n - 1)

    # 頂点
    verts = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                verts.append([i * step, j * step, k * step])
    verts = np.array(verts, dtype=np.float32)

    def idx(i, j, k):
        return k * n * n + j * n + i

    # 各 hex セルを 5 四面体に分割（偶奇パターン交互で整合性を保つ）
    tets = []
    for k in range(n - 1):
        for j in range(n - 1):
            for i in range(n - 1):
                v = [idx(i,   j,   k),   idx(i+1, j,   k),
                     idx(i+1, j+1, k),   idx(i,   j+1, k),
                     idx(i,   j,   k+1), idx(i+1, j,   k+1),
                     idx(i+1, j+1, k+1), idx(i,   j+1, k+1)]
                if (i + j + k) % 2 == 0:
                    tets += [
                        [v[0], v[1], v[3], v[4]],
                        [v[1], v[2], v[3], v[6]],
                        [v[1], v[4], v[5], v[6]],
                        [v[3], v[4], v[6], v[7]],
                        [v[1], v[3], v[4], v[6]],
                    ]
                else:
                    tets += [
                        [v[0], v[1], v[2], v[5]],
                        [v[0], v[2], v[3], v[7]],
                        [v[0], v[4], v[5], v[7]],
                        [v[2], v[5], v[6], v[7]],
                        [v[0], v[2], v[5], v[7]],
                    ]

    tets = np.array(tets, dtype=np.int32)
    inv_masses = np.full(len(verts), 1.0 / mass_per_particle, dtype=np.float32)
    return verts, tets, inv_masses


# ─── Bunny / OBJ (tetgen 使用) ────────────────────────────────────────────────

def _cap_boundary_holes(mesh):
    """
    開口した境界ループをファン三角形化で蓋する。
    各バウンダリループの重心を新頂点として追加し、ループ辺をすべて三角形に分割する。
    複数の穴があっても一括処理する。
    """
    import trimesh

    verts = list(mesh.vertices)
    faces = list(mesh.faces)

    # 境界辺 (片側だけに面が隣接する辺) を収集
    edge_to_faces = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            e = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_to_faces.setdefault(e, []).append(fi)

    boundary_edges = {e for e, fs in edge_to_faces.items() if len(fs) == 1}
    if not boundary_edges:
        return mesh

    # 境界辺をループにまとめる (next_vert[v] = 次の境界頂点)
    # 方向は面の向きに合わせる (反時計回りで外側に法線が向くよう)
    next_vert = {}
    for fi, face in enumerate(faces):
        for i in range(3):
            a, b = face[i], face[(i + 1) % 3]
            if tuple(sorted([a, b])) in boundary_edges:
                # この面から見て a→b が境界辺 → b→a が "外向き" 半辺
                next_vert[b] = a

    visited = set()
    new_faces = []

    for start in list(next_vert.keys()):
        if start in visited:
            continue
        # ループをトレース
        loop = []
        cur = start
        for _ in range(len(next_vert) + 1):
            if cur in visited and cur != start:
                break
            visited.add(cur)
            loop.append(cur)
            cur = next_vert.get(cur, -1)
            if cur == start or cur == -1:
                break

        if len(loop) < 3:
            continue

        # ループの重心を新頂点として追加
        loop_coords = np.array([verts[v] for v in loop], dtype=np.float32)
        centroid = loop_coords.mean(axis=0)
        ci = len(verts)
        verts.append(centroid)

        # ファン三角形 (反時計回り = 外向き法線)
        for i in range(len(loop)):
            a = loop[i]
            b = loop[(i + 1) % len(loop)]
            new_faces.append([ci, b, a])

    if not new_faces:
        return mesh

    all_verts = np.array(verts, dtype=np.float32)
    all_faces = np.array(faces + new_faces, dtype=np.int32)
    capped = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=True)
    trimesh.repair.fix_normals(capped)
    return capped


def make_bunny(obj_path, scale, max_vol, mass_per_particle=1.0):
    try:
        import trimesh
        import tetgen
    except ImportError:
        print("ERROR: bunny モードには trimesh と tetgen が必要です。")
        print("  pip install trimesh tetgen scipy")
        sys.exit(1)

    print(f"Loading mesh: {obj_path}")
    mesh = trimesh.load(obj_path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("メッシュの読み込みに失敗しました。")

    # スケールを適用
    mesh.apply_scale(scale)

    # 原点に移動（最小点を 0 に）
    mesh.apply_translation(-mesh.bounds[0])

    # 穴埋め修復:
    #  1. fill_holes を試す
    #  2. 失敗なら voxelize → marching cubes でバニー形状を保ったまま水密化
    trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight:
        print("Mesh is not watertight, attempting fill_holes...")
        trimesh.repair.fill_holes(mesh)

    if not mesh.is_watertight:
        print("fill_holes failed. Using voxel marching-cubes repair (shape-preserving)...")
        try:
            # ボクセル解像度: 最長辺を約 30 分割
            pitch = float(max(mesh.extents)) / 30.0
            vox   = mesh.voxelized(pitch=pitch).fill()
            mc    = vox.marching_cubes
            trimesh.repair.fix_normals(mc)
            if mc.is_watertight:
                mesh = mc
                print(f"  Voxel mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
            else:
                raise RuntimeError("marching cubes mesh is not watertight")
        except Exception as e:
            raise RuntimeError(
                f"ボクセル修復も失敗しました: {e}\n"
                "  水密な OBJ を用意するか MeshLab で修復してください。"
            )

    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    # TetGen で四面体化 (quality フラグなし → 入力頂点のみ使用、追加点なし)
    tet = tetgen.TetGen(mesh.vertices.astype(np.float64),
                        mesh.faces.astype(np.int32))
    result = tet.tetrahedralize(order=1, maxvolume=max_vol, verbose=False)
    # tetgen returns (nodes, elems, ...) — may have 2 or 4 elements depending on version
    nodes = np.asarray(result[0], dtype=np.float32)
    elems = np.asarray(result[1], dtype=np.int32)
    print(f"Tet mesh: {len(nodes)} nodes, {len(elems)} tets")

    inv_masses = np.full(len(nodes), 1.0 / mass_per_particle, dtype=np.float32)
    return nodes, elems, inv_masses


# ─── .sb バイナリ書き出し ────────────────────────────────────────────────────

def write_sb(out_path, nodes, tets, inv_masses):
    """
    nodes     : (N, 3) float32
    tets      : (T, 4) int32  (正の符号付き体積に修正済みであること)
    inv_masses: (N,) float32
    """
    n_verts = len(nodes)
    n_tets  = len(tets)

    # winding 修正
    tets = fix_tet_winding(nodes, tets)

    # エッジ抽出
    edges = extract_unique_edges(tets)
    n_edges = len(edges)
    print(f"  Edges: {n_edges}")

    # グラフ彩色
    print("  Coloring edges...")
    edge_colors = color_constraints(edges, n_verts, is_tet=False)
    print("  Coloring tets...")
    tet_tuples  = [tuple(t) for t in tets]
    tet_colors  = color_constraints(tet_tuples, n_verts, is_tet=True)

    n_edge_colors = max(edge_colors) + 1
    n_tet_colors  = max(tet_colors)  + 1
    print(f"  Edge colors: {n_edge_colors}, Tet colors: {n_tet_colors}")

    # 色順ソート
    sorted_edges, edge_color_batch, _ = sort_by_color(edges, edge_colors)
    sorted_tets,  tet_color_batch,  _ = sort_by_color(tet_tuples, tet_colors)

    # 静止長 / 静止体積の計算
    rest_lengths = []
    for a, b in sorted_edges:
        rest_lengths.append(float(np.linalg.norm(nodes[b] - nodes[a])))

    rest_vols = []
    for t in sorted_tets:
        v = tet_signed_vol(nodes[t[0]], nodes[t[1]], nodes[t[2]], nodes[t[3]])
        rest_vols.append(abs(float(v)))

    # エッジデータ: [p0, q0, floatBitsToUint(restLen)]
    import struct as st
    edge_data_flat = []
    for (a, b), rl in zip(sorted_edges, rest_lengths):
        bits = st.unpack('I', st.pack('f', rl))[0]
        edge_data_flat.extend([int(a), int(b), bits])

    # 四面体データ: [i0,i1,i2,i3]
    tet_data_flat = []
    for t in sorted_tets:
        tet_data_flat.extend([int(t[0]), int(t[1]), int(t[2]), int(t[3])])

    # 書き込み
    with open(out_path, "wb") as f:
        # マジック
        f.write(SB_MAGIC)

        # ヘッダー
        f.write(struct.pack("<IIIII",
                            n_verts, n_edges, n_tets,
                            n_edge_colors, n_tet_colors))

        # 頂点座標
        for p in nodes:
            f.write(struct.pack("<fff", float(p[0]), float(p[1]), float(p[2])))

        # 逆質量
        for m in inv_masses:
            f.write(struct.pack("<f", float(m)))

        # エッジ色バッチ
        for b in edge_color_batch:
            f.write(struct.pack("<I", b))

        # エッジデータ
        for v in edge_data_flat:
            f.write(struct.pack("<I", v))

        # 四面体色バッチ
        for b in tet_color_batch:
            f.write(struct.pack("<I", b))

        # 四面体データ
        for v in tet_data_flat:
            f.write(struct.pack("<I", v))

        # 静止体積
        for v in rest_vols:
            f.write(struct.pack("<f", v))

    print(f"Written: {out_path}")
    size_kb = Path(out_path).stat().st_size / 1024
    print(f"  File size: {size_kb:.1f} KB")
    print(f"  Particles: {n_verts}, Edges: {n_edges}, Tets: {n_tets}")


# ─── エントリポイント ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XPBD soft body .sb mesh generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cube",  action="store_true", help="Procedural cube (no extra deps)")
    group.add_argument("--bunny", action="store_true", help="Bunny from OBJ (needs tetgen, trimesh)")

    parser.add_argument("-o", "--output", required=True, help="Output .sb file path")

    # Cube オプション
    parser.add_argument("--grid", type=int,   default=5,    help="Cube grid resolution (default 5)")
    parser.add_argument("--size", type=float, default=1.5,  help="Cube side length [m] (default 1.5)")

    # Bunny オプション
    parser.add_argument("--input",   type=str,   default="",    help="Input OBJ/PLY path")
    parser.add_argument("--scale",   type=float, default=1.5,   help="Mesh scale factor")
    parser.add_argument("--max-vol", type=float, default=0.003, help="TetGen max tet volume")

    # 共通
    parser.add_argument("--mass", type=float, default=1.0, help="Mass per particle [kg]")

    args = parser.parse_args()

    if args.cube:
        print(f"Generating cube: grid={args.grid}, size={args.size}")
        nodes, tets, inv_masses = make_cube(args.grid, args.size, args.mass)
    else:
        if not args.input:
            parser.error("--bunny requires --input <obj_path>")
        nodes, tets, inv_masses = make_bunny(args.input, args.scale, args.max_vol, args.mass)

    write_sb(args.output, nodes, tets, inv_masses)


if __name__ == "__main__":
    main()
