#ifndef MPM_NANOVDB_COMMON_GLSL
#define MPM_NANOVDB_COMMON_GLSL

// ── PNanoVDB バインドレスバッファブリッジ ──────────────────────────────────
// mpm_common.glsl の後に include すること（buffers[], pc が必要）。
// pc.nanoVDBIdx != 0 のときのみ有効。

// pnanovdb_buf_data を bindless バッファ[nanoVDBIdx] の uint 配列にマップ
// PNanoVDB.h を include する前に定義する必要がある
#define pnanovdb_buf_data buffers[pc.nanoVDBIdx].data

#define PNANOVDB_GLSL
#include "PNanoVDB.h"   // Mac: ${CMAKE_BINARY_DIR}/nanovdb_glsl/PNanoVDB.h (auto-patched)
                        // other: third_party/openvdb/nanovdb/nanovdb/PNanoVDB.h

// ── NanoVDB SDF クエリ ────────────────────────────────────────────────────
// NanoVDB グリッドは MPM 座標系で作成済みであることを前提とする:
//   voxelSize = pc.cellSize,  origin = pc.worldMin
// → world → index 変換は (worldPos - worldMin) / cellSize のみで OK
// 戻り値: world 単位の SDF 値（正=外側, 負=内側, 0=表面）
float nanovdb_query_sdf(vec3 worldPos) {
    pnanovdb_buf_t buf;
    buf.unused = 0u;

    // グリッドはバッファの先頭 (byte offset 0) から始まる
    pnanovdb_grid_handle_t grid;
    grid.address.byte_offset = 0u;

    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, grid);
    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);

    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(PNANOVDB_REF(acc), root);

    // world → index 変換
    vec3 idx = (worldPos - vec3(pc.worldMin)) / pc.cellSize;
    pnanovdb_coord_t ijk = ivec3(floor(idx));

    pnanovdb_address_t addr = pnanovdb_readaccessor_get_value_address(
        PNANOVDB_GRID_TYPE_FLOAT, buf, PNANOVDB_REF(acc), PNANOVDB_REF(ijk));

    // SDF 値はインデックス空間 (voxel 単位) → world 単位に変換
    return pnanovdb_read_float(buf, addr) * pc.cellSize;
}

#endif // MPM_NANOVDB_COMMON_GLSL
