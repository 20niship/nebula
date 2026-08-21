#pragma once
#include <cstdint>
#include <glm/glm.hpp>

// MPM 専用 Push Constants — 176 bytes。GLSL側push_constantはvec3/uvec3を16byte境界に強制配置するため、各vec3系フィールド直前のスカラー数を4の倍数に揃えてパディングなしで境界を一致させている(フィールド順・スカラー数を変える場合は要再検証)。hashCellsは旧gridRes(スカラー)を改名したもので空間ハッシュ/MPMグリッドバッファの実要素数(=domain::hashCells()、cubeRes^3。nx*ny*nzではない)を表す。
struct MPMSimPC {
  // ── Bindless バッファインデックス (48 bytes、12個のスカラー) ──────────
  uint32_t posIdx;        // 0   vec4×N  (xyz=position, w=initial volume Vp)
  uint32_t velIdx;        // 4   vec4×N  (xyz=velocity, w=material id)
  uint32_t F0Idx;         // 8   vec4×N  F 列0 (xyz) + σ_xx (w)
  uint32_t F1Idx;         // 12  vec4×N  F 列1 (xyz) + σ_yy (w)
  uint32_t typeFlagIdx;   // 16  uint×N  (reserved)
  uint32_t particleCount; // 20  (ライブパーティクル数)
  uint32_t hashCells;     // 24  空間ハッシュ/MPMグリッドバッファの実要素数
  uint32_t F2Idx;         // 28  F 列2 (xyz) + σ_zz (w)
  uint32_t materialsIdx;  // 32  MaterialParams SSBO インデックス (0=無効)
  float dt;               // 36
  float cellSize;         // 40  全軸共通のセルサイズ [m]
  uint32_t forceBufIdx;   // 44  Force配列(ForceGPU×forceCount)のbindless index (issue #30; 旧gravity)

  // ── グリッド解像度 + 直後1個のスカラーで16byte境界を維持 ────────────────
  glm::uvec3 gridRes; // 48  各軸の実セル数 (nx,ny,nz) = domain::gridRes(domainSize, cellSize)
  float mu_lame;      // 60  グローバルデフォルト μ

  float lambda_lame;    // 64  グローバルデフォルト λ
  float particleVolume; // 68  初期パーティクル体積 V_p（グローバルデフォルト）
  float M_friction;     // 72  グローバルデフォルト DP M
  float q_cohesion;     // 76  グローバルデフォルト DP q_c

  // ── ワールド下限境界 + 直後1個のスカラーで16byte境界を維持 ──────────────
  glm::vec3 worldMin; // 80  ドメイン下限座標 [m] (通常は原点 (0,0,0))
  float q_max;        // 92  グローバルデフォルト VM q_max

  float flip_ratio;      // 96  0=PIC, 1=FLIP, -1=APIC (Phase 2)
  uint32_t colliderIdx;  // 100 Collider SSBO インデックス (0=無効, Phase 3)
  uint32_t colliderCount; // 104 コライダー数 (Phase 3)
  uint32_t B0Idx;         // 108 APIC B行列 列0 (xyz) + σ_xy (w)

  // ── ワールド上限境界 + 直後1個のスカラーで16byte境界を維持 ──────────────
  glm::vec3 worldMax; // 112 ドメイン上限座標 [m] (= worldMin + domainSize)
  uint32_t B1Idx;      // 124 APIC B行列 列1 (xyz) + σ_xz (w)

  uint32_t B2Idx;         // 128 APIC B行列 列2 (xyz) + σ_yz (w)
  uint32_t reserved144;   // 132 旧NanoVDB SDF境界条件用(mpm_nanovdb_bc.comp削除に伴い未使用化、colliderIdx/MESH_SDFに統一)
  uint32_t gridMomIdx;    // 136 vec4×CELLS グリッド運動量/速度
  uint32_t gridMassIdx;   // 140 float×CELLS グリッド質量
  float restitution;      // 144
  float wall_friction;    // 148
  uint32_t plasticModel;  // 152 グローバルモデル: 0=弾性,1=VM,2=DP（Phase 1 まで有効）
  uint32_t materialCount; // 156 materials SSBO のエントリ数
  float rho0;             // 160 グローバルデフォルト密度
  float p0_mcc;           // 164 MCC 予圧密圧力
  float xi_hard;          // 168 軟化パラメータ
  uint32_t forceCount;    // 172 有効なForce数 (issue #30; 旧maxParticlesFrac予約枠)
};
static_assert(sizeof(MPMSimPC) == 176, "MPMSimPC must be 176 bytes");
