#pragma once
#include <cstdint>
#include <glm/glm.hpp>

// MPM 専用 Push Constants — 188 bytes
// hash compat: cellCountIdx(20)/cellOffsetIdx(24)/hashCells(36) は SimPC と完全一致
// → hash_scan_local/global/add_base/zero_cells シェーダーを無修正で再利用可能
// (この4シェーダーが実際に参照するのはこの3フィールドのみ。他のフィールドはオフセットが
//  一致している必要はない)
//
// issue #46 (直方体ドメイン対応): worldMin/worldMax を vec3 化し、各軸独立の gridRes(uvec3)
// を追加した。hashCells は旧 gridRes(スカラー) を改名したもので、空間ハッシュ/MPMグリッド
// バッファの実要素数 (= domain::hashCells()、cubeRes^3。nx*ny*nzではない) を表す。
struct MPMSimPC {
  // ── Bindless バッファインデックス (48 bytes) ──────────────────────────
  uint32_t posIdx;        // 0   vec4×N  (xyz=position, w=initial volume Vp)
  uint32_t velIdx;        // 4   vec4×N  (xyz=velocity, w=material id)
  uint32_t F0Idx;         // 8   vec4×N  F 列0 (xyz) + σ_xx (w)
  uint32_t F1Idx;         // 12  vec4×N  F 列1 (xyz) + σ_yy (w)
  uint32_t typeFlagIdx;   // 16  uint×N  (reserved)
  uint32_t cellCountIdx;  // 20  ← hash compat ★
  uint32_t cellOffsetIdx; // 24  ← hash compat ★
  uint32_t sortedIdxIdx;  // 28  ← hash compat ★
  uint32_t particleCount; // 32  (ライブパーティクル数)
  uint32_t hashCells;     // 36  空間ハッシュ/MPMグリッドバッファの実要素数 ← hash compat ★
  uint32_t F2Idx;         // 40  F 列2 (xyz) + σ_zz (w)
  uint32_t materialsIdx;  // 44  MaterialParams SSBO インデックス (0=無効)

  // ── World / time (16 bytes) ──────────────────────────────────────────
  float dt;       // 48
  float cellSize; // 52  全軸共通のセルサイズ [m]
  uint32_t forceBufIdx; // 56  Force配列(ForceGPU×forceCount)のbindless index (issue #30; 旧gravity)
  float mu_lame;        // 60  グローバルデフォルト μ（material id ≥ 件数のフォールバック）

  // ── グリッド解像度 / ワールド境界 (48 bytes) ───────────────────────────
  glm::uvec3 gridRes; // 64  各軸の実セル数 (nx,ny,nz) = domain::gridRes(domainSize, cellSize)
  float lambda_lame;  // 76  グローバルデフォルト λ

  glm::vec3 worldMin;   // 80  ドメイン下限座標 [m] (通常は原点 (0,0,0))
  float particleVolume; // 92  初期パーティクル体積 V_p（グローバルデフォルト）

  glm::vec3 worldMax; // 96  ドメイン上限座標 [m] (= worldMin + domainSize)
  float M_friction;   // 108 グローバルデフォルト DP M

  // ── Plasticity parameters (16 bytes) ─────────────────────────────────
  float q_cohesion; // 112 グローバルデフォルト DP q_c
  float q_max;      // 116 グローバルデフォルト VM q_max
  float flip_ratio; // 120 0=PIC, 1=FLIP, -1=APIC (Phase 2)
  uint32_t colliderIdx; // 124 Collider SSBO インデックス (0=無効, Phase 3)

  // ── Extra buffer indices (16 bytes) ──────────────────────────────────
  uint32_t colliderCount; // 128 コライダー数 (Phase 3)
  uint32_t B0Idx;         // 132 APIC B行列 列0 (xyz) + σ_xy (w)
  uint32_t B1Idx;         // 136 APIC B行列 列1 (xyz) + σ_xz (w)
  uint32_t B2Idx;         // 140 APIC B行列 列2 (xyz) + σ_yz (w)

  // ── Grid buffer indices (16 bytes) ───────────────────────────────────
  uint32_t nanoVDBIdx;  // 144 NanoVDB SDF バッファ (0=無効)
  uint32_t gridMomIdx;  // 148 vec4×CELLS グリッド運動量/速度
  uint32_t gridMassIdx; // 152 float×CELLS グリッド質量
  float restitution;    // 156

  // ── Boundary / solver (16 bytes) ─────────────────────────────────────
  float wall_friction;    // 160
  uint32_t plasticModel;  // 164 グローバルモデル: 0=弾性,1=VM,2=DP（Phase 1 まで有効）
  uint32_t materialCount; // 168 materials SSBO のエントリ数
  float rho0;             // 172 グローバルデフォルト密度

  // ── Rest density / softening (16 bytes) ──────────────────────────────
  float p0_mcc;        // 176 MCC 予圧密圧力
  float xi_hard;       // 180 軟化パラメータ
  uint32_t forceCount; // 184 有効なForce数 (issue #30; 旧maxParticlesFrac予約枠)
};
static_assert(sizeof(MPMSimPC) == 188, "MPMSimPC must be 188 bytes");
static_assert(offsetof(MPMSimPC, cellCountIdx) == 20, "hash compat offset");
static_assert(offsetof(MPMSimPC, cellOffsetIdx) == 24, "hash compat offset");
static_assert(offsetof(MPMSimPC, hashCells) == 36, "hash compat offset");
