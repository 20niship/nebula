#pragma once
#include <cstdint>

// MPM 専用 Push Constants — 160 bytes
// hash compat: オフセット 20-60 は SimPC と完全一致
// → hash_scan_local/global/add_base/zero_cells シェーダーを無修正で再利用可能
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
  uint32_t particleCount; // 32  ← hash compat ★ (ライブパーティクル数)
  uint32_t gridRes;       // 36  ← hash compat ★
  uint32_t F2Idx;         // 40  F 列2 (xyz) + σ_zz (w)
  uint32_t materialsIdx;  // 44  MaterialParams SSBO インデックス (0=無効)

  // ── World / time (hash compat) (16 bytes) ────────────────────────────
  float dt;       // 48  ← hash compat ★
  float cellSize; // 52  ← hash compat ★
  float worldMin; // 56  ← hash compat ★
  float worldMax; // 60  ← hash compat ★

  // ── Material properties (16 bytes) ───────────────────────────────────
  float gravity;        // 64
  float mu_lame;        // 68  グローバルデフォルト μ（material id ≥ 件数のフォールバック）
  float lambda_lame;    // 72  グローバルデフォルト λ
  float particleVolume; // 76  初期パーティクル体積 V_p（グローバルデフォルト）

  // ── Plasticity parameters (16 bytes) ─────────────────────────────────
  float M_friction; // 80  グローバルデフォルト DP M
  float q_cohesion; // 84  グローバルデフォルト DP q_c
  float q_max;      // 88  グローバルデフォルト VM q_max
  float flip_ratio; // 92  0=PIC, 1=FLIP, -1=APIC (Phase 2)

  // ── Extra buffer indices (16 bytes) ──────────────────────────────────
  uint32_t colliderIdx;   // 96  Collider SSBO インデックス (0=無効, Phase 3)
  uint32_t colliderCount; // 100 コライダー数 (Phase 3)
  uint32_t B0Idx;         // 104 APIC B行列 列0 (xyz) + σ_xy (w)
  uint32_t B1Idx;         // 108 APIC B行列 列1 (xyz) + σ_xz (w)

  // ── Grid buffer indices (16 bytes) ───────────────────────────────────
  uint32_t B2Idx;       // 112 APIC B行列 列2 (xyz) + σ_yz (w)
  uint32_t nanoVDBIdx;  // 116 NanoVDB SDF バッファ (0=無効)
  uint32_t gridMomIdx;  // 120 vec4×CELLS グリッド運動量/速度
  uint32_t gridMassIdx; // 124 float×CELLS グリッド質量

  // ── Boundary / solver (16 bytes) ─────────────────────────────────────
  float restitution;      // 128
  float wall_friction;    // 132
  uint32_t plasticModel;  // 136 グローバルモデル: 0=弾性,1=VM,2=DP（Phase 1 まで有効）
  uint32_t materialCount; // 140 materials SSBO のエントリ数

  // ── Rest density / softening (16 bytes) ──────────────────────────────
  float rho0;             // 144 グローバルデフォルト密度
  float p0_mcc;           // 148 MCC 予圧密圧力
  float xi_hard;          // 152 軟化パラメータ
  float maxParticlesFrac; // 156 予約（将来用）
};
static_assert(sizeof(MPMSimPC) == 160, "MPMSimPC must be 160 bytes");
