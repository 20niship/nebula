#pragma once
#include <cstdint>
#include <glm/glm.hpp>

// MPM 専用 Push Constants — 172 bytes。各vec3系フィールド直前のスカラー数を4の倍数に揃え16byte境界に一致させている(要再検証)。hashCellsは空間ハッシュ/MPMグリッドバッファの実要素数(cubeRes^3)。worldMinは廃止(ワールド座標=ローカル座標)。colliderForceIdx/colliderTorqueIdxは末尾に追加。
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

  float q_max;            // 80  グローバルデフォルト VM q_max
  float flip_ratio;       // 84  0=PIC, 1=FLIP, -1=APIC (Phase 2)
  uint32_t colliderIdx;   // 88  Collider SSBO インデックス (0=無効, Phase 3)
  uint32_t colliderCount; // 92  コライダー数 (Phase 3)
  uint32_t B0Idx;         // 96  APIC B行列 列0 (xyz) + σ_xy (w)
  uint32_t B1Idx;         // 100 APIC B行列 列1 (xyz) + σ_xz (w)
  uint32_t B2Idx;         // 104 APIC B行列 列2 (xyz) + σ_yz (w)
  uint32_t reserved144;   // 108 旧NanoVDB SDF境界条件用(mpm_nanovdb_bc.comp削除に伴い未使用化、colliderIdx/MESH_SDFに統一)

  // ── ワールド上限境界 + 直後1個のスカラーで16byte境界を維持 ──────────────
  glm::vec3 worldMax;  // 112 ドメイン上限座標 [m] (= domainSize、下限は常に原点)
  uint32_t gridMomIdx; // 124 vec4×CELLS グリッド運動量/速度

  uint32_t gridMassIdx;   // 128 float×CELLS グリッド質量
  float restitution;      // 132
  float wall_friction;    // 136
  uint32_t plasticModel;  // 140 グローバルモデル: 0=弾性,1=VM,2=DP（Phase 1 まで有効）
  uint32_t materialCount; // 144 materials SSBO のエントリ数
  float rho0;             // 148 グローバルデフォルト密度
  float p0_mcc;           // 152 MCC 予圧密圧力
  float xi_hard;          // 156 軟化パラメータ
  uint32_t forceCount;    // 160 有効なForce数 (issue #30; 旧maxParticlesFrac予約枠)

  // ── コライダー反力/反トルク読み戻し (8 bytes) ─────────────────────────
  uint32_t colliderForceIdx;  // 164 vec4×64 コライダーが受け取った力積(fixed-point)
  uint32_t colliderTorqueIdx; // 168 vec4×64 コライダーが受け取った反トルク積(fixed-point)
};
static_assert(sizeof(MPMSimPC) == 172, "MPMSimPC must be 172 bytes");
