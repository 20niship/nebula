#pragma once
#include <cstdint>

// 全フェーズ共用 Push Constants — Vulkan 保証の 128 bytes ちょうど
struct SimPC {
  // ── Bindless バッファインデックス (32 bytes) ──────────────────────
  uint32_t posIdx;     // P         (vec4 × N)
  uint32_t velIdx;     // v         (vec4 × N)
  uint32_t predPIdx;   // predP     (vec4 × N)
  uint32_t invMassIdx; // invMass   (vec4 × N, .x = invMass)

  uint32_t typeFlagIdx;   // typeFlag  (uint × N)
  uint32_t cellCountIdx;  // cellCount (uint × CELLS)
  uint32_t cellOffsetIdx; // cellOffset(uint × CELLS+NGROUPS)
  uint32_t sortedIdxIdx;  // sortedIdx (uint × N)

  // ── パーティクル / グリッド定数 (16 bytes) ───────────────────────
  uint32_t particleCount;
  uint32_t gridRes;
  uint32_t stretchEdgesIdx; // エッジバッファ (uint×E×3: p,q,restLen)
  uint32_t lambdasIdx;      // XPBD λ (float × E、毎フレームゼロクリア)

  // ── ワールド / 時間 (16 bytes) ───────────────────────────────────
  float dt;
  float cellSize;
  float worldMin;
  float worldMax;

  // ── 環境衝突 (16 bytes) ──────────────────────────────────────────
  float gravity;
  float restitution;
  float friction;
  float particleRadius;

  // ── Phase 3: 布拘束 (32 bytes) ──────────────────────────────────
  uint32_t couplingForceIdx; // vec4×N 流体→布 連成力バッファ (Phase 5; Phase3/4は0)
  uint32_t clothVertexCount; // 布頂点数 (= gridN × gridN)
  uint32_t edgeCount;        // 全エッジ数
  uint32_t batchEdgeStart;   // 現色バッチの開始エッジ番号 (毎バッチ更新)

  uint32_t batchEdgeEnd;  // 現色バッチの終了エッジ番号 (毎バッチ更新)
  uint32_t densityIdx;    // float × N  SPH 密度 ρ_i   (PBF 流体用)
  uint32_t lambdaPbfIdx;  // float × N  PBF λ_i         (PBF 流体用)
  uint32_t boundaryStart; // 境界粒子の開始インデックス  (PBF 流体用)

  float stretchCompliance; // α_stretch (0=剛体)
  float bendCompliance;    // α_bend
  float windX;             // X方向風力
  float windZ;             // Z方向風力

  // ── PBF 流体専用 追加パラメータ (32 bytes) ─────────────────────────
  // 布/連成シェーダーは参照しない。SimPC pc{} のゼロ初期化で安全に無視される。
  float cfmEpsilon;       // CFM 緩和 ε (Macklin&Müller 式11; 0除算防止 + 拘束軟化)
  float scorrK;           // 人工圧力 k (式13; Tensile Instability 対策、0=無効)
  float vorticityEpsilon; // 渦度閉じ込め ε (式16; 0=無効)
  float linearDamping;    // 速度減衰係数 [1/s] (v *= exp(-d*dt); 0=減衰なし)
  uint32_t omegaIdx;         // 渦度 ω バッファ (vec4 × N) のバインドレス index
  float smokeRiseAccel;      // 煙の浮力加速度 [m/s²] (typeFlag==4 に適用)
  float smokeDamping;        // 煙の速度減衰係数 [1/s] (typeFlag==4 に適用)
  uint32_t pinnedTargetIdx;  // アニメーションピン目標位置バッファ (vec4 × N; ClothSceneEngine 専用)
};
static_assert(sizeof(SimPC) == 160, "SimPC must be 160 bytes");
