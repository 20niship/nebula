#pragma once
#include <cstdint>

// 全フェーズ共用 Push Constants。gravity/windX/windZ は issue #30 で Force システム
// (forceBufIdx/forceCount 経由の bindless SSBO) へ移行済み。
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

  // ── 環境衝突 (16 bytes; gravity は issue #30 で Force システムへ移行し削除、
  //    空いた1枠は forceBufIdx に充当) ────────────────────────────────
  float restitution;
  float friction;
  float particleRadius;
  uint32_t forceBufIdx; // Force配列 (ForceGPU×forceCount) の bindless index (0=無効)

  // ── Phase 3: 布拘束 (32 bytes) ──────────────────────────────────
  uint32_t couplingForceIdx; // vec4×N 流体→布 連成力バッファ (Phase 5; Phase3/4は0)
  uint32_t clothVertexCount; // 布頂点数 (= gridN × gridN)
  uint32_t edgeCount;        // 全エッジ数
  uint32_t batchEdgeStart;   // 現色バッチの開始エッジ番号 (毎バッチ更新)

  uint32_t batchEdgeEnd;  // 現色バッチの終了エッジ番号 (毎バッチ更新)
  uint32_t densityIdx;    // float × N  SPH 密度 ρ_i   (PBF 流体用)
  uint32_t lambdaPbfIdx;  // float × N  PBF λ_i         (PBF 流体用)
  uint32_t boundaryStart; // 境界粒子の開始インデックス  (PBF 流体用)

  float stretchCompliance;        // α_stretch (0=剛体)
  float bendCompliance;           // α_bend
  float particleCollisionRadius;  // 旧windX。SoftBodyEngine専用の粒子間衝突半径 (sb_particle_collision.comp)
  uint32_t forceCount;            // 有効な Force 数 (issue #30; 旧windZ の枠を充当)

  // ── PBF 流体専用 追加パラメータ (32 bytes) ─────────────────────────
  // 布/連成シェーダーは参照しない。SimPC pc{} のゼロ初期化で安全に無視される。
  float cfmEpsilon;         // CFM 緩和 ε (Macklin&Müller 式11; 0除算防止 + 拘束軟化)
  float scorrK;             // 人工圧力 k (式13; Tensile Instability 対策、0=無効)
  float vorticityEpsilon;   // 渦度閉じ込め ε (式16; 0=無効)
  float linearDamping;      // 速度減衰係数 [1/s] (v *= exp(-d*dt); 0=減衰なし)
  uint32_t omegaIdx;        // 渦度 ω バッファ (vec4 × N) のバインドレス index
  float smokeRiseAccel;     // 煙の浮力加速度 [m/s²] (typeFlag==4 に適用)
  float smokeDamping;       // 煙の速度減衰係数 [1/s] (typeFlag==4 に適用)
  uint32_t pinnedTargetIdx; // アニメーションピン目標位置バッファ (vec4 × N; ClothSceneEngine 専用)

  // ── 吸収ポート (fluid_absorb 専用; 他シェーダーは宣言のみで不使用) ──────
  uint32_t absorberBufIdx; // 吸収形状バッファの bindless index (8 floats × absorberCount)
  uint32_t absorberCount;  // 有効な吸収形状数 (0 = 吸収パスをスキップ)

  uint32_t fluidStart; // 流体パーティクル領域の開始オフセット (= FluidEngine の cfg_.max_boundary)。
                        // predict_sdf/sdf_collision/pbf_density/pbf_delta_p/update_velocity/
                        // pbf_viscosity/vorticity/absorb など「流体のみ」を対象とする FluidEngine
                        // 専用ディスパッチで i に加算し実バッファ添字を得る。既定値0=オフセットなし
                        // のため、このフィールドを設定しない他エンジンは影響を受けない。
};
static_assert(sizeof(SimPC) == 172, "SimPC must be 172 bytes");
