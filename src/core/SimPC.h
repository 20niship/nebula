#pragma once
#include <cstdint>
#include <glm/glm.hpp>

// 全フェーズ共用 Push Constants。gravity/windX/windZ は issue #30 で Force システム
// (forceBufIdx/forceCount 経由の bindless SSBO) へ移行済み。
//
// issue #46 (直方体ドメイン対応): worldMin/worldMax を vec3 化し、各軸独立の gridRes(uvec3)
// を追加した。offset 20/24/36 (cellCountIdx/cellOffsetIdx/hashCells) は MPMSimPC と完全一致
// させる「hash compat」制約があり (詳細は MPMSimPC.h 参照)、この3フィールドの型・オフセット
// は変更していない。hashCells は旧 gridRes(スカラー) を改名したもので、空間ハッシュバッファ
// の実要素数 (= domain::hashCells()、cubeRes^3。nx*ny*nzではない) を表す。
struct SimPC {
  // ── Bindless バッファインデックス (32 bytes) ──────────────────────
  uint32_t posIdx;     // P         (vec4 × N)
  uint32_t velIdx;     // v         (vec4 × N)
  uint32_t predPIdx;   // predP     (vec4 × N)
  uint32_t invMassIdx; // invMass   (vec4 × N, .x = invMass)

  uint32_t typeFlagIdx;   // typeFlag  (uint × N)
  uint32_t cellCountIdx;  // cellCount (uint × CELLS)      ← hash compat ★
  uint32_t cellOffsetIdx; // cellOffset(uint × CELLS+NGROUPS) ← hash compat ★
  uint32_t sortedIdxIdx;  // sortedIdx (uint × N)

  // ── パーティクル / グリッド定数 (16 bytes) ───────────────────────
  uint32_t particleCount;
  uint32_t hashCells; // 空間ハッシュバッファの実要素数 (= domain::hashCells(gridRes)) ← hash compat ★
  uint32_t stretchEdgesIdx; // エッジバッファ (uint×E×3: p,q,restLen)
  uint32_t lambdasIdx;      // XPBD λ (float × E、毎フレームゼロクリア)

  // ── ワールド / 時間 (16 bytes) ───────────────────────────────────
  float dt;
  float cellSize; // 全軸共通のセルサイズ [m]
  float restitution;
  float friction;

  // ── グリッド解像度 / 境界 (48 bytes) ──────────────────────────────
  glm::uvec3 gridRes; // 各軸の実セル数 (nx,ny,nz) = domain::gridRes(domainSize, cellSize)
  float particleRadius;

  glm::vec3 worldMin; // ドメイン下限座標 [m] (通常は原点 (0,0,0))
  uint32_t forceBufIdx; // Force配列 (ForceGPU×forceCount) の bindless index (0=無効)

  glm::vec3 worldMax; // ドメイン上限座標 [m] (= worldMin + domainSize)
  uint32_t couplingForceIdx; // vec4×N 流体→布 連成力バッファ (Phase 5; Phase3/4は0)

  // ── Phase 3: 布拘束 (16 bytes) ──────────────────────────────────
  uint32_t clothVertexCount; // 布頂点数 (= gridN × gridN)
  uint32_t edgeCount;        // 全エッジ数
  uint32_t batchEdgeStart;   // 現色バッチの開始エッジ番号 (毎バッチ更新)
  uint32_t batchEdgeEnd;     // 現色バッチの終了エッジ番号 (毎バッチ更新)

  uint32_t densityIdx;    // float × N  SPH 密度 ρ_i   (PBF 流体用)
  uint32_t lambdaPbfIdx;  // float × N  PBF λ_i         (PBF 流体用)
  uint32_t boundaryStart; // 境界粒子の開始インデックス  (PBF 流体用)
  float stretchCompliance; // α_stretch (0=剛体)

  float bendCompliance;           // α_bend
  float particleCollisionRadius;  // 旧windX。SoftBodyEngine専用の粒子間衝突半径 (sb_particle_collision.comp)
  uint32_t forceCount;            // 有効な Force 数 (issue #30; 旧windZ の枠を充当)
  float cfmEpsilon;                // CFM 緩和 ε (Macklin&Müller 式11; 0除算防止 + 拘束軟化)

  // ── PBF 流体専用 追加パラメータ (32 bytes) ─────────────────────────
  // 布/連成シェーダーは参照しない。SimPC pc{} のゼロ初期化で安全に無視される。
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

  // ── 泡 (foam/spray/bubble) 二次パーティクル (pbf_foam_generate/pbf_foam_advect 専用;
  //    他シェーダーは宣言のみで不使用。issue #47) ──────────────────────────
  uint32_t foamPosIdx;          // vec4 × maxDiffuseParticles (xyz=位置, w=残り寿命[s])
  uint32_t foamVelIdx;          // vec4 × maxDiffuseParticles (xyz=速度, w=初期寿命[s])
  uint32_t foamKindIdx;         // uint × (maxDiffuseParticles+1)。0=死/未使用,1=spray,2=foam,3=bubble。
                                 // 末尾1要素は生成スロット確保用の atomicAdd カーソル。
  uint32_t foamParamsIdx;       // FoamParams (16 floats) の bindless index
  uint32_t maxDiffuseParticles; // 泡バッファの固定容量 (0=無効)。pbf_foam_advect の境界チェックに使用
                                 // (pc.particleCount は同 substep 内で別用途のため流用不可)
};
static_assert(sizeof(SimPC) == 220, "SimPC must be 220 bytes"); // issue #46 直方体ドメイン対応 (200B) + issue #47 泡 (foamPosIdx等5フィールド, +20B)
static_assert(offsetof(SimPC, cellCountIdx) == 20, "hash compat offset");
static_assert(offsetof(SimPC, cellOffsetIdx) == 24, "hash compat offset");
static_assert(offsetof(SimPC, hashCells) == 36, "hash compat offset");
