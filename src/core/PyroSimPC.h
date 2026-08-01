#pragma once
#include <cstdint>
#include <glm/glm.hpp>

// Pyro (グリッドベース煙・火炎) 専用 Push Constants
// (issue #30: 末尾に forceBufIdx/forceCount を追加。
//  issue #46フォローアップ: 直方体ドメイン対応。worldMin/worldMax を vec3 化し、各軸独立の
//  gridRes(uvec3) を追加した。旧 gridRes(スカラー、Morton dispatch 用の立方体解像度)は
//  hashCells に改名した (MPMSimPC/mpm_p2g.comp/mpm_g2p.comp と同一の命名・用法。値は
//  domain::hashCells(gridRes()) = cubeRes^3)。
//  ComputePipeline の push constant range は sizeof(SimPC)=200 bytes で全エンジン共通
//  (src/core/ComputePipeline.h の static_assert(sizeof(PyroSimPC) <= sizeof(SimPC)) 参照)。
// MPMSimPC と同様、Bindless バッファインデックス + グリッド定数 + 物理パラメータを
// 固定サイズ構造体に詰める。velocity/density/temperature/fuel は front/back の
// ダブルバッファ (A/B) を持ち、CPU側 (PyroEngine::step) が毎フレーム役割を入れ替える。
//
// glm::vec3/uvec3 は GLSL push_constant ブロックの vec3 が要求する 16byte アライメントに
// 合わせるため、各 vec3/uvec3 直後にスカラーフィールドを1個「詰め物」として配置している
// (MPMSimPC.h の gridRes(64)+lambda_lame(76), worldMin(80)+particleVolume(92),
//  worldMax(96)+M_friction(108) と同一のテクニック)。
struct PyroSimPC {
  // ── フィールド バッファ (A/B ダブルバッファ) (16 bytes) ───────────────
  uint32_t velIdxA;     // 0   vec4×CELLS (xyz=速度)
  uint32_t velIdxB;     // 4
  uint32_t densityIdxA; // 8   float×CELLS
  uint32_t densityIdxB; // 12

  // (16 bytes)
  uint32_t temperatureIdxA; // 16  float×CELLS
  uint32_t temperatureIdxB; // 20
  uint32_t fuelIdxA;        // 24  float×CELLS
  uint32_t fuelIdxB;        // 28

  // ── 補助バッファ (16 bytes) ────────────────────────────────────────────
  uint32_t flameIdx;      // 32  float×CELLS (発光量、可視化用メタデータ)
  uint32_t pressureIdxA;  // 36  float×CELLS (Red-Black Gauss-Seidel、in-place更新)
  uint32_t gsColor;       // 40  Red-Black反復の色 (0=red/1=black、旧pressureIdxBを転用)
  uint32_t divergenceIdx; // 44  float×CELLS (スクラッチ)

  // ── コライダー / エミッタ / グリッド定数 (16 bytes) ────────────────────
  uint32_t colliderSDFIdx; // 48  float×CELLS (Morton SDF, 0=無効)
  uint32_t emittersIdx;    // 52  EmitterGPU×emitterCount (0=無効)
  uint32_t hashCells;      // 56  Morton dispatch用の稠密グリッド総セル数 (= cubeRes^3)
  uint32_t emitterCount;   // 60

  // ── グリッド解像度 / World (32 bytes) ────────────────────────────────
  glm::uvec3 gridRes; // 64  各軸の実セル数 (nx,ny,nz) = domain::gridRes(domainSize, cellSize)
  float dt;           // 76

  glm::vec3 worldMin; // 80  ドメイン下限座標 [m] (通常は原点 (0,0,0))
  float cellSize;     // 92  全軸共通のセルサイズ [m]

  glm::vec3 worldMax; // 96  ドメイン上限座標 [m] (= worldMin + domainSize)
  float buoyancyAlpha; // 108 温度浮力係数

  // ── 浮力 / 渦度 (16 bytes) ─────────────────────────────────────────────
  float buoyancyBeta;      // 112 密度による重さ (下向き) 係数
  float ambientTemp;       // 116 環境温度
  float vorticityEps;      // 120 渦度閉じ込め強度 (0=無効)
  float densityDissipation; // 124 密度減衰係数 [1/s]

  // ── 減衰 / 燃焼しきい値 (16 bytes) ─────────────────────────────────────
  float tempDissipation; // 128 温度減衰係数 [1/s] (環境温度への復帰)
  float ignitionTemp;    // 132 発火温度
  float burnRate;        // 136 燃料消費速度 [1/s]
  float heatRelease;     // 140 燃焼による温度上昇量 (burn量あたり)

  // ── 燃焼生成物 / Force (16 bytes) ──────────────────────────────────────
  float smokeYieldPerFuel; // 144 燃焼による密度生成量 (burn量あたり)
  float flameBrightness;   // 148 燃焼による発光量 (burn量あたり)
  uint32_t curlIdx;        // 152 vec4×CELLS 渦度 (curl) スクラッチ (渦度閉じ込め用)
  uint32_t forceBufIdx;    // 156 Force配列(ForceGPU×forceCount)のbindless index (0=無効)

  // ── Force (issue #30; 4 bytes) ─────────────────────────────────────────
  uint32_t forceCount; // 160 有効なForce数
};
static_assert(sizeof(PyroSimPC) == 164, "PyroSimPC must be 164 bytes");
static_assert(offsetof(PyroSimPC, gridRes) == 64, "vec3 alignment offset");
static_assert(offsetof(PyroSimPC, worldMin) == 80, "vec3 alignment offset");
static_assert(offsetof(PyroSimPC, worldMax) == 96, "vec3 alignment offset");
