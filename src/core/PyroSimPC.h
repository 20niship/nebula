#pragma once
#include <cstdint>

// Pyro (グリッドベース煙・火炎) 専用 Push Constants
// (issue #30: 末尾に forceBufIdx/forceCount を追加し 128→136 bytes に拡張。
//  ComputePipeline の push constant range は sizeof(SimPC)=172 bytes で
//  全エンジン共通のため、128超でも既存レイアウトの範囲内で安全)
// MPMSimPC と同様、Bindless バッファインデックス + グリッド定数 + 物理パラメータを
// 固定サイズ構造体に詰める。velocity/density/temperature/fuel は front/back の
// ダブルバッファ (A/B) を持ち、CPU側 (PyroEngine::step) が毎フレーム役割を入れ替える。
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
  uint32_t gridRes;        // 56
  uint32_t emitterCount;   // 60

  // ── World / time (16 bytes) ────────────────────────────────────────────
  float dt;       // 64
  float cellSize; // 68
  float worldMin; // 72
  float worldMax; // 76

  // ── 浮力 / 渦度 (16 bytes) ─────────────────────────────────────────────
  float buoyancyAlpha; // 80  温度浮力係数
  float buoyancyBeta;  // 84  密度による重さ (下向き) 係数
  float ambientTemp;   // 88  環境温度
  float vorticityEps;  // 92  渦度閉じ込め強度 (0=無効)

  // ── 減衰 / 燃焼しきい値 (16 bytes) ─────────────────────────────────────
  float densityDissipation; // 96  密度減衰係数 [1/s]
  float tempDissipation;    // 100 温度減衰係数 [1/s] (環境温度への復帰)
  float ignitionTemp;       // 104 発火温度
  float burnRate;           // 108 燃料消費速度 [1/s]

  // ── 燃焼生成物 (16 bytes) ──────────────────────────────────────────────
  float heatRelease;       // 112 燃焼による温度上昇量 (burn量あたり)
  float smokeYieldPerFuel; // 116 燃焼による密度生成量 (burn量あたり)
  float flameBrightness;   // 120 燃焼による発光量 (burn量あたり)
  uint32_t curlIdx;        // 124 vec4×CELLS 渦度 (curl) スクラッチ (渦度閉じ込め用)

  // ── Force (issue #30; 8 bytes) ─────────────────────────────────────────
  uint32_t forceBufIdx; // 128 Force配列(ForceGPU×forceCount)のbindless index (0=無効)
  uint32_t forceCount;  // 132 有効なForce数
};
static_assert(sizeof(PyroSimPC) == 136, "PyroSimPC must be 136 bytes");
