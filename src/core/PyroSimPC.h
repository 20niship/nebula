#pragma once
#include <cstdint>

// Pyro (グリッドベース煙・火炎) 専用 Push Constants — 128 bytes
// MPMSimPC と同様、Bindless バッファインデックス + グリッド定数 + 物理パラメータを
// 固定サイズ構造体に詰める。velocity/density/temperature/fuel は front/back の
// ダブルバッファ (A/B) を持ち、CPU側 (PyroEngine::step) が毎フレーム役割を入れ替える。
struct PyroSimPC {
  // ── フィールド バッファ (A/B ダブルバッファ) (16 bytes) ───────────────
  uint32_t velIdxA;         // 0   vec4×CELLS (xyz=速度)
  uint32_t velIdxB;         // 4
  uint32_t densityIdxA;     // 8   float×CELLS
  uint32_t densityIdxB;     // 12

  // (16 bytes)
  uint32_t temperatureIdxA; // 16  float×CELLS
  uint32_t temperatureIdxB; // 20
  uint32_t fuelIdxA;        // 24  float×CELLS
  uint32_t fuelIdxB;        // 28

  // ── 補助バッファ (16 bytes) ────────────────────────────────────────────
  // pressureIdxA/divergenceIdx/curlIdx (下記) は圧力投影フェーズ専用ではなく、
  // 同一ステップ内の移流フェーズ (pyro_advect.comp/pyro_advect_mc.comp) でも
  // MacCormack 中間値のスクラッチとして再利用する (両フェーズは同一ステップ内で
  // 時系列に重ならないため安全)。
  uint32_t flameIdx;        // 32  float×CELLS (発光量、可視化用メタデータ)
  uint32_t pressureIdxA;    // 36  float×CELLS (圧力=Red-Black GS in-place / 移流=density MacCormackスクラッチ)
  uint32_t gsColor;         // 40  0=red/1=black (pyro_pressure_gs.comp 専用、他シェーダーは無視)
  uint32_t divergenceIdx;   // 44  float×CELLS (圧力=divergenceスクラッチ / 移流=temperature MacCormackスクラッチ)

  // ── コライダー / ソース / グリッド定数 (16 bytes) ──────────────────────
  uint32_t colliderSDFIdx;  // 48  float×CELLS (Morton SDF, 0=無効)
  uint32_t sourcesIdx;      // 52  PyroSourceGPU×sourceCount (0=無効)
  uint32_t gridRes;         // 56
  uint32_t sourceCount;     // 60

  // ── World / time (16 bytes) ────────────────────────────────────────────
  float dt;                 // 64
  float cellSize;           // 68
  float worldMin;           // 72
  float worldMax;           // 76

  // ── 浮力 / 渦度 (16 bytes) ─────────────────────────────────────────────
  float buoyancyAlpha;      // 80  温度浮力係数
  float buoyancyBeta;       // 84  密度による重さ (下向き) 係数
  float ambientTemp;        // 88  環境温度
  float vorticityEps;       // 92  渦度閉じ込め強度 (0=無効)

  // ── 減衰 / 燃焼しきい値 (16 bytes) ─────────────────────────────────────
  float densityDissipation; // 96  密度減衰係数 [1/s]
  float tempDissipation;    // 100 温度減衰係数 [1/s] (環境温度への復帰)
  float ignitionTemp;       // 104 発火温度
  float burnRate;           // 108 燃料消費速度 [1/s]

  // ── 燃焼生成物 (16 bytes) ──────────────────────────────────────────────
  float heatRelease;        // 112 燃焼による温度上昇量 (burn量あたり)
  float smokeYieldPerFuel;  // 116 燃焼による密度生成量 (burn量あたり)
  float flameBrightness;    // 120 燃焼による発光量 (burn量あたり)
  uint32_t curlIdx;         // 124 vec4×CELLS (渦度閉じ込め=curlスクラッチ / 移流=velocity MacCormackスクラッチ)
};
static_assert(sizeof(PyroSimPC) == 128, "PyroSimPC must be 128 bytes");
