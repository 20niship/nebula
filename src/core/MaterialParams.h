#pragma once
#include <cstdint>

// マテリアルモデル enum（GLSL の model フィールドと一致させること）
enum class MaterialModel : uint32_t {
  ELASTIC          = 0, // Hencky 弾性 / Fixed-Corotated
  VON_MISES        = 1, // 金属的塑性
  DRUCKER_PRAGER   = 2, // 砂・土
  GRANULAR_POWDER  = 3, // 粉体（DP の低粘着・高摩擦バリアント）
  FLUID            = 4, // 弱圧縮流体（J ベース EOS）
  VISCOPLASTIC_MUD = 5, // 粘塑性泥
};

// GLSL std430 対応: 全フィールドは 4 byte 境界、struct サイズは 16 byte の倍数
// 計 64 bytes
struct MaterialParams {
  float mu;       // 0  ラメ係数 μ = E/(2*(1+ν))
  float lambda;   // 4  ラメ係数 λ = E*ν/((1+ν)*(1-2ν))
  float rho0;     // 8  基準密度 [kg/m^3] → mp = rho0 * Vp
  uint32_t model; // 12 構成則 enum

  float M_friction; // 16 Drucker-Prager M = tan(摩擦角)
  float q_cohesion; // 20 Drucker-Prager 粘着力
  float q_max;      // 24 Von Mises 降伏応力
  float bulkK;      // 28 流体 体積弾性率 K

  float fluidGamma; // 32 流体 EOS 指数 γ（Tait）
  float viscosity;  // 36 粘性係数（流体/泥）
  float hardening;  // 40 塑性硬化係数
  float xi;         // 44 MCC 軟化パラメータ

  float pad0; // 48
  float pad1; // 52
  float pad2; // 56
  float pad3; // 60
              // total: 64 bytes
};
static_assert(sizeof(MaterialParams) == 64, "MaterialParams must be 64 bytes");

// ── プリセット ───────────────────────────────────────────────────────────────

inline float calcMu(float E, float nu) { return E / (2.0f * (1.0f + nu)); }
inline float calcLambda(float E, float nu) { return E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu)); }

// ゼリー状弾性体
inline MaterialParams presetJelly(float E = 1e4f, float nu = 0.4f, float rho0 = 1000.0f) {
  MaterialParams m{};
  m.mu     = calcMu(E, nu);
  m.lambda = calcLambda(E, nu);
  m.rho0   = rho0;
  m.model  = uint32_t(MaterialModel::ELASTIC);
  return m;
}

// 砂（Drucker-Prager, 摩擦角 ~30°）
inline MaterialParams presetSand(float E = 1e5f, float nu = 0.3f, float rho0 = 1600.0f) {
  MaterialParams m{};
  m.mu         = calcMu(E, nu);
  m.lambda     = calcLambda(E, nu);
  m.rho0       = rho0;
  m.model      = uint32_t(MaterialModel::DRUCKER_PRAGER);
  m.M_friction = 0.577f; // tan(30°)
  m.q_cohesion = 0.0f;
  return m;
}

// 弱圧縮流体
inline MaterialParams presetWater(float rho0 = 1000.0f, float bulkK = 5e3f) {
  MaterialParams m{};
  m.mu         = 0.0f;
  m.lambda     = 0.0f;
  m.rho0       = rho0;
  m.model      = uint32_t(MaterialModel::FLUID);
  m.bulkK      = bulkK;
  m.fluidGamma = 7.0f;
  m.viscosity  = 0.01f;
  return m;
}

// 粘塑性泥
inline MaterialParams presetMud(float E = 8e4f, float nu = 0.35f, float rho0 = 1800.0f) {
  MaterialParams m{};
  m.mu         = calcMu(E, nu);
  m.lambda     = calcLambda(E, nu);
  m.rho0       = rho0;
  m.model      = uint32_t(MaterialModel::VISCOPLASTIC_MUD);
  m.M_friction = 0.4f;
  m.q_cohesion = 5e2f;
  m.viscosity  = 0.1f;
  return m;
}

// 粉体（高摩擦・低粘着 DP バリアント）
inline MaterialParams presetPowder(float E = 5e4f, float nu = 0.25f, float rho0 = 1200.0f) {
  MaterialParams m{};
  m.mu         = calcMu(E, nu);
  m.lambda     = calcLambda(E, nu);
  m.rho0       = rho0;
  m.model      = uint32_t(MaterialModel::GRANULAR_POWDER);
  m.M_friction = 0.8f;  // 高摩擦
  m.q_cohesion = 50.0f; // 微小粘着
  return m;
}
