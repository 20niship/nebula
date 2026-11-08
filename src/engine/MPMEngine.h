#pragma once

#include "ParticleGridEngine.h"

// MLS-MPM: 変形勾配 F + 構成則 (弾性/塑性/弱圧縮流体) による固体・粉体・流体。粒子-格子転写は ParticleGridEngine。
class MPMEngine : public ParticleGridEngine {
public:
  // 初期化時のデフォルトマテリアル(弾性体): Young率 E, ポアソン比 nu (密度は基底の rho0)
  float E  = 1e4f;
  float nu = 0.3f;

  uint32_t plasticModel = 0;      // 0=弾性, 1=VM, 2=DP（Phase 1 まではグローバル）
  float q_max           = 1e5f;   // VM 降伏応力
  float M_friction      = 0.577f; // DP: tan(30°)
  float q_cohesion      = 0.0f;

protected:
  uint32_t defaultSeedCount(const glm::uvec3& gr) const override { return gr.x * gr.y * gr.z; }
  void allocParticleExtras(uint32_t N, const glm::uvec3& gr) override;
  MaterialParams defaultMaterial() const override;
  void onInitParticles(const std::string& shaderDir) override;
  void initNewParticleExtras(uint32_t first, uint32_t count) override;
  void fillPC(MPMSimPC& pc) const override;
};
