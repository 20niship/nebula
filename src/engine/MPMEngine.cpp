#include "MPMEngine.h"

#include <algorithm>
#include <vector>

// F0-2: xyz = 変形勾配 F の列, w = 対角応力 (σ_xx / σ_yy / σ_zz)
// B0-2: xyz = APIC アフィン行列 B の列, w = 非対角応力 (σ_xy / σ_xz / σ_yz)
void MPMEngine::allocParticleExtras(uint32_t N, const glm::uvec3&) {
  pc_.F0Idx       = attrBuf_.addAttribute("F0", sizeof(glm::vec4), N);
  pc_.F1Idx       = attrBuf_.addAttribute("F1", sizeof(glm::vec4), N);
  pc_.F2Idx       = attrBuf_.addAttribute("F2", sizeof(glm::vec4), N);
  pc_.B0Idx       = attrBuf_.addAttribute("B0", sizeof(glm::vec4), N);
  pc_.B1Idx       = attrBuf_.addAttribute("B1", sizeof(glm::vec4), N);
  pc_.B2Idx       = attrBuf_.addAttribute("B2", sizeof(glm::vec4), N);
  pc_.mu_lame     = E / (2.0f * (1.0f + nu));
  pc_.lambda_lame = E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));
}

MaterialParams MPMEngine::defaultMaterial() const {
  MaterialParams m = presetJelly(E, nu, rho0);
  m.model          = 0; // ELASTIC をデフォルト
  m.M_friction     = M_friction;
  m.q_cohesion     = q_cohesion;
  m.q_max          = q_max;
  return m;
}

// maxParticles 未指定のとき、ドメイン中央付近に gridRes 解像度のブロックをシードする
void MPMEngine::onInitParticles(const std::string&) {
  if(nParticles_ == 0) return;
  const glm::uvec3 gr   = pc_.gridRes;
  const float blockFrac = 0.40f;
  const float minDomain = std::min({domainSize.x, domainSize.y, domainSize.z});
  const float sp        = minDomain * blockFrac / float(std::max({gr.x, gr.y, gr.z}));
  const float Vp        = sp * sp * sp;

  const float cx = domainSize.x * 0.5f;
  const float cz = domainSize.z * 0.5f;
  const float cy = domainSize.y * 0.70f;

  const float halfX = sp * float(gr.x - 1) / 2.0f;
  const float halfY = sp * float(gr.y - 1) / 2.0f;
  const float halfZ = sp * float(gr.z - 1) / 2.0f;

  std::vector<glm::vec4> pos(nParticles_);
  std::vector<glm::vec4> vel(nParticles_, glm::vec4(0.0f));
  uint32_t idx = 0;
  for(uint32_t iz = 0; iz < gr.z; iz++)
    for(uint32_t iy = 0; iy < gr.y; iy++)
      for(uint32_t ix = 0; ix < gr.x; ix++, idx++)
        pos[idx] = glm::vec4(cx - halfX + ix * sp, cy - halfY + iy * sp, cz - halfZ + iz * sp, Vp);
  uploadNewParticles_(pos.data(), vel.data(), nParticles_, 0);
}

void MPMEngine::initNewParticleExtras(uint32_t first, uint32_t count) {
  const VkDeviceSize byteOff = VkDeviceSize(first) * sizeof(glm::vec4);
  auto up                    = [&](const char* name, const std::vector<glm::vec4>& v) {
    attrBuf_.uploadAt(name, v.data(), VkDeviceSize(count) * sizeof(glm::vec4), byteOff, cmdPool_, queue_);
  };
  up("F0", std::vector<glm::vec4>(count, glm::vec4(1, 0, 0, 0)));
  up("F1", std::vector<glm::vec4>(count, glm::vec4(0, 1, 0, 0)));
  up("F2", std::vector<glm::vec4>(count, glm::vec4(0, 0, 1, 0)));
  const std::vector<glm::vec4> zero(count, glm::vec4(0));
  up("B0", zero);
  up("B1", zero);
  up("B2", zero);
}

void MPMEngine::fillPC(MPMSimPC& pc) const {
  pc.M_friction   = M_friction;
  pc.q_cohesion   = q_cohesion;
  pc.q_max        = q_max;
  pc.plasticModel = plasticModel;
}
