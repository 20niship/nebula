#include "FLIPEngine.h"
#include "../core/Profiling.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<std::pair<std::string, std::string>> FLIPEngine::shaderDefines() const {
  return {{"NO_DEFORMATION", "1"}, {"GRID_VMAX", std::to_string(maxGridSpeed)}};
}

void FLIPEngine::onInitParticles(const std::string& shaderDir) {
  const uint32_t NC = pc_.hashCells;
  pc_.flipGridIdx   = attrBuf_.addAttribute("flipGrid", sizeof(float), NC * 3);

  auto load = [&](ComputePipeline& k, const char* name) { k.init(device_, attrBuf_.descriptorSetLayout, shaderDir + "/" + name + ".spv"); };
  load(kClassify_, "flip_classify.comp");
  load(kDivergence_, "flip_divergence.comp");
  load(kPressure_, "flip_pressure_rb.comp");
  load(kProject_, "flip_project.comp");

  if(maxDiffuseParticles == 0) return;
  const uint32_t M = maxDiffuseParticles;
  foamPosIdx_      = attrBuf_.addAttribute("foamPos", sizeof(glm::vec4), M);
  foamVelIdx_      = attrBuf_.addAttribute("foamVel", sizeof(glm::vec4), M);
  foamKindIdx_     = attrBuf_.addAttribute("foamKind", sizeof(uint32_t), M + 1); // 末尾 1 要素は生成カーソル
  foamParamsIdx_   = attrBuf_.addAttribute("foamParams", sizeof(uint32_t), 32);
  load(kFoamGenerate_, "flip_foam_generate.comp");
  load(kFoamAdvect_, "flip_foam_advect.comp");
  resetParticles();
  setFoamParams(foamParams_);
}

void FLIPEngine::resetParticles() {
  ParticleGridEngine::resetParticles();
  if(maxDiffuseParticles == 0 || foamKindIdx_ == 0) return;
  const std::vector<uint32_t> zero(maxDiffuseParticles + 1, 0u);
  attrBuf_.upload("foamKind", zero.data(), zero.size() * sizeof(uint32_t), cmdPool_, queue_);
}

void FLIPEngine::setFoamParams(const FoamParams& p) {
  foamParams_ = p;
  if(foamParamsIdx_ == 0) return;
  std::vector<uint32_t> w(32, 0u);
  auto f = [&](size_t i, float v) { std::memcpy(&w[i], &v, sizeof(float)); };
  f(0, p.kTa);
  f(1, p.kWc);
  f(2, p.taLo);
  f(3, p.taHi);
  f(4, p.wcLo);
  f(5, p.wcHi);
  f(6, p.keLo);
  f(7, p.keHi);
  f(8, p.lifetimeMin);
  f(9, p.lifetimeMax);
  f(10, p.bubbleBuoyancy);
  f(11, p.dragCoeff);
  f(12, p.gravity.x);
  f(13, p.gravity.y);
  f(14, p.gravity.z);
  f(15, p.densityLo);
  f(16, p.densityHi);
  f(17, p.surfaceDensityRatio);
  w[18] = foamPosIdx_;
  w[19] = foamVelIdx_;
  w[20] = foamKindIdx_;
  w[21] = maxDiffuseParticles;
  attrBuf_.upload("foamParams", w.data(), w.size() * sizeof(uint32_t), cmdPool_, queue_);
}

void FLIPEngine::fillPC(MPMSimPC& pc) const { pc.xi_hard = densityCorrection; }

void FLIPEngine::projectGrid(VkCommandBuffer cmd, const MPMSimPC& pc) {
  const uint32_t NC = pc.hashCells;
  dispatchMPM(cmd, kClassify_, pc, NC);
  computeBarrier(cmd);
  dispatchMPM(cmd, kDivergence_, pc, NC);
  computeBarrier(cmd);
  for(int it = 0; it < pressureIterations; ++it) {
    for(uint32_t color = 0; color < 2; ++color) {
      MPMSimPC pcc = pc;
      pcc.pass     = color;
      dispatchMPM(cmd, kPressure_, pcc, NC);
      computeBarrier(cmd);
    }
  }
  dispatchMPM(cmd, kProject_, pc, NC);
  computeBarrier(cmd);
  syncGpuForProfiling(cmd);
}

// 泡パスは flipGridIdx に foamParams を、pass に乱数ソルトを渡す (圧力グリッドは不要なため)
void FLIPEngine::afterG2P(VkCommandBuffer cmd, const MPMSimPC& pc) {
  if(!foamEnabled || maxDiffuseParticles == 0) return;
  MPMSimPC pcf = pc;
  pcf.flipGridIdx = foamParamsIdx_;
  pcf.pass        = ++foamSalt_;
  dispatchMPM(cmd, kFoamGenerate_, pcf, pc.particleCount);
  computeBarrier(cmd);
  dispatchMPM(cmd, kFoamAdvect_, pcf, maxDiffuseParticles);
  computeBarrier(cmd);
  syncGpuForProfiling(cmd);
}

void FLIPEngine::cleanupDerived() {
  for(auto* k : {&kClassify_, &kDivergence_, &kPressure_, &kProject_, &kFoamGenerate_, &kFoamAdvect_}) k->cleanup();
}
