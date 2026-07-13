#pragma once
#include "AttributeBuffer.h"
#include "ComputePipeline.h"
#include "HeadlessCtx.h"
#include "SimPC.h"
#include <glm/glm.hpp>
#include <string>
#include <vector>

// GPU PBF fluid test harness (headless GPU).
// typeFlag=1 fluid + optional typeFlag=3 boundary particles.
class PBFHarness {
public:
  struct Config {
    uint32_t N             = 8;
    uint32_t boundaryCount = 0; // auto-set from boundaryPos in init()
    float worldMin         = 0.0f;
    float worldMax         = 10.0f;
    uint32_t gridRes       = 16; // h = worldSize / gridRes
    float gravity          = -9.8f;
    float rho0             = 1000.0f;
    float viscosityC       = 0.01f;
    float restitution      = 0.1f;
    int pbfIterations      = 4;
    int numSubsteps        = 4;
  };

  // boundaryPos: typeFlag=3, invMass=0 boundary particles (optional)
  void init(const HeadlessCtx& ctx, const Config& cfg, const std::string& shaderDir, const std::vector<glm::vec4>& initPos, const std::vector<glm::vec4>& initInvMass, const std::vector<glm::vec4>& boundaryPos = {});

  void step(float dt);

  glm::vec4 readPos(uint32_t i) const;
  glm::vec4 readVel(uint32_t i) const;
  float readDensity(uint32_t i) const;

  void cleanup();

private:
  const HeadlessCtx* ctx_ = nullptr;
  Config cfg_;
  uint32_t nCells_ = 0;

  AttributeBuffer attrBuf_;
  uint32_t posIdx_     = 0;
  uint32_t velIdx_     = 0;
  uint32_t predPIdx_   = 0;
  uint32_t invMIdx_    = 0;
  uint32_t typeFIdx_   = 0;
  uint32_t cellCntIdx_ = 0;
  uint32_t cellOffIdx_ = 0;
  uint32_t sortedIdx_  = 0;
  uint32_t densityIdx_ = 0;
  uint32_t lambdaIdx_  = 0;

  ComputePipeline kPredict_;
  ComputePipeline kSdf_;
  ComputePipeline kHashCnt_;
  ComputePipeline kScanLoc_;
  ComputePipeline kScanGlob_;
  ComputePipeline kSort_;
  ComputePipeline kPbfDensity_;
  ComputePipeline kPbfDeltaP_;
  ComputePipeline kPbfViscosity_;
  ComputePipeline kUpdateVel_;

  void computeBarrier(VkCommandBuffer cmd);
  void fillBarrier(VkCommandBuffer cmd, VkBuffer buf);
  void recordSubstep(VkCommandBuffer cmd, float subDt);
};
