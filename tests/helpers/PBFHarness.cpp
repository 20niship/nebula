#include "PBFHarness.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

void PBFHarness::computeBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier b{};
  b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void PBFHarness::fillBarrier(VkCommandBuffer cmd, VkBuffer buf) {
  VkBufferMemoryBarrier b{};
  b.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  b.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
  b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.buffer              = buf;
  b.offset              = 0;
  b.size                = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &b, 0, nullptr);
}

void PBFHarness::init(const HeadlessCtx& ctx, const Config& cfg, const std::string& shaderDir, const std::vector<glm::vec4>& initPos, const std::vector<glm::vec4>& initInvMass, const std::vector<glm::vec4>& boundaryPos) {
  ctx_               = &ctx;
  cfg_               = cfg;
  cfg_.boundaryCount = (uint32_t)boundaryPos.size();

  uint32_t totalN  = cfg_.N + cfg_.boundaryCount;
  nCells_          = cfg_.gridRes * cfg_.gridRes * cfg_.gridRes;
  uint32_t nGroups = (nCells_ + 255u) / 256u;

  attrBuf_.init(ctx.device, ctx.allocator, ctx.descriptorPool);

  posIdx_     = attrBuf_.addAttribute("P", sizeof(glm::vec4), totalN);
  velIdx_     = attrBuf_.addAttribute("v", sizeof(glm::vec4), totalN);
  predPIdx_   = attrBuf_.addAttribute("predP", sizeof(glm::vec4), totalN);
  invMIdx_    = attrBuf_.addAttribute("invMass", sizeof(glm::vec4), totalN);
  typeFIdx_   = attrBuf_.addAttribute("typeFlag", sizeof(uint32_t), totalN);
  cellCntIdx_ = attrBuf_.addAttribute("cellCnt", sizeof(uint32_t), nCells_);
  cellOffIdx_ = attrBuf_.addAttribute("cellOff", sizeof(uint32_t), nCells_ + nGroups);
  sortedIdx_  = attrBuf_.addAttribute("sorted", sizeof(uint32_t), totalN);
  densityIdx_ = attrBuf_.addAttribute("density", sizeof(float), totalN);
  lambdaIdx_  = attrBuf_.addAttribute("lambdaPbf", sizeof(float), totalN);

  auto pool  = ctx.commandPool;
  auto queue = ctx.computeQueue;

  // Fluid particle positions
  attrBuf_.upload("P", initPos.data(), sizeof(glm::vec4) * cfg_.N, pool, queue);
  attrBuf_.upload("predP", initPos.data(), sizeof(glm::vec4) * cfg_.N, pool, queue);

  // Velocity: zero for all particles
  std::vector<glm::vec4> zeroVec(totalN, glm::vec4(0.0f));
  attrBuf_.upload("v", zeroVec.data(), sizeof(glm::vec4) * totalN, pool, queue);

  // invMass: fluid particles use initInvMass, boundary get 0
  std::vector<glm::vec4> fullInvM(totalN, glm::vec4(0.0f));
  for(uint32_t k = 0; k < cfg_.N; ++k) fullInvM[k] = initInvMass[k];
  attrBuf_.upload("invMass", fullInvM.data(), sizeof(glm::vec4) * totalN, pool, queue);

  // typeFlag: fluid=1, boundary=3
  std::vector<uint32_t> typeFlags(totalN, 1u);
  for(uint32_t k = cfg_.N; k < totalN; ++k) typeFlags[k] = 3u;
  attrBuf_.upload("typeFlag", typeFlags.data(), sizeof(uint32_t) * totalN, pool, queue);

  // Boundary particle positions
  if(!boundaryPos.empty()) {
    uint32_t nb = cfg_.boundaryCount;
    attrBuf_.uploadAt("P", boundaryPos.data(), sizeof(glm::vec4) * nb, cfg_.N * sizeof(glm::vec4), pool, queue);
    attrBuf_.uploadAt("predP", boundaryPos.data(), sizeof(glm::vec4) * nb, cfg_.N * sizeof(glm::vec4), pool, queue);
  }

  auto load = [&](ComputePipeline& k, const char* name) { k.init(ctx.device, attrBuf_.descriptorSetLayout, shaderDir + "/" + name); };
  load(kPredict_, "predict.comp.spv");
  load(kSdf_, "sdf_collision.comp.spv");
  load(kHashCnt_, "hash_count.comp.spv");
  load(kScanLoc_, "hash_scan_local.comp.spv");
  load(kScanGlob_, "hash_scan_global.comp.spv");
  load(kSort_, "hash_sort.comp.spv");
  load(kPbfDensity_, "pbf_density.comp.spv");
  load(kPbfDeltaP_, "pbf_delta_p.comp.spv");
  load(kPbfViscosity_, "pbf_viscosity.comp.spv");
  load(kUpdateVel_, "update_velocity.comp.spv");
}

void PBFHarness::recordSubstep(VkCommandBuffer cmd, float subDt) {
  auto ds         = attrBuf_.descriptorSet;
  uint32_t totalN = cfg_.N + cfg_.boundaryCount;
  float h         = (cfg_.worldMax - cfg_.worldMin) / float(cfg_.gridRes);

  SimPC pc{};
  pc.posIdx            = posIdx_;
  pc.velIdx            = velIdx_;
  pc.predPIdx          = predPIdx_;
  pc.invMassIdx        = invMIdx_;
  pc.typeFlagIdx       = typeFIdx_;
  pc.cellCountIdx      = cellCntIdx_;
  pc.cellOffsetIdx     = cellOffIdx_;
  pc.sortedIdxIdx      = sortedIdx_;
  pc.particleCount     = totalN;
  pc.gridRes           = cfg_.gridRes;
  pc.stretchEdgesIdx   = 0;
  pc.lambdasIdx        = 0;
  pc.dt                = subDt;
  pc.cellSize          = h;
  pc.worldMin          = cfg_.worldMin;
  pc.worldMax          = cfg_.worldMax;
  pc.gravity           = cfg_.gravity;
  pc.restitution       = cfg_.restitution;
  pc.friction          = 0.05f;
  pc.particleRadius    = h * 0.5f;
  pc.couplingForceIdx  = 0;
  pc.clothVertexCount  = totalN;
  pc.edgeCount         = 0;
  pc.batchEdgeStart    = 0;
  pc.batchEdgeEnd      = 0;
  pc.stretchCompliance = cfg_.rho0;
  pc.bendCompliance    = cfg_.viscosityC;
  pc.windX             = 0.0f;
  pc.windZ             = 0.0f;
  pc.densityIdx        = densityIdx_;
  pc.lambdaPbfIdx      = lambdaIdx_;
  pc.boundaryStart     = cfg_.N;
  pc.cfmEpsilon        = cfg_.cfmEpsilon;
  pc.relaxOmega        = cfg_.relaxOmega;

  // 1. Predict (boundary particles with invMass=0 get pinned: predP = P)
  kPredict_.dispatch(cmd, ds, pc, totalN);
  computeBarrier(cmd);

  // 2. SDF collision
  kSdf_.dispatch(cmd, ds, pc, totalN);
  computeBarrier(cmd);

  // 3. Spatial hash
  vkCmdFillBuffer(cmd, attrBuf_.getBuffer("cellCnt"), 0, VK_WHOLE_SIZE, 0);
  fillBarrier(cmd, attrBuf_.getBuffer("cellCnt"));

  kHashCnt_.dispatch(cmd, ds, pc, totalN);
  computeBarrier(cmd);

  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kScanLoc_.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kScanLoc_.pipelineLayout, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, kScanLoc_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
    vkCmdDispatch(cmd, (nCells_ + 255u) / 256u, 1, 1);
  }
  computeBarrier(cmd);

  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kScanGlob_.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kScanGlob_.pipelineLayout, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, kScanGlob_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
    vkCmdDispatch(cmd, 1, 1, 1);
  }
  computeBarrier(cmd);

  kSort_.dispatch(cmd, ds, pc, totalN);
  computeBarrier(cmd);

  // 4. PBF incompressibility solver x pbfIterations
  for(int iter = 0; iter < cfg_.pbfIterations; ++iter) {
    kPbfDensity_.dispatch(cmd, ds, pc, totalN);
    computeBarrier(cmd);
    kPbfDeltaP_.dispatch(cmd, ds, pc, totalN);
    computeBarrier(cmd);
  }

  // 5. SDF re-apply
  kSdf_.dispatch(cmd, ds, pc, totalN);
  computeBarrier(cmd);

  // 6. Velocity update
  kUpdateVel_.dispatch(cmd, ds, pc, totalN);
  computeBarrier(cmd);

  // 7. XSPH viscosity
  kPbfViscosity_.dispatch(cmd, ds, pc, totalN);
  computeBarrier(cmd);
}

void PBFHarness::step(float dt) {
  float subDt         = dt / float(cfg_.numSubsteps);
  VkCommandBuffer cmd = ctx_->beginCmd();
  for(int s = 0; s < cfg_.numSubsteps; ++s) recordSubstep(cmd, subDt);
  ctx_->submitCmd(cmd);
}

glm::vec4 PBFHarness::readPos(uint32_t i) const {
  glm::vec4 r{};
  ctx_->readBuffer(attrBuf_.getBuffer("P"), i * sizeof(glm::vec4), &r, sizeof(r));
  return r;
}

glm::vec4 PBFHarness::readVel(uint32_t i) const {
  glm::vec4 r{};
  ctx_->readBuffer(attrBuf_.getBuffer("v"), i * sizeof(glm::vec4), &r, sizeof(r));
  return r;
}

float PBFHarness::readDensity(uint32_t i) const {
  float r = 0.0f;
  ctx_->readBuffer(attrBuf_.getBuffer("density"), i * sizeof(float), &r, sizeof(r));
  return r;
}

void PBFHarness::cleanup() {
  kPredict_.cleanup();
  kSdf_.cleanup();
  kHashCnt_.cleanup();
  kScanLoc_.cleanup();
  kScanGlob_.cleanup();
  kSort_.cleanup();
  kPbfDensity_.cleanup();
  kPbfDeltaP_.cleanup();
  kPbfViscosity_.cleanup();
  kUpdateVel_.cleanup();
  attrBuf_.cleanup();
}
