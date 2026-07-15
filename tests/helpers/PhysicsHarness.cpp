#include "PhysicsHarness.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

// ── Helpers ──────────────────────────────────────────────────────────────────

void PhysicsHarness::computeBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier b{};
  b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void PhysicsHarness::fillBarrier(VkCommandBuffer cmd, VkBuffer buf) {
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

// ── buildChainEdges ───────────────────────────────────────────────────────────

void PhysicsHarness::buildChainEdges(uint32_t N, float spacing, std::vector<uint32_t>& ed, uint32_t& c0End) {
  auto fb = [](float f) -> uint32_t {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
  };
  ed.clear();
  // Color 0: even edges (0-1, 2-3, 4-5, …) — no shared particles per batch
  for(uint32_t i = 0; i + 1 < N; i += 2) {
    ed.push_back(i);
    ed.push_back(i + 1);
    ed.push_back(fb(spacing));
  }
  c0End = static_cast<uint32_t>(ed.size() / 3);
  // Color 1: odd edges (1-2, 3-4, 5-6, …)
  for(uint32_t i = 1; i + 1 < N; i += 2) {
    ed.push_back(i);
    ed.push_back(i + 1);
    ed.push_back(fb(spacing));
  }
}

// ── init ─────────────────────────────────────────────────────────────────────

void PhysicsHarness::init(const HeadlessCtx& ctx, const Config& cfg, const std::string& shaderDir, const std::vector<glm::vec4>& initPos, const std::vector<glm::vec4>& initInvMass) {
  ctx_ = &ctx;
  cfg_ = cfg;

  nEdges_  = static_cast<uint32_t>(cfg.edgeData.size() / 3);
  nCells_  = cfg.gridRes * cfg.gridRes * cfg.gridRes;
  nGroups_ = (nCells_ + 255u) / 256u;

  attrBuf_.init(ctx.device, ctx.allocator, ctx.descriptorPool);

  posIdx_     = attrBuf_.addAttribute("P", sizeof(glm::vec4), cfg.N);
  velIdx_     = attrBuf_.addAttribute("v", sizeof(glm::vec4), cfg.N);
  predPIdx_   = attrBuf_.addAttribute("predP", sizeof(glm::vec4), cfg.N);
  invMIdx_    = attrBuf_.addAttribute("invMass", sizeof(glm::vec4), cfg.N);
  typeFIdx_   = attrBuf_.addAttribute("typeFlag", sizeof(uint32_t), cfg.N);
  cellCntIdx_ = attrBuf_.addAttribute("cellCnt", sizeof(uint32_t), nCells_);
  cellOffIdx_ = attrBuf_.addAttribute("cellOff", sizeof(uint32_t), nCells_ + nGroups_);
  sortedIdx_  = attrBuf_.addAttribute("sorted", sizeof(uint32_t), cfg.N);
  // Edge and lambda buffers need at least 1 element even when unused
  edgesIdx_  = attrBuf_.addAttribute("edges", sizeof(uint32_t), std::max(nEdges_ * 3u, 3u));
  lambdaIdx_ = attrBuf_.addAttribute("lambdas", sizeof(float), std::max(nEdges_, 1u));

  auto pool  = ctx.commandPool;
  auto queue = ctx.computeQueue;

  attrBuf_.upload("P", initPos.data(), sizeof(glm::vec4) * cfg.N, pool, queue);
  attrBuf_.upload("predP", initPos.data(), sizeof(glm::vec4) * cfg.N, pool, queue);

  std::vector<glm::vec4> zeroVec(cfg.N, glm::vec4(0.0f));
  attrBuf_.upload("v", zeroVec.data(), sizeof(glm::vec4) * cfg.N, pool, queue);
  attrBuf_.upload("invMass", initInvMass.data(), sizeof(glm::vec4) * cfg.N, pool, queue);

  std::vector<uint32_t> typeFlags(cfg.N, cfg.typeFlag);
  attrBuf_.upload("typeFlag", typeFlags.data(), sizeof(uint32_t) * cfg.N, pool, queue);

  if(nEdges_ > 0) {
    attrBuf_.upload("edges", cfg.edgeData.data(), sizeof(uint32_t) * cfg.edgeData.size(), pool, queue);
  }
  std::vector<float> zeros(std::max(nEdges_, 1u), 0.0f);
  attrBuf_.upload("lambdas", zeros.data(), sizeof(float) * zeros.size(), pool, queue);

  // Load shaders
  auto load = [&](ComputePipeline& k, const char* name) { k.init(ctx.device, attrBuf_.descriptorSetLayout, shaderDir + "/" + name); };
  load(kPredict_, "predict.comp.spv");
  load(kSdf_, "sdf_collision.comp.spv");
  load(kHashCnt_, "hash_count.comp.spv");
  load(kScanLoc_, "hash_scan_local.comp.spv");
  load(kScanGlob_, "hash_scan_global.comp.spv");
  load(kAddBase_, "hash_add_base.comp.spv");
  load(kSort_, "hash_sort.comp.spv");
  load(kSolveDen_, "solve_density.comp.spv");
  load(kSolveSt_, "solve_stretch.comp.spv");
  load(kUpdateVel_, "update_velocity.comp.spv");
}

// ── recordSubstep ─────────────────────────────────────────────────────────────

void PhysicsHarness::recordSubstep(VkCommandBuffer cmd, float subDt) {
  auto ds = attrBuf_.descriptorSet;
  auto N  = cfg_.N;

  SimPC pc{};
  pc.posIdx            = posIdx_;
  pc.velIdx            = velIdx_;
  pc.predPIdx          = predPIdx_;
  pc.invMassIdx        = invMIdx_;
  pc.typeFlagIdx       = typeFIdx_;
  pc.cellCountIdx      = cellCntIdx_;
  pc.cellOffsetIdx     = cellOffIdx_;
  pc.sortedIdxIdx      = sortedIdx_;
  pc.particleCount     = N;
  pc.gridRes           = cfg_.gridRes;
  pc.stretchEdgesIdx   = edgesIdx_;
  pc.lambdasIdx        = lambdaIdx_;
  pc.dt                = subDt;
  pc.cellSize          = (cfg_.worldMax - cfg_.worldMin) / float(cfg_.gridRes);
  pc.worldMin          = cfg_.worldMin;
  pc.worldMax          = cfg_.worldMax;
  pc.gravity           = cfg_.gravity;
  pc.restitution       = cfg_.restitution;
  pc.friction          = cfg_.friction;
  pc.particleRadius    = cfg_.radius;
  pc.couplingForceIdx  = 0;
  pc.clothVertexCount  = N;
  pc.edgeCount         = nEdges_;
  pc.stretchCompliance = cfg_.stretchCompliance;
  pc.windX             = cfg_.windX;
  pc.windZ             = cfg_.windZ;
  pc.bendCompliance    = 0.0f;
  pc.densityIdx = pc.lambdaPbfIdx = pc.boundaryStart = 0;
  pc.batchEdgeStart                                  = 0;
  pc.batchEdgeEnd                                    = 0;

  // ① Predict (gravity + wind)
  kPredict_.dispatch(cmd, ds, pc, N);
  computeBarrier(cmd);

  // ② SDF boundary collision
  kSdf_.dispatch(cmd, ds, pc, N);
  computeBarrier(cmd);

  // ③ Stretch constraints (if edges exist)
  if(nEdges_ > 0) {
    vkCmdFillBuffer(cmd, attrBuf_.getBuffer("lambdas"), 0, VK_WHOLE_SIZE, 0);
    fillBarrier(cmd, attrBuf_.getBuffer("lambdas"));

    uint32_t c0End = cfg_.edgeColor0End;
    uint32_t total = nEdges_;

    for(int iter = 0; iter < cfg_.solverIterations; ++iter) {
      auto dispatchColor = [&](uint32_t start, uint32_t end) {
        if(end <= start) return;
        pc.batchEdgeStart = start;
        pc.batchEdgeEnd   = end;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kSolveSt_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kSolveSt_.pipelineLayout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, kSolveSt_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
        vkCmdDispatch(cmd, ((end - start) + 255u) / 256u, 1, 1);
        computeBarrier(cmd);
      };
      dispatchColor(0, c0End);
      dispatchColor(c0End, total);
    }

    // Re-apply SDF after stretch
    kSdf_.dispatch(cmd, ds, pc, N);
    computeBarrier(cmd);
  }

  // ④ Self-collision: spatial hash + solve_density
  if(cfg_.selfCollision) {
    vkCmdFillBuffer(cmd, attrBuf_.getBuffer("cellCnt"), 0, VK_WHOLE_SIZE, 0);
    fillBarrier(cmd, attrBuf_.getBuffer("cellCnt"));

    kHashCnt_.dispatch(cmd, ds, pc, N);
    computeBarrier(cmd);
    kScanLoc_.dispatch(cmd, ds, pc, nCells_); // dispatch one workgroup per 256 cells
    computeBarrier(cmd);
    kScanGlob_.dispatch(cmd, ds, pc, nGroups_); // one workgroup (1024 threads) for all groups
    computeBarrier(cmd); // exclusive prefix を書き戻してから kAddBase_ が読む
    kAddBase_.dispatch(cmd, ds, pc, nCells_);
    computeBarrier(cmd);
    kSort_.dispatch(cmd, ds, pc, N);
    computeBarrier(cmd);
    kSolveDen_.dispatch(cmd, ds, pc, N);
    computeBarrier(cmd);

    kSdf_.dispatch(cmd, ds, pc, N);
    computeBarrier(cmd);
  }

  // ⑤ Update velocity
  kUpdateVel_.dispatch(cmd, ds, pc, N);
  computeBarrier(cmd);
}

// ── step ─────────────────────────────────────────────────────────────────────

void PhysicsHarness::step(float dt) {
  float subDt         = dt / float(cfg_.numSubsteps);
  VkCommandBuffer cmd = ctx_->beginCmd();
  for(int s = 0; s < cfg_.numSubsteps; ++s) {
    recordSubstep(cmd, subDt);
  }
  ctx_->submitCmd(cmd);
}

// ── readback ─────────────────────────────────────────────────────────────────

glm::vec4 PhysicsHarness::readPos(uint32_t i) const {
  glm::vec4 r{};
  ctx_->readBuffer(attrBuf_.getBuffer("P"), i * sizeof(glm::vec4), &r, sizeof(r));
  return r;
}

glm::vec4 PhysicsHarness::readVel(uint32_t i) const {
  glm::vec4 r{};
  ctx_->readBuffer(attrBuf_.getBuffer("v"), i * sizeof(glm::vec4), &r, sizeof(r));
  return r;
}

// ── cleanup ──────────────────────────────────────────────────────────────────

void PhysicsHarness::cleanup() {
  kPredict_.cleanup();
  kSdf_.cleanup();
  kHashCnt_.cleanup();
  kScanLoc_.cleanup();
  kScanGlob_.cleanup();
  kAddBase_.cleanup();
  kSort_.cleanup();
  kSolveDen_.cleanup();
  kSolveSt_.cleanup();
  kUpdateVel_.cleanup();
  attrBuf_.cleanup();
}
