#include "FluidEngine.h"
#include "BoundaryParticles.h"

#include <cmath>
#include <cstdlib>
#include <glm/glm.hpp>
#include <random>
#include <stdexcept>
#include <vector>

// ── バリアヘルパー ───────────────────────────────────────────────────────────

void FluidEngine::computeBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier b{};
  b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}


// ── 初期化 ────────────────────────────────────────────────────────────────────

void FluidEngine::init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, const FluidConfig& cfg) {
  cfg_       = cfg;
  device_    = device;
  allocator_ = allocator;
  cmdPool_   = cmdPool;
  queue_     = queue;

  // h/d 比に対応した静止密度を自動設定（ImGui から上書き可能）
  // rho0 = cfg_.computeRestDensity();

  attrBuf_.init(device, allocator, descriptorPool);

  const uint32_t N_GROUPS = (cfg_.totalCells() + 255u) / 256u;

  // 全バッファを N_TOTAL_MAX で確保（流体 + 境界粒子）
  // vec4 系は FP16 packed（8 bytes/粒子）でメモリ帯域幅を半減
  posIdx         = attrBuf_.addAttribute("P", sizeof(glm::vec4), cfg_.nTotalMax());
  velIdx         = attrBuf_.addAttribute("v", sizeof(glm::vec4), cfg_.nTotalMax());
  predPIdx_      = attrBuf_.addAttribute("predP", sizeof(glm::vec4), cfg_.nTotalMax());
  invMassIdx_    = attrBuf_.addAttribute("invMass", sizeof(glm::vec4), cfg_.nTotalMax());
  typeFlagIdx_   = attrBuf_.addAttribute("typeFlag", sizeof(uint32_t), cfg_.nTotalMax());
  cellCountIdx_  = attrBuf_.addAttribute("cellCnt", sizeof(uint32_t), cfg_.totalCells());
  cellOffsetIdx_ = attrBuf_.addAttribute("cellOff", sizeof(uint32_t), cfg_.totalCells() + N_GROUPS);
  sortedIdxIdx_  = attrBuf_.addAttribute("sorted", sizeof(uint32_t), cfg_.nTotalMax());
  densityIdx_    = attrBuf_.addAttribute("density", sizeof(float), cfg_.nTotalMax());
  lambdaPbfIdx_  = attrBuf_.addAttribute("lambdaPbf", sizeof(float), cfg_.nTotalMax());
  omegaIdx_      = attrBuf_.addAttribute("omega", sizeof(glm::vec4), cfg_.nTotalMax());

  nFluid_ = 0;
  sources_.clear();
  sourceStepsDone_.clear();

  auto load = [&](ComputePipeline& k, const char* name) { k.init(device, attrBuf_.descriptorSetLayout, shaderDir + "/" + name + ".spv"); };
  load(kPredictSdf_, "predict_sdf.comp");
  load(kSdfCollision_, "sdf_collision.comp");
  load(kHashCount_, "hash_count.comp");
  load(kHashScanLocal_, "hash_scan_local.comp");
  load(kHashScanGlobal_, "hash_scan_global.comp");
  load(kHashAddBase_, "hash_add_base.comp");
  load(kHashSort_, "hash_sort.comp");
  load(kPbfDensity_, "pbf_density.comp");
  load(kPbfDeltaP_, "pbf_delta_p.comp");
  load(kPbfViscosity_, "pbf_viscosity.comp");
  load(kUpdateVelocity_, "update_velocity.comp");
  load(kZeroCells_, "zero_cells.comp");
  load(kVorticityOmega_, "pbf_vorticity_omega.comp");
  load(kVorticityForce_, "pbf_vorticity_force.comp");

  descriptorSetLayout = attrBuf_.descriptorSetLayout;
  descriptorSet       = attrBuf_.descriptorSet;
}

VkBuffer FluidEngine::getPositionBuffer() const { return attrBuf_.getBuffer("P"); }

// ── 境界粒子ロード ────────────────────────────────────────────────────────────

void FluidEngine::loadBoundary(const std::string& objPath, float spacing) {
  BoundaryParticles bp;
  auto pts = bp.loadOBJ(objPath, spacing);

  if(pts.empty()) return;

  uint32_t n           = static_cast<uint32_t>(std::min(pts.size(), size_t(cfg_.max_boundary)));
  VkDeviceSize byteOff = cfg_.fluidCount() * sizeof(glm::vec4);

  // 位置を [FLUID_COUNT, FLUID_COUNT+n) に書き込む
  attrBuf_.uploadAt("P", pts.data(), sizeof(glm::vec4) * n, byteOff, cmdPool_, queue_);
  attrBuf_.uploadAt("predP", pts.data(), sizeof(glm::vec4) * n, byteOff, cmdPool_, queue_);

  // invMass = 0 (固定)
  std::vector<glm::vec4> zeroMass(n, glm::vec4(0.0f));
  attrBuf_.uploadAt("invMass", zeroMass.data(), sizeof(glm::vec4) * n, byteOff, cmdPool_, queue_);

  // 速度 = 0
  std::vector<glm::vec4> zeroVel(n, glm::vec4(0.0f));
  attrBuf_.uploadAt("v", zeroVel.data(), sizeof(glm::vec4) * n, byteOff, cmdPool_, queue_);

  // typeFlag = 3 (境界粒子)
  std::vector<uint32_t> flags(n, 3u);
  VkDeviceSize flagOff = cfg_.fluidCount() * sizeof(uint32_t);
  attrBuf_.uploadAt("typeFlag", flags.data(), sizeof(uint32_t) * n, flagOff, cmdPool_, queue_);

  nBoundary = n;
  boundaryTriVerts_.clear();
}

void FluidEngine::loadBoundary(const std::string& objPath, float spacing, float scale, glm::vec3 offset, bool yup_to_zup) {
  BoundaryParticles bp;
  BoundaryMesh mesh = bp.loadOBJ(objPath, spacing, scale, offset, yup_to_zup);

  if(mesh.particles.empty()) return;

  uint32_t n           = static_cast<uint32_t>(std::min(mesh.particles.size(), size_t(cfg_.max_boundary)));
  VkDeviceSize byteOff = cfg_.fluidCount() * sizeof(glm::vec4);

  // 位置を [FLUID_COUNT, FLUID_COUNT+n) に書き込む
  attrBuf_.uploadAt("P", mesh.particles.data(), sizeof(glm::vec4) * n, byteOff, cmdPool_, queue_);
  attrBuf_.uploadAt("predP", mesh.particles.data(), sizeof(glm::vec4) * n, byteOff, cmdPool_, queue_);

  // invMass = 0 (固定)
  std::vector<glm::vec4> zeroMass(n, glm::vec4(0.0f));
  attrBuf_.uploadAt("invMass", zeroMass.data(), sizeof(glm::vec4) * n, byteOff, cmdPool_, queue_);

  // 速度 = 0
  std::vector<glm::vec4> zeroVel(n, glm::vec4(0.0f));
  attrBuf_.uploadAt("v", zeroVel.data(), sizeof(glm::vec4) * n, byteOff, cmdPool_, queue_);

  // typeFlag = 3 (境界粒子)
  std::vector<uint32_t> flags(n, 3u);
  VkDeviceSize flagOff = cfg_.fluidCount() * sizeof(uint32_t);
  attrBuf_.uploadAt("typeFlag", flags.data(), sizeof(uint32_t) * n, flagOff, cmdPool_, queue_);

  nBoundary         = n;
  boundaryTriVerts_ = std::move(mesh.triVerts);
}

void FluidEngine::loadBoundaryParticles(const std::vector<glm::vec4>& pts) {
  if(pts.empty()) return;

  uint32_t n           = static_cast<uint32_t>(std::min(pts.size(), size_t(cfg_.max_boundary)));
  VkDeviceSize byteOff = cfg_.fluidCount() * sizeof(glm::vec4);

  attrBuf_.uploadAt("P",     pts.data(), sizeof(glm::vec4) * n, byteOff, cmdPool_, queue_);
  attrBuf_.uploadAt("predP", pts.data(), sizeof(glm::vec4) * n, byteOff, cmdPool_, queue_);

  std::vector<glm::vec4> zeroMass(n, glm::vec4(0.0f));
  attrBuf_.uploadAt("invMass", zeroMass.data(), sizeof(glm::vec4) * n, byteOff, cmdPool_, queue_);

  std::vector<glm::vec4> zeroVel(n, glm::vec4(0.0f));
  attrBuf_.uploadAt("v", zeroVel.data(), sizeof(glm::vec4) * n, byteOff, cmdPool_, queue_);

  std::vector<uint32_t> flags(n, 3u);
  VkDeviceSize flagOff = cfg_.fluidCount() * sizeof(uint32_t);
  attrBuf_.uploadAt("typeFlag", flags.data(), sizeof(uint32_t) * n, flagOff, cmdPool_, queue_);

  nBoundary = n;
}

void FluidEngine::clearBoundary() {
  nBoundary = 0;
  boundaryTriVerts_.clear();
}

// ── ソース管理 ───────────────────────────────────────────────────────────────

void FluidEngine::addSource(std::shared_ptr<Source> src) {
  sources_.push_back(std::move(src));
  sourceStepsDone_.push_back(0);
}

void FluidEngine::clearSources() {
  sources_.clear();
  sourceStepsDone_.clear();
}

void FluidEngine::emitSources(float dt) {
  for(size_t i = 0; i < sources_.size(); ++i) {
    Source& src = *sources_[i];
    int& done   = sourceStepsDone_[i];

    bool shouldEmit = false;
    if(src.step_count == -1)
      shouldEmit = (done == 0);
    else if(src.step_count == 0)
      shouldEmit = true;
    else
      shouldEmit = (done < src.step_count);

    if(!shouldEmit) continue;

    int available = int(cfg_.fluidCount()) - int(nFluid_);
    int nNew      = std::min(src.particles_per_step, available);
    if(nNew <= 0) { done++; continue; }

    std::vector<glm::vec4> pos(nNew);

    if(auto* aabb = dynamic_cast<AABBSource*>(&src)) {
      glm::vec3 half = aabb->size * 0.5f;
      std::uniform_real_distribution<float> dx(-half.x, half.x);
      std::uniform_real_distribution<float> dy(-half.y, half.y);
      std::uniform_real_distribution<float> dz(-half.z, half.z);
      for(int j = 0; j < nNew; ++j)
        pos[j] = glm::vec4(aabb->center.x + dx(sourceRng_), aabb->center.y + dy(sourceRng_), aabb->center.z + dz(sourceRng_), 1.0f);
    } else if(auto* sphere = dynamic_cast<SphereSource*>(&src)) {
      std::uniform_real_distribution<float> dr(0.0f, 1.0f);
      std::uniform_real_distribution<float> dtheta(0.0f, 3.14159265f);
      std::uniform_real_distribution<float> dphi(0.0f, 2.0f * 3.14159265f);
      for(int j = 0; j < nNew; ++j) {
        float r     = sphere->radius * std::cbrt(dr(sourceRng_));
        float theta = dtheta(sourceRng_);
        float phi   = dphi(sourceRng_);
        pos[j]      = glm::vec4(sphere->center.x + r * std::sin(theta) * std::cos(phi), sphere->center.y + r * std::sin(theta) * std::sin(phi), sphere->center.z + r * std::cos(theta), 1.0f);
      }
    } else {
      for(int j = 0; j < nNew; ++j)
        pos[j] = glm::vec4(src.center, 1.0f);
    }

    std::vector<glm::vec4> vel(nNew, glm::vec4(src.vel, 0.0f));
    std::vector<glm::vec4> invM(nNew, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    std::vector<uint32_t>  flags(nNew, src.particleType);

    VkDeviceSize byteOff = nFluid_ * sizeof(glm::vec4);
    VkDeviceSize flagOff = nFluid_ * sizeof(uint32_t);

    attrBuf_.uploadAt("P",        pos.data(),  sizeof(glm::vec4)  * nNew, byteOff, cmdPool_, queue_);
    attrBuf_.uploadAt("predP",    pos.data(),  sizeof(glm::vec4)  * nNew, byteOff, cmdPool_, queue_);
    attrBuf_.uploadAt("v",        vel.data(),  sizeof(glm::vec4)  * nNew, byteOff, cmdPool_, queue_);
    attrBuf_.uploadAt("invMass",  invM.data(), sizeof(glm::vec4)  * nNew, byteOff, cmdPool_, queue_);
    attrBuf_.uploadAt("typeFlag", flags.data(),sizeof(uint32_t)   * nNew, flagOff, cmdPool_, queue_);

    nFluid_ += uint32_t(nNew);
    done++;

    // ソース中心を移動
    src.center += src.center_vel * dt;
  }
}

// ── 粒子リセット ─────────────────────────────────────────────────────────────

void FluidEngine::resetParticles() {
  if(device_ == VK_NULL_HANDLE) return;
  vkDeviceWaitIdle(device_);
  nFluid_ = 0;
  std::fill(sourceStepsDone_.begin(), sourceStepsDone_.end(), 0);
  sourceRng_.seed(12345);
}

// ── クリーンアップ ───────────────────────────────────────────────────────────

void FluidEngine::cleanup() {
  cleanupKinematicBoundaryStaging();
  kPredictSdf_.cleanup();
  kSdfCollision_.cleanup();
  kHashCount_.cleanup();
  kHashScanLocal_.cleanup();
  kHashScanGlobal_.cleanup();
  kHashAddBase_.cleanup();
  kHashSort_.cleanup();
  kPbfDensity_.cleanup();
  kPbfDeltaP_.cleanup();
  kPbfViscosity_.cleanup();
  kUpdateVelocity_.cleanup();
  kZeroCells_.cleanup();
  kVorticityOmega_.cleanup();
  kVorticityForce_.cleanup();
  attrBuf_.cleanup();
}

// ── TC8: kinematic 境界粒子 staging ────────────────────────────────────────────

void FluidEngine::initKinematicBoundaryStaging(uint32_t maxBoundaryCount) {
  if(maxBoundaryCount == 0) return;
  kinStagingMaxCount_ = maxBoundaryCount;

  VkBufferCreateInfo bufInfo{};
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufInfo.size  = maxBoundaryCount * sizeof(glm::vec4) * 2; // 前半: pos, 後半: vel
  bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  VmaAllocationCreateInfo allocInfo{};
  allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
  allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

  for(uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i) {
    VmaAllocationInfo info{};
    vmaCreateBuffer(allocator_, &bufInfo, &allocInfo, &kinStagingBuf_[i], &kinStagingAlloc_[i], &info);
    kinStagingMapped_[i] = info.pMappedData;
  }
}

void FluidEngine::recordKinematicBoundaryUpdate(
    VkCommandBuffer cmd, uint32_t frameIndex,
    uint32_t boundaryOffset, uint32_t count,
    const glm::vec4* positions, const glm::vec4* velocities) {
  if(kinStagingMaxCount_ == 0 || count == 0) return;
  uint32_t fi = frameIndex % MAX_CONCURRENT_FRAMES;

  // positions → staging 前半, velocities → staging 後半
  VkDeviceSize posBytes = count * sizeof(glm::vec4);
  memcpy(kinStagingMapped_[fi], positions, posBytes);
  memcpy(static_cast<char*>(kinStagingMapped_[fi]) + posBytes, velocities, posBytes);
  vmaFlushAllocation(allocator_, kinStagingAlloc_[fi], 0, posBytes * 2);

  VkDeviceSize dstByteOff = boundaryOffset * sizeof(glm::vec4);

  // P バッファへコピー
  VkBufferCopy cpPos{};
  cpPos.srcOffset = 0;
  cpPos.dstOffset = dstByteOff;
  cpPos.size      = posBytes;
  vkCmdCopyBuffer(cmd, kinStagingBuf_[fi], attrBuf_.getBuffer("P"), 1, &cpPos);

  // predP バッファへコピー（PBF density が predPIdx を読む）
  vkCmdCopyBuffer(cmd, kinStagingBuf_[fi], attrBuf_.getBuffer("predP"), 1, &cpPos);

  // v バッファへコピー（XSPH が velIdx を読む）
  VkBufferCopy cpVel{};
  cpVel.srcOffset = posBytes;
  cpVel.dstOffset = dstByteOff;
  cpVel.size      = posBytes;
  vkCmdCopyBuffer(cmd, kinStagingBuf_[fi], attrBuf_.getBuffer("v"), 1, &cpVel);

  // compute シェーダーが読む前に TRANSFER_WRITE → SHADER_READ バリア
  VkMemoryBarrier bar{};
  bar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &bar, 0, nullptr, 0, nullptr);
}

void FluidEngine::cleanupKinematicBoundaryStaging() {
  if(kinStagingMaxCount_ == 0) return;
  for(uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i) {
    if(kinStagingBuf_[i] != VK_NULL_HANDLE) {
      vmaDestroyBuffer(allocator_, kinStagingBuf_[i], kinStagingAlloc_[i]);
      kinStagingBuf_[i]    = VK_NULL_HANDLE;
      kinStagingMapped_[i] = nullptr;
    }
  }
  kinStagingMaxCount_ = 0;
}

// ── 1フレームのシミュレーション ───────────────────────────────────────────────

void FluidEngine::step(VkCommandBuffer cmd, float dt) {
  emitSources(dt);

  if(nFluid_ == 0 && nBoundary == 0) return;

  auto ds         = attrBuf_.descriptorSet;
  float subDt     = dt / float(std::max(1, numSubsteps));
  uint32_t totalN = nFluid_ + nBoundary;

  for(int sub = 0; sub < numSubsteps; ++sub) {
    SimPC pc{};
    pc.posIdx            = posIdx;
    pc.velIdx            = velIdx;
    pc.predPIdx          = predPIdx_;
    pc.invMassIdx        = invMassIdx_;
    pc.typeFlagIdx       = typeFlagIdx_;
    pc.cellCountIdx      = cellCountIdx_;
    pc.cellOffsetIdx     = cellOffsetIdx_;
    pc.sortedIdxIdx      = sortedIdxIdx_;
    pc.particleCount     = totalN;
    pc.gridRes           = cfg_.grid_res;
    pc.stretchEdgesIdx   = 0;
    pc.lambdasIdx        = 0;
    pc.dt                = subDt;
    pc.cellSize          = cfg_.cellSize();
    pc.worldMin          = 0.0f;
    pc.worldMax          = cfg_.world_size;
    pc.gravity           = gravity;
    pc.restitution       = restitution;
    pc.friction          = friction;
    pc.particleRadius    = cfg_.cellSize() * 0.5f;
    pc.couplingForceIdx  = 0;
    pc.clothVertexCount  = 0; // 流体では未使用（布頂点数ではない）
    pc.edgeCount         = 0;
    pc.batchEdgeStart    = 0;
    pc.batchEdgeEnd      = 0;
    pc.stretchCompliance = rho0;
    pc.bendCompliance    = viscosityC;
    pc.windX             = 0.0f;
    pc.windZ             = 0.0f;
    pc.densityIdx        = densityIdx_;
    pc.lambdaPbfIdx      = lambdaPbfIdx_;
    pc.boundaryStart     = cfg_.fluidCount();
    // PBF 論文準拠パラメータ
    pc.cfmEpsilon       = cfmEpsilon;
    pc.scorrK           = scorrK;
    pc.vorticityEpsilon = vorticityEpsilon;
    pc.linearDamping    = linearDamping;
    pc.omegaIdx         = omegaIdx_;
    // 煙・粉体パラメータ
    pc.smokeRiseAccel   = smokeRiseAccel;
    pc.smokeDamping     = smokeDamping;
    // pc.powderFriction → SimPC では pinnedTargetIdx に転用。FluidEngine では未使用 (0のまま)。

    // ① Predict + SDF 壁衝突 (merged: 1 dispatch instead of 2)
    // 境界粒子は invMass==0 で predP=pos のまま変化しないため nFluid_ のみ
    kPredictSdf_.dispatch(cmd, ds, pc, nFluid_);
    // kPredictSdf_ は predP を書く。kZeroCells_ は cellCount を書く（predP 不使用）
    // → 依存関係なしのためバリア不要。MoltenVK encoder switch を 1 回削減。

    // ② 空間ハッシュ構築 (全粒子: 流体 + 境界)
    // compute シェーダーでゼロクリア (blit エンコーダー遷移を排除)
    kZeroCells_.dispatch(cmd, ds, pc, cfg_.totalCells());
    computeBarrier(cmd); // kHashCount_ が cellCount を読むため必要

    kHashCount_.dispatch(cmd, ds, pc, totalN);
    computeBarrier(cmd);

    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanLocal_.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanLocal_.pipelineLayout, 0, 1, &ds, 0, nullptr);
      vkCmdPushConstants(cmd, kHashScanLocal_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
      vkCmdDispatch(cmd, (cfg_.totalCells() + 255u) / 256u, 1, 1);
    }
    computeBarrier(cmd);

    {
      // Pass 2b-1: グループサムの prefix scan（1 workgroup × 1024 threads）
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanGlobal_.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanGlobal_.pipelineLayout, 0, 1, &ds, 0, nullptr);
      vkCmdPushConstants(cmd, kHashScanGlobal_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
      vkCmdDispatch(cmd, 1, 1, 1);
    }
    computeBarrier(cmd); // exclusive prefix を書き戻してから kHashAddBase_ が読む

    // Pass 2b-2: ベース加算を 1024 wg × 256 threads で並列化（旧: 1 wg × 1024 threads × 256 sequential writes）
    kHashAddBase_.dispatch(cmd, ds, pc, cfg_.totalCells());
    computeBarrier(cmd);

    kHashSort_.dispatch(cmd, ds, pc, totalN);
    computeBarrier(cmd);

    // ④ PBF 不圧縮ソルバー × pbfIterations
    for(int iter = 0; iter < pbfIterations; ++iter) {
      kPbfDensity_.dispatch(cmd, ds, pc, nFluid_);
      computeBarrier(cmd);
      kPbfDeltaP_.dispatch(cmd, ds, pc, nFluid_);
      computeBarrier(cmd);
    }

    // ⑤ SDF 再適用 (PBF 補正後の境界貫通防止; 境界粒子は固定位置のため除外)
    kSdfCollision_.dispatch(cmd, ds, pc, nFluid_);
    computeBarrier(cmd);

    // ⑥ 速度更新（境界粒子は invMass==0 で速度変化なし、除外)
    kUpdateVelocity_.dispatch(cmd, ds, pc, nFluid_);
    computeBarrier(cmd);

    // ⑦ 渦度閉じ込め (式15-16; ON/OFF 切替可能)
    if(vorticityEnabled) {
      kVorticityOmega_.dispatch(cmd, ds, pc, nFluid_);
      computeBarrier(cmd);
      kVorticityForce_.dispatch(cmd, ds, pc, nFluid_);
      computeBarrier(cmd);
    }

    // ⑧ XSPH 粘性 (1 回のみ; 境界粒子は typeFlag チェックで除外)
    kPbfViscosity_.dispatch(cmd, ds, pc, nFluid_);
    // 次の substep がある場合のみバリアが必要（次の kPredictSdf_ が vel を読む）
    if(sub < numSubsteps - 1) {
      computeBarrier(cmd);
    }
  }
}
