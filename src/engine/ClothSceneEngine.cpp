#include "ClothSceneEngine.h"
#include "../core/DefineShaderCompiler.h"

#include <algorithm>
#include <cstring>
#include <glm/glm.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

// ─── シーン構築 ────────────────────────────────────────────────────────────

uint32_t ClothSceneEngine::addCloth(const ClothMesh& mesh) {
  uint32_t offset = totalCount_;
  meshOffsets_.push_back(offset);
  meshes_.push_back(mesh);
  totalCount_ += mesh.vertexCount();
  return offset;
}

void ClothSceneEngine::addConstraint(const ClothConstraint& c) { constraints_.push_back(c); }

// ─── 初期化 ────────────────────────────────────────────────────────────────

void ClothSceneEngine::buildCombinedEdges() {
  nColors_ = 12;
  combinedColorBatch_.assign(nColors_ + 1, 0);

  uint32_t edgeCursor = 0;
  for(int c = 0; c < nColors_; ++c) {
    combinedColorBatch_[c] = edgeCursor;
    for(uint32_t mi = 0; mi < (uint32_t)meshes_.size(); ++mi) {
      const ClothMesh& mesh = meshes_[mi];
      uint32_t offset       = meshOffsets_[mi];
      uint32_t start        = mesh.colorBatch[c] * 3;
      uint32_t end          = mesh.colorBatch[c + 1] * 3;
      for(uint32_t k = start; k < end; k += 3) {
        combinedEdgeData_.push_back(mesh.edgeData[k] + offset);     // p
        combinedEdgeData_.push_back(mesh.edgeData[k + 1] + offset); // q
        combinedEdgeData_.push_back(mesh.edgeData[k + 2]);          // restLen bits
        ++edgeCursor;
      }
    }
  }
  combinedColorBatch_[nColors_] = edgeCursor;
  totalEdgeCount_               = edgeCursor;
}

void ClothSceneEngine::createStagingBuffer() {
  VkDeviceSize size = totalCount_ * sizeof(glm::vec4);

  VkBufferCreateInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size  = size;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  VmaAllocationCreateInfo ai{};
  ai.usage = VMA_MEMORY_USAGE_CPU_ONLY;
  ai.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VmaAllocationInfo info{};
  vmaCreateBuffer(allocator_, &bi, &ai, &stagingBuf_, &stagingAlloc_, &info);
  stagingMapped_ = info.pMappedData;
}

void ClothSceneEngine::initBuffers(VkCommandPool cmdPool, VkQueue queue) {
  const uint32_t N = totalCount_;

  // 統合パーティクルデータを構築
  std::vector<glm::vec4> positions(N);
  std::vector<glm::vec4> invMasses(N);
  std::vector<uint32_t> typeFlags(N);

  for(uint32_t mi = 0; mi < (uint32_t)meshes_.size(); ++mi) {
    const ClothMesh& mesh = meshes_[mi];
    uint32_t offset       = meshOffsets_[mi];
    uint32_t cnt          = mesh.vertexCount();
    for(uint32_t i = 0; i < cnt; ++i) {
      positions[offset + i] = mesh.positions[i];
      invMasses[offset + i] = mesh.invMasses[i];
      typeFlags[offset + i] = mesh.typeFlags[i];
    }
  }

  // 制約を適用: invMass=0 + pinnedTarget CPU バッファを設定
  pinnedTargetCpu_.resize(N);
  for(uint32_t i = 0; i < N; ++i) pinnedTargetCpu_[i] = positions[i];

  for(const auto& c : constraints_) {
    invMasses[c.particleIdx].x      = 0.0f;
    pinnedTargetCpu_[c.particleIdx] = glm::vec4(c.targetPos, 1.0f);
    if(c.type == ClothConstraint::Type::PinAnimated) {
      hasPinAnimated_ = true;
      animatedPinParticles_.push_back(c.particleIdx);
    }
  }

  // バッファ登録
  const uint32_t N_GROUPS = (totalCells() + 255u) / 256u;
  posIdx                  = attrBuf_.addAttribute("P", sizeof(glm::vec4), N);
  velIdx                  = attrBuf_.addAttribute("v", sizeof(glm::vec4), N);
  predPIdx_               = attrBuf_.addAttribute("predP", sizeof(glm::vec4), N);
  invMassIdx_             = attrBuf_.addAttribute("invMass", sizeof(glm::vec4), N);
  typeFlagIdx_            = attrBuf_.addAttribute("typeFlag", sizeof(uint32_t), N);
  cellCountIdx_           = attrBuf_.addAttribute("cellCount", sizeof(uint32_t), totalCells());
  cellOffsetIdx_          = attrBuf_.addAttribute("cellOffset", sizeof(uint32_t), totalCells() + N_GROUPS);
  sortedIdxIdx_           = attrBuf_.addAttribute("sortedIdx", sizeof(uint32_t), N);

  std::vector<glm::vec4> zeros(N, glm::vec4(0.0f));
  attrBuf_.upload("v", zeros.data(), sizeof(glm::vec4) * N, cmdPool, queue);
  attrBuf_.upload("P", positions.data(), sizeof(glm::vec4) * N, cmdPool, queue);
  attrBuf_.upload("invMass", invMasses.data(), sizeof(glm::vec4) * N, cmdPool, queue);
  attrBuf_.upload("typeFlag", typeFlags.data(), sizeof(uint32_t) * N, cmdPool, queue);

  // エッジバッファ
  buildCombinedEdges();
  uint32_t E       = totalEdgeCount_;
  stretchEdgesIdx_ = attrBuf_.addAttribute("stretchEdges", sizeof(uint32_t), E * 3);
  lambdasIdx_      = attrBuf_.addAttribute("lambdas", sizeof(float), E);
  attrBuf_.upload("stretchEdges", combinedEdgeData_.data(), sizeof(uint32_t) * combinedEdgeData_.size(), cmdPool, queue);

  // PinAnimated 用バッファ
  if(hasPinAnimated_) {
    pinnedTargetIdx_ = attrBuf_.addAttribute("pinnedTarget", sizeof(glm::vec4), N);
    attrBuf_.upload("pinnedTarget", pinnedTargetCpu_.data(), sizeof(glm::vec4) * N, cmdPool, queue);
    createStagingBuffer();
    memcpy(stagingMapped_, pinnedTargetCpu_.data(), sizeof(glm::vec4) * N);
    pinnedTargetDirty_ = false;
  }
}

void ClothSceneEngine::init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, glm::vec3 domainSize, float cellSize) {
  domainSize_ = domainSize;
  cellSize_   = cellSize;

  if(meshes_.empty()) throw std::runtime_error("ClothSceneEngine: addCloth() before init()");

  initEngineBase(device, allocator, descriptorPool, cmdPool, queue);
  initBuffers(cmdPool, queue);

  // Force (issue #30): 既定では空リスト。重力・風が必要な場合は呼び出し側が
  // addForce(GravityForce::FromDirection(...)) 等で登録する。
  initForces();

  auto load = [&](ComputePipeline& k, const std::string& name) { k.init(device, attrBuf_.descriptorSetLayout, shaderDir + "/" + name + ".spv"); };

  load(kSdfCollision_, "sdf_collision.comp");
  load(kHashScanLocal_, "hash_scan_local.comp");
  load(kHashScanGlobal_, "hash_scan_global.comp");
  load(kHashAddBase_, "hash_add_base.comp");
  load(kSolveStretch_, "solve_stretch.comp");
  load(kUpdateVelocity_, "update_velocity.comp");
  load(kZeroLambdas_, "zero_lambdas.comp");
  load(kZeroCells_, "zero_cells.comp");
  if(hasPinAnimated_) load(kMovePins_, "move_pins.comp");

  // 空間ハッシュ近傍探索シェーダー (cellId()/mortonAxisTriples() 使用) はアダプティブ
  // (直方体)Morton定数をドメイン形状から算出し#defineで注入する必要があるため、
  // 実行時コンパイルする (FluidEngine::init() と同じ理由・同じパターン)。
  const domain::AdaptiveMortonParams morton = domain::computeAdaptiveMortonParams(gridRes());
  const std::vector<std::pair<std::string, std::string>> mortonDefines = {
      {"ADAPTIVE_MASK", std::to_string(morton.mask) + "u"},
      {"ADAPTIVE_COMMON_BITS", std::to_string(morton.commonBits) + "u"},
      {"ADAPTIVE_SHIFT_X", std::to_string(morton.shiftX) + "u"},
      {"ADAPTIVE_SHIFT_Y", std::to_string(morton.shiftY) + "u"},
      {"ADAPTIVE_SHIFT_Z", std::to_string(morton.shiftZ) + "u"},
  };
  auto loadAdaptive = [&](ComputePipeline& k, const std::string& name) {
    std::vector<uint32_t> spirv = DefineShaderCompiler::compile(name, mortonDefines);
    k.initFromSpirv(device, attrBuf_.descriptorSetLayout, spirv);
  };
  loadAdaptive(kHashCount_, "hash_count.comp");
  loadAdaptive(kHashSort_, "hash_sort.comp");
  loadAdaptive(kSolveDensity_, "solve_density.comp");

  descriptorSetLayout = attrBuf_.descriptorSetLayout;
  descriptorSet       = attrBuf_.descriptorSet;

  std::cout << "[ClothScene] " << meshes_.size() << " meshes, " << totalCount_ << " particles, " << totalEdgeCount_ << " edges\n";
}

VkBuffer ClothSceneEngine::getPositionBuffer() const { return attrBuf_.getBuffer("P"); }

// ─── 制約更新 ──────────────────────────────────────────────────────────────

void ClothSceneEngine::updateConstraint(uint32_t idx, const glm::vec3& newTarget) {
  if(idx >= (uint32_t)animatedPinParticles_.size()) return;
  uint32_t pi          = animatedPinParticles_[idx];
  pinnedTargetCpu_[pi] = glm::vec4(newTarget, 1.0f);
  pinnedTargetDirty_   = true;
}

// ─── クリーンアップ ────────────────────────────────────────────────────────

void ClothSceneEngine::cleanup() {
  kPredict_.cleanup();
  kSdfCollision_.cleanup();
  kHashCount_.cleanup();
  kHashScanLocal_.cleanup();
  kHashScanGlobal_.cleanup();
  kHashAddBase_.cleanup();
  kHashSort_.cleanup();
  kSolveDensity_.cleanup();
  kSolveStretch_.cleanup();
  kUpdateVelocity_.cleanup();
  kZeroLambdas_.cleanup();
  kZeroCells_.cleanup();
  if(hasPinAnimated_) kMovePins_.cleanup();
  if(stagingBuf_) vmaDestroyBuffer(allocator_, stagingBuf_, stagingAlloc_);
  cleanupEngineBase();
}

// ─── バリア ────────────────────────────────────────────────────────────────

void ClothSceneEngine::computeBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier b{};
  b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void ClothSceneEngine::transferBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier b{};
  b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

// ─── 1フレームのシミュレーション ──────────────────────────────────────────

void ClothSceneEngine::step(VkCommandBuffer cmd, float dt) {
  auto ds = attrBuf_.descriptorSet;

  // PinAnimated ターゲットをGPUへ転送
  if(hasPinAnimated_ && pinnedTargetDirty_) {
    VkDeviceSize size = totalCount_ * sizeof(glm::vec4);
    memcpy(stagingMapped_, pinnedTargetCpu_.data(), (size_t)size);
    VkBuffer devBuf = attrBuf_.getBuffer("pinnedTarget");
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, stagingBuf_, devBuf, 1, &region);
    transferBarrier(cmd);
    pinnedTargetDirty_ = false;
  }

  uploadForces(dt);

  const float subDt = dt / std::max(1, numSubsteps);

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
    pc.particleCount     = totalCount_;
    pc.hashCells         = totalCells();
    pc.stretchEdgesIdx   = stretchEdgesIdx_;
    pc.lambdasIdx        = lambdasIdx_;
    pc.dt                = subDt;
    pc.cellSize          = cellSize_;
    pc.gridRes           = gridRes();
    pc.worldMin          = glm::vec3(0.0f);
    pc.worldMax          = domainSize_;
    pc.restitution       = restitution;
    pc.friction          = friction;
    pc.particleRadius    = cellSize_ * 0.5f;
    pc.forceBufIdx       = forcesIdx_;
    pc.couplingForceIdx  = 0;
    pc.clothVertexCount  = totalCount_;
    pc.edgeCount         = totalEdgeCount_;
    pc.stretchCompliance = stretchCompliance;
    pc.bendCompliance    = bendCompliance;
    pc.forceCount        = (uint32_t)forces_.size();
    pc.linearDamping     = 0.02f;
    pc.pinnedTargetIdx   = pinnedTargetIdx_;

    // ① アニメーションピン位置を pos/predP に適用
    if(hasPinAnimated_) {
      kMovePins_.dispatch(cmd, ds, pc, totalCount_);
      computeBarrier(cmd);
    }

    // ② Predict (重力・風力; 静的ピンは pos→predP コピー)
    kPredict_.dispatch(cmd, ds, pc, totalCount_);
    computeBarrier(cmd);

    // ③ SDF 衝突
    kSdfCollision_.dispatch(cmd, ds, pc, totalCount_);
    computeBarrier(cmd);

    // ④ λ リセット
    kZeroLambdas_.dispatch(cmd, ds, pc, totalEdgeCount_);
    computeBarrier(cmd);

    // ⑤ XPBD 距離拘束 (グラフ彩色バッチ × solverIterations 反復)
    for(int iter = 0; iter < solverIterations; ++iter) {
      for(int color = 0; color < nColors_; ++color) {
        uint32_t start = combinedColorBatch_[color];
        uint32_t end   = combinedColorBatch_[color + 1];
        uint32_t cnt   = end - start;
        if(cnt == 0) continue;

        pc.batchEdgeStart    = start;
        pc.batchEdgeEnd      = end;
        pc.stretchCompliance = (color >= 8) ? bendCompliance : stretchCompliance;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kSolveStretch_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kSolveStretch_.pipelineLayout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, kSolveStretch_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
        vkCmdDispatch(cmd, (cnt + 255) / 256, 1, 1);
      }
      computeBarrier(cmd);
    }

    // ⑥ SDF 再適用
    kSdfCollision_.dispatch(cmd, ds, pc, totalCount_);
    computeBarrier(cmd);

    // ⑦ 布-布 & 自己衝突 (solve_density.comp が fi==2&&fj==2 の布間衝突も解く)
    if(enableSelfCollision) {
      kZeroCells_.dispatch(cmd, ds, pc, totalCells());
      computeBarrier(cmd);
      kHashCount_.dispatch(cmd, ds, pc, totalCount_);
      computeBarrier(cmd);
      {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanLocal_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanLocal_.pipelineLayout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, kHashScanLocal_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
        vkCmdDispatch(cmd, (totalCells() + 255u) / 256u, 1, 1);
      }
      computeBarrier(cmd);
      {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanGlobal_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanGlobal_.pipelineLayout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, kHashScanGlobal_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
        vkCmdDispatch(cmd, 1, 1, 1);
      }
      computeBarrier(cmd); // exclusive prefix を書き戻してから kHashAddBase_ が読む
      kHashAddBase_.dispatch(cmd, ds, pc, totalCells());
      computeBarrier(cmd);
      kHashSort_.dispatch(cmd, ds, pc, totalCount_);
      computeBarrier(cmd);

      for(int i = 0; i < 2; ++i) {
        kSolveDensity_.dispatch(cmd, ds, pc, totalCount_);
        computeBarrier(cmd);
      }
      kSdfCollision_.dispatch(cmd, ds, pc, totalCount_);
      computeBarrier(cmd);
    }

    // ⑧ 速度更新
    kUpdateVelocity_.dispatch(cmd, ds, pc, totalCount_);
    computeBarrier(cmd);
  }
}
