#include "SoftBodyEngine.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <glm/glm.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

static constexpr uint32_t SB_MAGIC = 0x42544653u; // "SFTB" little-endian

// ─── public ──────────────────────────────────────────────────────────────────

uint32_t SoftBodyEngine::addInstance(const SoftBodyInstance& inst) {
  InstanceData data;
  data.offset         = inst.offset;
  data.scale          = inst.scale;
  data.sbPath         = inst.sbPath;
  data.particleOffset = totalCount_;

  loadSBFile(data);

  totalCount_ += data.n_particles;
  totalEdgeCount_ += data.n_edges;
  totalTetCount_ += data.n_tets;

  instances_.push_back(std::move(data));
  return instances_.back().particleOffset;
}

void SoftBodyEngine::init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, float worldSize, uint32_t gridRes) {
  worldSize_ = worldSize;
  gridRes_   = gridRes;

  buildCombinedBuffers();

  initEngineBase(device, allocator, descriptorPool, cmdPool, queue);
  initGPUBuffers(cmdPool, queue);
  initForces();

  auto load = [&](ComputePipeline& k, const std::string& name) { k.init(device, attrBuf_.descriptorSetLayout, shaderDir + "/" + name + ".spv"); };
  load(kSdfCollision_, "sdf_collision.comp");
  load(kSolveEdge_, "solve_stretch.comp");
  load(kZeroEdgeLambda_, "zero_lambdas.comp");
  load(kSolveVolume_, "sb_solve_volume.comp");
  load(kZeroVolLambda_, "sb_zero_vol_lambdas.comp");
  load(kParticleCollision_, "sb_particle_collision.comp");
  load(kUpdateVelocity_, "update_velocity.comp");

  descriptorSetLayout = attrBuf_.descriptorSetLayout;
  descriptorSet       = attrBuf_.descriptorSet;

  std::cout << "[SoftBody] " << totalCount_ << " particles, " << totalEdgeCount_ << " edges (" << nEdgeColors_ << " colors), " << totalTetCount_ << " tets (" << nTetColors_ << " colors)" << std::endl;
}

void SoftBodyEngine::cleanup() {
  kPredict_.cleanup();
  kSdfCollision_.cleanup();
  kSolveEdge_.cleanup();
  kZeroEdgeLambda_.cleanup();
  kSolveVolume_.cleanup();
  kZeroVolLambda_.cleanup();
  kParticleCollision_.cleanup();
  kUpdateVelocity_.cleanup();
  cleanupEngineBase();
}

void SoftBodyEngine::step(VkCommandBuffer cmd, float dt) {
  if(totalCount_ == 0) return;

  uploadForces(dt);

  VkDescriptorSet ds = attrBuf_.descriptorSet;
  const float subDt  = dt / float(std::max(1, numSubsteps));

  for(int sub = 0; sub < numSubsteps; ++sub) {
    SimPC pc{};
    // 粒子共通
    pc.posIdx         = posIdx;
    pc.velIdx         = velIdx;
    pc.predPIdx       = predPIdx_;
    pc.invMassIdx     = invMassIdx_;
    pc.typeFlagIdx    = typeFlagIdx_;
    pc.particleCount  = totalCount_;
    pc.dt             = subDt;
    pc.cellSize       = worldSize_ / float(gridRes_);
    pc.worldMin       = 0.0f;
    pc.worldMax       = worldSize_;
    pc.restitution    = restitution;
    pc.friction       = friction;
    pc.particleRadius = particleRadius;
    pc.forceBufIdx    = forcesIdx_;

    // エッジ距離拘束 (solve_stretch.comp / zero_lambdas.comp が参照)
    pc.stretchEdgesIdx   = edgeDataIdx_;
    pc.lambdasIdx        = edgeLambdaIdx_;
    pc.edgeCount         = totalEdgeCount_;
    pc.stretchCompliance = stretchCompliance;

    // 四面体体積拘束 (SimPC エイリアス)
    pc.couplingForceIdx = tetDataIdx_;    // tetDataIdx
    pc.clothVertexCount = tetLambdaIdx_;  // tetLambdaIdx
    pc.boundaryStart    = totalTetCount_; // tetCount
    pc.bendCompliance   = volCompliance;  // volCompliance
    pc.omegaIdx         = tetRestVolIdx_; // tetRestVolIdx

    // 速度減衰 (update_velocity.comp)
    pc.linearDamping = linearDamping;

    // 粒子間衝突半径 (sb_particle_collision.comp 専用)
    pc.particleCollisionRadius = particleCollisionRadius;
    pc.forceCount               = (uint32_t)forces_.size();

    // ① Predict: 重力 → predP
    kPredict_.dispatch(cmd, ds, pc, totalCount_);
    computeBarrier(cmd);

    // ② λ リセット (エッジ / 四面体)
    kZeroEdgeLambda_.dispatch(cmd, ds, pc, totalEdgeCount_);
    computeBarrier(cmd);
    kZeroVolLambda_.dispatch(cmd, ds, pc, totalTetCount_);
    computeBarrier(cmd);

    // ③ XPBD ソルバーループ
    for(int iter = 0; iter < solverIterations; ++iter) {
      // エッジ距離拘束 (グラフ彩色バッチ)
      for(int c = 0; c < nEdgeColors_; ++c) {
        uint32_t start = edgeColorBatch_[c];
        uint32_t end   = edgeColorBatch_[c + 1];
        uint32_t cnt   = end - start;
        if(cnt == 0) continue;

        pc.batchEdgeStart = start;
        pc.batchEdgeEnd   = end;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kSolveEdge_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kSolveEdge_.pipelineLayout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, kSolveEdge_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
        vkCmdDispatch(cmd, (cnt + 255u) / 256u, 1, 1);
      }

      // 四面体体積拘束 (グラフ彩色バッチ)
      for(int c = 0; c < nTetColors_; ++c) {
        uint32_t start = tetColorBatch_[c];
        uint32_t end   = tetColorBatch_[c + 1];
        uint32_t cnt   = end - start;
        if(cnt == 0) continue;

        pc.densityIdx   = start; // batchTetStart
        pc.lambdaPbfIdx = end;   // batchTetEnd

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kSolveVolume_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kSolveVolume_.pipelineLayout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, kSolveVolume_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
        vkCmdDispatch(cmd, (cnt + 255u) / 256u, 1, 1);
      }
      computeBarrier(cmd);
    }

    // ④ 粒子間衝突 (ボディ間反発)
    kParticleCollision_.dispatch(cmd, ds, pc, totalCount_);
    computeBarrier(cmd);

    // ⑤ SDF 境界衝突
    kSdfCollision_.dispatch(cmd, ds, pc, totalCount_);
    computeBarrier(cmd);

    // ⑥ XPBD 速度更新: v = (predP - P) / dt
    kUpdateVelocity_.dispatch(cmd, ds, pc, totalCount_);
    computeBarrier(cmd);
  }
}

// ─── private ─────────────────────────────────────────────────────────────────

void SoftBodyEngine::loadSBFile(InstanceData& inst) {
  std::ifstream f(inst.sbPath, std::ios::binary);
  if(!f.is_open()) throw std::runtime_error("Cannot open .sb file: " + inst.sbPath);

  auto readU32 = [&]() {
    uint32_t v;
    f.read(reinterpret_cast<char*>(&v), 4);
    return v;
  };
  auto readF32 = [&]() {
    float v;
    f.read(reinterpret_cast<char*>(&v), 4);
    return v;
  };

  // マジック検証
  uint32_t magic = readU32();
  if(magic != SB_MAGIC) throw std::runtime_error("Invalid .sb file (bad magic): " + inst.sbPath);

  inst.n_particles   = readU32();
  inst.n_edges       = readU32();
  inst.n_tets        = readU32();
  inst.n_edge_colors = int(readU32());
  inst.n_tet_colors  = int(readU32());

  // 頂点座標 (scale + offset 適用)
  inst.positions.resize(inst.n_particles);
  for(auto& p : inst.positions) {
    p.x = readF32() * inst.scale + inst.offset.x;
    p.y = readF32() * inst.scale + inst.offset.y;
    p.z = readF32() * inst.scale + inst.offset.z;
  }

  // 逆質量
  inst.invMasses.resize(inst.n_particles);
  for(auto& m : inst.invMasses) m = readF32();

  // エッジ
  inst.edgeColorBatch.resize(inst.n_edge_colors + 1);
  for(auto& b : inst.edgeColorBatch) b = readU32();
  inst.edgeData.resize(inst.n_edges * 3);
  for(auto& v : inst.edgeData) v = readU32();

  // 四面体
  inst.tetColorBatch.resize(inst.n_tet_colors + 1);
  for(auto& b : inst.tetColorBatch) b = readU32();
  inst.tetData.resize(inst.n_tets * 4);
  for(auto& v : inst.tetData) v = readU32();
  inst.tetRestVols.resize(inst.n_tets);
  for(auto& v : inst.tetRestVols) v = readF32();
}

void SoftBodyEngine::buildCombinedBuffers() {
  if(instances_.empty()) return;

  // 最大色数を決定
  for(const auto& inst : instances_) {
    nEdgeColors_ = std::max(nEdgeColors_, inst.n_edge_colors);
    nTetColors_  = std::max(nTetColors_, inst.n_tet_colors);
  }

  // 位置・invMass・typeFlag
  combinedPos_.reserve(totalCount_);
  combinedInvMass_.reserve(totalCount_);
  combinedTypeFlag_.reserve(totalCount_);
  for(const auto& inst : instances_) {
    for(uint32_t i = 0; i < inst.n_particles; ++i) {
      combinedPos_.emplace_back(inst.positions[i], 0.0f);
      combinedInvMass_.emplace_back(inst.invMasses[i], 0.0f, 0.0f, 0.0f);
      combinedTypeFlag_.push_back(6u); // typeFlag = 6 (soft body)
    }
  }

  // エッジバッファ: 各インスタンスの色 c のエッジを連結
  edgeColorBatch_.resize(nEdgeColors_ + 1, 0);
  combinedEdgeData_.reserve(totalEdgeCount_ * 3);

  for(int c = 0; c < nEdgeColors_; ++c) {
    for(const auto& inst : instances_) {
      if(c >= inst.n_edge_colors) continue;
      uint32_t s       = inst.edgeColorBatch[c];
      uint32_t e       = inst.edgeColorBatch[c + 1];
      uint32_t pOffset = inst.particleOffset;
      for(uint32_t k = s; k < e; ++k) {
        combinedEdgeData_.push_back(inst.edgeData[k * 3] + pOffset);
        combinedEdgeData_.push_back(inst.edgeData[k * 3 + 1] + pOffset);
        combinedEdgeData_.push_back(inst.edgeData[k * 3 + 2]); // restLen は bits のまま
      }
    }
    edgeColorBatch_[c + 1] = uint32_t(combinedEdgeData_.size() / 3);
  }

  // 四面体バッファ: 各インスタンスの色 c の四面体を連結
  tetColorBatch_.resize(nTetColors_ + 1, 0);
  combinedTetData_.reserve(totalTetCount_ * 4);
  combinedTetRestVol_.reserve(totalTetCount_);

  for(int c = 0; c < nTetColors_; ++c) {
    for(const auto& inst : instances_) {
      if(c >= inst.n_tet_colors) continue;
      uint32_t s       = inst.tetColorBatch[c];
      uint32_t e       = inst.tetColorBatch[c + 1];
      uint32_t pOffset = inst.particleOffset;
      for(uint32_t k = s; k < e; ++k) {
        combinedTetData_.push_back(inst.tetData[k * 4] + pOffset);
        combinedTetData_.push_back(inst.tetData[k * 4 + 1] + pOffset);
        combinedTetData_.push_back(inst.tetData[k * 4 + 2] + pOffset);
        combinedTetData_.push_back(inst.tetData[k * 4 + 3] + pOffset);
        combinedTetRestVol_.push_back(inst.tetRestVols[k]);
      }
    }
    tetColorBatch_[c + 1] = uint32_t(combinedTetData_.size() / 4);
  }
}

void SoftBodyEngine::initGPUBuffers(VkCommandPool cmdPool, VkQueue queue) {
  const uint32_t N = totalCount_;
  const uint32_t E = totalEdgeCount_;
  const uint32_t T = totalTetCount_;

  // バッファ確保 (10 スロット / 16)
  posIdx       = attrBuf_.addAttribute("P", sizeof(glm::vec4), N);
  velIdx       = attrBuf_.addAttribute("v", sizeof(glm::vec4), N);
  predPIdx_    = attrBuf_.addAttribute("predP", sizeof(glm::vec4), N);
  invMassIdx_  = attrBuf_.addAttribute("invMass", sizeof(glm::vec4), N);
  typeFlagIdx_ = attrBuf_.addAttribute("typeFlag", sizeof(uint32_t), N);

  edgeDataIdx = edgeDataIdx_ = attrBuf_.addAttribute("edgeData", sizeof(uint32_t), E * 3);
  edgeLambdaIdx_             = attrBuf_.addAttribute("edgeLambda", sizeof(float), E);
  tetDataIdx_                = attrBuf_.addAttribute("tetData", sizeof(uint32_t), T * 4);
  tetLambdaIdx_              = attrBuf_.addAttribute("tetLambda", sizeof(float), T);
  tetRestVolIdx_             = attrBuf_.addAttribute("tetRestVol", sizeof(float), T);

  auto up = [&](const std::string& name, const void* data, size_t bytes) { attrBuf_.upload(name, data, bytes, cmdPool, queue); };

  // 初期データをアップロード
  up("P", combinedPos_.data(), N * sizeof(glm::vec4));
  up("invMass", combinedInvMass_.data(), N * sizeof(glm::vec4));
  up("typeFlag", combinedTypeFlag_.data(), N * sizeof(uint32_t));
  up("edgeData", combinedEdgeData_.data(), E * 3 * sizeof(uint32_t));
  up("tetData", combinedTetData_.data(), T * 4 * sizeof(uint32_t));
  up("tetRestVol", combinedTetRestVol_.data(), T * sizeof(float));

  // vel / predP / edgeLambda / tetLambda は 0 クリアのためデータ不要
  // (AttributeBuffer は VMA で確保するので未初期化のまま。
  //  シェーダー実行前に zero_lambdas がクリアし、predict が predP を設定する)
  std::vector<glm::vec4> zeros4(N, glm::vec4(0.0f));
  up("v", zeros4.data(), N * sizeof(glm::vec4));
  up("predP", zeros4.data(), N * sizeof(glm::vec4));

  std::vector<float> zerosF(std::max(E, T), 0.0f);
  up("edgeLambda", zerosF.data(), E * sizeof(float));
  up("tetLambda", zerosF.data(), T * sizeof(float));

  // CPU バッファは不要になったので解放
  combinedPos_.clear();
  combinedPos_.shrink_to_fit();
  combinedInvMass_.clear();
  combinedInvMass_.shrink_to_fit();
  combinedTypeFlag_.clear();
  combinedTypeFlag_.shrink_to_fit();
  combinedEdgeData_.clear();
  combinedEdgeData_.shrink_to_fit();
  combinedTetData_.clear();
  combinedTetData_.shrink_to_fit();
  combinedTetRestVol_.clear();
  combinedTetRestVol_.shrink_to_fit();
}

void SoftBodyEngine::computeBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier barrier{};
  barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}
