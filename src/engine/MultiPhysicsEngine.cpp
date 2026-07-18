#include "MultiPhysicsEngine.h"

#include <algorithm>
#include <glm/glm.hpp>
#include <stdexcept>
#include <vector>

// ── バリアヘルパー ────────────────────────────────────────────────────────────

void MultiPhysicsEngine::computeBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier b{};
  b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void MultiPhysicsEngine::fillBarrier(VkCommandBuffer cmd, VkBuffer buf) {
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

// ── Push Constants ビルド ─────────────────────────────────────────────────────

SimPC MultiPhysicsEngine::buildPC(float subDt) const {
  SimPC pc{};
  pc.posIdx            = posIdx;
  pc.velIdx            = velIdx;
  pc.predPIdx          = predPIdx_;
  pc.invMassIdx        = invMassIdx_;
  pc.typeFlagIdx       = typeFlagIdx_;
  pc.cellCountIdx      = cellCountIdx_;
  pc.cellOffsetIdx     = cellOffsetIdx_;
  pc.sortedIdxIdx      = sortedIdxIdx_;
  pc.particleCount     = cfg_.totalMax();
  pc.hashCells         = cfg_.totalCells();
  pc.stretchEdgesIdx   = stretchEdgesIdx_;
  pc.lambdasIdx        = clothLambdasIdx_;
  pc.dt                = subDt;
  pc.cellSize          = cfg_.cellSize;
  pc.gridRes           = cfg_.gridRes();
  pc.worldMin          = glm::vec3(0.0f);
  pc.worldMax          = cfg_.domainSize;
  pc.restitution       = restitution;
  pc.friction          = friction;
  pc.particleRadius    = cfg_.cellSize * 0.5f;
  pc.forceBufIdx       = forcesIdx_;
  pc.couplingForceIdx  = couplingForceIdx_;
  pc.clothVertexCount  = cfg_.clothCount();
  pc.edgeCount         = (uint32_t)clothMesh_.edgeCount();
  pc.stretchCompliance = rho0; // coupling_cloth / pbf_density が参照
  pc.bendCompliance    = viscosityC;
  pc.forceCount        = (uint32_t)forces_.size();
  pc.densityIdx        = densityIdx_;
  pc.lambdaPbfIdx      = lambdaPbfIdx_;
  pc.boundaryStart     = cfg_.fluidStart(); // 境界粒子なし; 流体開始でも使用される
  pc.linearDamping     = 0.6f;              // 布の従来挙動を維持（update_velocity 共用）
  pc.cfmEpsilon        = 100.0f;            // pbf_density の CFM ε（0 除算回避）
  return pc;
}

// ── 初期化 ───────────────────────────────────────────────────────────────────

void MultiPhysicsEngine::init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, const MultiPhysicsConfig& cfg) {
  cfg_ = cfg;
  initEngineBase(device, allocator, descriptorPool, cmdPool, queue);

  const uint32_t N_GROUPS = (cfg_.totalCells() + 255u) / 256u;

  // 共通属性バッファ
  posIdx            = attrBuf_.addAttribute("P", sizeof(glm::vec4), cfg_.totalMax());
  velIdx            = attrBuf_.addAttribute("v", sizeof(glm::vec4), cfg_.totalMax());
  predPIdx_         = attrBuf_.addAttribute("predP", sizeof(glm::vec4), cfg_.totalMax());
  invMassIdx_       = attrBuf_.addAttribute("invMass", sizeof(glm::vec4), cfg_.totalMax());
  typeFlagIdx_      = attrBuf_.addAttribute("typeFlag", sizeof(uint32_t), cfg_.totalMax());
  cellCountIdx_     = attrBuf_.addAttribute("cellCount", sizeof(uint32_t), cfg_.totalCells());
  cellOffsetIdx_    = attrBuf_.addAttribute("cellOffset", sizeof(uint32_t), cfg_.totalCells() + N_GROUPS);
  sortedIdxIdx_     = attrBuf_.addAttribute("sortedIdx", sizeof(uint32_t), cfg_.totalMax());
  densityIdx_       = attrBuf_.addAttribute("density", sizeof(float), cfg_.totalMax());
  lambdaPbfIdx_     = attrBuf_.addAttribute("lambdaPbf", sizeof(float), cfg_.totalMax());
  couplingForceIdx_ = attrBuf_.addAttribute("couplingF", sizeof(glm::vec4), cfg_.clothCount());

  initClothParticles(cmdPool, queue);
  initFluidParticles(cmdPool, queue);

  // Force (issue #30): 既定では空リスト。重力・風が必要な場合は呼び出し側が
  // addForce(GravityForce::FromDirection(...)) 等で登録する。
  initForces();

  auto load = [&](ComputePipeline& k, const char* name) { k.init(device, attrBuf_.descriptorSetLayout, shaderDir + "/" + name + ".spv"); };
  load(kSdfCollision_, "sdf_collision.comp");
  load(kHashCount_, "hash_count.comp");
  load(kHashScanLocal_, "hash_scan_local.comp");
  load(kHashScanGlobal_, "hash_scan_global.comp");
  load(kHashAddBase_, "hash_add_base.comp");
  load(kHashSort_, "hash_sort.comp");
  load(kPbfDensity_, "pbf_density.comp");
  load(kPbfDeltaP_, "pbf_delta_p.comp");
  load(kCouplingCloth_, "coupling_cloth.comp");
  load(kSolveStretch_, "solve_stretch.comp");
  load(kPbfViscosity_, "pbf_viscosity.comp");
  load(kUpdateVelocity_, "update_velocity.comp");

  descriptorSetLayout = attrBuf_.descriptorSetLayout;
  descriptorSet       = attrBuf_.descriptorSet;
}

void MultiPhysicsEngine::initClothParticles(VkCommandPool cmdPool, VkQueue queue) {
  // 布: XY 平面、Z = domainSize.z * 0.78 付近
  // spacing は XY 最短軸を基準にし、非立方体ドメインでも布が短辺からはみ出さないようにする
  const float minXY = std::min(cfg_.domainSize.x, cfg_.domainSize.y);
  clothMesh_.build(cfg_.cloth_grid_n, minXY / float(cfg_.cloth_grid_n) * 0.9f, cfg_.domainSize.x * 0.5f, cfg_.domainSize.y * 0.5f, cfg_.domainSize.z * 0.78f);

  // 上端全行 (i==0) をピン留め (invMass=0)
  for(int j = 0; j < (int)cfg_.cloth_grid_n; ++j) clothMesh_.invMasses[clothMesh_.idx(0, j)].x = 0.0f;

  nColors_        = clothMesh_.nColors;
  colorBatch_cpu_ = clothMesh_.colorBatch;

  uint32_t N = clothMesh_.vertexCount();

  // typeFlag を uint 配列で格納
  std::vector<uint32_t> flags(N, 2u); // typeFlag = 2 (Cloth) // typeFlag = 2 (Cloth)

  attrBuf_.upload("P", clothMesh_.positions.data(), sizeof(glm::vec4) * N, cmdPool, queue);
  attrBuf_.upload("v", std::vector<glm::vec4>(N, glm::vec4(0.0f)).data(), sizeof(glm::vec4) * N, cmdPool, queue);
  attrBuf_.upload("invMass", clothMesh_.invMasses.data(), sizeof(glm::vec4) * N, cmdPool, queue);
  attrBuf_.upload("typeFlag", flags.data(), sizeof(uint32_t) * N, cmdPool, queue);

  // 布エッジバッファ
  uint32_t E       = clothMesh_.edgeCount();
  stretchEdgesIdx_ = attrBuf_.addAttribute("stretchEdges", sizeof(uint32_t), E * 3);
  clothLambdasIdx_ = attrBuf_.addAttribute("clothLambdas", sizeof(float), E);

  attrBuf_.upload("stretchEdges", clothMesh_.edgeData.data(), sizeof(uint32_t) * clothMesh_.edgeData.size(), cmdPool, queue);
}

void MultiPhysicsEngine::initFluidParticles(VkCommandPool cmdPool, VkQueue queue) {
  // 流体: XY平面に32×32、Z方向に16層積み上げ (Z-up座標系)
  // 布は Z≈7.8 に配置。流体は Z=0.5 から積み上げ → 重力で底(Z≈0)に溜まる
  const float spacing = cfg_.domainSize.x / float(cfg_.fluid_nx);
  const float startX  = (cfg_.domainSize.x - (cfg_.fluid_nx - 1) * spacing) * 0.5f;
  const float startY  = (cfg_.domainSize.y - (cfg_.fluid_nz - 1) * spacing) * 0.5f;
  const float startZ  = 0.5f; // 布(Z=7.8)の下から積み上げ

  std::vector<glm::vec4> pos(cfg_.fluidCount());
  std::vector<glm::vec4> invM(cfg_.fluidCount(), glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
  std::vector<uint32_t> flags(cfg_.fluidCount(), 1u); // typeFlag = 1 (Fluid)
  std::vector<glm::vec4> vel(cfg_.fluidCount(), glm::vec4(0.0f));

  uint32_t idx = 0;
  for(uint32_t layer = 0; layer < cfg_.fluid_ny; ++layer)
    for(uint32_t iy = 0; iy < cfg_.fluid_nz; ++iy)
      for(uint32_t ix = 0; ix < cfg_.fluid_nx; ++ix) {
        pos[idx++] = glm::vec4(startX + ix * spacing, startY + iy * spacing, startZ + layer * spacing, 1.0f);
      }

  // 流体粒子はバッファの [MP_FLUID_START, MP_TOTAL_MAX) に配置
  VkDeviceSize byteOff = cfg_.fluidStart() * sizeof(glm::vec4);
  VkDeviceSize flagOff = cfg_.fluidStart() * sizeof(uint32_t);

  attrBuf_.uploadAt("P", pos.data(), sizeof(glm::vec4) * cfg_.fluidCount(), byteOff, cmdPool, queue);
  attrBuf_.uploadAt("v", vel.data(), sizeof(glm::vec4) * cfg_.fluidCount(), byteOff, cmdPool, queue);
  attrBuf_.uploadAt("invMass", invM.data(), sizeof(glm::vec4) * cfg_.fluidCount(), byteOff, cmdPool, queue);
  attrBuf_.uploadAt("typeFlag", flags.data(), sizeof(uint32_t) * cfg_.fluidCount(), flagOff, cmdPool, queue);
}

VkBuffer MultiPhysicsEngine::getPositionBuffer() const { return attrBuf_.getBuffer("P"); }

// ── クリーンアップ ────────────────────────────────────────────────────────────

void MultiPhysicsEngine::cleanup() {
  kPredict_.cleanup();
  kSdfCollision_.cleanup();
  kHashCount_.cleanup();
  kHashScanLocal_.cleanup();
  kHashScanGlobal_.cleanup();
  kHashAddBase_.cleanup();
  kHashSort_.cleanup();
  kPbfDensity_.cleanup();
  kPbfDeltaP_.cleanup();
  kCouplingCloth_.cleanup();
  kSolveStretch_.cleanup();
  kPbfViscosity_.cleanup();
  kUpdateVelocity_.cleanup();
  cleanupEngineBase();
}

// ── 1フレームのシミュレーション ───────────────────────────────────────────────

void MultiPhysicsEngine::step(VkCommandBuffer cmd, float dt) {
  auto ds = attrBuf_.descriptorSet;

  uploadForces(dt);

  float subDt = dt / float(std::max(1, numSubsteps));

  for(int sub = 0; sub < numSubsteps; ++sub) {
    SimPC pc = buildPC(subDt);

    // ① 各サブステップ先頭: λ クリア
    vkCmdFillBuffer(cmd, attrBuf_.getBuffer("clothLambdas"), 0, VK_WHOLE_SIZE, 0);
    fillBarrier(cmd, attrBuf_.getBuffer("clothLambdas"));

    vkCmdFillBuffer(cmd, attrBuf_.getBuffer("cellCount"), 0, VK_WHOLE_SIZE, 0);
    fillBarrier(cmd, attrBuf_.getBuffer("cellCount"));

    // ② Predict (重力・風力・CFL クランプ)
    kPredict_.dispatch(cmd, ds, pc, cfg_.totalMax());
    computeBarrier(cmd);

    // ③ SDF ボックス衝突
    kSdfCollision_.dispatch(cmd, ds, pc, cfg_.totalMax());
    computeBarrier(cmd);

    // ④ 空間ハッシュ構築 (全粒子)
    kHashCount_.dispatch(cmd, ds, pc, cfg_.totalMax());
    computeBarrier(cmd);

    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanLocal_.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanLocal_.pipelineLayout, 0, 1, &ds, 0, nullptr);
      vkCmdPushConstants(cmd, kHashScanLocal_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
      vkCmdDispatch(cmd, (cfg_.totalCells() + 255u) / 256u, 1, 1);
    }
    computeBarrier(cmd);

    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanGlobal_.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanGlobal_.pipelineLayout, 0, 1, &ds, 0, nullptr);
      vkCmdPushConstants(cmd, kHashScanGlobal_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
      vkCmdDispatch(cmd, 1, 1, 1);
    }
    computeBarrier(cmd); // exclusive prefix を書き戻してから kHashAddBase_ が読む

    kHashAddBase_.dispatch(cmd, ds, pc, cfg_.totalCells());
    computeBarrier(cmd);

    kHashSort_.dispatch(cmd, ds, pc, cfg_.totalMax());
    computeBarrier(cmd);

    // ⑤ PBF + XPBD 連立ループ
    for(int iter = 0; iter < pbfIterations; ++iter) {
      // PBF: 流体密度・λ 計算
      kPbfDensity_.dispatch(cmd, ds, pc, cfg_.totalMax());
      computeBarrier(cmd);

      // PBF: 流体 ΔP 適用 (布粒子を動く境界として扱う)
      kPbfDeltaP_.dispatch(cmd, ds, pc, cfg_.totalMax());
      computeBarrier(cmd);

      // 連成: 流体 lambda → 布頂点 ΔP
      if(enableCoupling) {
        pc.stretchCompliance = rho0; // coupling_cloth が rho0 として参照
        kCouplingCloth_.dispatch(cmd, ds, pc, cfg_.clothCount());
        computeBarrier(cmd);
      }

      // XPBD: 布距離拘束 (全色バッチ)
      for(int color = 0; color < nColors_; ++color) {
        uint32_t start = colorBatch_cpu_[color];
        uint32_t end   = colorBatch_cpu_[color + 1];
        uint32_t cnt   = end - start;
        if(cnt == 0) continue;

        pc.batchEdgeStart = start;
        pc.batchEdgeEnd   = end;
        // 色 8-11 はベンドエッジ
        pc.stretchCompliance = (color >= 8) ? bendCompliance : stretchCompliance;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kSolveStretch_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kSolveStretch_.pipelineLayout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, kSolveStretch_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
        vkCmdDispatch(cmd, (cnt + 255u) / 256u, 1, 1);
        computeBarrier(cmd);
      }

      // 連成後の布が SDF 壁を貫通しないよう補正
      kSdfCollision_.dispatch(cmd, ds, pc, cfg_.clothCount());
      computeBarrier(cmd);
    }

    // ⑥ XSPH 粘性 (流体のみ)
    pc.stretchCompliance = rho0;
    kPbfViscosity_.dispatch(cmd, ds, pc, cfg_.totalMax());
    computeBarrier(cmd);

    // ⑦ 速度更新 (全粒子)
    kUpdateVelocity_.dispatch(cmd, ds, pc, cfg_.totalMax());
    computeBarrier(cmd);
  }
}
