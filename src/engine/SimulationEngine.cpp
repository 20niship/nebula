#include "SimulationEngine.h"

#include <algorithm>
#include <cstring>
#include <glm/glm.hpp>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

// ─── 初期化 ────────────────────────────────────────────────────────────────

void SimulationEngine::init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, const ClothConfig& cfg) {
  cfg_           = cfg;
  particleRadius = cfg_.cellSize() * 0.5f;
  initEngineBase(device, allocator, descriptorPool, cmdPool, queue);

  initParticleBuffers(cmdPool, queue);
  initClothBuffers(cmdPool, queue);

  // Force (issue #30): 既定では空リスト。重力・風が必要な場合は呼び出し側が
  // addForce(GravityForce::FromDirection(...)) 等で登録する。
  initForces();

  auto load = [&](ComputePipeline& k, const std::string& name) { k.init(device, attrBuf_.descriptorSetLayout, shaderDir + "/" + name + ".spv"); };

  load(kSdfCollision_, "sdf_collision.comp");
  load(kHashCount_, "hash_count.comp");
  load(kHashScanLocal_, "hash_scan_local.comp");
  load(kHashScanGlobal_, "hash_scan_global.comp");
  load(kHashAddBase_, "hash_add_base.comp");
  load(kHashSort_, "hash_sort.comp");
  load(kSolveDensity_, "solve_density.comp");
  load(kSolveStretch_, "solve_stretch.comp");
  load(kUpdateVelocity_, "update_velocity.comp");
  load(kZeroLambdas_, "zero_lambdas.comp");
  load(kZeroCells_, "zero_cells.comp");

  descriptorSetLayout = attrBuf_.descriptorSetLayout;
  descriptorSet       = attrBuf_.descriptorSet;

  std::cout << "[Sim] Phase3: " << cfg_.clothVertCount() << " particles (" << cfg_.clothVertCount() << " cloth), " << clothMesh_.edgeCount() << " edges, " << clothMesh_.nColors << " colors" << std::endl;
}

void SimulationEngine::initParticleBuffers(VkCommandPool cmdPool, VkQueue queue) {
  // 共通粒子属性 (布 + 砂 共有)
  posIdx       = attrBuf_.addAttribute("P", sizeof(glm::vec4), cfg_.clothVertCount());
  velIdx       = attrBuf_.addAttribute("v", sizeof(glm::vec4), cfg_.clothVertCount());
  predPIdx_    = attrBuf_.addAttribute("predP", sizeof(glm::vec4), cfg_.clothVertCount());
  invMassIdx_  = attrBuf_.addAttribute("invMass", sizeof(glm::vec4), cfg_.clothVertCount());
  typeFlagIdx_ = attrBuf_.addAttribute("typeFlag", sizeof(uint32_t), cfg_.clothVertCount());

  const uint32_t N_GROUPS = (cfg_.totalCells() + 255u) / 256u;
  cellCountIdx_           = attrBuf_.addAttribute("cellCount", sizeof(uint32_t), cfg_.totalCells());
  cellOffsetIdx_          = attrBuf_.addAttribute("cellOffset", sizeof(uint32_t), cfg_.totalCells() + N_GROUPS);
  sortedIdxIdx_           = attrBuf_.addAttribute("sortedIdx", sizeof(uint32_t), cfg_.clothVertCount());

  // 布頂点の初期データは initClothBuffers() で上書きするのでここでは 0 クリア
  std::vector<glm::vec4> zeros(cfg_.clothVertCount(), glm::vec4(0.0f));
  attrBuf_.upload("v", zeros.data(), sizeof(glm::vec4) * cfg_.clothVertCount(), cmdPool, queue);
}

void SimulationEngine::initClothBuffers(VkCommandPool cmdPool, VkQueue queue) {
  // ClothMesh 生成
  float clothZ = cfg_.world_size * 0.85f; // 箱上部付近 (Z-up)
  clothMesh_.build(cfg_.cloth_grid_n, 0.065f, cfg_.world_size * 0.5f, cfg_.world_size * 0.5f, clothZ);

  // 上端全行 (i==0) をピン留め (invMass=0)
  for(int j = 0; j < (int)cfg_.cloth_grid_n; ++j) clothMesh_.invMasses[clothMesh_.idx(0, j)].x = 0.0f;

  nColors_        = clothMesh_.nColors;
  colorBatch_cpu_ = clothMesh_.colorBatch;

  // 布頂点データをアップロード (オフセット 0 = 布が先頭)
  uint32_t N = clothMesh_.vertexCount();

  // typeFlag: uint32 per 粒子で格納 (readUint(typeFlagIdx, i) = data[i] が正しく機能する)
  std::vector<uint32_t> typeVec(N);
  for(uint32_t i = 0; i < N; ++i) typeVec[i] = clothMesh_.typeFlags[i]; // 2 = Cloth

  attrBuf_.upload("P", clothMesh_.positions.data(), sizeof(glm::vec4) * N, cmdPool, queue);
  attrBuf_.upload("invMass", clothMesh_.invMasses.data(), sizeof(glm::vec4) * N, cmdPool, queue);
  attrBuf_.upload("typeFlag", typeVec.data(), sizeof(uint32_t) * N, cmdPool, queue);

  // 布専用バッファ
  uint32_t E       = clothMesh_.edgeCount();
  stretchEdgesIdx_ = attrBuf_.addAttribute("stretchEdges", sizeof(uint32_t), E * 3);
  lambdasIdx_      = attrBuf_.addAttribute("lambdas", sizeof(float), E);

  attrBuf_.upload("stretchEdges", clothMesh_.edgeData.data(), sizeof(uint32_t) * clothMesh_.edgeData.size(), cmdPool, queue);
  // lambdas はゼロ初期化 (毎フレーム FillBuffer でリセット)
}

VkBuffer SimulationEngine::getPositionBuffer() const { return attrBuf_.getBuffer("P"); }

// ─── デバッグ: 代表頂点の位置・速度を stdout に出力 ─────────────────────────

void SimulationEngine::debugPrintVertices(VkCommandPool cmdPool, VkQueue queue) const {
  const uint32_t N = cfg_.cloth_grid_n;

  struct DebugVert {
    const char* name;
    uint32_t idx;
  };
  static const DebugVert verts[] = {
    {"corner[0,0]    (pinned)", (uint32_t)(0 * N + 0)},     {"corner[0,N-1]  (pinned)", (uint32_t)(0 * N + N - 1)},       {"corner[N-1,0]          ", (uint32_t)((N - 1) * N + 0)}, {"corner[N-1,N-1]        ", (uint32_t)((N - 1) * N + N - 1)},
    {"mid_top                ", (uint32_t)(0 * N + N / 2)}, {"mid_bottom             ", (uint32_t)((N - 1) * N + N / 2)}, {"mid_left               ", (uint32_t)((N / 2) * N + 0)}, {"mid_right              ", (uint32_t)((N / 2) * N + N - 1)},
  };
  constexpr uint32_t NVERT = (uint32_t)(sizeof(verts) / sizeof(verts[0]));

  // ステージングバッファ: pos × NVERT + vel × NVERT (各 vec4 = 16 bytes)
  VkDeviceSize stagingSize = (VkDeviceSize)NVERT * 2 * sizeof(glm::vec4);

  VkBufferCreateInfo bufInfo{};
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufInfo.size  = stagingSize;
  bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  VmaAllocationCreateInfo allocCI{};
  allocCI.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;

  VkBuffer stageBuf;
  VmaAllocation stageAlloc;
  vmaCreateBuffer(allocator_, &bufInfo, &allocCI, &stageBuf, &stageAlloc, nullptr);

  // 1ショットコマンドバッファ
  VkCommandBufferAllocateInfo ai{};
  ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool        = cmdPool;
  ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(device_, &ai, &cmd);

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);

  // pos と vel をそれぞれ NVERT 頂点分コピー
  VkBuffer posBuf = attrBuf_.getBuffer("P");
  VkBuffer velBuf = attrBuf_.getBuffer("v");

  std::vector<VkBufferCopy> posRegions(NVERT), velRegions(NVERT);
  for(uint32_t i = 0; i < NVERT; ++i) {
    VkDeviceSize src = (VkDeviceSize)verts[i].idx * sizeof(glm::vec4);
    posRegions[i]    = {src, (VkDeviceSize)i * sizeof(glm::vec4), sizeof(glm::vec4)};
    velRegions[i]    = {src, (VkDeviceSize)(i + NVERT) * sizeof(glm::vec4), sizeof(glm::vec4)};
  }
  vkCmdCopyBuffer(cmd, posBuf, stageBuf, NVERT, posRegions.data());
  vkCmdCopyBuffer(cmd, velBuf, stageBuf, NVERT, velRegions.data());

  vkEndCommandBuffer(cmd);

  VkSubmitInfo si{};
  si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers    = &cmd;
  vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);

  vkFreeCommandBuffers(device_, cmdPool, 1, &cmd);

  // マップして読み出し
  void* mapped;
  vmaMapMemory(allocator_, stageAlloc, &mapped);
  auto* posData = reinterpret_cast<glm::vec4*>(mapped);
  auto* velData = posData + NVERT;

  static int printCount = 0;
  printf("\n=== debugPrintVertices #%d ===\n", ++printCount);
  printf("%-32s  pos(x,y,z)                     |vel|(m/s)\n", "vertex");
  printf("%-32s  %-30s  %s\n", "--------------------------------", "------------------------------", "----------");
  for(uint32_t i = 0; i < NVERT; ++i) {
    glm::vec4 p = posData[i];
    glm::vec4 v = velData[i];
    float speed = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    printf("%-32s  (%8.4f, %8.4f, %8.4f)    %10.5f\n", verts[i].name, p.x, p.y, p.z, speed);
  }
  fflush(stdout);

  vmaUnmapMemory(allocator_, stageAlloc);
  vmaDestroyBuffer(allocator_, stageBuf, stageAlloc);
}

// ─── クリーンアップ ─────────────────────────────────────────────────────────

void SimulationEngine::cleanup() {
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
  cleanupEngineBase();
}

// ─── バリア ────────────────────────────────────────────────────────────────

void SimulationEngine::computeBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier b{};
  b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

// ─── 1フレームのシミュレーション ──────────────────────────────────────────

void SimulationEngine::step(VkCommandBuffer cmd, float dt) {
  auto ds = attrBuf_.descriptorSet;

  uploadForces(cmd, dt);

  float subDt = dt / std::max(1, numSubsteps);

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
    pc.particleCount     = cfg_.clothVertCount();
    pc.gridRes           = cfg_.grid_res;
    pc.stretchEdgesIdx   = stretchEdgesIdx_;
    pc.lambdasIdx        = lambdasIdx_;
    pc.dt                = subDt;
    pc.cellSize          = cfg_.cellSize();
    pc.worldMin          = 0.0f;
    pc.worldMax          = cfg_.world_size;
    pc.restitution       = restitution;
    pc.friction          = friction;
    pc.particleRadius    = particleRadius;
    pc.forceBufIdx       = forcesIdx_;
    pc.couplingForceIdx  = 0;
    pc.clothVertexCount  = cfg_.clothVertCount();
    pc.edgeCount         = (uint32_t)clothMesh_.edgeCount();
    pc.stretchCompliance = stretchCompliance;
    pc.bendCompliance    = bendCompliance;
    pc.forceCount        = (uint32_t)forces_.size();
    pc.linearDamping     = 0.02f; // 布の従来挙動を維持（update_velocity 共用シェーダー）

    // ① Predict (重力 + 風力、ピン留め)
    kPredict_.dispatch(cmd, ds, pc, cfg_.clothVertCount());
    computeBarrier(cmd);

    // ② SDF 衝突
    kSdfCollision_.dispatch(cmd, ds, pc, cfg_.clothVertCount());
    computeBarrier(cmd);

    // ③ λ リセット (compute シェーダーで一括ゼロクリア)
    // vkCmdFillBuffer は blit エンコーダーを起動して MoltenVK のコスト高いため代替
    kZeroLambdas_.dispatch(cmd, ds, pc, pc.edgeCount);
    computeBarrier(cmd);

    // ④ XPBD 距離拘束 (全色 × solverIterations 反復)
    // 最適化: グラフ彩色により同一反復内の異なる色は頂点を共有しない
    // → 色ループ内バリアを廃止し反復ごとに 1 バリアのみ (120→10 に削減)
    for(int iter = 0; iter < solverIterations; ++iter) {
      for(int color = 0; color < nColors_; ++color) {
        uint32_t start = colorBatch_cpu_[color];
        uint32_t end   = colorBatch_cpu_[color + 1];
        uint32_t cnt   = end - start;
        if(cnt == 0) continue;

        pc.batchEdgeStart = start;
        pc.batchEdgeEnd   = end;
        // 色 8–11 はベンドエッジ: bendCompliance を使う
        pc.stretchCompliance = (color >= 8) ? bendCompliance : stretchCompliance;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kSolveStretch_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kSolveStretch_.pipelineLayout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, kSolveStretch_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
        vkCmdDispatch(cmd, (cnt + 255) / 256, 1, 1);
        // バリア不要: 同一反復内で異なる色の辺は頂点非共有
      }
      // 反復間のバリア: 次の反復が今回の位置書き込みを読む
      computeBarrier(cmd);
    }

    // ⑤ SDF 再適用 (拘束後の境界修正)
    kSdfCollision_.dispatch(cmd, ds, pc, cfg_.clothVertCount());
    computeBarrier(cmd);

    // ⑥ 自己衝突 (enableSelfCollision が true の場合のみ)
    if(enableSelfCollision) {
      // compute シェーダーでゼロクリア (blit エンコーダー遷移を排除)
      kZeroCells_.dispatch(cmd, ds, pc, cfg_.totalCells());
      computeBarrier(cmd);
      kHashCount_.dispatch(cmd, ds, pc, cfg_.clothVertCount());
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
      kHashSort_.dispatch(cmd, ds, pc, cfg_.clothVertCount());
      computeBarrier(cmd);

      for(int i = 0; i < 2; ++i) {
        kSolveDensity_.dispatch(cmd, ds, pc, cfg_.clothVertCount());
        computeBarrier(cmd);
      }
      kSdfCollision_.dispatch(cmd, ds, pc, cfg_.clothVertCount());
      computeBarrier(cmd);
    }

    // ⑦ Update Velocity
    kUpdateVelocity_.dispatch(cmd, ds, pc, cfg_.clothVertCount());
    computeBarrier(cmd);
  }
}
