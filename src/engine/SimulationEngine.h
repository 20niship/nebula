#pragma once

#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "../core/Domain.h"
#include "ClothMesh.h"
#include "ComputePipeline.h"
#include "EngineBase.h"
#include "SimPC.h"

// issue #46: ドメインは domainSize(vec3, 物理サイズ[m]) + cellSize(float, 全軸共通の
// セルサイズ[m]) で指定する。各軸の実セル数は gridRes() (= domain::gridRes()) から導出。
struct ClothConfig {
  uint32_t cloth_grid_n = 128;

  glm::vec3 domainSize{10.0f, 10.0f, 10.0f}; // ドメイン物理サイズ [m] (旧 world_size)
  float cellSize = 10.0f / 64.0f;            // 全軸共通のセルサイズ [m] (旧 grid_res の逆算値)

  uint32_t clothVertCount() const { return cloth_grid_n * cloth_grid_n; }
  glm::uvec3 gridRes() const { return domain::gridRes(domainSize, cellSize); }
  // 空間ハッシュバッファの実要素数 (= cubeRes^3。gridRes.x*y*zではない点に注意)
  uint32_t totalCells() const { return domain::hashCells(gridRes()); }
};

// 重力・風は addForce() で GravityForce/ConstantWindForce を登録すること
// (issue #30 レビュー対応: gravity/windX/windZ の public メンバは廃止)。
class SimulationEngine : public EngineBase {
public:
  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, const ClothConfig& cfg = {});
  void cleanup();

  void step(VkCommandBuffer cmd, float dt);

  const ClothConfig& config() const { return cfg_; }

  // ImGui から書き換え可能
  float restitution        = 0.3f;
  float friction           = 0.1f;
  float particleRadius     = 0.0f; // init() で cfg_.cellSize*0.5 に設定
  float stretchCompliance  = 1e-4f;
  float bendCompliance     = 1e-2f;
  int solverIterations     = 3;
  int numSubsteps          = 15;
  bool enableSelfCollision = true;

  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet             = VK_NULL_HANDLE;
  uint32_t posIdx                           = 0;
  uint32_t velIdx                           = 0;

  VkBuffer getPositionBuffer() const;
  const ClothMesh& getClothMesh() const { return clothMesh_; }
  void debugPrintVertices(VkCommandPool cmdPool, VkQueue queue) const;

protected:
  ComputePipeline& forceTargetPipeline() override { return kPredict_; }
  const char* forceShaderName() const override { return "predict.comp"; }

private:
  ClothConfig cfg_;
  ClothMesh clothMesh_;

  uint32_t predPIdx_        = 0;
  uint32_t invMassIdx_      = 0;
  uint32_t typeFlagIdx_     = 0;
  uint32_t cellCountIdx_    = 0;
  uint32_t cellOffsetIdx_   = 0;
  uint32_t sortedIdxIdx_    = 0;
  uint32_t stretchEdgesIdx_ = 0;
  uint32_t lambdasIdx_      = 0;

  std::vector<uint32_t> colorBatch_cpu_;
  int nColors_ = 0;

  ComputePipeline kPredict_;
  ComputePipeline kSdfCollision_;
  ComputePipeline kHashCount_;
  ComputePipeline kHashScanLocal_;
  ComputePipeline kHashScanGlobal_;
  ComputePipeline kHashAddBase_;
  ComputePipeline kHashSort_;
  ComputePipeline kSolveDensity_;
  ComputePipeline kSolveStretch_;
  ComputePipeline kUpdateVelocity_;
  ComputePipeline kZeroLambdas_;
  ComputePipeline kZeroCells_;

  void initParticleBuffers(VkCommandPool cmdPool, VkQueue queue);
  void initClothBuffers(VkCommandPool cmdPool, VkQueue queue);
  void computeBarrier(VkCommandBuffer cmd);
};
