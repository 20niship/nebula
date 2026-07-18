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
// セルサイズ[m]) で指定する。h = cellSize, 流体粒子間隔 d = domainSize.x/fluid_nx
// 必要: h >= 2d → cellSize >= 2 * domainSize.x/fluid_nx
// cloth(XPBD)とfluid(PBF)は同一Domainを共有する (coupling_cloth.compが両者を同じ
// セルで近傍探索するため)。
struct MultiPhysicsConfig {
  uint32_t cloth_grid_n = 64;
  uint32_t fluid_nx     = 32;
  uint32_t fluid_ny     = 16;
  uint32_t fluid_nz     = 32;

  glm::vec3 domainSize{10.0f, 10.0f, 10.0f}; // ドメイン物理サイズ [m] (旧 world_size)
  float cellSize = 10.0f / 16.0f;            // 全軸共通のセルサイズ [m] (旧 grid_res の逆算値)

  uint32_t clothCount() const { return cloth_grid_n * cloth_grid_n; }
  uint32_t fluidCount() const { return fluid_nx * fluid_ny * fluid_nz; }
  uint32_t totalMax() const { return clothCount() + fluidCount(); }
  glm::uvec3 gridRes() const { return domain::gridRes(domainSize, cellSize); }
  // 空間ハッシュバッファの実要素数 (= cubeRes^3。gridRes.x*y*zではない点に注意)
  uint32_t totalCells() const { return domain::hashCells(gridRes()); }
  uint32_t fluidStart() const { return clothCount(); }
};

// 重力・風は addForce() で GravityForce/ConstantWindForce を登録すること
// (issue #30 レビュー対応: gravity/windX/windZ の public メンバは廃止)。
class MultiPhysicsEngine : public EngineBase {
public:
  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, const MultiPhysicsConfig& cfg = {});
  void cleanup();

  void step(VkCommandBuffer cmd, float dt);

  const MultiPhysicsConfig& config() const { return cfg_; }

  // ImGui から調整可能なパラメータ
  float restitution       = 0.2f;
  float friction          = 0.1f;
  float rho0              = 200.0f;
  float viscosityC        = 0.01f;
  float stretchCompliance = 1e-4f;
  float bendCompliance    = 1e-2f;
  int pbfIterations       = 4;
  int numSubsteps         = 6;
  bool enableCoupling     = true;

  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet             = VK_NULL_HANDLE;
  uint32_t posIdx                           = 0;
  uint32_t velIdx                           = 0;

  VkBuffer getPositionBuffer() const;
  const ClothMesh& getClothMesh() const { return clothMesh_; }

protected:
  ComputePipeline& forceTargetPipeline() override { return kPredict_; }
  const char* forceShaderName() const override { return "predict.comp"; }

private:
  MultiPhysicsConfig cfg_;
  ClothMesh clothMesh_;

  uint32_t predPIdx_         = 0;
  uint32_t invMassIdx_       = 0;
  uint32_t typeFlagIdx_      = 0;
  uint32_t cellCountIdx_     = 0;
  uint32_t cellOffsetIdx_    = 0;
  uint32_t sortedIdxIdx_     = 0;
  uint32_t densityIdx_       = 0;
  uint32_t lambdaPbfIdx_     = 0;
  uint32_t stretchEdgesIdx_  = 0;
  uint32_t clothLambdasIdx_  = 0;
  uint32_t couplingForceIdx_ = 0;

  std::vector<uint32_t> colorBatch_cpu_;
  int nColors_ = 0;

  ComputePipeline kPredict_;
  ComputePipeline kSdfCollision_;
  ComputePipeline kHashCount_;
  ComputePipeline kHashScanLocal_;
  ComputePipeline kHashScanGlobal_;
  ComputePipeline kHashAddBase_;
  ComputePipeline kHashSort_;
  ComputePipeline kPbfDensity_;
  ComputePipeline kPbfDeltaP_;
  ComputePipeline kCouplingCloth_;
  ComputePipeline kSolveStretch_;
  ComputePipeline kPbfViscosity_;
  ComputePipeline kUpdateVelocity_;

  void initClothParticles(VkCommandPool cmdPool, VkQueue queue);
  void initFluidParticles(VkCommandPool cmdPool, VkQueue queue);
  void computeBarrier(VkCommandBuffer cmd);
  void fillBarrier(VkCommandBuffer cmd, VkBuffer buf);
  SimPC buildPC(float subDt) const;
};
