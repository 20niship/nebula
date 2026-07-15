#pragma once

#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "AttributeBuffer.h"
#include "ClothMesh.h"
#include "ComputePipeline.h"
#include "Force.h"
#include "SimPC.h"

// h = cellSize = world_size/grid_res, 流体粒子間隔 d = world_size/fluid_nx
// 必要: h >= 2d → grid_res <= fluid_nx/2
struct MultiPhysicsConfig {
  uint32_t cloth_grid_n = 64;
  uint32_t fluid_nx     = 32;
  uint32_t fluid_ny     = 16;
  uint32_t fluid_nz     = 32;
  float world_size      = 10.0f;
  uint32_t grid_res     = 16;

  uint32_t clothCount() const { return cloth_grid_n * cloth_grid_n; }
  uint32_t fluidCount() const { return fluid_nx * fluid_ny * fluid_nz; }
  uint32_t totalMax() const { return clothCount() + fluidCount(); }
  uint32_t totalCells() const { return grid_res * grid_res * grid_res; }
  float cellSize() const { return world_size / float(grid_res); }
  uint32_t fluidStart() const { return clothCount(); }
};

class MultiPhysicsEngine {
public:
  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, const MultiPhysicsConfig& cfg = {});
  void cleanup();

  void step(VkCommandBuffer cmd, float dt);

  const MultiPhysicsConfig& config() const { return cfg_; }

  // ImGui から調整可能なパラメータ
  float gravity           = -9.8f;
  float restitution       = 0.2f;
  float friction          = 0.1f;
  float rho0              = 200.0f;
  float viscosityC        = 0.01f;
  float stretchCompliance = 1e-4f;
  float bendCompliance    = 1e-2f;
  float windX             = 0.0f;
  float windZ             = 0.0f;
  int pbfIterations       = 4;
  int numSubsteps         = 6;
  bool enableCoupling     = true;

  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet             = VK_NULL_HANDLE;
  uint32_t posIdx                           = 0;
  uint32_t velIdx                           = 0;

  VkBuffer getPositionBuffer() const;
  const ClothMesh& getClothMesh() const { return clothMesh_; }

  // Force (issue #30): gravity/windX/windZ 以外の任意の力を追加する
  void addForce(std::shared_ptr<Force> f);
  void removeForce(const std::shared_ptr<Force>& f);
  void setForces(std::vector<std::shared_ptr<Force>> forces);
  void clearForces();

private:
  MultiPhysicsConfig cfg_;
  VkDevice device_        = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkCommandPool cmdPool_  = VK_NULL_HANDLE;
  VkQueue queue_          = VK_NULL_HANDLE;

  AttributeBuffer attrBuf_;
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

  // Force (issue #30): gravity/windX/windZ互換の既定Forceを常時登録
  static constexpr uint32_t kMaxForces = 32;
  std::vector<std::shared_ptr<Force>> forces_;
  std::shared_ptr<GravityForce> legacyGravity_;
  std::shared_ptr<ConstantWindForce> legacyWind_;
  uint32_t forcesIdx_ = 0;
  void rebuildForceShader();
  void uploadForces();

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
