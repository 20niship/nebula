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

struct ClothConfig {
  uint32_t cloth_grid_n = 128;
  float world_size      = 10.0f;
  uint32_t grid_res     = 64;

  uint32_t clothVertCount() const { return cloth_grid_n * cloth_grid_n; }
  uint32_t totalCells() const { return grid_res * grid_res * grid_res; }
  float cellSize() const { return world_size / float(grid_res); }
};

class SimulationEngine {
public:
  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, const ClothConfig& cfg = {});
  void cleanup();

  void step(VkCommandBuffer cmd, float dt);

  const ClothConfig& config() const { return cfg_; }

  // ImGui から書き換え可能
  float gravity            = -9.8f;
  float restitution        = 0.3f;
  float friction           = 0.1f;
  float particleRadius     = 0.0f; // init() で cfg_.cellSize()*0.5 に設定
  float stretchCompliance  = 1e-4f;
  float bendCompliance     = 1e-2f;
  float windX              = 0.0f;
  float windZ              = 0.0f;
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

  // ── Force (issue #30): 重力(gravity)・風(windX/windZ)以外の任意の力を追加する ──
  // 追加/削除のたびに predict.comp を無条件で再生成・再コンパイルする。
  void addForce(std::shared_ptr<Force> f);
  void removeForce(const std::shared_ptr<Force>& f);
  void setForces(std::vector<std::shared_ptr<Force>> forces);
  void clearForces();

private:
  ClothConfig cfg_;
  VkDevice device_        = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkCommandPool cmdPool_  = VK_NULL_HANDLE;
  VkQueue queue_          = VK_NULL_HANDLE;

  AttributeBuffer attrBuf_;
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

  // Force (issue #30): gravity/windX/windZ は互換のため既定Forceとして常時登録し、
  // 値は毎フレーム forces_ 経由でバッファへ反映する (シェーダー再生成は不要)。
  // 追加の Force (斜め重力・複数風・Turbulence・Noise 等) は addForce() で積む。
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
  ComputePipeline kSolveDensity_;
  ComputePipeline kSolveStretch_;
  ComputePipeline kUpdateVelocity_;
  ComputePipeline kZeroLambdas_;
  ComputePipeline kZeroCells_;

  void initParticleBuffers(VkCommandPool cmdPool, VkQueue queue);
  void initClothBuffers(VkCommandPool cmdPool, VkQueue queue);
  void computeBarrier(VkCommandBuffer cmd);
};
