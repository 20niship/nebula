#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "AttributeBuffer.h"
#include "ComputePipeline.h"
#include "Force.h"
#include "SimPC.h"

struct SoftBodyInstance {
  std::string sbPath;
  glm::vec3 offset = {0.0f, 0.0f, 0.0f};
  float scale      = 1.0f;
};

class SoftBodyEngine {
public:
  // init() 前に呼ぶ。戻り値はインスタンスの粒子開始インデックス
  uint32_t addInstance(const SoftBodyInstance& inst);

  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, float worldSize = 10.0f, uint32_t gridRes = 64);
  void step(VkCommandBuffer cmd, float dt);
  void cleanup();

  VkBuffer getPositionBuffer() const { return attrBuf_.getBuffer("P"); }

  // Renderer インターフェース (mpm_elastic.cpp と同パターン)
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet             = VK_NULL_HANDLE;
  uint32_t posIdx                           = 0;
  uint32_t velIdx                           = 0;
  uint32_t edgeDataIdx                      = 0; // ワイヤーフレーム描画用
  uint32_t totalParticleCount() const { return totalCount_; }
  uint32_t totalEdgeCount() const { return totalEdgeCount_; }

  // ImGui から直接書き換え可能
  float gravity                 = -9.8f;
  float restitution             = 0.5f;
  float friction                = 0.15f;
  float stretchCompliance       = 1e-5f;
  float volCompliance           = 5e-4f;
  float linearDamping           = 0.03f;
  float particleRadius          = 0.02f;
  float particleCollisionRadius = 0.25f; // ボディ間衝突半径 [m]
  int solverIterations          = 5;
  int numSubsteps               = 15;

  // Force (issue #30): gravity 以外の任意の力を追加する
  void addForce(std::shared_ptr<Force> f);
  void removeForce(const std::shared_ptr<Force>& f);
  void setForces(std::vector<std::shared_ptr<Force>> forces);
  void clearForces();

private:
  struct InstanceData {
    std::string sbPath;
    uint32_t n_particles = 0;
    uint32_t n_edges     = 0;
    uint32_t n_tets      = 0;
    int n_edge_colors    = 0;
    int n_tet_colors     = 0;

    std::vector<glm::vec3> positions;
    std::vector<float> invMasses;
    std::vector<uint32_t> edgeColorBatch; // (n_edge_colors+1) 要素
    std::vector<uint32_t> edgeData;       // n_edges*3 uint
    std::vector<uint32_t> tetColorBatch;  // (n_tet_colors+1) 要素
    std::vector<uint32_t> tetData;        // n_tets*4 uint
    std::vector<float> tetRestVols;       // n_tets float

    glm::vec3 offset        = {};
    float scale             = 1.0f;
    uint32_t particleOffset = 0; // 結合バッファ内でのこのインスタンスの粒子開始オフセット
  };

  void loadSBFile(InstanceData& inst);
  void buildCombinedBuffers();
  void initGPUBuffers(VkCommandPool cmdPool, VkQueue queue);
  void computeBarrier(VkCommandBuffer cmd);

  std::vector<InstanceData> instances_;

  // 結合バッファ (全インスタンス連結)
  uint32_t totalCount_     = 0;
  uint32_t totalEdgeCount_ = 0;
  uint32_t totalTetCount_  = 0;
  int nEdgeColors_         = 0;
  int nTetColors_          = 0;
  std::vector<uint32_t> edgeColorBatch_; // (nEdgeColors_+1) 要素
  std::vector<uint32_t> tetColorBatch_;  // (nTetColors_+1) 要素
  std::vector<glm::vec4> combinedPos_;
  std::vector<glm::vec4> combinedInvMass_;
  std::vector<uint32_t> combinedTypeFlag_;
  std::vector<uint32_t> combinedEdgeData_;
  std::vector<uint32_t> combinedTetData_;
  std::vector<float> combinedTetRestVol_;

  // AttributeBuffer バインドレスインデックス
  uint32_t predPIdx_      = 0;
  uint32_t invMassIdx_    = 0;
  uint32_t typeFlagIdx_   = 0;
  uint32_t edgeDataIdx_   = 0;
  uint32_t edgeLambdaIdx_ = 0;
  uint32_t tetDataIdx_    = 0;
  uint32_t tetLambdaIdx_  = 0;
  uint32_t tetRestVolIdx_ = 0;

  float worldSize_  = 10.0f;
  uint32_t gridRes_ = 64;

  // Force (issue #30): gravity互換の既定Forceを常時登録 (soft bodyにwindはない)
  static constexpr uint32_t kMaxForces = 32;
  std::vector<std::shared_ptr<Force>> forces_;
  std::shared_ptr<GravityForce> legacyGravity_;
  uint32_t forcesIdx_ = 0;
  void rebuildForceShader();
  void uploadForces();

  AttributeBuffer attrBuf_;
  VkDevice device_        = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkCommandPool cmdPool_  = VK_NULL_HANDLE;
  VkQueue queue_          = VK_NULL_HANDLE;

  ComputePipeline kPredict_;
  ComputePipeline kSdfCollision_;
  ComputePipeline kSolveEdge_;
  ComputePipeline kZeroEdgeLambda_;
  ComputePipeline kSolveVolume_;
  ComputePipeline kZeroVolLambda_;
  ComputePipeline kParticleCollision_;
  ComputePipeline kUpdateVelocity_;
};
