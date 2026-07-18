#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "../core/Domain.h"
#include "ComputePipeline.h"
#include "EngineBase.h"
#include "SimPC.h"

struct SoftBodyInstance {
  std::string sbPath;
  glm::vec3 offset = {0.0f, 0.0f, 0.0f};
  float scale      = 1.0f;
};

// 重力は addForce() で GravityForce を登録すること (issue #30 レビュー対応:
// gravity の public メンバは廃止)。
class SoftBodyEngine : public EngineBase {
public:
  // init() 前に呼ぶ。戻り値はインスタンスの粒子開始インデックス
  uint32_t addInstance(const SoftBodyInstance& inst);

  // issue #46: domainSize(vec3, ドメイン物理サイズ[m]) + cellSize(float, 全軸共通の
  // セルサイズ[m]) で指定する。
  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, glm::vec3 domainSize = {10.0f, 10.0f, 10.0f}, float cellSize = 10.0f / 64.0f);
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
  float restitution             = 0.5f;
  float friction                = 0.15f;
  float stretchCompliance       = 1e-5f;
  float volCompliance           = 5e-4f;
  float linearDamping           = 0.03f;
  float particleRadius          = 0.02f;
  float particleCollisionRadius = 0.25f; // ボディ間衝突半径 [m]
  int solverIterations          = 5;
  int numSubsteps               = 15;

protected:
  ComputePipeline& forceTargetPipeline() override { return kPredict_; }
  const char* forceShaderName() const override { return "predict.comp"; }

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

  glm::vec3 domainSize_{10.0f, 10.0f, 10.0f};
  float cellSize_ = 10.0f / 64.0f;

  ComputePipeline kPredict_;
  ComputePipeline kSdfCollision_;
  ComputePipeline kSolveEdge_;
  ComputePipeline kZeroEdgeLambda_;
  ComputePipeline kSolveVolume_;
  ComputePipeline kZeroVolLambda_;
  ComputePipeline kParticleCollision_;
  ComputePipeline kUpdateVelocity_;
};
