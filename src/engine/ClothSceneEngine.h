#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "ClothConstraint.h"
#include "ClothMesh.h"
#include "ComputePipeline.h"
#include "EngineBase.h"
#include "SimPC.h"

// 複数布メッシュを1統合バッファで解くVellumスタイルエンジン。
// addCloth() → addConstraint() → init() の順に呼ぶ。
// 重力・風は addForce() で GravityForce/ConstantWindForce を登録すること
// (issue #30 レビュー対応: gravity/windX/windZ の public メンバは廃止)。
class ClothSceneEngine : public EngineBase {
public:
  // ── シーン構築 (init() 前) ───────────────────────────────────────────────
  // 戻り値: このメッシュのグローバル頂点開始インデックス
  uint32_t addCloth(const ClothMesh& mesh);
  void addConstraint(const ClothConstraint& c);

  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, float worldSize, uint32_t gridRes);
  void cleanup();

  // 1フレーム進める (compute command buffer に積む)
  void step(VkCommandBuffer cmd, float dt);

  // PinAnimated 制約の目標位置をフレーム毎に更新
  // idx = addConstraint() で追加した制約の通し番号 (PinAnimated のみカウント)
  void updateConstraint(uint32_t idx, const glm::vec3& newTarget);

  // ── ImGui パラメータ ─────────────────────────────────────────────────────
  float restitution        = 0.3f;
  float friction           = 0.1f;
  float stretchCompliance  = 1e-4f;
  float bendCompliance     = 1e-2f;
  int solverIterations     = 3;
  int numSubsteps          = 6;
  bool enableSelfCollision = true;

  // ── レンダラー用 ─────────────────────────────────────────────────────────
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet             = VK_NULL_HANDLE;
  uint32_t posIdx                           = 0;
  uint32_t velIdx                           = 0;

  uint32_t totalParticleCount() const { return totalCount_; }
  uint32_t clothCount() const { return (uint32_t)meshes_.size(); }
  uint32_t meshOffset(uint32_t i) const { return meshOffsets_[i]; }
  const ClothMesh& getMesh(uint32_t i) const { return meshes_[i]; }

  VkBuffer getPositionBuffer() const;

protected:
  ComputePipeline& forceTargetPipeline() override { return kPredict_; }
  const char* forceShaderName() const override { return "predict.comp"; }

private:
  // ── シーン入力 ───────────────────────────────────────────────────────────
  std::vector<ClothMesh> meshes_;
  std::vector<uint32_t> meshOffsets_;
  std::vector<ClothConstraint> constraints_;

  uint32_t totalCount_ = 0;
  float worldSize_     = 10.0f;
  uint32_t gridRes_    = 64;

  uint32_t predPIdx_        = 0;
  uint32_t invMassIdx_      = 0;
  uint32_t typeFlagIdx_     = 0;
  uint32_t cellCountIdx_    = 0;
  uint32_t cellOffsetIdx_   = 0;
  uint32_t sortedIdxIdx_    = 0;
  uint32_t stretchEdgesIdx_ = 0;
  uint32_t lambdasIdx_      = 0;
  uint32_t pinnedTargetIdx_ = 0;

  bool hasPinAnimated_    = false;
  bool pinnedTargetDirty_ = false;

  // CPU-side pinnedTarget (vec4 × totalCount)
  std::vector<glm::vec4> pinnedTargetCpu_;
  // PinAnimated 制約のグローバル粒子インデックス一覧 (updateConstraint() 用)
  std::vector<uint32_t> animatedPinParticles_;

  // stagingバッファ (persistently mapped, CPU_ONLY)
  VkBuffer stagingBuf_        = VK_NULL_HANDLE;
  VmaAllocation stagingAlloc_ = VK_NULL_HANDLE;
  void* stagingMapped_        = nullptr;

  // 統合エッジデータ
  std::vector<uint32_t> combinedEdgeData_;
  std::vector<uint32_t> combinedColorBatch_; // size = nColors+1
  int nColors_             = 12;
  uint32_t totalEdgeCount_ = 0;

  // ── コンピュートパイプライン ─────────────────────────────────────────────
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
  ComputePipeline kMovePins_;

  void buildCombinedEdges();
  void initBuffers(VkCommandPool cmdPool, VkQueue queue);
  void createStagingBuffer();
  void computeBarrier(VkCommandBuffer cmd);
  void transferBarrier(VkCommandBuffer cmd);
  uint32_t totalCells() const { return gridRes_ * gridRes_ * gridRes_; }
  float cellSize() const { return worldSize_ / float(gridRes_); }
};
