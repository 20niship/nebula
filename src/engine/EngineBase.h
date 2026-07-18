#pragma once

#include <memory>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "../core/Force.h"
#include "AttributeBuffer.h"
#include "ComputePipeline.h"

// 7つの物理エンジン (SimulationEngine/ClothSceneEngine/MultiPhysicsEngine/
// FluidEngine/SoftBodyEngine/MPMEngine/PyroEngine) に共通するVulkanリソース
// 保持パターンとForce管理を統合する基底クラス (issue #30 レビュー対応)。
//
// 各派生Engineは forceTargetPipeline()/forceShaderName() の2つの仮想関数のみを
// 実装すればよい。addForce() 系メソッドはここで完結する。
//
// 使い方: 派生Engineの init() 冒頭で initEngineBase() を呼び、粒子/グリッド
// バッファを確保し終えた直後に initForces() を呼ぶ。cleanup() の最後に
// cleanupEngineBase() を呼ぶ。step() の冒頭で uploadForces(dt) を呼ぶ。
class EngineBase {
public:
  virtual ~EngineBase() = default;

  void addForce(std::shared_ptr<Force> f);
  void removeForce(const std::shared_ptr<Force>& f);
  void setForces(std::vector<std::shared_ptr<Force>> forces);
  void clearForces();
  const std::vector<std::shared_ptr<Force>>& forces() const { return forces_; }

protected:
  static constexpr uint32_t kMaxForces = 32;

  void initEngineBase(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue);
  // 派生Engineが粒子/グリッドバッファを確保し終えた後に呼ぶこと。
  // rebuildForceShader() が forceTargetPipeline() (純粋仮想) を呼ぶため、
  // 派生オブジェクトが完全に構築された init() 内から呼ぶ必要がある
  // (コンストラクタ内での仮想呼び出しは避けること)。
  void initForces(std::vector<std::shared_ptr<Force>> defaultForces = {});
  void cleanupEngineBase();
  // forces_ の各要素に advanceTime(dt) を適用してから pack() し SSBO へアップロードする。
  void uploadForces(float dt);

  // Force対象のComputePipelineへの参照 (Engineごとに kPredict_/kPredictSdf_/
  // kGridUpdate_/kForces_ 等、既存メンバをそのまま返す)。
  virtual ComputePipeline& forceTargetPipeline() = 0;
  // ForceShaderCompiler::compile() に渡すシェーダーファイル名 (例: "predict.comp")。
  virtual const char* forceShaderName() const = 0;

  VkDevice device_        = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkCommandPool cmdPool_  = VK_NULL_HANDLE;
  VkQueue queue_          = VK_NULL_HANDLE;
  AttributeBuffer attrBuf_;

  std::vector<std::shared_ptr<Force>> forces_;
  uint32_t forcesIdx_ = 0;

private:
  // 登録済みForce群に応じて対象シェーダーを実行時に再生成・再コンパイルする。
  // キャッシュはせず、addForce/removeForce/setForces/clearForces/initForcesの
  // たびに無条件で行う。
  void rebuildForceShader();
};
