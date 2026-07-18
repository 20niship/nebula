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
// cleanupEngineBase() を呼ぶ。step() の冒頭で uploadForces(cmd, dt) を呼ぶ。
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
  // cmd に記録中のコマンドバッファへ vkCmdUpdateBuffer で埋め込む (呼び出し側で
  // 別途 submit/wait しない)。AttributeBuffer::upload() の使い捨てステージング
  // バッファ+vkQueueWaitIdle方式は、対象データがForce配列程度(最大 kMaxForces×
  // sizeof(ForceGPU) 以下)でも毎フレームキュー全体を完全ドレインさせてしまい
  // 致命的に高コストだったため置き換えた (詳細は呼び出し元コミットメッセージ参照)。
  void uploadForces(VkCommandBuffer cmd, float dt);

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
  // uploadForces() が毎フレーム書き込み先を切り替えた後の「今フレーム使うべき」
  // bindless index。各Engineの buildPC() はこれをそのまま pc.forceBufIdx に使う。
  uint32_t forcesIdx_ = 0;

private:
  // 登録済みForce群に応じて対象シェーダーを実行時に再生成・再コンパイルする。
  // キャッシュはせず、addForce/removeForce/setForces/clearForces/initForcesの
  // たびに無条件で行う。
  void rebuildForceShader();

  // forcesバッファのダブルバッファ (ping-pong)。windowed example は MAX_FRAMES=2
  // のフェンスでdouble-bufferingしており、uploadForces()がvkCmdUpdateBufferで
  // 単一バッファに毎フレーム書き込むと、フレームNの書き込みとフレームN+1の書き込みが
  // (両者の compute submission 間に順序保証が無いため) SYNC-HAZARD-WRITE-AFTER-WRITE
  // を起こす。2バッファを交互に使うことでフレームNとN+2が同じバッファを使うまで
  // 猶予ができ、既存のフェンス機構と整合する。
  uint32_t forcesBufIdx_[2] = {0, 0};
  int forcesBufCur_         = 0;
};
