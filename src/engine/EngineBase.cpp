#include "EngineBase.h"

#include "ForceShaderCompiler.h"
#include <algorithm>

void EngineBase::initEngineBase(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue) {
  device_    = device;
  allocator_ = allocator;
  cmdPool_   = cmdPool;
  queue_     = queue;
  attrBuf_.init(device, allocator, descriptorPool);
}

void EngineBase::initForces(std::vector<std::shared_ptr<Force>> defaultForces) {
  // ダブルバッファ (ping-pong)。uploadForces() 参照。
  forcesBufIdx_[0] = attrBuf_.addAttribute("forces0", sizeof(ForceGPU), kMaxForces);
  forcesBufIdx_[1] = attrBuf_.addAttribute("forces1", sizeof(ForceGPU), kMaxForces);
  forcesBufCur_    = 0;
  forcesIdx_       = forcesBufIdx_[0];
  forces_          = std::move(defaultForces);
  rebuildForceShader();
}

void EngineBase::cleanupEngineBase() { attrBuf_.cleanup(); }

void EngineBase::addForce(std::shared_ptr<Force> f) {
  forces_.push_back(std::move(f));
  rebuildForceShader();
}

void EngineBase::removeForce(const std::shared_ptr<Force>& f) {
  forces_.erase(std::remove(forces_.begin(), forces_.end(), f), forces_.end());
  rebuildForceShader();
}

void EngineBase::setForces(std::vector<std::shared_ptr<Force>> forces) {
  forces_ = std::move(forces);
  rebuildForceShader();
}

void EngineBase::clearForces() {
  forces_.clear();
  rebuildForceShader();
}

void EngineBase::rebuildForceShader() {
  std::vector<uint32_t> spirv = ForceShaderCompiler::compile(forces_, forceShaderName());
  ComputePipeline& k          = forceTargetPipeline();
  k.cleanup(); // 未初期化 (device_==VK_NULL_HANDLE) でも安全
  k.initFromSpirv(device_, attrBuf_.descriptorSetLayout, spirv);
}

void EngineBase::uploadForces(VkCommandBuffer cmd, float dt) {
  std::vector<ForceGPU> packed;
  packed.reserve(forces_.size());
  for(const auto& f : forces_) {
    f->advanceTime(dt);
    packed.push_back(f->pack());
  }
  if(packed.empty()) return;

  // ping-pong: フレームごとに書き込み先を切り替える (このstep()呼び出しが
  // 「今フレーム」に対応する)。windowed exampleのMAX_FRAMES=2フェンスと
  // 整合し、フレームNとN+2が同じバッファを使うまでに1フレーム分の猶予ができる。
  forcesBufCur_ ^= 1;
  forcesIdx_       = forcesBufIdx_[forcesBufCur_];
  VkBuffer buf     = attrBuf_.getBuffer(forcesBufCur_ == 0 ? "forces0" : "forces1");

  // vkCmdUpdateBuffer は 65536 byte 未満のデータをコマンドバッファに直接埋め込める
  // (別途ステージングバッファ確保やsubmit/waitが不要)。kMaxForces(32) ×
  // sizeof(ForceGPU)(32B) = 最大1024Bなので余裕を持って収まる。
  VkDeviceSize byteSize = sizeof(ForceGPU) * packed.size();
  vkCmdUpdateBuffer(cmd, buf, 0, byteSize, packed.data());

  // 直後のcompute dispatchがforcesバッファを読む前に転送を完了させる
  VkMemoryBarrier b{};
  b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}
