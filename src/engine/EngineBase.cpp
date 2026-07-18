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
  forcesIdx_ = attrBuf_.addAttribute("forces", sizeof(ForceGPU), kMaxForces);
  forces_    = std::move(defaultForces);
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

void EngineBase::uploadForces(float dt) {
  std::vector<ForceGPU> packed;
  packed.reserve(forces_.size());
  for(const auto& f : forces_) {
    f->advanceTime(dt);
    packed.push_back(f->pack());
  }
  if(!packed.empty()) attrBuf_.upload("forces", packed.data(), sizeof(ForceGPU) * packed.size(), cmdPool_, queue_);
}
