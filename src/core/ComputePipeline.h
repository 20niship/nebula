#pragma once

#include "SimPC.h"
#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

class ComputePipeline {
public:
  void init(VkDevice device, VkDescriptorSetLayout bindlessLayout, const std::string& shaderPath);
  // SPIR-V バイト列(shaderc等でランタイム生成したもの)から直接パイプラインを構築する。
  // issue #30: Force動的シェーダー生成が rebuild のたびに呼ぶ想定。init() と同じレイアウト
  // (push constant size = sizeof(SimPC)) で構築する。
  void initFromSpirv(VkDevice device, VkDescriptorSetLayout bindlessLayout, const std::vector<uint32_t>& spirv);
  void cleanup();

  // フレームごとに dispatch (barrier は呼び出し側で挿入)
  void dispatch(VkCommandBuffer cmd, VkDescriptorSet bindlessSet, const SimPC& pc, uint32_t particleCount);
  // SimPC以外のpush constant構造体(MPMSimPC/PyroSimPC等)を使う場合の汎用dispatch
  void dispatchRaw(VkCommandBuffer cmd, VkDescriptorSet bindlessSet, const void* pcData, uint32_t pcSize, uint32_t groupCountX);

  VkPipeline pipeline             = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

private:
  VkDevice device_                          = VK_NULL_HANDLE;
  VkDescriptorSetLayout bindlessSetLayout_   = VK_NULL_HANDLE;
  VkShaderModule loadShader(const std::string& path);
  VkShaderModule createShaderModuleFromSpirv(const uint32_t* code, size_t wordCount);
  void buildPipeline(VkShaderModule shaderModule);
};
