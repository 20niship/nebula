#pragma once

#include "SimPC.h"
#include <string>
#include <vulkan/vulkan.h>

class ComputePipeline {
public:
  void init(VkDevice device, VkDescriptorSetLayout bindlessLayout, const std::string& shaderPath);
  void cleanup();

  // フレームごとに dispatch (barrier は呼び出し側で挿入)
  void dispatch(VkCommandBuffer cmd, VkDescriptorSet bindlessSet, const SimPC& pc, uint32_t particleCount);

  VkPipeline pipeline             = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkShaderModule loadShader(const std::string& path);
};
