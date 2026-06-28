#pragma once

#include "SimPC.h"
#include <string>
#include <vulkan/vulkan.h>

class GraphicsPipeline {
public:
  void init(VkDevice device, VkRenderPass renderPass, VkDescriptorSetLayout bindlessLayout, const std::string& vertPath, const std::string& fragPath);
  void cleanup();

  void draw(VkCommandBuffer cmd, VkDescriptorSet bindlessSet, const SimPC& pc, uint32_t particleCount);

  VkPipeline pipeline             = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkShaderModule loadShader(const std::string& path);
};
