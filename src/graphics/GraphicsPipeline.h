#pragma once

#include "SimPC.h"
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

class GraphicsPipeline {
public:
  void init(VkDevice device, VkRenderPass renderPass, VkDescriptorSetLayout bindlessLayout, const std::string& vertPath, const std::string& fragPath, VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST, bool enableBlend = false);

  void initVertFromSpirv(VkDevice device, VkRenderPass renderPass, VkDescriptorSetLayout bindlessLayout, const std::vector<uint32_t>& vertSpirv, const std::string& fragPath, VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST, bool enableBlend = false);
  void cleanup();

  void draw(VkCommandBuffer cmd, VkDescriptorSet bindlessSet, const SimPC& pc, uint32_t particleCount);

  VkPipeline pipeline             = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

private:
  VkDevice device_ = VK_NULL_HANDLE;

  VkShaderModule loadShader(const std::string& path);
  VkShaderModule loadShader(const std::vector<uint8_t>& code);

  void buildPipeline(VkRenderPass renderPass, VkDescriptorSetLayout bindlessLayout, VkShaderModule vertMod, VkShaderModule fragMod, VkPrimitiveTopology topology, bool enableBlend);
};
