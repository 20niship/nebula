#pragma once

#include "SimPC.h"
#include <string>
#include <vulkan/vulkan.h>

class GraphicsPipeline {
public:
  // enableBlend: true の場合、標準 alpha over ブレンド (srcAlpha, oneMinusSrcAlpha) を有効化する
  // (泡の半透明描画用; issue #47)。既定 false は既存呼び出し元と同一の不透明描画。
  void init(VkDevice device, VkRenderPass renderPass, VkDescriptorSetLayout bindlessLayout, const std::string& vertPath, const std::string& fragPath, VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST, bool enableBlend = false);
  void cleanup();

  void draw(VkCommandBuffer cmd, VkDescriptorSet bindlessSet, const SimPC& pc, uint32_t particleCount);

  VkPipeline pipeline             = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkShaderModule loadShader(const std::string& path);
};
