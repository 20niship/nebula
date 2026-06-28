#pragma once

#include "SimPC.h"
#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

// 布メッシュを三角形で描画するグラフィクスパイプライン
class ClothRenderer {
public:
  void init(VkDevice device, VmaAllocator allocator, VkRenderPass renderPass, VkDescriptorSetLayout bindlessLayout, const std::string& shaderDir);
  void cleanup();

  // 三角形インデックスをアップロード (初期化時に一度だけ)
  void uploadIndices(const std::vector<uint32_t>& triIndices, VkCommandPool cmdPool, VkQueue queue);

  // vertexOffset: 複数メッシュ使用時のグローバル頂点オフセット (vkCmdDrawIndexed の vertexOffset)
  void draw(VkCommandBuffer cmd, VkDescriptorSet bindlessSet, const SimPC& pc, uint32_t vertexCount, int32_t vertexOffset = 0);

private:
  VkDevice device_        = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;

  VkPipeline pipeline_             = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;

  VkBuffer indexBuf_        = VK_NULL_HANDLE;
  VmaAllocation indexAlloc_ = VK_NULL_HANDLE;
  uint32_t indexCount_      = 0;

  VkShaderModule loadShader(const std::string& path);
};
