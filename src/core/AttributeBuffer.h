#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

// MoltenVK (Apple M2) maxPerStageDescriptorStorageBuffers = 31
// MPMEngine が既に16スロット使い切っているため、Force用バッファ (issue #30) 追加分の
// 余裕を持たせて20に引き上げ (31に対し十分な余裕を確保)
static constexpr uint32_t MAX_BINDLESS_BUFFERS = 20;

// SoAバッファマネージャ。addAttribute()でVMAバッファを確保し、
// Bindlessディスクリプタ配列へ自動登録してインデックスを返す。
class AttributeBuffer {
public:
  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool);
  void cleanup();

  // 属性バッファを追加。返値がBindlessインデックス
  uint32_t addAttribute(const std::string& name, VkDeviceSize elementSize, uint32_t count);

  // データをGPUへ転送（ステージング経由）
  void upload(const std::string& name, const void* data, VkDeviceSize byteSize, VkCommandPool cmdPool, VkQueue queue);

  // dstOffset バイト目から転送（境界粒子の追記用）
  void uploadAt(const std::string& name, const void* data, VkDeviceSize byteSize, VkDeviceSize dstOffset, VkCommandPool cmdPool, VkQueue queue);

  // packed な count 要素を dstIndices[j] の要素位置へ1 submitのmulti-region copyで散布転送する(スロット再利用の穴埋め用; 単位は要素index)。
  void uploadScattered(const std::string& name, const void* packedData, VkDeviceSize elemSize, const std::vector<uint32_t>& dstIndices, VkCommandPool cmdPool, VkQueue queue);

  // 既存データ（先頭からのバイト列）を保持したまま容量を newCount 要素に再確保する。
  // Bindless index は維持されるため、他の保持済みインデックスは変更不要。
  void resizeAttribute(const std::string& name, uint32_t newCount, VkCommandPool cmdPool, VkQueue queue);

  VkBuffer getBuffer(const std::string& name) const;
  uint32_t getIndex(const std::string& name) const;
  uint32_t getCount(const std::string& name) const;

  // DescriptorSetLayout / DescriptorSet (Bindless配列)
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet             = VK_NULL_HANDLE;

private:
  struct Attribute {
    VkBuffer buffer          = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint32_t count           = 0;
    uint32_t bindlessIndex   = 0;
    VkDeviceSize elementSize = 0;
  };

  VkDevice device_        = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  uint32_t nextIndex_     = 0;

  std::unordered_map<std::string, Attribute> attributes_;

  void createDescriptorSetLayout();
  void createDescriptorSet(VkDescriptorPool pool);
  void registerBuffer(uint32_t index, VkBuffer buffer);
};
