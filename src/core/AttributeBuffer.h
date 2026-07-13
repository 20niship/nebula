#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

// MoltenVK (Apple M2) maxPerStageDescriptorStorageBuffers = 31。
// 24 はその範囲内で安全マージンを残しつつ機能追加の余地を確保する値
// (Pyroエンジンだけで既に14/16枠を消費しており、旧上限16は逼迫していた)。
// device が実際にこの数を確保できない場合は vkCreateDescriptorSetLayout /
// vkCreateDescriptorPool が VK_SUCCESS 以外を返し、AttributeBuffer 側の
// 既存エラーハンドリング (createDescriptorSetLayout/createDescriptorSet) が
// std::runtime_error を投げて起動時に明確に検出できる。
// 本プロジェクトは Apple Silicon / MoltenVK を主対象としており、31という
// 実機値が既知のため、VkPhysicalDeviceProperties を都度クエリして動的に
// 決定するのではなく、検証済みの安全な固定値を採用している
// (低スペックGPUへの動的な段階的縮退が必要になった場合は要再検討)。
static constexpr uint32_t MAX_BINDLESS_BUFFERS = 24;

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
