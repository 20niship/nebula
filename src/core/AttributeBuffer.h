#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

// MoltenVK上限31に対し、FluidEngineのfoam有効時最大27スロット(issue #87)を確保。
static constexpr uint32_t MAX_BINDLESS_BUFFERS = 27;

// SoAバッファマネージャ。addAttribute()でVMAバッファを確保し、
// Bindlessディスクリプタ配列へ自動登録してインデックスを返す。
class AttributeBuffer {
public:
  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool);
  void cleanup();

  // 属性バッファを追加。返値がBindlessインデックス
  uint32_t addAttribute(const std::string& name, VkDeviceSize elementSize, uint32_t count);

  // GPU書き込み・CPU読み取り用の常時mapped属性バッファを追加(GPU_TO_CPU)。vkQueueWaitIdle無しでCPU側から読める。
  uint32_t addHostVisibleAttribute(const std::string& name, VkDeviceSize elementSize, uint32_t count, void** outMappedPtr);

  // addHostVisibleAttributeで得たmappedポインタをCPUから読む前にキャッシュを無効化する(コヒーレント保証がない環境向け)。
  void invalidateHostVisible(const std::string& name) const;

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
    void* mapped             = nullptr; // addHostVisibleAttribute専用。VMAが破棄時に自動unmapする
  };

  VkDevice device_        = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  uint32_t nextIndex_     = 0;

  std::unordered_map<std::string, Attribute> attributes_;

  void createDescriptorSetLayout();
  void createDescriptorSet(VkDescriptorPool pool);
  void registerBuffer(uint32_t index, VkBuffer buffer);
};
