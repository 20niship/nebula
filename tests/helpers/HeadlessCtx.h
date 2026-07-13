#pragma once
#include <cstdint>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

// Minimum headless Vulkan context for GPU compute tests — no window, no swapchain.
struct HeadlessCtx {
  VkInstance instance             = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device                 = VK_NULL_HANDLE;
  VkQueue computeQueue            = VK_NULL_HANDLE;
  uint32_t computeFamily          = UINT32_MAX;
  VmaAllocator allocator          = VK_NULL_HANDLE;
  VkCommandPool commandPool       = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

  void init();
  void cleanup();

  // Allocate, begin, end/submit/wait/free — one-shot compute command
  VkCommandBuffer beginCmd() const;
  void submitCmd(VkCommandBuffer cmd) const;

  // Blocking GPU→CPU readback of `size` bytes from src at srcOffset
  void readBuffer(VkBuffer src, VkDeviceSize srcOffset, void* dst, VkDeviceSize size) const;
};
