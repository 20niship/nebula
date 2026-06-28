#pragma once
#include <vulkan/vulkan.hpp> // vulkanのインクルードが先
/// ---
#include <GLFW/glfw3.h>

#include <string>
#include <vector>
#include <vk_mem_alloc.h>

class VulkanContext {
public:
  void init(GLFWwindow* window);
  void cleanup();

  // Swapchain rebuild on resize
  void recreateSwapchain();

  // ── Core handles ────────────────────────────────────────────────────
  VkInstance instance             = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device                 = VK_NULL_HANDLE;
  VkSurfaceKHR surface            = VK_NULL_HANDLE;
  VmaAllocator allocator          = VK_NULL_HANDLE;

  // ── Queues ───────────────────────────────────────────────────────────
  VkQueue graphicsQueue   = VK_NULL_HANDLE;
  VkQueue computeQueue    = VK_NULL_HANDLE;
  uint32_t graphicsFamily = UINT32_MAX;
  uint32_t computeFamily  = UINT32_MAX;

  // VSync 制御 (init() 前に設定する)
  bool vsync = true;

  // ── Swapchain ────────────────────────────────────────────────────────
  VkSwapchainKHR swapchain   = VK_NULL_HANDLE;
  VkFormat swapchainFormat   = VK_FORMAT_UNDEFINED;
  VkExtent2D swapchainExtent = {};
  std::vector<VkImage> swapchainImages;
  std::vector<VkImageView> swapchainImageViews;

  // ── RenderPass / Framebuffers ────────────────────────────────────────
  VkRenderPass renderPass = VK_NULL_HANDLE;
  std::vector<VkFramebuffer> framebuffers;

  // ── Command pools ────────────────────────────────────────────────────
  VkCommandPool graphicsCommandPool = VK_NULL_HANDLE;
  VkCommandPool computeCommandPool  = VK_NULL_HANDLE;

  // ── Helpers ──────────────────────────────────────────────────────────
  VkCommandBuffer beginSingleTimeCommands(VkCommandPool pool);
  void endSingleTimeCommands(VkCommandPool pool, VkCommandBuffer cmd, VkQueue queue);

  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props);

  // Screenshot: call after vkWaitForFences on the frame's graphics fence.
  // Saves a PPM file. Image must be in PRESENT_SRC_KHR layout.
  void saveScreenshotPPM(const std::string& path, uint32_t imageIdx);

private:
  GLFWwindow* window_                      = nullptr;
  VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;

  void createInstance();
  void createDebugMessenger();
  void createSurface();
  void selectPhysicalDevice();
  void createLogicalDevice();
  void createAllocator();
  void createSwapchain();
  void createImageViews();
  void createRenderPass();
  void createFramebuffers();
  void createCommandPools();

  void cleanupSwapchain();

  bool checkValidationLayerSupport();
  std::vector<const char*> getRequiredInstanceExtensions();
};
