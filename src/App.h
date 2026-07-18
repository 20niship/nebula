#pragma once

#include <GLFW/glfw3.h>
#include <array>
#include <chrono>
#include <string>
#include <vulkan/vulkan.h>

#include "core/VulkanContext.h"

struct FrameData {
  VkSemaphore imageAvailable    = VK_NULL_HANDLE;
  VkSemaphore renderFinished    = VK_NULL_HANDLE;
  VkSemaphore timelineSemaphore = VK_NULL_HANDLE;
  VkFence inFlightFence         = VK_NULL_HANDLE;
  VkCommandBuffer graphicsCmd   = VK_NULL_HANDLE;
  VkCommandBuffer computeCmd    = VK_NULL_HANDLE;
  uint64_t timelineValue        = 0;
};

class BaseApp {
public:
  static constexpr uint32_t MAX_FRAMES = 2;

  GLFWwindow* window = nullptr;
  VulkanContext ctx;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

  std::array<FrameData, MAX_FRAMES> frames;
  uint32_t currentFrame   = 0;
  bool framebufferResized = false;

  // Screenshot
  std::string screenshotDir;
  int screenshotsTaken = 0;
  bool shouldExit      = false;

  // task perf 用計測 (screenshotDir が空でも n_shots フレームで終了し、
  // stdout に PERF_RESULT 行を出力する。screenshotDir 指定時はPPM保存も行う)。
  std::chrono::steady_clock::time_point perfStartTime_;

  void initWindow(const char* title, int w = 640, int h = 480);
  void createDescriptorPool();
  void createFrameData();
  void initImGui();
  void cleanupBase();
  // fence wait 済みの状態で呼ぶこと。nShots 枚で shouldExit=true
  void saveScreenshot(uint32_t imageIdx, int nShots);

  static void framebufferResizeCallback(GLFWwindow* w, int, int);
};
