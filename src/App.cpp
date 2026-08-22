#include "App.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "core/AttributeBuffer.h"

// ── Window ────────────────────────────────────────────────────────────────────

void BaseApp::initWindow(const char* title, int w, int h) {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window = glfwCreateWindow(w, h, title, nullptr, nullptr);
  glfwSetWindowUserPointer(window, this);
  glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void BaseApp::framebufferResizeCallback(GLFWwindow* w, int, int) {
  auto* app               = reinterpret_cast<BaseApp*>(glfwGetWindowUserPointer(w));
  app->framebufferResized = true;
}

// ── Descriptor pool ───────────────────────────────────────────────────────────

void BaseApp::createDescriptorPool() {
  // MoltenVK (Metal3 argument buffer) は UPDATE_AFTER_BIND 時に内部的に
  // より多くの descriptor slot を消費するため、余裕を持たせる
  static constexpr uint32_t POOL_DESCRIPTOR_COUNT = MAX_BINDLESS_BUFFERS * 64;

  VkDescriptorPoolSize sz{};
  sz.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sz.descriptorCount = POOL_DESCRIPTOR_COUNT;

  VkDescriptorPoolCreateInfo info{};
  info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  info.maxSets       = 16;
  info.poolSizeCount = 1;
  info.pPoolSizes    = &sz;
  info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;

  if(vkCreateDescriptorPool(ctx.device, &info, nullptr, &descriptorPool) != VK_SUCCESS) throw std::runtime_error("BaseApp: failed to create descriptor pool");
}

// ── Per-frame sync ────────────────────────────────────────────────────────────

void BaseApp::createFrameData() {
  VkSemaphoreTypeCreateInfo typeInfo{};
  typeInfo.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  typeInfo.initialValue  = 0;

  VkSemaphoreCreateInfo binInfo{};
  binInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkSemaphoreCreateInfo tlInfo{};
  tlInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  tlInfo.pNext = &typeInfo;

  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for(uint32_t i = 0; i < MAX_FRAMES; ++i) {
    auto& f         = frames[i];
    f.timelineValue = 0;
    vkCreateSemaphore(ctx.device, &binInfo, nullptr, &f.imageAvailable);
    vkCreateSemaphore(ctx.device, &binInfo, nullptr, &f.renderFinished);
    vkCreateSemaphore(ctx.device, &tlInfo, nullptr, &f.timelineSemaphore);
    vkCreateFence(ctx.device, &fenceInfo, nullptr, &f.inFlightFence);

    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    ai.commandPool = ctx.graphicsCommandPool;
    vkAllocateCommandBuffers(ctx.device, &ai, &f.graphicsCmd);
    ai.commandPool = ctx.computeCommandPool;
    vkAllocateCommandBuffers(ctx.device, &ai, &f.computeCmd);
  }
}

// ── Screenshot ────────────────────────────────────────────────────────────────

void BaseApp::saveScreenshot(uint32_t imageIdx, int nShots) {
  if(nShots <= 0 || screenshotsTaken >= nShots) return;
  if(screenshotsTaken == 0) perfStartTime_ = std::chrono::steady_clock::now();

  if(!screenshotDir.empty()) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/frame%04d.ppm", screenshotDir.c_str(), screenshotsTaken + 1);
    ctx.saveScreenshotPPM(buf, imageIdx);
  }

  ++screenshotsTaken;
  if(screenshotsTaken >= nShots) {
    shouldExit       = true;
    double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - perfStartTime_).count();
    std::printf("PERF_RESULT frames=%d elapsed_s=%.6f ms_per_frame=%.6f\n", nShots, elapsed_s, elapsed_s * 1000.0 / double(nShots));
    std::fflush(stdout);
  }
}

// ── Cleanup ───────────────────────────────────────────────────────────────────

void BaseApp::cleanupBase() {
  for(auto& f : frames) {
    vkDestroySemaphore(ctx.device, f.imageAvailable, nullptr);
    vkDestroySemaphore(ctx.device, f.renderFinished, nullptr);
    vkDestroySemaphore(ctx.device, f.timelineSemaphore, nullptr);
    vkDestroyFence(ctx.device, f.inFlightFence, nullptr);
  }

  vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
  ctx.cleanup();

  glfwDestroyWindow(window);
  glfwTerminate();
}
