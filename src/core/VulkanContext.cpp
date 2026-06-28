#define VMA_IMPLEMENTATION
#include "VulkanContext.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

// ── Debug messenger ───────────────────────────────────────────────────────
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
  if(severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) std::cerr << "[Vulkan] " << data->pMessage << "\n";
  return VK_FALSE;
}

static VkResult createDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, VkDebugUtilsMessengerEXT* pMessenger) {
  auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  return fn ? fn(instance, pCreateInfo, nullptr, pMessenger) : VK_ERROR_EXTENSION_NOT_PRESENT;
}

static void destroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT messenger) {
  auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if(fn) fn(instance, messenger, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────

void VulkanContext::init(GLFWwindow* window) {
  window_ = window;
  createInstance();
#ifdef VULKAN_VALIDATION
  createDebugMessenger();
#endif
  createSurface();
  selectPhysicalDevice();
  createLogicalDevice();
  createAllocator();
  createSwapchain();
  createImageViews();
  createRenderPass();
  createFramebuffers();
  createCommandPools();
}

void VulkanContext::cleanup() {
  vkDeviceWaitIdle(device);

  cleanupSwapchain();

  vkDestroyCommandPool(device, graphicsCommandPool, nullptr);
  if(computeFamily != graphicsFamily) vkDestroyCommandPool(device, computeCommandPool, nullptr);

  vmaDestroyAllocator(allocator);
  vkDestroyDevice(device, nullptr);

#ifdef VULKAN_VALIDATION
  destroyDebugUtilsMessengerEXT(instance, debugMessenger_);
#endif
  vkDestroySurfaceKHR(instance, surface, nullptr);
  vkDestroyInstance(instance, nullptr);
}

// ── Instance ──────────────────────────────────────────────────────────────
void VulkanContext::createInstance() {
  VkApplicationInfo appInfo{};
  appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName   = "vulkan-sim";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName        = "No Engine";
  appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion         = VK_API_VERSION_1_2;

  auto extensions = getRequiredInstanceExtensions();

  const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef VULKAN_VALIDATION
  if(!checkValidationLayerSupport()) throw std::runtime_error("Validation layers not available");
#endif

  VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
  debugInfo.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  debugInfo.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  debugInfo.pfnUserCallback = debugCallback;

  VkInstanceCreateInfo info{};
  info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  info.pApplicationInfo        = &appInfo;
  info.enabledExtensionCount   = (uint32_t)extensions.size();
  info.ppEnabledExtensionNames = extensions.data();
  // macOS: enumerate portability
  info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

#ifdef VULKAN_VALIDATION
  info.enabledLayerCount   = (uint32_t)validationLayers.size();
  info.ppEnabledLayerNames = validationLayers.data();
  info.pNext               = &debugInfo;
#endif

  if(vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS) throw std::runtime_error("Failed to create Vulkan instance");
}

std::vector<const char*> VulkanContext::getRequiredInstanceExtensions() {
  uint32_t glfwCount    = 0;
  const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwCount);
  std::vector<const char*> exts(glfwExts, glfwExts + glfwCount);

  exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  exts.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#ifdef VULKAN_VALIDATION
  exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
  return exts;
}

bool VulkanContext::checkValidationLayerSupport() {
  uint32_t count;
  vkEnumerateInstanceLayerProperties(&count, nullptr);
  std::vector<VkLayerProperties> layers(count);
  vkEnumerateInstanceLayerProperties(&count, layers.data());

  for(auto& l : layers)
    if(strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) return true;
  return false;
}

// ── Debug messenger ───────────────────────────────────────────────────────
void VulkanContext::createDebugMessenger() {
  VkDebugUtilsMessengerCreateInfoEXT info{};
  info.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  info.pfnUserCallback = debugCallback;

  if(createDebugUtilsMessengerEXT(instance, &info, &debugMessenger_) != VK_SUCCESS) throw std::runtime_error("Failed to create debug messenger");
}

// ── Surface ───────────────────────────────────────────────────────────────
void VulkanContext::createSurface() {
  if(glfwCreateWindowSurface(instance, window_, nullptr, &surface) != VK_SUCCESS) throw std::runtime_error("Failed to create window surface");
}

// ── Physical device ───────────────────────────────────────────────────────
void VulkanContext::selectPhysicalDevice() {
  uint32_t count;
  vkEnumeratePhysicalDevices(instance, &count, nullptr);
  if(count == 0) throw std::runtime_error("No Vulkan-capable GPU found");

  std::vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(instance, &count, devices.data());

  for(auto& pd : devices) {
    // Check required features
    VkPhysicalDeviceDescriptorIndexingFeatures indexingFeatures{};
    indexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;

    VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures{};
    timelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    timelineFeatures.pNext = &indexingFeatures;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &timelineFeatures;

    vkGetPhysicalDeviceFeatures2(pd, &features2);

    if(!timelineFeatures.timelineSemaphore) continue;
    if(!indexingFeatures.runtimeDescriptorArray) continue;
    if(!indexingFeatures.descriptorBindingVariableDescriptorCount) continue;
    if(!indexingFeatures.shaderStorageBufferArrayNonUniformIndexing) continue;

    // Check queue families
    uint32_t qCount;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qProps(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, qProps.data());

    int gfx = -1, cmp = -1;
    for(uint32_t i = 0; i < qCount; i++) {
      VkBool32 present = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present);
      if((qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present && gfx < 0) gfx = (int)i;
      if((qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && cmp < 0) cmp = (int)i;
    }
    if(gfx < 0 || cmp < 0) continue;

    physicalDevice = pd;
    graphicsFamily = (uint32_t)gfx;
    computeFamily  = (uint32_t)cmp;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    std::cout << "[Vulkan] GPU: " << props.deviceName << std::endl;
    return;
  }
  throw std::runtime_error("No suitable GPU found");
}

// ── Logical device ────────────────────────────────────────────────────────
void VulkanContext::createLogicalDevice() {
  std::set<uint32_t> uniqueFamilies = {graphicsFamily, computeFamily};
  float priority                    = 1.0f;

  std::vector<VkDeviceQueueCreateInfo> queueInfos;
  for(uint32_t f : uniqueFamilies) {
    VkDeviceQueueCreateInfo qi{};
    qi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi.queueFamilyIndex = f;
    qi.queueCount       = 1;
    qi.pQueuePriorities = &priority;
    queueInfos.push_back(qi);
  }

  // Feature chain
  VkPhysicalDeviceDescriptorIndexingFeatures indexingFeatures{};
  indexingFeatures.sType                                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
  indexingFeatures.runtimeDescriptorArray                     = VK_TRUE;
  indexingFeatures.descriptorBindingVariableDescriptorCount   = VK_TRUE;
  indexingFeatures.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
  indexingFeatures.descriptorBindingPartiallyBound            = VK_TRUE;

  VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures{};
  timelineFeatures.sType             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
  timelineFeatures.timelineSemaphore = VK_TRUE;
  timelineFeatures.pNext             = &indexingFeatures;

  VkPhysicalDeviceFeatures2 features2{};
  features2.sType                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.features.shaderInt64 = VK_FALSE;
  features2.pNext                = &timelineFeatures;

  std::vector<const char*> deviceExts = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    "VK_KHR_portability_subset",
    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
  };

  VkDeviceCreateInfo info{};
  info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  info.pNext                   = &features2;
  info.queueCreateInfoCount    = (uint32_t)queueInfos.size();
  info.pQueueCreateInfos       = queueInfos.data();
  info.enabledExtensionCount   = (uint32_t)deviceExts.size();
  info.ppEnabledExtensionNames = deviceExts.data();

  if(vkCreateDevice(physicalDevice, &info, nullptr, &device) != VK_SUCCESS) throw std::runtime_error("Failed to create logical device");

  vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);
  vkGetDeviceQueue(device, computeFamily, 0, &computeQueue);
}

// ── VMA ───────────────────────────────────────────────────────────────────
void VulkanContext::createAllocator() {
  VmaAllocatorCreateInfo info{};
  info.physicalDevice   = physicalDevice;
  info.device           = device;
  info.instance         = instance;
  info.vulkanApiVersion = VK_API_VERSION_1_2;

  if(vmaCreateAllocator(&info, &allocator) != VK_SUCCESS) throw std::runtime_error("Failed to create VMA allocator");
}

// ── Swapchain ─────────────────────────────────────────────────────────────
void VulkanContext::createSwapchain() {
  // Choose surface format
  uint32_t fmtCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &fmtCount, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(fmtCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &fmtCount, formats.data());

  VkSurfaceFormatKHR chosen = formats[0];
  for(auto& f : formats)
    if(f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) chosen = f;
  swapchainFormat = chosen.format;

  // Capabilities
  VkSurfaceCapabilitiesKHR caps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &caps);

  int w, h;
  glfwGetFramebufferSize(window_, &w, &h);
  swapchainExtent = {std::clamp((uint32_t)w, caps.minImageExtent.width, caps.maxImageExtent.width), std::clamp((uint32_t)h, caps.minImageExtent.height, caps.maxImageExtent.height)};

  uint32_t imgCount = caps.minImageCount + 1;
  if(caps.maxImageCount > 0 && imgCount > caps.maxImageCount) imgCount = caps.maxImageCount;

  VkSwapchainCreateInfoKHR info{};
  info.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  info.surface          = surface;
  info.minImageCount    = imgCount;
  info.imageFormat      = chosen.format;
  info.imageColorSpace  = chosen.colorSpace;
  info.imageExtent      = swapchainExtent;
  info.imageArrayLayers = 1;
  info.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  info.preTransform     = caps.currentTransform;
  info.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  // VSync OFF 時: MAILBOX → IMMEDIATE → FIFO の優先順で選択
  VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
  if(!vsync) {
    uint32_t pmCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &pmCount, nullptr);
    std::vector<VkPresentModeKHR> pms(pmCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &pmCount, pms.data());
    for(auto pm : pms) {
      if(pm == VK_PRESENT_MODE_MAILBOX_KHR) {
        presentMode = pm;
        break;
      }
      if(pm == VK_PRESENT_MODE_IMMEDIATE_KHR) presentMode = pm;
    }
  }
  info.presentMode = presentMode;
  info.clipped     = VK_TRUE;

  uint32_t indices[] = {graphicsFamily, computeFamily};
  if(graphicsFamily != computeFamily) {
    info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
    info.queueFamilyIndexCount = 2;
    info.pQueueFamilyIndices   = indices;
  } else {
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  if(vkCreateSwapchainKHR(device, &info, nullptr, &swapchain) != VK_SUCCESS) throw std::runtime_error("Failed to create swapchain");

  uint32_t n;
  vkGetSwapchainImagesKHR(device, swapchain, &n, nullptr);
  swapchainImages.resize(n);
  vkGetSwapchainImagesKHR(device, swapchain, &n, swapchainImages.data());
}

void VulkanContext::createImageViews() {
  swapchainImageViews.resize(swapchainImages.size());
  for(size_t i = 0; i < swapchainImages.size(); i++) {
    VkImageViewCreateInfo info{};
    info.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image                       = swapchainImages[i];
    info.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
    info.format                      = swapchainFormat;
    info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    info.subresourceRange.levelCount = 1;
    info.subresourceRange.layerCount = 1;
    if(vkCreateImageView(device, &info, nullptr, &swapchainImageViews[i]) != VK_SUCCESS) throw std::runtime_error("Failed to create image view");
  }
}

// ── RenderPass ────────────────────────────────────────────────────────────
void VulkanContext::createRenderPass() {
  VkAttachmentDescription color{};
  color.format         = swapchainFormat;
  color.samples        = VK_SAMPLE_COUNT_1_BIT;
  color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
  color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
  color.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference colorRef{};
  colorRef.attachment = 0;
  colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments    = &colorRef;

  VkSubpassDependency dep{};
  dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
  dep.dstSubpass    = 0;
  dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dep.srcAccessMask = 0;
  dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo info{};
  info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  info.attachmentCount = 1;
  info.pAttachments    = &color;
  info.subpassCount    = 1;
  info.pSubpasses      = &subpass;
  info.dependencyCount = 1;
  info.pDependencies   = &dep;

  if(vkCreateRenderPass(device, &info, nullptr, &renderPass) != VK_SUCCESS) throw std::runtime_error("Failed to create render pass");
}

void VulkanContext::createFramebuffers() {
  framebuffers.resize(swapchainImageViews.size());
  for(size_t i = 0; i < swapchainImageViews.size(); i++) {
    VkFramebufferCreateInfo info{};
    info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    info.renderPass      = renderPass;
    info.attachmentCount = 1;
    info.pAttachments    = &swapchainImageViews[i];
    info.width           = swapchainExtent.width;
    info.height          = swapchainExtent.height;
    info.layers          = 1;
    if(vkCreateFramebuffer(device, &info, nullptr, &framebuffers[i]) != VK_SUCCESS) throw std::runtime_error("Failed to create framebuffer");
  }
}

// ── Command pools ─────────────────────────────────────────────────────────
void VulkanContext::createCommandPools() {
  auto makePool = [&](uint32_t family) {
    VkCommandPoolCreateInfo info{};
    info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.queueFamilyIndex = family;
    info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool;
    if(vkCreateCommandPool(device, &info, nullptr, &pool) != VK_SUCCESS) throw std::runtime_error("Failed to create command pool");
    return pool;
  };

  graphicsCommandPool = makePool(graphicsFamily);
  computeCommandPool  = (computeFamily == graphicsFamily) ? graphicsCommandPool : makePool(computeFamily);
}

// ── Swapchain recreation ──────────────────────────────────────────────────
void VulkanContext::cleanupSwapchain() {
  for(auto fb : framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
  for(auto iv : swapchainImageViews) vkDestroyImageView(device, iv, nullptr);
  vkDestroySwapchainKHR(device, swapchain, nullptr);
  vkDestroyRenderPass(device, renderPass, nullptr);
}

void VulkanContext::recreateSwapchain() {
  vkDeviceWaitIdle(device);
  cleanupSwapchain();
  createSwapchain();
  createImageViews();
  createRenderPass();
  createFramebuffers();
}

// ── Helpers ───────────────────────────────────────────────────────────────
VkCommandBuffer VulkanContext::beginSingleTimeCommands(VkCommandPool pool) {
  VkCommandBufferAllocateInfo ai{};
  ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool        = pool;
  ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(device, &ai, &cmd);

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);
  return cmd;
}

void VulkanContext::endSingleTimeCommands(VkCommandPool pool, VkCommandBuffer cmd, VkQueue queue) {
  vkEndCommandBuffer(cmd);
  VkSubmitInfo si{};
  si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers    = &cmd;
  vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);
  vkFreeCommandBuffers(device, pool, 1, &cmd);
}

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
  for(uint32_t i = 0; i < memProps.memoryTypeCount; i++)
    if((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) return i;
  throw std::runtime_error("Failed to find suitable memory type");
}

void VulkanContext::saveScreenshotPPM(const std::string& path, uint32_t imageIdx) {
  uint32_t w            = swapchainExtent.width;
  uint32_t h            = swapchainExtent.height;
  VkDeviceSize byteSize = (VkDeviceSize)w * h * 4;

  VkBuffer stagingBuf;
  VmaAllocation stagingAlloc;
  {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size  = byteSize;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    if(vmaCreateBuffer(allocator, &bi, &ai, &stagingBuf, &stagingAlloc, nullptr) != VK_SUCCESS) {
      std::cerr << "[Screenshot] Failed to create staging buffer\n";
      return;
    }
  }

  VkCommandBuffer cmd = beginSingleTimeCommands(graphicsCommandPool);

  // PRESENT_SRC_KHR -> TRANSFER_SRC_OPTIMAL
  {
    VkImageMemoryBarrier b{};
    b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image               = swapchainImages[imageIdx];
    b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcAccessMask       = VK_ACCESS_MEMORY_READ_BIT;
    b.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  }

  // Copy image to staging buffer
  {
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {w, h, 1};
    vkCmdCopyImageToBuffer(cmd, swapchainImages[imageIdx], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuf, 1, &region);
  }

  // Buffer barrier for host read
  {
    VkBufferMemoryBarrier b{};
    b.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask       = VK_ACCESS_HOST_READ_BIT;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer              = stagingBuf;
    b.size                = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &b, 0, nullptr);
  }

  // TRANSFER_SRC_OPTIMAL -> PRESENT_SRC_KHR (restore for present)
  {
    VkImageMemoryBarrier b{};
    b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image               = swapchainImages[imageIdx];
    b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    b.dstAccessMask       = VK_ACCESS_MEMORY_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  }

  endSingleTimeCommands(graphicsCommandPool, cmd, graphicsQueue);
  vmaInvalidateAllocation(allocator, stagingAlloc, 0, VK_WHOLE_SIZE);

  void* mapped;
  vmaMapMemory(allocator, stagingAlloc, &mapped);
  const uint8_t* px = static_cast<const uint8_t*>(mapped);

  bool isBGRA = (swapchainFormat == VK_FORMAT_B8G8R8A8_SRGB || swapchainFormat == VK_FORMAT_B8G8R8A8_UNORM || swapchainFormat == VK_FORMAT_B8G8R8A8_SNORM);

  FILE* fp = std::fopen(path.c_str(), "wb");
  if(fp) {
    std::fprintf(fp, "P6\n%u %u\n255\n", w, h);
    for(uint32_t i = 0; i < w * h; i++) {
      uint8_t rgb[3];
      if(isBGRA) {
        rgb[0] = px[i * 4 + 2];
        rgb[1] = px[i * 4 + 1];
        rgb[2] = px[i * 4 + 0];
      } else {
        rgb[0] = px[i * 4 + 0];
        rgb[1] = px[i * 4 + 1];
        rgb[2] = px[i * 4 + 2];
      }
      std::fwrite(rgb, 1, 3, fp);
    }
    std::fclose(fp);
    std::printf("[Screenshot] %s (%ux%u)\n", path.c_str(), w, h);
    std::fflush(stdout);
  } else {
    std::cerr << "[Screenshot] Cannot open: " << path << "\n";
  }

  vmaUnmapMemory(allocator, stagingAlloc);
  vmaDestroyBuffer(allocator, stagingBuf, stagingAlloc);
}
