#define VMA_IMPLEMENTATION
#include "HeadlessCtx.h"
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef __APPLE__
// ObjC++ wrapper: catches NSException from MoltenVK/Metal and rethrows as std::runtime_error
extern "C" VkResult tryVkCreateInstance(const VkInstanceCreateInfo*, const VkAllocationCallbacks*, VkInstance*);
#endif

void HeadlessCtx::init() {
  // ── Instance ──────────────────────────────────────────────────────────
  fprintf(stderr, "[HCtx] step 1: vkCreateInstance\n");
  fflush(stderr);
  VkApplicationInfo appInfo{};
  appInfo.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "vulkan_tests";
  appInfo.apiVersion       = VK_API_VERSION_1_2;

  VkInstanceCreateInfo instCI{};
  instCI.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instCI.pApplicationInfo = &appInfo;
#ifdef __APPLE__
  const char* instExts[] = {
    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
  };
  instCI.enabledExtensionCount   = 2;
  instCI.ppEnabledExtensionNames = instExts;
  instCI.flags                   = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  if(tryVkCreateInstance(&instCI, nullptr, &instance) != VK_SUCCESS)
#else
  if(vkCreateInstance(&instCI, nullptr, &instance) != VK_SUCCESS)
#endif
    throw std::runtime_error("HeadlessCtx: vkCreateInstance failed");
  fprintf(stderr, "[HCtx] step 1 OK\n");
  fflush(stderr);

  // ── Physical device ───────────────────────────────────────────────────
  fprintf(stderr, "[HCtx] step 2: vkEnumeratePhysicalDevices\n");
  fflush(stderr);
  uint32_t devCount = 0;
  vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
  if(devCount == 0) throw std::runtime_error("HeadlessCtx: no Vulkan GPU");
  std::vector<VkPhysicalDevice> devs(devCount);
  vkEnumeratePhysicalDevices(instance, &devCount, devs.data());
  physicalDevice = devs[0];
  fprintf(stderr, "[HCtx] step 2 OK: %u device(s)\n", devCount);
  fflush(stderr);

  // ── Compute queue family ──────────────────────────────────────────────
  fprintf(stderr, "[HCtx] step 3: queue family\n");
  fflush(stderr);
  uint32_t qfCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, nullptr);
  std::vector<VkQueueFamilyProperties> qf(qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, qf.data());
  for(uint32_t i = 0; i < qfCount; ++i) {
    if(qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      computeFamily = i;
      break;
    }
  }
  if(computeFamily == UINT32_MAX) throw std::runtime_error("HeadlessCtx: no compute queue");
  fprintf(stderr, "[HCtx] step 3 OK: computeFamily=%u\n", computeFamily);
  fflush(stderr);

  // ── Memory diagnostics ────────────────────────────────────────────────
  {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mp);
    fprintf(stderr, "[HCtx] memHeaps=%u memTypes=%u\n", mp.memoryHeapCount, mp.memoryTypeCount);
    for(uint32_t i = 0; i < mp.memoryHeapCount; i++) fprintf(stderr, "[HCtx]   heap[%u] size=%llu flags=0x%x\n", i, (unsigned long long)mp.memoryHeaps[i].size, mp.memoryHeaps[i].flags);
    for(uint32_t i = 0; i < mp.memoryTypeCount; i++) fprintf(stderr, "[HCtx]   type[%u] heap=%u props=0x%x\n", i, mp.memoryTypes[i].heapIndex, mp.memoryTypes[i].propertyFlags);
    fflush(stderr);
  }

  // ── Logical device ────────────────────────────────────────────────────
  fprintf(stderr, "[HCtx] step 4: vkCreateDevice\n");
  fflush(stderr);
  float pri = 1.0f;
  VkDeviceQueueCreateInfo queueCI{};
  queueCI.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCI.queueFamilyIndex = computeFamily;
  queueCI.queueCount       = 1;
  queueCI.pQueuePriorities = &pri;

  // Vulkan 1.2 promotes descriptor-indexing to core; all 1.2 devices support it.
  // Use VkPhysicalDeviceVulkan12Features to enable required bindless flags
  // without a separate VkPhysicalDeviceDescriptorIndexingFeatures struct that
  // can crash older or alternate MoltenVK builds via pNext mishandling.
  VkPhysicalDeviceVulkan12Features vk12{};
  vk12.sType                                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  vk12.runtimeDescriptorArray                     = VK_TRUE;
  vk12.descriptorBindingVariableDescriptorCount   = VK_TRUE;
  vk12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
  vk12.descriptorBindingPartiallyBound            = VK_TRUE;
  // Required by VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT in descriptor pool
  vk12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;

  VkDeviceCreateInfo devCI{};
  devCI.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  devCI.pNext                = &vk12;
  devCI.queueCreateInfoCount = 1;
  devCI.pQueueCreateInfos    = &queueCI;
#ifdef __APPLE__
  const char* devExts[]         = {"VK_KHR_portability_subset"};
  devCI.enabledExtensionCount   = 1;
  devCI.ppEnabledExtensionNames = devExts;
#endif
  if(vkCreateDevice(physicalDevice, &devCI, nullptr, &device) != VK_SUCCESS) throw std::runtime_error("HeadlessCtx: vkCreateDevice failed");
  fprintf(stderr, "[HCtx] step 4 OK\n");
  fflush(stderr);

  vkGetDeviceQueue(device, computeFamily, 0, &computeQueue);

  // ── VMA ───────────────────────────────────────────────────────────────
  fprintf(stderr, "[HCtx] step 5: vmaCreateAllocator\n");
  fflush(stderr);
  VmaAllocatorCreateInfo vmaCI{};
  vmaCI.physicalDevice   = physicalDevice;
  vmaCI.device           = device;
  vmaCI.instance         = instance;
  vmaCI.vulkanApiVersion = VK_API_VERSION_1_2;
  // Limit block size: VMA defaults can exceed the ~32 MB of VRAM available
  // in paravirtualized CI GPU environments (AppleParavirtDevice).
  vmaCI.preferredLargeHeapBlockSize = 4ull * 1024 * 1024; // 4 MB
  if(vmaCreateAllocator(&vmaCI, &allocator) != VK_SUCCESS) throw std::runtime_error("HeadlessCtx: vmaCreateAllocator failed");
  fprintf(stderr, "[HCtx] step 5 OK\n");
  fflush(stderr);

  // ── Command pool (compute) ────────────────────────────────────────────
  fprintf(stderr, "[HCtx] step 6: vkCreateCommandPool\n");
  fflush(stderr);
  VkCommandPoolCreateInfo poolCI{};
  poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolCI.queueFamilyIndex = computeFamily;
  poolCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  if(vkCreateCommandPool(device, &poolCI, nullptr, &commandPool) != VK_SUCCESS) throw std::runtime_error("HeadlessCtx: vkCreateCommandPool failed");
  fprintf(stderr, "[HCtx] step 6 OK\n");
  fflush(stderr);

  // ── Descriptor pool (bindless storage buffers) ────────────────────────
  fprintf(stderr, "[HCtx] step 7: vkCreateDescriptorPool\n");
  fflush(stderr);
  VkDescriptorPoolSize ps{};
  ps.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 512;
  VkDescriptorPoolCreateInfo dpCI{};
  dpCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpCI.maxSets       = 16;
  dpCI.poolSizeCount = 1;
  dpCI.pPoolSizes    = &ps;
  dpCI.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
  if(vkCreateDescriptorPool(device, &dpCI, nullptr, &descriptorPool) != VK_SUCCESS) throw std::runtime_error("HeadlessCtx: vkCreateDescriptorPool failed");
  fprintf(stderr, "[HCtx] step 7 OK: HeadlessCtx init complete\n");
  fflush(stderr);
}

void HeadlessCtx::cleanup() {
  vkDeviceWaitIdle(device);
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyCommandPool(device, commandPool, nullptr);
  vmaDestroyAllocator(allocator);
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}

VkCommandBuffer HeadlessCtx::beginCmd() const {
  VkCommandBufferAllocateInfo ai{};
  ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool        = commandPool;
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

void HeadlessCtx::submitCmd(VkCommandBuffer cmd) const {
  vkEndCommandBuffer(cmd);
  VkSubmitInfo si{};
  si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers    = &cmd;
  vkQueueSubmit(computeQueue, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(computeQueue);
  vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

void HeadlessCtx::readBuffer(VkBuffer src, VkDeviceSize srcOffset, void* dst, VkDeviceSize size) const {
  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size  = size;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
  aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VkBuffer stageBuf;
  VmaAllocation stageAlloc;
  VmaAllocationInfo info;
  vmaCreateBuffer(allocator, &bci, &aci, &stageBuf, &stageAlloc, &info);

  VkCommandBuffer cmd = beginCmd();
  VkBufferCopy region{srcOffset, 0, size};
  vkCmdCopyBuffer(cmd, src, stageBuf, 1, &region);
  submitCmd(cmd);

  memcpy(dst, info.pMappedData, size);
  vmaDestroyBuffer(allocator, stageBuf, stageAlloc);
}
