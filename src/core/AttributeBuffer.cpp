#include "AttributeBuffer.h"
#include <cstring>
#include <stdexcept>

// Declared in tests/helpers/VkWrapper.mm (ObjC++ NSException catcher).
// On non-Apple or non-test builds this symbol may not exist; keep it isolated here.
#ifdef __APPLE__
extern "C" VkResult tryVmaCreateBuffer(VmaAllocator, const VkBufferCreateInfo*, const VmaAllocationCreateInfo*, VkBuffer*, VmaAllocation*, VmaAllocationInfo*);
#define SAFE_VMA_CREATE_BUFFER(alloc, bci, aci, buf, mem, info) tryVmaCreateBuffer(alloc, bci, aci, buf, mem, info)
#else
#define SAFE_VMA_CREATE_BUFFER(alloc, bci, aci, buf, mem, info) vmaCreateBuffer(alloc, bci, aci, buf, mem, info)
#endif

void AttributeBuffer::init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool) {
  device_    = device;
  allocator_ = allocator;
  createDescriptorSetLayout();
  createDescriptorSet(descriptorPool);
}

void AttributeBuffer::cleanup() {
  for(auto& [name, attr] : attributes_) vmaDestroyBuffer(allocator_, attr.buffer, attr.allocation);
  vkDestroyDescriptorSetLayout(device_, descriptorSetLayout, nullptr);
}

uint32_t AttributeBuffer::addAttribute(const std::string& name, VkDeviceSize elementSize, uint32_t count) {
  if(nextIndex_ >= MAX_BINDLESS_BUFFERS) throw std::runtime_error("Bindless buffer limit exceeded");

  VkBufferCreateInfo bufInfo{};
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufInfo.size  = elementSize * count;
  bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  VmaAllocationCreateInfo allocInfo{};
  allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

  Attribute attr{};
  attr.count         = count;
  attr.bindlessIndex = nextIndex_++;
  attr.elementSize   = elementSize;

  VkResult r = SAFE_VMA_CREATE_BUFFER(allocator_, &bufInfo, &allocInfo, &attr.buffer, &attr.allocation, nullptr);
  if(r != VK_SUCCESS || attr.buffer == VK_NULL_HANDLE) {
    attr.buffer     = VK_NULL_HANDLE;
    attr.allocation = VK_NULL_HANDLE;
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    r               = SAFE_VMA_CREATE_BUFFER(allocator_, &bufInfo, &allocInfo, &attr.buffer, &attr.allocation, nullptr);
    if(r != VK_SUCCESS) throw std::runtime_error("Failed to create attribute buffer (both GPU_ONLY and CPU_TO_GPU): " + name);
  }
  if(attr.buffer == VK_NULL_HANDLE) throw std::runtime_error("vmaCreateBuffer success but null buffer: " + name);

  registerBuffer(attr.bindlessIndex, attr.buffer);
  attributes_[name] = attr;
  return attr.bindlessIndex;
}

void AttributeBuffer::upload(const std::string& name, const void* data, VkDeviceSize byteSize, VkCommandPool cmdPool, VkQueue queue) {
  auto it = attributes_.find(name);
  if(it == attributes_.end()) throw std::runtime_error("Attribute not found: " + name);

  // Staging buffer
  VkBufferCreateInfo stageBufInfo{};
  stageBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  stageBufInfo.size  = byteSize;
  stageBufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  VmaAllocationCreateInfo stageAllocInfo{};
  stageAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

  VkBuffer stageBuf;
  VmaAllocation stageAlloc;
  vmaCreateBuffer(allocator_, &stageBufInfo, &stageAllocInfo, &stageBuf, &stageAlloc, nullptr);

  void* mapped;
  vmaMapMemory(allocator_, stageAlloc, &mapped);
  memcpy(mapped, data, byteSize);
  vmaUnmapMemory(allocator_, stageAlloc);

  // Copy via single-time command
  VkCommandBufferAllocateInfo ai{};
  ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool        = cmdPool;
  ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(device_, &ai, &cmd);

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);

  VkBufferCopy region{};
  region.size = byteSize;
  vkCmdCopyBuffer(cmd, stageBuf, it->second.buffer, 1, &region);

  vkEndCommandBuffer(cmd);

  VkSubmitInfo si{};
  si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers    = &cmd;
  vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);
  vkFreeCommandBuffers(device_, cmdPool, 1, &cmd);

  vmaDestroyBuffer(allocator_, stageBuf, stageAlloc);
}

void AttributeBuffer::uploadAt(const std::string& name, const void* data, VkDeviceSize byteSize, VkDeviceSize dstOffset, VkCommandPool cmdPool, VkQueue queue) {
  auto it = attributes_.find(name);
  if(it == attributes_.end()) throw std::runtime_error("Attribute not found: " + name);

  VkBufferCreateInfo stageBufInfo{};
  stageBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  stageBufInfo.size  = byteSize;
  stageBufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  VmaAllocationCreateInfo stageAllocInfo{};
  stageAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

  VkBuffer stageBuf;
  VmaAllocation stageAlloc;
  vmaCreateBuffer(allocator_, &stageBufInfo, &stageAllocInfo, &stageBuf, &stageAlloc, nullptr);

  void* mapped;
  vmaMapMemory(allocator_, stageAlloc, &mapped);
  memcpy(mapped, data, byteSize);
  vmaUnmapMemory(allocator_, stageAlloc);

  VkCommandBufferAllocateInfo ai{};
  ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool        = cmdPool;
  ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(device_, &ai, &cmd);

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);

  VkBufferCopy region{};
  region.srcOffset = 0;
  region.dstOffset = dstOffset;
  region.size      = byteSize;
  vkCmdCopyBuffer(cmd, stageBuf, it->second.buffer, 1, &region);

  vkEndCommandBuffer(cmd);

  VkSubmitInfo si{};
  si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers    = &cmd;
  vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);
  vkFreeCommandBuffers(device_, cmdPool, 1, &cmd);
  vmaDestroyBuffer(allocator_, stageBuf, stageAlloc);
}

void AttributeBuffer::resizeAttribute(const std::string& name, uint32_t newCount, VkCommandPool cmdPool, VkQueue queue) {
  auto it = attributes_.find(name);
  if(it == attributes_.end()) throw std::runtime_error("Attribute not found: " + name);
  Attribute& attr = it->second;
  if(newCount == attr.count) return;

  VkBufferCreateInfo bufInfo{};
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufInfo.size  = attr.elementSize * newCount;
  bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  VmaAllocationCreateInfo allocInfo{};
  allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

  VkBuffer newBuffer;
  VmaAllocation newAlloc;
  VkResult r = SAFE_VMA_CREATE_BUFFER(allocator_, &bufInfo, &allocInfo, &newBuffer, &newAlloc, nullptr);
  if(r != VK_SUCCESS || newBuffer == VK_NULL_HANDLE) {
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    r               = SAFE_VMA_CREATE_BUFFER(allocator_, &bufInfo, &allocInfo, &newBuffer, &newAlloc, nullptr);
    if(r != VK_SUCCESS) throw std::runtime_error("Failed to resize attribute buffer: " + name);
  }

  VkDeviceSize copyBytes = attr.elementSize * std::min(attr.count, newCount);
  if(copyBytes > 0 && attr.buffer != VK_NULL_HANDLE) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = cmdPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device_, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    VkBufferCopy region{};
    region.size = copyBytes;
    vkCmdCopyBuffer(cmd, attr.buffer, newBuffer, 1, &region);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device_, cmdPool, 1, &cmd);
  }

  if(attr.buffer != VK_NULL_HANDLE) vmaDestroyBuffer(allocator_, attr.buffer, attr.allocation);

  attr.buffer     = newBuffer;
  attr.allocation = newAlloc;
  attr.count      = newCount;

  registerBuffer(attr.bindlessIndex, attr.buffer);
}

VkBuffer AttributeBuffer::getBuffer(const std::string& name) const { return attributes_.at(name).buffer; }

uint32_t AttributeBuffer::getIndex(const std::string& name) const { return attributes_.at(name).bindlessIndex; }

uint32_t AttributeBuffer::getCount(const std::string& name) const { return attributes_.at(name).count; }

// ── Bindless Descriptor ────────────────────────────────────────────────────
void AttributeBuffer::createDescriptorSetLayout() {
  VkDescriptorBindingFlags bindingFlags = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

  VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
  flagsInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
  flagsInfo.bindingCount  = 1;
  flagsInfo.pBindingFlags = &bindingFlags;

  VkDescriptorSetLayoutBinding binding{};
  binding.binding         = 0;
  binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = MAX_BINDLESS_BUFFERS;
  binding.stageFlags      = VK_SHADER_STAGE_ALL;

  VkDescriptorSetLayoutCreateInfo info{};
  info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  info.pNext        = &flagsInfo;
  info.bindingCount = 1;
  info.pBindings    = &binding;
  info.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;

  if(vkCreateDescriptorSetLayout(device_, &info, nullptr, &descriptorSetLayout) != VK_SUCCESS) throw std::runtime_error("Failed to create bindless descriptor set layout");
}

void AttributeBuffer::createDescriptorSet(VkDescriptorPool pool) {
  uint32_t varCount = MAX_BINDLESS_BUFFERS;

  VkDescriptorSetVariableDescriptorCountAllocateInfo varInfo{};
  varInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
  varInfo.descriptorSetCount = 1;
  varInfo.pDescriptorCounts  = &varCount;

  VkDescriptorSetAllocateInfo ai{};
  ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.pNext              = &varInfo;
  ai.descriptorPool     = pool;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts        = &descriptorSetLayout;

  if(vkAllocateDescriptorSets(device_, &ai, &descriptorSet) != VK_SUCCESS) throw std::runtime_error("Failed to allocate bindless descriptor set");
}

void AttributeBuffer::registerBuffer(uint32_t index, VkBuffer buffer) {
  VkDescriptorBufferInfo bufInfo{};
  bufInfo.buffer = buffer;
  bufInfo.offset = 0;
  bufInfo.range  = VK_WHOLE_SIZE;

  VkWriteDescriptorSet write{};
  write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet          = descriptorSet;
  write.dstBinding      = 0;
  write.dstArrayElement = index;
  write.descriptorCount = 1;
  write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo     = &bufInfo;

  vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}
