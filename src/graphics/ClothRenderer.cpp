#include "ClothRenderer.h"
#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

VkShaderModule ClothRenderer::loadShader(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if(!file.is_open()) throw std::runtime_error("Cannot open shader: " + path);
  size_t size = file.tellg();
  std::vector<char> code(size);
  file.seekg(0);
  file.read(code.data(), size);

  VkShaderModuleCreateInfo info{};
  info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  info.codeSize = size;
  info.pCode    = reinterpret_cast<const uint32_t*>(code.data());
  VkShaderModule mod;
  if(vkCreateShaderModule(device_, &info, nullptr, &mod) != VK_SUCCESS) throw std::runtime_error("Failed to create cloth shader: " + path);
  return mod;
}

void ClothRenderer::init(VkDevice device, VmaAllocator allocator, VkRenderPass renderPass, VkDescriptorSetLayout bindlessLayout, const std::string& shaderDir) {
  device_    = device;
  allocator_ = allocator;

  // Pipeline layout (同じ Bindless セット + SimPC push constants)
  VkPushConstantRange pcRange{};
  pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  pcRange.offset     = 0;
  pcRange.size       = sizeof(SimPC);

  VkPipelineLayoutCreateInfo layoutInfo{};
  layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount         = 1;
  layoutInfo.pSetLayouts            = &bindlessLayout;
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges    = &pcRange;
  if(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) throw std::runtime_error("Failed to create cloth pipeline layout");

  VkShaderModule vertMod = loadShader(shaderDir + "/cloth.vert.spv");
  VkShaderModule fragMod = loadShader(shaderDir + "/cloth.frag.spv");

  std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
  stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vertMod;
  stages[0].pName  = "main";
  stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fragMod;
  stages[1].pName  = "main";

  VkPipelineVertexInputStateCreateInfo vertexInput{};
  vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo assembly{};
  assembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount  = 1;

  VkPipelineRasterizationStateCreateInfo raster{};
  raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode    = VK_CULL_MODE_NONE; // 両面描画
  raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  raster.lineWidth   = 1.0f;

  VkPipelineMultisampleStateCreateInfo ms{};
  ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState blendAttach{};
  blendAttach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

  VkPipelineColorBlendStateCreateInfo blend{};
  blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  blend.attachmentCount = 1;
  blend.pAttachments    = &blendAttach;

  std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynState{};
  dynState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynState.dynamicStateCount = (uint32_t)dynStates.size();
  dynState.pDynamicStates    = dynStates.data();

  VkGraphicsPipelineCreateInfo pipeInfo{};
  pipeInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeInfo.stageCount          = (uint32_t)stages.size();
  pipeInfo.pStages             = stages.data();
  pipeInfo.pVertexInputState   = &vertexInput;
  pipeInfo.pInputAssemblyState = &assembly;
  pipeInfo.pViewportState      = &viewportState;
  pipeInfo.pRasterizationState = &raster;
  pipeInfo.pMultisampleState   = &ms;
  pipeInfo.pColorBlendState    = &blend;
  pipeInfo.pDynamicState       = &dynState;
  pipeInfo.layout              = pipelineLayout_;
  pipeInfo.renderPass          = renderPass;
  pipeInfo.subpass             = 0;

  if(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline_) != VK_SUCCESS) throw std::runtime_error("Failed to create cloth graphics pipeline");

  vkDestroyShaderModule(device_, vertMod, nullptr);
  vkDestroyShaderModule(device_, fragMod, nullptr);
}

void ClothRenderer::uploadIndices(const std::vector<uint32_t>& triIndices, VkCommandPool cmdPool, VkQueue queue) {
  indexCount_       = (uint32_t)triIndices.size();
  VkDeviceSize size = indexCount_ * sizeof(uint32_t);

  // ステージングバッファ
  VkBufferCreateInfo stagingInfo{};
  stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  stagingInfo.size  = size;
  stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  VmaAllocationCreateInfo stagingAllocInfo{};
  stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

  VkBuffer stageBuf;
  VmaAllocation stageAlloc;
  vmaCreateBuffer(allocator_, &stagingInfo, &stagingAllocInfo, &stageBuf, &stageAlloc, nullptr);

  void* mapped;
  vmaMapMemory(allocator_, stageAlloc, &mapped);
  memcpy(mapped, triIndices.data(), size);
  vmaUnmapMemory(allocator_, stageAlloc);

  // デバイスローカルバッファ
  VkBufferCreateInfo devInfo{};
  devInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  devInfo.size  = size;
  devInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  VmaAllocationCreateInfo devAllocInfo{};
  devAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  vmaCreateBuffer(allocator_, &devInfo, &devAllocInfo, &indexBuf_, &indexAlloc_, nullptr);

  // コピー
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

  VkBufferCopy copy{0, 0, size};
  vkCmdCopyBuffer(cmd, stageBuf, indexBuf_, 1, &copy);
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

void ClothRenderer::draw(VkCommandBuffer cmd, VkDescriptorSet bindlessSet, const SimPC& pc, uint32_t /*vertexCount*/, int32_t vertexOffset) {
  if(indexCount_ == 0) return;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1, &bindlessSet, 0, nullptr);
  vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SimPC), &pc);
  vkCmdBindIndexBuffer(cmd, indexBuf_, 0, VK_INDEX_TYPE_UINT32);
  vkCmdDrawIndexed(cmd, indexCount_, 1, 0, vertexOffset, 0);
}

void ClothRenderer::cleanup() {
  if(indexBuf_) vmaDestroyBuffer(allocator_, indexBuf_, indexAlloc_);
  if(pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
  if(pipelineLayout_) vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
}
