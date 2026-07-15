#include "ComputePipeline.h"
#include <fstream>
#include <stdexcept>
#include <vector>

VkShaderModule ComputePipeline::loadShader(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if(!file.is_open()) throw std::runtime_error("Cannot open shader: " + path);
  size_t size = file.tellg();
  std::vector<char> code(size);
  file.seekg(0);
  file.read(code.data(), size);

  return createShaderModuleFromSpirv(reinterpret_cast<const uint32_t*>(code.data()), size / sizeof(uint32_t));
}

VkShaderModule ComputePipeline::createShaderModuleFromSpirv(const uint32_t* code, size_t wordCount) {
  VkShaderModuleCreateInfo info{};
  info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  info.codeSize = wordCount * sizeof(uint32_t);
  info.pCode    = code;

  VkShaderModule mod;
  if(vkCreateShaderModule(device_, &info, nullptr, &mod) != VK_SUCCESS) throw std::runtime_error("Failed to create shader module");
  return mod;
}

void ComputePipeline::buildPipeline(VkShaderModule shaderModule) {
  VkPushConstantRange pcRange{};
  pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcRange.offset     = 0;
  pcRange.size       = sizeof(SimPC);

  VkPipelineLayoutCreateInfo layoutInfo{};
  layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount         = 1;
  layoutInfo.pSetLayouts            = &bindlessSetLayout_;
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges    = &pcRange;

  if(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) throw std::runtime_error("Failed to create compute pipeline layout");

  VkPipelineShaderStageCreateInfo stageInfo{};
  stageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.module = shaderModule;
  stageInfo.pName  = "main";

  VkComputePipelineCreateInfo pipeInfo{};
  pipeInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeInfo.stage  = stageInfo;
  pipeInfo.layout = pipelineLayout;

  if(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline) != VK_SUCCESS) throw std::runtime_error("Failed to create compute pipeline");

  vkDestroyShaderModule(device_, shaderModule, nullptr);
}

void ComputePipeline::init(VkDevice device, VkDescriptorSetLayout bindlessLayout, const std::string& shaderPath) {
  device_             = device;
  bindlessSetLayout_  = bindlessLayout;
  buildPipeline(loadShader(shaderPath));
}

void ComputePipeline::initFromSpirv(VkDevice device, VkDescriptorSetLayout bindlessLayout, const std::vector<uint32_t>& spirv) {
  device_            = device;
  bindlessSetLayout_ = bindlessLayout;
  buildPipeline(createShaderModuleFromSpirv(spirv.data(), spirv.size()));
}

void ComputePipeline::cleanup() {
  // rebuildForceShader() 系 (issue #30) は init/initFromSpirv 未実行の状態でも
  // 無条件で cleanup() を呼ぶため、未初期化時 (device_ == VK_NULL_HANDLE) は
  // 何もせず安全に抜ける。
  if(device_ == VK_NULL_HANDLE) return;
  if(pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline, nullptr);
  if(pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
  pipeline       = VK_NULL_HANDLE;
  pipelineLayout = VK_NULL_HANDLE;
}

void ComputePipeline::dispatch(VkCommandBuffer cmd, VkDescriptorSet bindlessSet, const SimPC& pc, uint32_t particleCount) {
  uint32_t groups = (particleCount + 255) / 256;
  dispatchRaw(cmd, bindlessSet, &pc, sizeof(SimPC), groups);
}

void ComputePipeline::dispatchRaw(VkCommandBuffer cmd, VkDescriptorSet bindlessSet, const void* pcData, uint32_t pcSize, uint32_t groupCountX) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
  vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pcData);
  vkCmdDispatch(cmd, groupCountX, 1, 1);
}
