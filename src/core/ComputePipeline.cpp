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

  VkShaderModuleCreateInfo info{};
  info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  info.codeSize = size;
  info.pCode    = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule mod;
  if(vkCreateShaderModule(device_, &info, nullptr, &mod) != VK_SUCCESS) throw std::runtime_error("Failed to create shader module: " + path);
  return mod;
}

void ComputePipeline::init(VkDevice device, VkDescriptorSetLayout bindlessLayout, const std::string& shaderPath) {
  device_ = device;

  VkPushConstantRange pcRange{};
  pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcRange.offset     = 0;
  pcRange.size       = sizeof(SimPC);

  VkPipelineLayoutCreateInfo layoutInfo{};
  layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount         = 1;
  layoutInfo.pSetLayouts            = &bindlessLayout;
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges    = &pcRange;

  if(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) throw std::runtime_error("Failed to create compute pipeline layout");

  VkShaderModule shaderModule = loadShader(shaderPath);

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

void ComputePipeline::cleanup() {
  vkDestroyPipeline(device_, pipeline, nullptr);
  vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
}

void ComputePipeline::dispatch(VkCommandBuffer cmd, VkDescriptorSet bindlessSet, const SimPC& pc, uint32_t particleCount) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
  vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);

  uint32_t groups = (particleCount + 255) / 256;
  vkCmdDispatch(cmd, groups, 1, 1);
}
