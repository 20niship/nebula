#include "GraphicsPipeline.h"
#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

VkShaderModule GraphicsPipeline::loadShader(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if(!file.is_open()) throw std::runtime_error("Cannot open shader: " + path);
  size_t size = file.tellg();
  std::vector<uint8_t> code(size);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(code.data()), size);
  if(!file) throw std::runtime_error("Failed to read shader: " + path);
  return loadShader(code);
}

VkShaderModule GraphicsPipeline::loadShader(const std::vector<uint8_t>& code) {
  if(code.size() % sizeof(uint32_t) != 0) throw std::runtime_error("Invalid SPIR-V size (not a multiple of 4)");
  VkShaderModuleCreateInfo info{};
  info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  info.codeSize = code.size();
  info.pCode    = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule mod;
  if(vkCreateShaderModule(device_, &info, nullptr, &mod) != VK_SUCCESS) throw std::runtime_error("Failed to create shader module from spirv");
  return mod;
}

void GraphicsPipeline::init(VkDevice device, VkRenderPass renderPass, VkDescriptorSetLayout bindlessLayout, const std::string& vertPath, const std::string& fragPath, VkPrimitiveTopology topology, bool enableBlend) {
  device_                = device;
  VkShaderModule vertMod = loadShader(vertPath);
  VkShaderModule fragMod = loadShader(fragPath);
  buildPipeline(renderPass, bindlessLayout, vertMod, fragMod, topology, enableBlend);
}

void GraphicsPipeline::initVertFromSpirv(VkDevice device, VkRenderPass renderPass, VkDescriptorSetLayout bindlessLayout, const std::vector<uint32_t>& vertSpirv, const std::string& fragPath, VkPrimitiveTopology topology, bool enableBlend) {
  device_ = device;
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(vertSpirv.data());
  VkShaderModule vertMod = loadShader(std::vector<uint8_t>(bytes, bytes + vertSpirv.size() * sizeof(uint32_t)));
  VkShaderModule fragMod = loadShader(fragPath);
  buildPipeline(renderPass, bindlessLayout, vertMod, fragMod, topology, enableBlend);
}

void GraphicsPipeline::buildPipeline(VkRenderPass renderPass, VkDescriptorSetLayout bindlessLayout, VkShaderModule vertMod, VkShaderModule fragMod, VkPrimitiveTopology topology, bool enableBlend) {
  // Pipeline layout
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

  if(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) throw std::runtime_error("Failed to create graphics pipeline layout");

  std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
  stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vertMod;
  stages[0].pName  = "main";
  stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fragMod;
  stages[1].pName  = "main";

  // No vertex input (positions from SSBO)
  VkPipelineVertexInputStateCreateInfo vertexInput{};
  vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo assembly{};
  assembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  assembly.topology = topology;

  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount  = 1;

  VkPipelineRasterizationStateCreateInfo raster{};
  raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode    = VK_CULL_MODE_NONE;
  raster.frontFace   = VK_FRONT_FACE_CLOCKWISE;
  raster.lineWidth   = 1.0f;

  VkPipelineMultisampleStateCreateInfo ms{};
  ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState blendAttach{};
  blendAttach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  if(enableBlend) {
    blendAttach.blendEnable         = VK_TRUE;
    blendAttach.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttach.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttach.colorBlendOp        = VK_BLEND_OP_ADD;
    blendAttach.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttach.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttach.alphaBlendOp        = VK_BLEND_OP_ADD;
  }

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
  pipeInfo.layout              = pipelineLayout;
  pipeInfo.renderPass          = renderPass;
  pipeInfo.subpass             = 0;

  if(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline) != VK_SUCCESS) throw std::runtime_error("Failed to create graphics pipeline");

  vkDestroyShaderModule(device_, vertMod, nullptr);
  vkDestroyShaderModule(device_, fragMod, nullptr);
}

void GraphicsPipeline::cleanup() {
  vkDestroyPipeline(device_, pipeline, nullptr);
  vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
}

void GraphicsPipeline::draw(VkCommandBuffer cmd, VkDescriptorSet bindlessSet, const SimPC& pc, uint32_t particleCount) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
  vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SimPC), &pc);
  vkCmdDraw(cmd, particleCount, 1, 0, 0);
}
