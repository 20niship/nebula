#include "ComputePipeline.h"
#include <filesystem>
#include <fstream>
#include <memory>
#include <shaderc/shaderc.hpp>
#include <stdexcept>
#include <vector>

namespace {

// common.glsl 等の #include を SHADER_SRC_DIR から解決する
// (ForceShaderCompiler.cpp の ForceGlslIncluder と同一実装)。
class GlslIncluder : public shaderc::CompileOptions::IncluderInterface {
public:
  shaderc_include_result* GetInclude(const char* requestedSource, shaderc_include_type, const char*, size_t) override {
    std::string path = std::string(SHADER_SRC_DIR) + "/" + requestedSource;
    std::ifstream file(path);
    auto* content  = new std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    auto* name     = new std::string(requestedSource);
    auto* result   = new shaderc_include_result{};
    result->source_name        = name->c_str();
    result->source_name_length = name->size();
    result->content             = content->c_str();
    result->content_length      = content->size();
    result->user_data           = new std::pair<std::string*, std::string*>(name, content);
    return result;
  }

  void ReleaseInclude(shaderc_include_result* data) override {
    auto* owned = static_cast<std::pair<std::string*, std::string*>*>(data->user_data);
    delete owned->first;
    delete owned->second;
    delete owned;
    delete data;
  }
};

} // namespace

std::vector<uint32_t> ComputePipeline::compileShader(const std::string& shaderSourceName, const ShaderDefines& defines, bool isVertexShader) {
  std::string path = std::string(SHADER_SRC_DIR) + "/" + shaderSourceName;
  std::ifstream file(path);
  if(!file.is_open()) throw std::runtime_error("ComputePipeline: cannot open " + path);
  std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  shaderc::Compiler compiler;
  shaderc::CompileOptions options;
  options.SetIncluder(std::make_unique<GlslIncluder>());
  options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
  options.SetOptimizationLevel(shaderc_optimization_level_performance);
  for(const auto& [name, value] : defines) options.AddMacroDefinition(name, value);

  const shaderc_shader_kind kind = isVertexShader ? shaderc_glsl_vertex_shader : shaderc_compute_shader;
  shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(source, kind, shaderSourceName.c_str(), options);
  if(result.GetCompilationStatus() != shaderc_compilation_status_success)
    throw std::runtime_error("ComputePipeline: compile failed (" + shaderSourceName + "): " + result.GetErrorMessage());

  return {result.cbegin(), result.cend()};
}

VkShaderModule ComputePipeline::loadShader(const std::string& path, const ShaderDefines& defines) {
  if(!defines.empty()) {
    const std::vector<uint32_t> spirv = compileShader(std::filesystem::path(path).stem().string(), defines);
    return createShaderModuleFromSpirv(spirv.data(), spirv.size());
  }
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if(!file.is_open()) throw std::runtime_error("Cannot open shader: " + path);
  size_t size = file.tellg();
  std::vector<uint8_t> code(size);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(code.data()), size);
  if(!file) throw std::runtime_error("Failed to read shader: " + path);

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

void ComputePipeline::init(VkDevice device, VkDescriptorSetLayout bindlessLayout, const std::string& shaderPath, const ShaderDefines& defines) {
  device_             = device;
  bindlessSetLayout_  = bindlessLayout;
  buildPipeline(loadShader(shaderPath, defines));
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
