#include "DefineShaderCompiler.h"

#include <fstream>
#include <memory>
#include <shaderc/shaderc.hpp>
#include <stdexcept>

namespace {

// common.glsl 等の #include を SHADER_SRC_DIR から解決する
// (ForceShaderCompiler.cpp の ForceGlslIncluder と同一実装)。
class DefineGlslIncluder : public shaderc::CompileOptions::IncluderInterface {
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

std::vector<uint32_t> DefineShaderCompiler::compile(const std::string& shaderSourceName,
                                                      const std::vector<std::pair<std::string, std::string>>& defines) {
  std::string path = std::string(SHADER_SRC_DIR) + "/" + shaderSourceName;
  std::ifstream file(path);
  if(!file.is_open()) throw std::runtime_error("DefineShaderCompiler: cannot open " + path);
  std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  shaderc::Compiler compiler;
  shaderc::CompileOptions options;
  options.SetIncluder(std::make_unique<DefineGlslIncluder>());
  options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
  options.SetOptimizationLevel(shaderc_optimization_level_performance);
  for(const auto& [name, value] : defines) options.AddMacroDefinition(name, value);

  shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(source, shaderc_compute_shader, shaderSourceName.c_str(), options);
  if(result.GetCompilationStatus() != shaderc_compilation_status_success)
    throw std::runtime_error("DefineShaderCompiler: compile failed (" + shaderSourceName + "): " + result.GetErrorMessage());

  return {result.cbegin(), result.cend()};
}
