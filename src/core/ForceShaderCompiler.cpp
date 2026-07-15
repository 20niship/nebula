#include "ForceShaderCompiler.h"

#include <fstream>
#include <set>
#include <shaderc/shaderc.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

constexpr const char* kBeginMarker = "FORCE_AUTOGEN_BEGIN";
constexpr const char* kEndMarker   = "FORCE_AUTOGEN_END";

// force_common.glsl / force_noise.glsl 等の #include を SHADER_SRC_DIR から解決する。
class ForceGlslIncluder : public shaderc::CompileOptions::IncluderInterface {
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

std::string ForceShaderCompiler::forceSnippet(ForceType t) {
  switch(t) {
  case ForceType::GRAVITY:
    return R"(        if (ftype == FORCE_TYPE_GRAVITY) {
            v.xyz += FORCE_DIR(pc.forceBufIdx, fbase) * FORCE_STRENGTH(pc.forceBufIdx, fbase) * pc.dt;
        })";
  case ForceType::CONSTANT_WIND:
    return R"(        if (ftype == FORCE_TYPE_CONSTANT_WIND) {
            v.xyz += FORCE_DIR(pc.forceBufIdx, fbase) * FORCE_STRENGTH(pc.forceBufIdx, fbase) * pc.dt;
        })";
  case ForceType::TURBULENCE:
    return R"(        if (ftype == FORCE_TYPE_TURBULENCE) {
            vec3 curl = forceCurlNoise3D(forcePos * FORCE_FREQUENCY(pc.forceBufIdx, fbase) + FORCE_SEED(pc.forceBufIdx, fbase), FORCE_OCTAVES(pc.forceBufIdx, fbase));
            v.xyz += curl * FORCE_STRENGTH(pc.forceBufIdx, fbase) * pc.dt;
        })";
  case ForceType::NOISE:
    return R"(        if (ftype == FORCE_TYPE_NOISE) {
            float n = forceFbmNoise3D(forcePos * FORCE_FREQUENCY(pc.forceBufIdx, fbase) + FORCE_SEED(pc.forceBufIdx, fbase), FORCE_OCTAVES(pc.forceBufIdx, fbase));
            v.xyz += FORCE_DIR(pc.forceBufIdx, fbase) * (n * FORCE_STRENGTH(pc.forceBufIdx, fbase) * pc.dt);
        })";
  }
  return "";
}

std::string ForceShaderCompiler::buildForceBlock(const std::vector<std::shared_ptr<Force>>& forces) {
  // 実際に登録されている型のみ分岐を生成する (未使用の型はソーステキストに現れない)
  std::set<ForceType> types;
  for(const auto& f : forces) types.insert(f->type());

  std::ostringstream ss;
  ss << "    for (uint fi = 0u; fi < pc.forceCount; fi++) {\n";
  ss << "        uint fbase = fi * 16u;\n";
  ss << "        uint ftype = FORCE_TYPE_AT(pc.forceBufIdx, fbase);\n";
  ss << "        uint fmask = FORCE_AFFECT_MASK(pc.forceBufIdx, fbase);\n";
  ss << "        if (fmask != 0u && ((fmask >> forceTypeFlag) & 1u) == 0u) continue;\n";
  for(ForceType t : types) {
    std::string snippet = forceSnippet(t);
    if(!snippet.empty()) ss << snippet << "\n";
  }
  ss << "    }\n";
  return ss.str();
}

std::vector<uint32_t> ForceShaderCompiler::compile(const std::vector<std::shared_ptr<Force>>& forces, const std::string& shaderSourceName) {
  std::string path = std::string(SHADER_SRC_DIR) + "/" + shaderSourceName;
  std::ifstream file(path);
  if(!file.is_open()) throw std::runtime_error("ForceShaderCompiler: cannot open " + path);
  std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  size_t beginPos = source.find(kBeginMarker);
  size_t endPos    = source.find(kEndMarker);
  if(beginPos == std::string::npos || endPos == std::string::npos || endPos < beginPos)
    throw std::runtime_error("ForceShaderCompiler: FORCE_AUTOGEN markers not found in " + shaderSourceName);

  // マーカーのコメント行全体 (プレースホルダ含む) を生成コードで置換する
  size_t lineStart = source.rfind('\n', beginPos) + 1; // rfind失敗時 npos+1==0 でファイル先頭扱い
  size_t lineEnd    = source.find('\n', endPos);
  if(lineEnd == std::string::npos) lineEnd = source.size();

  std::string generated = source.substr(0, lineStart) + buildForceBlock(forces) + source.substr(lineEnd + 1);

  shaderc::Compiler compiler;
  shaderc::CompileOptions options;
  options.SetIncluder(std::make_unique<ForceGlslIncluder>());
  options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
  options.SetOptimizationLevel(shaderc_optimization_level_performance);

  shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(generated, shaderc_compute_shader, shaderSourceName.c_str(), options);
  if(result.GetCompilationStatus() != shaderc_compilation_status_success)
    throw std::runtime_error("ForceShaderCompiler: compile failed (" + shaderSourceName + "): " + result.GetErrorMessage());

  return {result.cbegin(), result.cend()};
}
