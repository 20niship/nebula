#include "ForceShaderCompiler.h"

#include <fstream>
#include <shaderc/shaderc.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

constexpr const char* kBeginMarker     = "FORCE_AUTOGEN_BEGIN";
constexpr const char* kEndMarker       = "FORCE_AUTOGEN_END";
constexpr const char* kPostBeginMarker = "FORCE_AUTOGEN_POST_BEGIN";
constexpr const char* kPostEndMarker   = "FORCE_AUTOGEN_POST_END";

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

// ForceVar::kind に応じて FORCE_DATA_*(pc.forceBufIdx, fbase, offset) 呼び出しへ変換する。
std::string argExpr(const ForceVar& var) {
  std::ostringstream ss;
  switch(var.kind) {
  case ForceVar::FLOAT: ss << "FORCE_DATA_FLOAT(pc.forceBufIdx, fbase, " << var.offset << "u)"; break;
  case ForceVar::VEC3: ss << "FORCE_DATA_VEC3(pc.forceBufIdx, fbase, " << var.offset << "u)"; break;
  case ForceVar::UINT: ss << "FORCE_DATA_UINT(pc.forceBufIdx, fbase, " << var.offset << "u)"; break;
  }
  return ss.str();
}

} // namespace

std::map<ForceType, std::shared_ptr<Force>> ForceShaderCompiler::uniqueByType(const std::vector<std::shared_ptr<Force>>& forces) {
  std::map<ForceType, std::shared_ptr<Force>> byType;
  for(const auto& f : forces) byType.try_emplace(f->type(), f); // 先勝ち: 同一type()は同一GLSL文字列という前提
  return byType;
}

std::string ForceShaderCompiler::buildForceFunctions(const std::vector<std::shared_ptr<Force>>& forces) {
  std::ostringstream ss;
  for(const auto& [type, f] : uniqueByType(forces)) {
    std::string vel  = f->glslApplyVelocity();
    std::string cons = f->glslApplyConstraint();
    if(!vel.empty()) ss << vel << "\n";
    if(!cons.empty()) ss << cons << "\n";
  }
  return ss.str();
}

std::string ForceShaderCompiler::buildVelocityCallBlock(const std::vector<std::shared_ptr<Force>>& forces) {
  auto byType = uniqueByType(forces);
  std::ostringstream ss;
  ss << "    {\n";
  ss << "        vec3 fVel = v.xyz;\n";
  ss << "        for (uint fi = 0u; fi < pc.forceCount; fi++) {\n";
  ss << "            uint fbase = fi * 16u;\n";
  ss << "            uint ftype = FORCE_TYPE_AT(pc.forceBufIdx, fbase);\n";
  ss << "            uint fmask = FORCE_AFFECT_MASK(pc.forceBufIdx, fbase);\n";
  ss << "            if (fmask != 0u && ((fmask >> forceTypeFlag) & 1u) == 0u) continue;\n";
  for(const auto& [type, f] : byType) {
    if(f->glslApplyVelocity().empty()) continue;
    ss << "            if (ftype == " << uint32_t(type) << "u) {\n";
    ss << "                forceApplyVelocity_" << f->glslFunctionSuffix() << "(fVel, forcePos, forceTypeFlag, pc.dt";
    for(const auto& var : f->variables()) ss << ", " << argExpr(var);
    ss << ");\n            }\n";
  }
  ss << "        }\n";
  ss << "        v.xyz = fVel;\n";
  ss << "    }\n";
  return ss.str();
}

std::string ForceShaderCompiler::buildConstraintCallBlock(const std::vector<std::shared_ptr<Force>>& forces) {
  auto byType = uniqueByType(forces);
  bool any    = false;
  for(const auto& [type, f] : byType)
    if(!f->glslApplyConstraint().empty()) {
      any = true;
      break;
    }
  if(!any) return ""; // 既定: 空 (何もしない)

  std::ostringstream ss;
  ss << "    for (uint fi = 0u; fi < pc.forceCount; fi++) {\n";
  ss << "        uint fbase = fi * 16u;\n";
  ss << "        uint ftype = FORCE_TYPE_AT(pc.forceBufIdx, fbase);\n";
  ss << "        uint fmask = FORCE_AFFECT_MASK(pc.forceBufIdx, fbase);\n";
  ss << "        if (fmask != 0u && ((fmask >> forceTypeFlag) & 1u) == 0u) continue;\n";
  for(const auto& [type, f] : byType) {
    if(f->glslApplyConstraint().empty()) continue;
    ss << "        if (ftype == " << uint32_t(type) << "u) {\n";
    ss << "            forceApplyConstraint_" << f->glslFunctionSuffix() << "(predP, p, v.xyz, forcePos, forceTypeFlag, pc.dt";
    for(const auto& var : f->variables()) ss << ", " << argExpr(var);
    ss << ");\n        }\n";
  }
  ss << "    }\n";
  return ss.str();
}

std::string ForceShaderCompiler::replaceMarkerBlock(const std::string& source, const std::string& beginMarker, const std::string& endMarker,
                                                      const std::string& replacement, bool required, const std::string& shaderSourceName) {
  size_t beginPos = source.find(beginMarker);
  size_t endPos    = source.find(endMarker);
  if(beginPos == std::string::npos || endPos == std::string::npos || endPos < beginPos) {
    if(required) throw std::runtime_error("ForceShaderCompiler: " + beginMarker + "/" + endMarker + " markers not found in " + shaderSourceName);
    return source; // 任意マーカーが無いシェーダーはそのまま (何もしない)
  }

  // マーカーのコメント行全体 (プレースホルダ含む) を生成コードで置換する
  size_t lineStart = source.rfind('\n', beginPos) + 1; // rfind失敗時 npos+1==0 でファイル先頭扱い
  size_t lineEnd    = source.find('\n', endPos);
  if(lineEnd == std::string::npos) lineEnd = source.size();

  return source.substr(0, lineStart) + replacement + source.substr(lineEnd + 1);
}

std::vector<uint32_t> ForceShaderCompiler::compile(const std::vector<std::shared_ptr<Force>>& forces, const std::string& shaderSourceName) {
  std::string path = std::string(SHADER_SRC_DIR) + "/" + shaderSourceName;
  std::ifstream file(path);
  if(!file.is_open()) throw std::runtime_error("ForceShaderCompiler: cannot open " + path);
  std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  // ① 速度への力 (必須マーカー)
  source = replaceMarkerBlock(source, kBeginMarker, kEndMarker, buildVelocityCallBlock(forces), /*required=*/true, shaderSourceName);
  // ② 位置制約 (任意マーカー; predict.comp/predict_sdf.compのみ存在する)
  source = replaceMarkerBlock(source, kPostBeginMarker, kPostEndMarker, buildConstraintCallBlock(forces), /*required=*/false, shaderSourceName);

  // ③ 各Force型のGLSL関数定義を void main() の直前 (グローバルスコープ) に挿入する
  size_t mainPos = source.find("void main(");
  if(mainPos == std::string::npos) throw std::runtime_error("ForceShaderCompiler: void main() not found in " + shaderSourceName);
  source = source.substr(0, mainPos) + buildForceFunctions(forces) + "\n" + source.substr(mainPos);

  shaderc::Compiler compiler;
  shaderc::CompileOptions options;
  options.SetIncluder(std::make_unique<ForceGlslIncluder>());
  options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
  options.SetOptimizationLevel(shaderc_optimization_level_performance);

  shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(source, shaderc_compute_shader, shaderSourceName.c_str(), options);
  if(result.GetCompilationStatus() != shaderc_compilation_status_success)
    throw std::runtime_error("ForceShaderCompiler: compile failed (" + shaderSourceName + "): " + result.GetErrorMessage());

  return {result.cbegin(), result.cend()};
}
