#pragma once

#include "MPMSimPC.h"
#include "PyroSimPC.h"
#include "SimPC.h"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>

// buildPipeline() は全パイプライン共通で push constant range を sizeof(SimPC) に固定する
// (下記 initFromSpirv のコメント参照)。MPMSimPC/PyroSimPC がこの範囲に収まることを保証する。
static_assert(sizeof(MPMSimPC) <= sizeof(SimPC), "MPMSimPC must fit within the shared push constant range (sizeof(SimPC))");
static_assert(sizeof(PyroSimPC) <= sizeof(SimPC), "PyroSimPC must fit within the shared push constant range (sizeof(SimPC))");

using ShaderDefines = std::vector<std::pair<std::string, std::string>>;

class ComputePipeline {
public:
  // defines が空なら shaderPath (.spv) をそのまま読む。非空なら shaderPath のファイル名から
  // ".spv" を除いたソース名 (例: "hash_count.comp") を SHADER_SRC_DIR から読み、
  // defines を #define 注入して実行時コンパイルする (ドメイン形状依存のMorton定数等の焼き込み用)。
  void init(VkDevice device, VkDescriptorSetLayout bindlessLayout, const std::string& shaderPath, const ShaderDefines& defines = {});
  // SHADER_SRC_DIR からの相対ソース名を defines 付きで SPIR-V にコンパイルする。
  // 失敗時 (ファイルが開けない・コンパイルエラー) は std::runtime_error。
  // isVertexShader: true なら .vert、false なら .comp としてコンパイルする (頂点シェーダーは GraphicsPipeline::initVertFromSpirv 用)。
  static std::vector<uint32_t> compileShader(const std::string& shaderSourceName, const ShaderDefines& defines, bool isVertexShader = false);
  // SPIR-V バイト列(shaderc等でランタイム生成したもの)から直接パイプラインを構築する。
  // issue #30: Force動的シェーダー生成が rebuild のたびに呼ぶ想定。init() と同じレイアウト
  // (push constant size = sizeof(SimPC)) で構築する。
  void initFromSpirv(VkDevice device, VkDescriptorSetLayout bindlessLayout, const std::vector<uint32_t>& spirv);
  void cleanup();

  // フレームごとに dispatch (barrier は呼び出し側で挿入)
  void dispatch(VkCommandBuffer cmd, VkDescriptorSet bindlessSet, const SimPC& pc, uint32_t particleCount);
  // SimPC以外のpush constant構造体(MPMSimPC/PyroSimPC等)を使う場合の汎用dispatch
  void dispatchRaw(VkCommandBuffer cmd, VkDescriptorSet bindlessSet, const void* pcData, uint32_t pcSize, uint32_t groupCountX);

  VkPipeline pipeline             = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

private:
  VkDevice device_                          = VK_NULL_HANDLE;
  VkDescriptorSetLayout bindlessSetLayout_   = VK_NULL_HANDLE;
  VkShaderModule loadShader(const std::string& path, const ShaderDefines& defines);
  VkShaderModule createShaderModuleFromSpirv(const uint32_t* code, size_t wordCount);
  void buildPipeline(VkShaderModule shaderModule);
};
