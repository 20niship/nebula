#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// #define 定数を注入してシェーダーを実行時コンパイルする汎用ヘルパー。
// ForceShaderCompiler (issue #30) と同じ shaderc セットアップ (Includer/
// TargetEnvironment/OptimizationLevel) を使うが、FORCE_AUTOGENマーカーの
// 文字列置換は行わない。ドメイン形状(gridRes)に応じたアダプティブMorton定数
// (ADAPTIVE_MASK 等、src/core/Domain.h の AdaptiveMortonParams 参照) のように
// エンジン初期化時に一度だけ決まる値をシェーダーへ焼き込む用途。
class DefineShaderCompiler {
public:
  // shaderSourceName: SHADER_SRC_DIR からの相対ファイル名 (例: "hash_count.comp")。
  // defines: マクロ名→値 (例: {"ADAPTIVE_MASK", "15u"})。
  // 失敗時 (ファイルが開けない・コンパイルエラー) は std::runtime_error を投げる。
  static std::vector<uint32_t> compile(const std::string& shaderSourceName,
                                        const std::vector<std::pair<std::string, std::string>>& defines);
};
