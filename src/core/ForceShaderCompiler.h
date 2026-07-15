#pragma once

#include "Force.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// issue #30: Engine::addForce()/removeForce()/setForces() が呼ばれるたびに、
// 登録済み Force 一覧に応じて GLSL ソースの FORCE_AUTOGEN_BEGIN/END マーカー間を
// 実行時に置換し、shaderc で SPIR-V へコンパイルする。キャッシュは行わず、
// 呼び出しのたびに無条件で再生成・再コンパイルする。
//
// マーカーを含むホストシェーダー (predict.comp 等) は、マーカーより前に
// 以下をスコープ内に用意しておくこと:
//   vec4 v             - 力を適用する速度 (v.xyz がその場で書き換えられる)
//   vec3 forcePos      - 空間ノイズ計算用のワールド座標
//   uint forceTypeFlag - affectMask によるフィルタ対象の typeFlag
class ForceShaderCompiler {
public:
  // shaderSourceName: SHADER_SRC_DIR からの相対ファイル名 (例: "predict.comp")。
  // 失敗時 (マーカー未検出・コンパイルエラー) は std::runtime_error を投げる。
  static std::vector<uint32_t> compile(const std::vector<std::shared_ptr<Force>>& forces, const std::string& shaderSourceName);

private:
  static std::string buildForceBlock(const std::vector<std::shared_ptr<Force>>& forces);
  static std::string forceSnippet(ForceType t);
};
