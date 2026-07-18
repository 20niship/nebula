#pragma once

#include "Force.h"
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

// issue #30: Engine::addForce()/removeForce()/setForces() が呼ばれるたびに、
// 登録済み Force 一覧の variables()/glslApplyVelocity()/glslApplyConstraint()
// から GLSL ソースを動的に組み立て、FORCE_AUTOGEN(_POST) マーカー間の置換と
// 関数定義の挿入を行い、shaderc で SPIR-V へ実行時コンパイルする。キャッシュは
// 行わず、呼び出しのたびに無条件で再生成・再コンパイルする。
//
// マーカーを含むホストシェーダー (predict.comp 等) は、マーカーより前に
// 以下をスコープ内に用意しておくこと:
//   vec4/vec3 v         - 力を適用する速度 (v.xyz がその場で書き換えられる)
//   vec3 forcePos       - 空間ノイズ計算用のワールド座標
//   uint forceTypeFlag  - affectMask によるフィルタ対象の typeFlag
// FORCE_AUTOGEN_POST_BEGIN/END (任意、predict.comp/predict_sdf.compのみ) は
// 以下を追加でスコープ内に用意しておくこと:
//   vec4 predP          - 位置制約で書き換え可能な次フレーム予測位置
//   vec4 p              - 現在位置 (読み取り専用)
class ForceShaderCompiler {
public:
  // shaderSourceName: SHADER_SRC_DIR からの相対ファイル名 (例: "predict.comp")。
  // 失敗時 (BEGIN/ENDマーカー未検出・コンパイルエラー) は std::runtime_error を投げる。
  // POST_BEGIN/POST_ENDマーカーは任意 (見つからなければ何もしない)。
  static std::vector<uint32_t> compile(const std::vector<std::shared_ptr<Force>>& forces, const std::string& shaderSourceName);

private:
  // type() ごとに代表インスタンスを1つ選ぶ (先勝ち; 同一type()は同一GLSL文字列という前提)
  static std::map<ForceType, std::shared_ptr<Force>> uniqueByType(const std::vector<std::shared_ptr<Force>>& forces);
  // 各代表インスタンスの glslApplyVelocity()/glslApplyConstraint() (空でなければ) を
  // 連結し、void main() の直前 (グローバルスコープ) に挿入する関数定義群を生成する。
  static std::string buildForceFunctions(const std::vector<std::shared_ptr<Force>>& forces);
  // FORCE_AUTOGEN_BEGIN/END を置換する速度適用ループ。
  static std::string buildVelocityCallBlock(const std::vector<std::shared_ptr<Force>>& forces);
  // FORCE_AUTOGEN_POST_BEGIN/END を置換する位置制約ループ。
  // glslApplyConstraint() を持つForceが1つもなければ空文字列 (何もしない) を返す。
  static std::string buildConstraintCallBlock(const std::vector<std::shared_ptr<Force>>& forces);
  // マーカー行全体を置換する。required=falseでマーカー未検出時はsourceをそのまま返す。
  static std::string replaceMarkerBlock(const std::string& source, const std::string& beginMarker, const std::string& endMarker,
                                         const std::string& replacement, bool required, const std::string& shaderSourceName);
};
