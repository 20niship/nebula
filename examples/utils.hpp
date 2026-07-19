#pragma once
// examples 共通 ImGui UI ヘルパー。
// FluidEngine / SimulationEngine / MultiPhysicsEngine のパラメータを
// 日本語 tooltip 付きで描画する inline 関数群。
//
// 使い方: 各 example .cpp で #include "utils.hpp" を追加し、
//         drawFrame() 内の ImGui::Begin()〜End() の中で呼ぶ。

#include "core/Force.h"
#include "engine/FluidEngine.h"
#include "engine/MultiPhysicsEngine.h"
#include "engine/SimulationEngine.h"
#include <imgui.h>

namespace sim_ui {

// 直前の ImGui ウィジェットにホバー時 tooltip を設定する短縮マクロ
#define SIM_TIP(text)                                                                                                                                                                                                                                                                                      \
  do {                                                                                                                                                                                                                                                                                                     \
    if(ImGui::IsItemHovered()) ImGui::SetTooltip(text);                                                                                                                                                                                                                                                    \
  } while(0)

// ── リセット ─────────────────────────────────────────────────────────────────

// 赤いリセットボタン。押下時に流体粒子を初期位置へ戻し simTime を 0 にリセット。
inline void fluid_reset_button(FluidEngine& e, float& simTime) {
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.70f, 0.20f, 0.20f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.90f, 0.30f, 0.30f, 1.0f));
  if(ImGui::Button("  リセット  ")) {
    e.resetParticles();
    simTime = 0.0f;
  }
  ImGui::PopStyleColor(2);
  SIM_TIP("粒子を初期位置・速度にリセットします。\n境界メッシュは維持されます。");
}

// ── 流体パラメータ (FluidEngine) ──────────────────────────────────────────────

// PBF 流体の全パラメータを tooltip 付きで描画する。
// issue #30 レビュー対応: gravity は FluidEngine の public メンバとしては廃止された
// ため、呼び出し側が addForce() で登録した GravityForce への参照を直接受け取る。
inline void fluid_params(FluidEngine& e, GravityForce& gravity) {
  ImGui::Text("基本");
  ImGui::SliderFloat("重力", &gravity.strength, 0.0f, 20.0f);
  SIM_TIP("重力加速度 [m/s²]。大きくすると液体が速く落下する。");
  ImGui::SliderFloat("静止密度 rho0", &e.rho0, 100.0f, 5000.0f);
  SIM_TIP("流体の目標密度 [kg/m³]。\n初期化時に格子配置から自動計算した値が推奨。\n高すぎると過圧縮、低すぎると膨張が起きる。");
  ImGui::SliderFloat("粘性係数", &e.viscosityC, 0.0f, 0.1f, "%.4f");
  SIM_TIP("XSPH粘性係数。粒子間の速度を平均化し滑らかな流れを作る。\n大きいほど蜂蜜のような粘い流体になる。0で無粘性（水に近い）。");
  ImGui::SliderInt("PBFイテレーション", &e.pbfIterations, 1, 10);
  SIM_TIP("Position-Based Fluids の密度拘束の反復回数。\n多いほど不圧縮性が正確になるが計算コストが増える。推奨: 2〜4。");
  ImGui::SliderInt("サブステップ", &e.numSubsteps, 1, 10);
  SIM_TIP("1フレームを分割するサブステップ数。\n多いほど安定するが計算コストが増える。推奨: 1〜4。");

  ImGui::Separator();
  ImGui::Text("PBF 詳細");
  ImGui::SliderFloat("CFM epsilon", &e.cfmEpsilon, 0.1f, 3000.0f, "%.2f");
  SIM_TIP("制約力混合（CFM）の緩和パラメータ (Macklin & Müller 式11)。\n大きいほど密度拘束が「柔らかく」なり発散を防ぐ。\n小さすぎると爆発的に発散する。推奨: 100〜3000。");
  ImGui::SliderFloat("人工圧力 k", &e.scorrK, 0.0f, 1.0f, "%.5f");
  SIM_TIP("人工圧力の強さ (Macklin & Müller 式13)。\n負圧による粒子クラスタリング（Tensile Instability）を補正する。\n0で無効。推奨: 0（通常）〜 0.001（引力が強い場合）。");
  ImGui::SliderFloat("線形ダンピング", &e.linearDamping, 0.0f, 2.0f, "%.3f");
  SIM_TIP("速度の減衰率 [1/s]。v *= exp(-d*dt) で計算。\n大きいほど粒子が素早く減速・停止する。\n0で減衰なし、0.6が標準値。");
  ImGui::Checkbox("渦度閉じ込め", &e.vorticityEnabled);
  SIM_TIP("渦度閉じ込め法 (Macklin & Müller 式15-16) を有効化。\n回転する流れ（渦）を数値散逸から保護し、リアルな渦巻き挙動を強化する。\n計算コストがやや増える。");
  if(e.vorticityEnabled) {
    ImGui::SliderFloat("渦度 epsilon", &e.vorticityEpsilon, 0.0f, 5.0f, "%.4f");
    SIM_TIP("渦度閉じ込めの強さ (式16の ε)。\n大きいほど渦が強調されてダイナミックな流れになる。\n大きすぎると不安定になる場合がある。推奨: 0.1〜1.0。");
  }
}

// ── 泡 (spray/foam/bubble) パラメータ (issue #47) ─────────────────────────────
// setFoamParams() は GPU 同期を伴う重い呼び出しのため毎フレームは呼ばない。
// 戻り値 true のときだけ呼び出し側で e.setFoamParams(p) を呼ぶこと。
inline bool foam_params(FluidEngine& e, FluidEngine::FoamParams& p) {
  ImGui::Checkbox("泡を有効化 (spray/foam/bubble)", &e.foamEnabled);
  SIM_TIP("水しぶき(spray)/泡(foam)/気泡(bubble)の二次パーティクル生成を有効化。\n--max-diffuse=0 で起動した場合は容量が無くバッファが確保されないため無効のまま。");
  if(!e.foamEnabled) return false;

  bool changed = false;
  changed |= ImGui::SliderFloat("生成係数 (trapped-air)", &p.kTa, 0.0f, 20000.0f, "%.0f");
  SIM_TIP("空気を巻き込む勢いに応じた泡の生成量係数 (Ihmsen et al. 2012 式2)。大きいほど激しい飛沫が発生する。");
  changed |= ImGui::SliderFloat("生成係数 (wave-crest)", &p.kWc, 0.0f, 20000.0f, "%.0f");
  SIM_TIP("波頭・表面から飛び出す勢いに応じた泡の生成量係数。");
  changed |= ImGui::SliderFloat("寿命 最小 [s]", &p.lifetimeMin, 0.1f, 10.0f);
  changed |= ImGui::SliderFloat("寿命 最大 [s]", &p.lifetimeMax, 0.1f, 10.0f);
  SIM_TIP("泡パーティクルの生存時間の範囲 [s]。切れると消滅する。");
  changed |= ImGui::SliderFloat("気泡浮力", &p.bubbleBuoyancy, 0.0f, 20.0f);
  SIM_TIP("水中に沈んだ気泡(bubble)が浮上する加速度 [m/s²]。");
  changed |= ImGui::SliderFloat("流体追従係数", &p.dragCoeff, 0.0f, 1.0f);
  SIM_TIP("foam/bubble が周囲流体の速度へ追従する割合。1で完全追従、0で追従しない。");
  changed |= ImGui::SliderFloat("表面密度比しきい値", &p.surfaceDensityRatio, 0.5f, 1.0f);
  SIM_TIP("この比率(rho_i/rho0)未満の低密度(表面)粒子のみ泡生成の対象にする。");
  return changed;
}

// ── 布パラメータ (SimulationEngine) ──────────────────────────────────────────
// issue #30 レビュー対応: gravity/windX/windZ は SimulationEngine の public
// メンバとしては廃止されたため、呼び出し側が addForce() で登録した
// GravityForce/ConstantWindForce への参照を直接受け取り、そのフィールドを
// スライダーで書き換える (Engine への追加同期呼び出しは不要。EngineBase::
// uploadForces() が毎フレーム forces_ の現在値を再アップロードする)。

inline void cloth_params(SimulationEngine& sim, GravityForce& gravity, ConstantWindForce& wind) {
  ImGui::SliderFloat("重力", &gravity.strength, 0.0f, 20.0f);
  SIM_TIP("重力加速度 [m/s²]。");
  ImGui::SliderFloat("反発係数", &sim.restitution, 0.0f, 1.0f);
  SIM_TIP("壁・床との衝突時の反発係数。0で完全吸収（跳ねない）、1で完全弾性。");
  ImGui::SliderFloat("摩擦係数", &sim.friction, 0.0f, 1.0f);
  SIM_TIP("壁・床との衝突時の摩擦係数。0で摩擦なし、1で最大摩擦。");
  ImGui::Separator();
  ImGui::SliderFloat("風 X", &wind.direction.x, -10.0f, 10.0f);
  SIM_TIP("X軸方向の風力 [N/m²]。布を横方向に押す。");
  ImGui::SliderFloat("風 Z", &wind.direction.y, -10.0f, 10.0f);
  SIM_TIP("Z軸方向の風力 [N/m²]。");
  ImGui::Separator();
  ImGui::SliderFloat("伸び剛性", &sim.stretchCompliance, 0.0f, 1e-2f, "%.6f");
  SIM_TIP("布の伸び拘束のコンプライアンス（柔らかさ）。\n0で完全剛体、大きいほど伸びやすくなる。");
  ImGui::SliderFloat("曲げ剛性", &sim.bendCompliance, 0.0f, 1e-1f, "%.6f");
  SIM_TIP("布の曲げ拘束のコンプライアンス。\n大きいほど曲がりやすく（柔らかく）なる。");
  ImGui::SliderInt("ソルバー反復回数", &sim.solverIterations, 1, 10);
  SIM_TIP("拘束ソルバーの反復回数。\n多いほど拘束が正確になるが計算コストが増える。");
  ImGui::SliderInt("サブステップ", &sim.numSubsteps, 1, 20);
  SIM_TIP("1フレームを分割するサブステップ数。\n多いほど安定するが計算コストが増える。");
  ImGui::Checkbox("自己衝突", &sim.enableSelfCollision);
  SIM_TIP("布の自己衝突を有効化。有効にすると布が自分自身を貫通しなくなる。\n計算コストが増える。");
}

// ── Multi-Physics 用（布・流体サブセット + 連成） ─────────────────────────────

// issue #30 レビュー対応: gravity/windX/windZ は MultiPhysicsEngine の public
// メンバとしては廃止されたため、呼び出し側が addForce() で登録した
// GravityForce/ConstantWindForce への参照を直接受け取る。
inline void multi_physics_params(MultiPhysicsEngine& e, GravityForce& gravity, ConstantWindForce& wind) {
  ImGui::Text("シミュレーション共通");
  ImGui::SliderFloat("重力", &gravity.strength, 0.0f, 20.0f);
  SIM_TIP("重力加速度 [m/s²]。");
  ImGui::SliderFloat("反発係数", &e.restitution, 0.0f, 1.0f);
  SIM_TIP("壁・床との衝突時の反発係数。0で完全吸収、1で完全弾性。");
  ImGui::SliderInt("サブステップ", &e.numSubsteps, 1, 20);
  SIM_TIP("1フレームを分割するサブステップ数。多いほど安定するが計算コストが増える。");
  ImGui::SliderInt("PBFイテレーション", &e.pbfIterations, 1, 10);
  SIM_TIP("PBF 流体の密度拘束の反復回数。多いほど不圧縮性が正確になる。推奨: 2〜4。");

  ImGui::Separator();
  ImGui::Text("流体");
  ImGui::SliderFloat("静止密度 rho0", &e.rho0, 100.0f, 5000.0f);
  SIM_TIP("流体の目標密度 [kg/m³]。高すぎると過圧縮、低すぎると膨張が起きる。");
  ImGui::SliderFloat("粘性係数", &e.viscosityC, 0.0f, 0.1f, "%.4f");
  SIM_TIP("XSPH粘性係数。大きいほど粘い流体になる。0で無粘性（水に近い）。");

  ImGui::Separator();
  ImGui::Text("布");
  ImGui::SliderFloat("伸び剛性", &e.stretchCompliance, 0.0f, 1e-2f, "%.6f");
  SIM_TIP("布の伸び拘束のコンプライアンス（柔らかさ）。0で完全剛体、大きいほど伸びやすい。");
  ImGui::SliderFloat("曲げ剛性", &e.bendCompliance, 0.0f, 1e-1f, "%.6f");
  SIM_TIP("布の曲げ拘束のコンプライアンス。大きいほど曲がりやすくなる。");
  ImGui::SliderFloat("風 X", &wind.direction.x, -10.0f, 10.0f);
  SIM_TIP("X軸方向の風力 [N/m²]。");
  ImGui::SliderFloat("風 Z", &wind.direction.y, -10.0f, 10.0f);
  SIM_TIP("Z軸方向の風力 [N/m²]。");

  ImGui::Separator();
  ImGui::Text("連成");
  ImGui::Checkbox("流体-布 連成を有効化", &e.enableCoupling);
  SIM_TIP("流体と布の双方向連成を有効化。\n有効にすると流体が布を押す/布が流体を押す相互作用が生じる。\n計算コストが増える。");
}

#undef SIM_TIP

} // namespace sim_ui
