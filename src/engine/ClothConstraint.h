#pragma once
#include <cstdint>
#include <glm/glm.hpp>

// Vellum スタイルの制約定義。ClothSceneEngine::addConstraint() で登録する。
// 距離拘束 (Stretch/Bend) は ClothMesh が自動生成するためここでは定義しない。
struct ClothConstraint {
  enum class Type {
    Pin,         // 静的固定 (invMass=0, 初期位置に固定)
    PinAnimated, // アニメーション固定 (invMass=0, 毎フレーム updateConstraint() で更新)
  };

  Type type;
  uint32_t particleIdx; // グローバル粒子インデックス (addCloth() 戻り値 + ローカル idx)
  glm::vec3 targetPos;  // Pin: 固定位置 / PinAnimated: 現在の目標位置
};
