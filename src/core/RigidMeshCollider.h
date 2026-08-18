#pragma once
#include "MeshSDF.h" // MeshTriangle/closestPtOnTriangleを共有(ワールド空間ベイクのPyro用と同じ幾何プリミティブ)
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

// MeshSDF.h(ワールド空間・毎回全ドメイン再ベイク、Pyro障害物用)と異なり、こちらはメッシュローカル空間の小さいグリッドを一度だけ焼き、移動/回転はシェーダー側で毎フレームクォータニオン変換して参照する(MPMのColliderPrimitive MESH_SDF用)。
struct LocalMeshSDF {
  std::vector<float> data; // res*res*res, ix + iy*res + iz*res*res (ローカル空間, dense)
  glm::vec3 localMin{0.0f};
  float cellSize = 0.0f;
  uint32_t res   = 0;
};

LocalMeshSDF bakeLocalMeshSDF(const std::vector<MeshTriangle>& tris, uint32_t res = 48, float marginCells = 2.0f);
