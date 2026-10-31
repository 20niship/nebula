#pragma once
#include "MeshSDF.h" // MeshTriangle/closestPtOnTriangleを共有(ワールド空間ベイクのPyro用と同じ幾何プリミティブ)
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

// MeshSDF.h(ワールド空間・毎回全ドメイン再ベイク、Pyro障害物用)と異なり、こちらはメッシュローカル空間の小さいグリッドを一度だけ焼き、移動/回転はシェーダー側で毎フレームクォータニオン変換して参照する(MPMのColliderPrimitive MESH_SDF用)。
struct LocalMeshSDF {
  std::vector<float> data; // res*res*res, ix + iy*res + iz*res*res (ローカル空間, dense)
  glm::vec3 localMin{0.0f};
  float cellSize = 0.0f;
  uint32_t res   = 0;
};

LocalMeshSDF bakeLocalMeshSDF(const std::vector<MeshTriangle>& tris, uint32_t res = 48, float marginCells = 2.0f);

// コライダープリミティブタイプ
enum class ColliderType : uint32_t {
  PLANE    = 0u,
  SPHERE   = 1u,
  BOX      = 2u,
  CAPSULE  = 3u,
  MESH_SDF = 4u,
  CYLINDER = 5u, // 平底円柱、ローカルY軸=軸方向
};

// GPU互換コライダープリミティブ(96 bytes = 24 uint32_t、buffers[colliderIdx].data[i*24 + field]): [0]type [1-3]pos [4-6]nrm/halfExt/軸/半高さ/localMin [7]radius/cellSize [8-10]線速度 [11]restitution [12]friction [13-15]pad(MESH_SDF: bufIdx/res/予約) [16-19]姿勢クォータニオン(BOX/CYLINDER/MESH_SDFが使用、他は(0,0,0,1)) [20-22]角速度(同上、他は0) [23]pad
struct alignas(16) ColliderPrimitive {
  uint32_t type;           // [0]  ColliderType
  float px, py, pz;        // [1-3] 位置 / 平面上の点 / カプセル始点 / メッシュ原点(世界座標)
  float nx, ny, nz;        // [4-6] 法線(平面) / カプセル軸ベクトル / 半辺長(ボックス) / 半高さ(円柱、nyのみ使用) / ローカルAABB最小(MESH_SDF)
  float radius;            // [7]  球/カプセル/円柱半径 / ローカルセルサイズ(MESH_SDF)
  float vx, vy, vz;        // [8-10] コライダー線速度 (移動境界)
  float restitution;       // [11]
  float friction;          // [12]
  float pad0, pad1, pad2;  // [13-15] MESH_SDF: bindlessバッファindex(pad0)/グリッド解像度(pad1)/予約(pad2)。他typeは未使用
  float qx, qy, qz, qw;    // [16-19] 姿勢クォータニオン(BOX/CYLINDER/MESH_SDF)。他typeは(0,0,0,1)=無回転
  float wx, wy, wz;        // [20-22] 角速度[rad/s](BOX/CYLINDER/MESH_SDF)。他typeは(0,0,0)
  float pad3;              // [23] 予約(16byteアライメント維持)
};
static_assert(sizeof(ColliderPrimitive) == 96, "ColliderPrimitive must be 96 bytes");

// CPU 側コライダーセット管理
class ColliderSet {
public:
  // 平面: point = 平面上の点, normal = 外向き法線
  void addPlane(glm::vec3 point, glm::vec3 normal, float restitution = 0.3f, float friction = 0.0f, glm::vec3 vel = {0, 0, 0});
  // 球
  void addSphere(glm::vec3 center, float radius, float restitution = 0.3f, float friction = 0.0f, glm::vec3 vel = {0, 0, 0});
  // ボックス (center + half-extents)。rot/angVelは末尾に追加(既存呼び出しは非破壊)、回転は解析幾何のままでSDFグリッドは経由しない
  void addBox(glm::vec3 center, glm::vec3 halfExt, float restitution = 0.3f, float friction = 0.0f, glm::vec3 vel = {0, 0, 0},
               glm::quat rot = glm::quat(1, 0, 0, 0), glm::vec3 angVel = {0, 0, 0});
  // 平底円柱 (center, ローカルY軸=軸方向)。rot/angVelで回転(解析幾何のまま)
  void addCylinder(glm::vec3 center, float radius, float halfHeight, float restitution = 0.3f, float friction = 0.0f, glm::vec3 vel = {0, 0, 0},
                     glm::quat rot = glm::quat(1, 0, 0, 0), glm::vec3 angVel = {0, 0, 0});
  // カプセル: start = 始点, axisVec = 軸方向×長さ, radius = 半径
  void addCapsule(glm::vec3 start, glm::vec3 axisVec, float radius, float restitution = 0.3f, float friction = 0.0f, glm::vec3 vel = {0, 0, 0});
  // 任意メッシュ(bakeMeshSDFで焼いたローカルSDF)。回転なし利用は rot=glm::quat(1,0,0,0), angVel={0,0,0}。
  void addMeshSDF(uint32_t sdfBufIdx, const LocalMeshSDF& grid, glm::vec3 worldPos, glm::quat rot,
                   glm::vec3 linVel = {0, 0, 0}, glm::vec3 angVel = {0, 0, 0}, float restitution = 0.3f, float friction = 0.0f);

  void clear() { primitives_.clear(); }
  uint32_t count() const { return uint32_t(primitives_.size()); }
  bool empty() const { return primitives_.empty(); }
  const std::vector<ColliderPrimitive>& data() const { return primitives_; }

private:
  std::vector<ColliderPrimitive> primitives_;
};

// ── インライン実装 ─────────────────────────────────────────────────────────────

inline void ColliderSet::addPlane(glm::vec3 point, glm::vec3 normal, float restitution, float friction, glm::vec3 vel) {
  ColliderPrimitive p{};
  p.type        = uint32_t(ColliderType::PLANE);
  p.px          = point.x;
  p.py          = point.y;
  p.pz          = point.z;
  glm::vec3 n   = glm::normalize(normal);
  p.nx          = n.x;
  p.ny          = n.y;
  p.nz          = n.z;
  p.vx          = vel.x;
  p.vy          = vel.y;
  p.vz          = vel.z;
  p.restitution = restitution;
  p.friction    = friction;
  primitives_.push_back(p);
}

inline void ColliderSet::addSphere(glm::vec3 center, float radius, float restitution, float friction, glm::vec3 vel) {
  ColliderPrimitive p{};
  p.type        = uint32_t(ColliderType::SPHERE);
  p.px          = center.x;
  p.py          = center.y;
  p.pz          = center.z;
  p.radius      = radius;
  p.vx          = vel.x;
  p.vy          = vel.y;
  p.vz          = vel.z;
  p.restitution = restitution;
  p.friction    = friction;
  primitives_.push_back(p);
}

inline void ColliderSet::addBox(glm::vec3 center, glm::vec3 halfExt, float restitution, float friction, glm::vec3 vel, glm::quat rot, glm::vec3 angVel) {
  ColliderPrimitive p{};
  p.type        = uint32_t(ColliderType::BOX);
  p.px          = center.x;
  p.py          = center.y;
  p.pz          = center.z;
  p.nx          = halfExt.x;
  p.ny          = halfExt.y;
  p.nz          = halfExt.z;
  p.vx          = vel.x;
  p.vy          = vel.y;
  p.vz          = vel.z;
  p.restitution = restitution;
  p.friction    = friction;
  glm::quat nq  = glm::normalize(rot);
  p.qx          = nq.x;
  p.qy          = nq.y;
  p.qz          = nq.z;
  p.qw          = nq.w;
  p.wx          = angVel.x;
  p.wy          = angVel.y;
  p.wz          = angVel.z;
  primitives_.push_back(p);
}

inline void ColliderSet::addCylinder(glm::vec3 center, float radius, float halfHeight, float restitution, float friction, glm::vec3 vel, glm::quat rot, glm::vec3 angVel) {
  ColliderPrimitive p{};
  p.type        = uint32_t(ColliderType::CYLINDER);
  p.px          = center.x;
  p.py          = center.y;
  p.pz          = center.z;
  p.ny          = halfHeight;
  p.radius      = radius;
  p.vx          = vel.x;
  p.vy          = vel.y;
  p.vz          = vel.z;
  p.restitution = restitution;
  p.friction    = friction;
  glm::quat nq  = glm::normalize(rot);
  p.qx          = nq.x;
  p.qy          = nq.y;
  p.qz          = nq.z;
  p.qw          = nq.w;
  p.wx          = angVel.x;
  p.wy          = angVel.y;
  p.wz          = angVel.z;
  primitives_.push_back(p);
}

inline void ColliderSet::addMeshSDF(uint32_t sdfBufIdx, const LocalMeshSDF& grid, glm::vec3 worldPos, glm::quat rot,
                                     glm::vec3 linVel, glm::vec3 angVel, float restitution, float friction) {
  ColliderPrimitive p{};
  p.type        = uint32_t(ColliderType::MESH_SDF);
  p.px          = worldPos.x;
  p.py          = worldPos.y;
  p.pz          = worldPos.z;
  p.nx          = grid.localMin.x;
  p.ny          = grid.localMin.y;
  p.nz          = grid.localMin.z;
  p.radius      = grid.cellSize;
  p.vx          = linVel.x;
  p.vy          = linVel.y;
  p.vz          = linVel.z;
  p.restitution = restitution;
  p.friction    = friction;
  p.pad0        = glm::uintBitsToFloat(sdfBufIdx);
  p.pad1        = glm::uintBitsToFloat(grid.res);
  glm::quat nq  = glm::normalize(rot);
  p.qx          = nq.x;
  p.qy          = nq.y;
  p.qz          = nq.z;
  p.qw          = nq.w;
  p.wx          = angVel.x;
  p.wy          = angVel.y;
  p.wz          = angVel.z;
  primitives_.push_back(p);
}

inline void ColliderSet::addCapsule(glm::vec3 start, glm::vec3 axisVec, float radius, float restitution, float friction, glm::vec3 vel) {
  ColliderPrimitive p{};
  p.type        = uint32_t(ColliderType::CAPSULE);
  p.px          = start.x;
  p.py          = start.y;
  p.pz          = start.z;
  p.nx          = axisVec.x;
  p.ny          = axisVec.y;
  p.nz          = axisVec.z;
  p.radius      = radius;
  p.vx          = vel.x;
  p.vy          = vel.y;
  p.vz          = vel.z;
  p.restitution = restitution;
  p.friction    = friction;
  primitives_.push_back(p);
}
