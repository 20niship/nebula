#pragma once
#include <cstdint>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

// コライダープリミティブタイプ
enum class ColliderType : uint32_t {
    PLANE   = 0u,
    SPHERE  = 1u,
    BOX     = 2u,
    CAPSULE = 3u,
};

// GPU 互換コライダープリミティブ (64 bytes = 16 uint32_t)
// GLSL 側では buffers[colliderIdx].data[i*16 + field] で読む
// [0]=type [1-3]=pos [4-6]=nrm/halfExt/axis [7]=radius
// [8-10]=velocity [11]=restitution [12]=friction [13-15]=pad
struct alignas(16) ColliderPrimitive {
    uint32_t type;             // [0]  ColliderType
    float    px, py, pz;       // [1-3] 位置 / 平面上の点 / カプセル始点
    float    nx, ny, nz;       // [4-6] 法線(平面) / カプセル軸ベクトル / 半辺長(ボックス)
    float    radius;           // [7]  球/カプセル半径
    float    vx, vy, vz;       // [8-10] コライダー速度 (移動境界)
    float    restitution;      // [11]
    float    friction;         // [12]
    float    pad0, pad1, pad2; // [13-15] 予約
};
static_assert(sizeof(ColliderPrimitive) == 64, "ColliderPrimitive must be 64 bytes");

// CPU 側コライダーセット管理
class ColliderSet {
public:
    // 平面: point = 平面上の点, normal = 外向き法線
    void addPlane(glm::vec3 point, glm::vec3 normal,
                  float restitution = 0.3f, float friction = 0.0f,
                  glm::vec3 vel = {0, 0, 0});
    // 球
    void addSphere(glm::vec3 center, float radius,
                   float restitution = 0.3f, float friction = 0.0f,
                   glm::vec3 vel = {0, 0, 0});
    // 軸平行ボックス (center + half-extents)
    void addBox(glm::vec3 center, glm::vec3 halfExt,
                float restitution = 0.3f, float friction = 0.0f,
                glm::vec3 vel = {0, 0, 0});
    // カプセル: start = 始点, axisVec = 軸方向×長さ, radius = 半径
    void addCapsule(glm::vec3 start, glm::vec3 axisVec, float radius,
                    float restitution = 0.3f, float friction = 0.0f,
                    glm::vec3 vel = {0, 0, 0});

    void clear()               { primitives_.clear(); }
    uint32_t count()     const { return uint32_t(primitives_.size()); }
    bool     empty()     const { return primitives_.empty(); }
    const std::vector<ColliderPrimitive>& data() const { return primitives_; }

private:
    std::vector<ColliderPrimitive> primitives_;
};

// ── インライン実装 ─────────────────────────────────────────────────────────────

inline void ColliderSet::addPlane(glm::vec3 point, glm::vec3 normal,
                                  float restitution, float friction, glm::vec3 vel) {
    ColliderPrimitive p{};
    p.type = uint32_t(ColliderType::PLANE);
    p.px = point.x;  p.py = point.y;  p.pz = point.z;
    glm::vec3 n = glm::normalize(normal);
    p.nx = n.x;  p.ny = n.y;  p.nz = n.z;
    p.vx = vel.x;  p.vy = vel.y;  p.vz = vel.z;
    p.restitution = restitution;  p.friction = friction;
    primitives_.push_back(p);
}

inline void ColliderSet::addSphere(glm::vec3 center, float radius,
                                   float restitution, float friction, glm::vec3 vel) {
    ColliderPrimitive p{};
    p.type = uint32_t(ColliderType::SPHERE);
    p.px = center.x;  p.py = center.y;  p.pz = center.z;
    p.radius = radius;
    p.vx = vel.x;  p.vy = vel.y;  p.vz = vel.z;
    p.restitution = restitution;  p.friction = friction;
    primitives_.push_back(p);
}

inline void ColliderSet::addBox(glm::vec3 center, glm::vec3 halfExt,
                                float restitution, float friction, glm::vec3 vel) {
    ColliderPrimitive p{};
    p.type = uint32_t(ColliderType::BOX);
    p.px = center.x;  p.py = center.y;  p.pz = center.z;
    p.nx = halfExt.x;  p.ny = halfExt.y;  p.nz = halfExt.z;
    p.vx = vel.x;  p.vy = vel.y;  p.vz = vel.z;
    p.restitution = restitution;  p.friction = friction;
    primitives_.push_back(p);
}

inline void ColliderSet::addCapsule(glm::vec3 start, glm::vec3 axisVec, float radius,
                                    float restitution, float friction, glm::vec3 vel) {
    ColliderPrimitive p{};
    p.type = uint32_t(ColliderType::CAPSULE);
    p.px = start.x;  p.py = start.y;  p.pz = start.z;
    p.nx = axisVec.x;  p.ny = axisVec.y;  p.nz = axisVec.z;
    p.radius = radius;
    p.vx = vel.x;  p.vy = vel.y;  p.vz = vel.z;
    p.restitution = restitution;  p.friction = friction;
    primitives_.push_back(p);
}
