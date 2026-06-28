#pragma once

#pragma once

#include <glm/glm.hpp>

struct Source {
  virtual ~Source() = default;
  glm::vec3 center;
  glm::vec3 vel;
  glm::vec3 center_vel = {0.0f, 0.0f, 0.0f}; // ソース自体の移動速度 [m/s]
  int particles_per_step = 1024;
  int step_count          = -1; // 0=無限, -1=最初の1回のみ, >0=指定ステップ数
  uint32_t particleType   = 1u; // 1=流体, 4=煙, 5=粉体
};

struct AABBSource : public Source {
  glm::vec3 size;
  static AABBSource FromAABB(const glm::vec3& min, const glm::vec3& max, const glm::vec3& vel, int particles_per_step) {
    AABBSource source;
    source.center             = (min + max) * 0.5f;
    source.size               = max - min;
    source.vel                = vel;
    source.particles_per_step = particles_per_step;
    return source;
  }
};

struct SphereSource : public Source {
  float radius;
  static SphereSource FromSphere(const glm::vec3& center, float radius, const glm::vec3& vel, int particles_per_step) {
    SphereSource source;
    source.center             = center;
    source.radius             = radius;
    source.vel                = vel;
    source.particles_per_step = particles_per_step;
    return source;
  }
};

// XY 平面上の楕円領域に粒子を生成するソース (rejection sampling)
struct EllipseSource : public Source {
  float semi_a = 1.0f; // X 方向の半径 [m]
  float semi_b = 1.0f; // Y 方向の半径 [m]
};
