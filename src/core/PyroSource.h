#pragma once
#include <cstdint>
#include <glm/glm.hpp>

// Pyro グリッドソルバー用の発生源。MPM の Source (src/core/source.h) と異なり、
// パーティクルを生成するのではなく、形状内のセルへ毎フレーム
// density/temperature/fuel/velocity を連続的に加算する。
enum class PyroSourceShape : uint32_t {
  AABB    = 0u, // size = 半辺長 (halfExtent)
  SPHERE  = 1u, // size.x = 半径
  ELLIPSE = 2u, // XZ 平面上の楕円; size.x = 半径a (X), size.z = 半径b (Z), size.y = 半高さ (Y)
};

struct PyroSource {
  PyroSourceShape shape = PyroSourceShape::SPHERE;
  glm::vec3 center{0.0f};
  glm::vec3 size{1.0f};
  glm::vec3 center_vel{0.0f};      // ソース自体の移動速度 [m/s] (MPM Source と同じ意味論)
  glm::vec3 inflowVelocity{0.0f};  // 放出時に加える初速 [m/s]
  float     densityRate     = 0.0f; // 密度注入速度 [1/s]
  float     temperatureRate = 0.0f; // 温度注入速度 [K/s]
  float     fuelRate        = 0.0f; // 燃料注入速度 [1/s]
  // 0=無限, -1=最初の1ステップのみ, >0=指定ステップ数 (MPM Source::step_count と同じ意味論)
  int       step_count = 0;
};

// GPU 転送用 packed struct (64 bytes = 16 uint32_t、ColliderPrimitive と同じ規約)
// [0]=shape [1-3]=center [4-6]=size [7]=densityRate
// [8-10]=inflowVelocity [11]=temperatureRate [12]=fuelRate [13-15]=pad
struct alignas(16) PyroSourceGPU {
  uint32_t shape;
  float    cx, cy, cz;
  float    sx, sy, sz;
  float    densityRate;
  float    ivx, ivy, ivz;
  float    temperatureRate;
  float    fuelRate;
  float    pad0, pad1, pad2;
};
static_assert(sizeof(PyroSourceGPU) == 64, "PyroSourceGPU must be 64 bytes");

inline PyroSourceGPU toPyroSourceGPU(const PyroSource& s) {
  PyroSourceGPU g{};
  g.shape           = static_cast<uint32_t>(s.shape);
  g.cx = s.center.x; g.cy = s.center.y; g.cz = s.center.z;
  g.sx = s.size.x;   g.sy = s.size.y;   g.sz = s.size.z;
  g.densityRate     = s.densityRate;
  g.ivx = s.inflowVelocity.x; g.ivy = s.inflowVelocity.y; g.ivz = s.inflowVelocity.z;
  g.temperatureRate = s.temperatureRate;
  g.fuelRate        = s.fuelRate;
  return g;
}
