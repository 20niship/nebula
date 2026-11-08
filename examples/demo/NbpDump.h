#pragma once
// Blender 側 (render/nbp.py) が読む粒子フレーム形式 .nbp。位置はドメイン AABB に対し uint16 量子化 (1000 万粒子でも 1 フレーム約 70MB)。
//   "NBP1" | u32 n | f32 min[3] | f32 max[3] | f32 speedMax | u16 xyz[n][3] | u8 speed[n]
#include <glm/glm.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

inline void writeNbp(const std::string& path, const std::vector<glm::vec4>& pos, const std::vector<glm::vec4>& vel, glm::vec3 lo, glm::vec3 hi, float speedMax) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if(!f) throw std::runtime_error("cannot open " + path);
  const uint32_t n = uint32_t(pos.size());
  std::fwrite("NBP1", 1, 4, f);
  std::fwrite(&n, 4, 1, f);
  std::fwrite(&lo, 4, 3, f);
  std::fwrite(&hi, 4, 3, f);
  std::fwrite(&speedMax, 4, 1, f);
  std::vector<uint16_t> q(size_t(n) * 3);
  std::vector<uint8_t> s(n);
  const glm::vec3 inv = 65535.0f / (hi - lo);
  for(uint32_t i = 0; i < n; i++) {
    for(int a = 0; a < 3; a++) q[size_t(i) * 3 + a] = uint16_t(std::clamp((pos[i][a] - lo[a]) * inv[a], 0.0f, 65535.0f));
    s[i] = uint8_t(std::clamp(glm::length(glm::vec3(vel[i])) / speedMax, 0.0f, 1.0f) * 255.0f);
  }
  std::fwrite(q.data(), 2, q.size(), f);
  std::fwrite(s.data(), 1, n, f);
  std::fclose(f);
}

// 泡粒子 (spray/foam/bubble) 用。種別 (1=spray, 2=foam, 3=bubble) を持つ。
//   "NBF1" | u32 n | f32 min[3] | f32 max[3] | u16 xyz[n][3] | u8 kind[n]
inline void writeNbf(const std::string& path, const std::vector<glm::vec4>& pos, const std::vector<uint8_t>& kind, glm::vec3 lo, glm::vec3 hi) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if(!f) throw std::runtime_error("cannot open " + path);
  const uint32_t n = uint32_t(pos.size());
  std::fwrite("NBF1", 1, 4, f);
  std::fwrite(&n, 4, 1, f);
  std::fwrite(&lo, 4, 3, f);
  std::fwrite(&hi, 4, 3, f);
  std::vector<uint16_t> q(size_t(n) * 3);
  const glm::vec3 inv = 65535.0f / (hi - lo);
  for(uint32_t i = 0; i < n; i++)
    for(int a = 0; a < 3; a++) q[size_t(i) * 3 + a] = uint16_t(std::clamp((pos[i][a] - lo[a]) * inv[a], 0.0f, 65535.0f));
  std::fwrite(q.data(), 2, q.size(), f);
  std::fwrite(kind.data(), 1, n, f);
  std::fclose(f);
}
