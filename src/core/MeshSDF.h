#pragma once
// 任意メッシュ (STL) から Morton 順 SDF グリッドを構築するヘッダオンリーユーティリティ。
// examples/mpm_stl_drop.cpp のロジックを MPMConfig 非依存の形に一般化したもの。
// MPMEngine::setColliderSDF / PyroEngine::setColliderSDF どちらにも使える。

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct MeshTriangle {
  glm::vec3 n;
  glm::vec3 v[3];
};

// ── Binary STL ローダー ─────────────────────────────────────────────────────

inline std::vector<MeshTriangle> loadBinarySTL(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if(!f) throw std::runtime_error("STL open failed: " + path);

  char header[80];
  f.read(header, 80);
  uint32_t count;
  f.read(reinterpret_cast<char*>(&count), 4);

  std::vector<MeshTriangle> tris(count);
  for(uint32_t i = 0; i < count; ++i) {
    float buf[12];
    f.read(reinterpret_cast<char*>(buf), 48);
    uint16_t attr;
    f.read(reinterpret_cast<char*>(&attr), 2);
    tris[i].n    = {buf[0], buf[1], buf[2]};
    tris[i].v[0] = {buf[3], buf[4], buf[5]};
    tris[i].v[1] = {buf[6], buf[7], buf[8]};
    tris[i].v[2] = {buf[9], buf[10], buf[11]};
  }
  return tris;
}

// ── 点→三角形 最近傍点 (Ericson RTCD) ───────────────────────────────────────

inline glm::vec3 closestPtOnTriangle(glm::vec3 p, glm::vec3 a, glm::vec3 b, glm::vec3 c) {
  glm::vec3 ab = b - a, ac = c - a, ap = p - a;
  float d1 = glm::dot(ab, ap), d2 = glm::dot(ac, ap);
  if(d1 <= 0.0f && d2 <= 0.0f) return a;

  glm::vec3 bp = p - b;
  float d3 = glm::dot(ab, bp), d4 = glm::dot(ac, bp);
  if(d3 >= 0.0f && d4 <= d3) return b;

  float vc = d1 * d4 - d3 * d2;
  if(vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    return a + (d1 / (d1 - d3)) * ab;
  }

  glm::vec3 cp2 = p - c;
  float d5 = glm::dot(ab, cp2), d6 = glm::dot(ac, cp2);
  if(d6 >= 0.0f && d5 <= d6) return c;

  float vb = d5 * d2 - d1 * d6;
  if(vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    return a + (d2 / (d2 - d6)) * ac;
  }

  float va  = d3 * d6 - d5 * d4;
  float sum = va + vb + vc;
  return a + (vb / sum) * ab + (vc / sum) * ac;
}

// ── Morton 符号化 (mpm_common.glsl/pyro_common.glsl と同一ロジック) ─────────

inline uint32_t meshSdfMortonExpand(uint32_t v) {
  v = (v | (v << 16u)) & 0x030000FFu;
  v = (v | (v << 8u)) & 0x0300F00Fu;
  v = (v | (v << 4u)) & 0x030C30C3u;
  v = (v | (v << 2u)) & 0x09249249u;
  return v;
}
inline uint32_t meshSdfMortonEncode(uint32_t x, uint32_t y, uint32_t z) { return meshSdfMortonExpand(x) | (meshSdfMortonExpand(y) << 1u) | (meshSdfMortonExpand(z) << 2u); }

// ── メッシュ→SDF バッファ構築 (O(realRes.x*y*z × 三角形数) のブルートフォース) ───
// 三角形数・解像度が大きいと重いため、動的障害物として毎フレーム呼ぶ場合は
// 低ポリメッシュ + 適度な解像度 + 再構築間隔 (interval) で実用速度に収めること。
//
// issue #46フォローアップ: 直方体ドメイン対応。realRes(各軸の実セル数)+totalCells(=Morton
// dispatch用の立方体セル総数、cubeRes^3。PyroConfig::totalCells()/MPMConfig::totalCells()と
// 同じ値)+cellSize(全軸共通セルサイズ)を受け取る。出力バッファは常に totalCells 要素確保し、
// realRes 範囲内のみ SDF を計算する(範囲外=パディング領域はデフォルト値 1e9f=無衝突のまま)。
// これにより PyroEngine::setColliderSDF() が要求するサイズ契約 (cfg().totalCells()) と
// 型レベルで一致する。

inline std::vector<float> buildMeshSDF(const std::vector<MeshTriangle>& tris, const glm::uvec3& realRes, uint32_t totalCells, float cellSize, bool verbose = true) {
  const float cs = cellSize;
  std::vector<float> sdf(size_t(totalCells), 1e9f);

  if(verbose) {
    std::printf("  SDF 構築中: %u×%u×%u セル (cube総数%u) × %zu 三角形 ...\n", realRes.x, realRes.y, realRes.z, totalCells, tris.size());
    std::fflush(stdout);
  }

  for(uint32_t iz = 0; iz < realRes.z; ++iz) {
    if(verbose && iz % std::max(1u, realRes.z / 8) == 0) {
      std::printf("  SDF %u/%u\r", iz, realRes.z);
      std::fflush(stdout);
    }
    for(uint32_t iy = 0; iy < realRes.y; ++iy)
      for(uint32_t ix = 0; ix < realRes.x; ++ix) {
        glm::vec3 p = {(ix + 0.5f) * cs, (iy + 0.5f) * cs, (iz + 0.5f) * cs};
        float minD  = 1e9f;
        bool outer  = true;
        for(const auto& tri : tris) {
          glm::vec3 cp = closestPtOnTriangle(p, tri.v[0], tri.v[1], tri.v[2]);
          float d      = glm::length(p - cp);
          if(d < minD) {
            minD  = d;
            outer = glm::dot(p - cp, tri.n) >= 0.0f;
          }
        }
        sdf[meshSdfMortonEncode(ix, iy, iz)] = outer ? minD : -minD;
      }
  }
  if(verbose) std::printf("  SDF 完了                    \n");
  return sdf;
}

// ── 剛体変換 (動的障害物の移動・回転に使用) ─────────────────────────────────

inline std::vector<MeshTriangle> transformTriangles(const std::vector<MeshTriangle>& tris, const glm::vec3& translation, const glm::quat& rotation) {
  std::vector<MeshTriangle> out(tris.size());
  for(size_t i = 0; i < tris.size(); ++i) {
    out[i].n = rotation * tris[i].n;
    for(int j = 0; j < 3; ++j) out[i].v[j] = rotation * tris[i].v[j] + translation;
  }
  return out;
}
