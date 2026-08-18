#include "RigidMeshCollider.h"

#include <algorithm>
#include <cmath>

namespace {

// 三角形をバケット化した簡易アクセラレータ(MeshSDF.h側のbuildMeshSDFはO(cells*tris)のブルートフォースで遅すぎるため探索はこちらで加速する)
struct TriBuckets {
  static constexpr int kRes = 32;
  glm::vec3 bmin, bmax;
  glm::ivec3 cellsOf(const glm::vec3& p) const {
    glm::vec3 t = (p - bmin) / (bmax - bmin);
    glm::ivec3 c = glm::ivec3(t * float(kRes));
    return glm::clamp(c, glm::ivec3(0), glm::ivec3(kRes - 1));
  }
  std::vector<std::vector<uint32_t>> buckets{size_t(kRes) * kRes * kRes};
  int idx(int x, int y, int z) const { return (z * kRes + y) * kRes + x; }

  void build(const std::vector<MeshTriangle>& tris) {
    for(size_t ti = 0; ti < tris.size(); ++ti) {
      const auto& t = tris[ti];
      glm::vec3 tmin = glm::min(t.v[0], glm::min(t.v[1], t.v[2]));
      glm::vec3 tmax = glm::max(t.v[0], glm::max(t.v[1], t.v[2]));
      glm::ivec3 c0 = cellsOf(tmin), c1 = cellsOf(tmax);
      for(int z = c0.z; z <= c1.z; ++z)
        for(int y = c0.y; y <= c1.y; ++y)
          for(int x = c0.x; x <= c1.x; ++x)
            buckets[idx(x, y, z)].push_back(uint32_t(ti));
    }
  }
};

} // namespace

LocalMeshSDF bakeLocalMeshSDF(const std::vector<MeshTriangle>& tris, uint32_t res, float marginCells) {
  LocalMeshSDF grid;
  grid.res = res;
  if(tris.empty()) return grid;

  glm::vec3 meshMin(1e30f), meshMax(-1e30f);
  for(const auto& t : tris)
    for(int i = 0; i < 3; ++i) {
      meshMin = glm::min(meshMin, t.v[i]);
      meshMax = glm::max(meshMax, t.v[i]);
    }
  glm::vec3 extent = meshMax - meshMin;
  float maxExtent  = std::max({extent.x, extent.y, extent.z, 1e-6f});
  grid.cellSize     = maxExtent / std::max(1.0f, float(res) - 2.0f * marginCells);
  glm::vec3 center  = (meshMin + meshMax) * 0.5f;
  grid.localMin      = center - 0.5f * float(res) * grid.cellSize;

  TriBuckets bk;
  bk.bmin = grid.localMin;
  bk.bmax = grid.localMin + float(res) * grid.cellSize;
  bk.build(tris);

  grid.data.assign(size_t(res) * res * res, 1e6f);

  for(uint32_t iz = 0; iz < res; ++iz) {
    for(uint32_t iy = 0; iy < res; ++iy) {
      for(uint32_t ix = 0; ix < res; ++ix) {
        glm::vec3 p = grid.localMin + (glm::vec3(ix, iy, iz) + 0.5f) * grid.cellSize;

        // バケットを輪状に広げ探索し、見つかった半径+1リングまで確定させる(隣接バケットのより近い三角形を潰す)
        glm::ivec3 c = bk.cellsOf(p);
        float bestDist2 = 1e30f;
        bool outer = true;
        int foundRing = -1;
        for(int r = 0; r <= TriBuckets::kRes; ++r) {
          if(foundRing >= 0 && r > foundRing + 1) break;
          bool any = false;
          for(int dz = -r; dz <= r; ++dz)
            for(int dy = -r; dy <= r; ++dy)
              for(int dx = -r; dx <= r; ++dx) {
                if(std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != r) continue;
                int x = c.x + dx, y = c.y + dy, z = c.z + dz;
                if(x < 0 || y < 0 || z < 0 || x >= TriBuckets::kRes || y >= TriBuckets::kRes || z >= TriBuckets::kRes) continue;
                any = true;
                for(uint32_t ti : bk.buckets[bk.idx(x, y, z)]) {
                  const auto& t = tris[ti];
                  glm::vec3 cp  = closestPtOnTriangle(p, t.v[0], t.v[1], t.v[2]);
                  glm::vec3 d   = p - cp;
                  float d2      = glm::dot(d, d);
                  if(d2 < bestDist2) {
                    bestDist2 = d2;
                    outer     = glm::dot(d, t.n) >= 0.0f;
                  }
                }
              }
          if(bestDist2 < 1e30f && foundRing < 0) foundRing = r;
          if(!any && r > 0 && foundRing < 0 && r > TriBuckets::kRes) break;
        }

        float dist = std::sqrt(bestDist2 < 1e30f ? bestDist2 : (maxExtent * maxExtent));
        grid.data[(size_t(iz) * res + iy) * res + ix] = outer ? dist : -dist;
      }
    }
  }

  return grid;
}
