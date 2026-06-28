#include "ClothMesh.h"

#include <cmath>
#include <cstring>
#include <glm/glm.hpp>

void ClothMesh::build(int n, float sp, float originX, float originY, float originZ) {
  gridN   = n;
  spacing = sp;

  const int N       = gridN;
  const float halfW = (N - 1) * spacing * 0.5f;

  // ── 頂点データ ────────────────────────────────────────────────────────
  positions.resize(N * N);
  invMasses.resize(N * N, glm::vec4(1.0f / mass, 0.0f, 0.0f, 0.0f));
  typeFlags.resize(N * N, 2u); // 2 = Cloth

  for(int i = 0; i < N; ++i)
    for(int j = 0; j < N; ++j) {
      float x              = originX - halfW + j * spacing;
      float y              = originY - halfW + i * spacing;
      positions[idx(i, j)] = glm::vec4(x, y, originZ, 1.0f);
    }

  // ── エッジ & グラフ彩色 ───────────────────────────────────────────────
  buildEdgesAndColoring();

  // ── 三角形インデックス ────────────────────────────────────────────────
  buildTriIndices();
}

void ClothMesh::buildEdgesAndColoring() {
  const int N = gridN;
  // 12 色: StretchH×2, StretchV×2, ShearA×2, ShearB×2, BendH×2, BendV×2
  nColors = 12;
  coloredEdges_.assign(nColors, {});

  auto addE = [&](int p, int q, float len, int color) {
    auto& v = coloredEdges_[color];
    v.push_back((uint32_t)p);
    v.push_back((uint32_t)q);
    // restLen をビット変換して uint として格納
    uint32_t bits;
    memcpy(&bits, &len, 4);
    v.push_back(bits);
  };

  const float diagLen  = spacing * 1.41421356f;
  const float bendLen  = spacing * 2.0f;
  const float bendDiag = spacing * 2.82842712f;

  for(int i = 0; i < N; ++i)
    for(int j = 0; j < N; ++j) {
      // ── Stretch 横 (i,j)→(i,j+1)  色 = j%2  (0 or 1) ────────────
      if(j + 1 < N) addE(idx(i, j), idx(i, j + 1), spacing, j % 2);

      // ── Stretch 縦 (i,j)→(i+1,j)  色 = 2 + i%2 ─────────────────
      if(i + 1 < N) addE(idx(i, j), idx(i + 1, j), spacing, 2 + i % 2);

      // ── Shear ↘ (i,j)→(i+1,j+1)  色 = 4 + i%2 ──────────────────
      // (i+j)%2 だと (i,j)→(i+1,j+1) と (i+1,j+1)→(i+2,j+2) が同色で頂点共有 → 競合
      if(i + 1 < N && j + 1 < N) addE(idx(i, j), idx(i + 1, j + 1), diagLen, 4 + i % 2);

      // ── Shear ↗ (i,j+1)→(i+1,j)  色 = 6 + i%2 ──────────────────
      if(i + 1 < N && j + 1 < N) addE(idx(i, j + 1), idx(i + 1, j), diagLen, 6 + i % 2);

      // ── Bend 横 (i,j)→(i,j+2)  色 = 8 + (j/2)%2 ────────────────
      if(j + 2 < N) addE(idx(i, j), idx(i, j + 2), bendLen, 8 + (j / 2) % 2);

      // ── Bend 縦 (i,j)→(i+2,j)  色 = 10 + (i/2)%2 ───────────────
      if(i + 2 < N) addE(idx(i, j), idx(i + 2, j), bendLen, 10 + (i / 2) % 2);
    }

  // ── 色順に edgeData と colorBatch を構築 ─────────────────────────────
  edgeData.clear();
  colorBatch.clear();
  colorBatch.push_back(0);
  for(int c = 0; c < nColors; ++c) {
    for(uint32_t x : coloredEdges_[c]) edgeData.push_back(x);
    // coloredEdges_[c].size() は stride=3 なので エッジ数 = size/3
    colorBatch.push_back((uint32_t)(edgeData.size() / 3));
  }
  coloredEdges_.clear();
}

void ClothMesh::buildTriIndices() {
  const int N = gridN;
  triIndices.clear();
  triIndices.reserve((N - 1) * (N - 1) * 6);
  for(int i = 0; i < N - 1; ++i)
    for(int j = 0; j < N - 1; ++j) {
      uint32_t a = idx(i, j);
      uint32_t b = idx(i, j + 1);
      uint32_t c = idx(i + 1, j);
      uint32_t d = idx(i + 1, j + 1);
      triIndices.push_back(a);
      triIndices.push_back(b);
      triIndices.push_back(c);
      triIndices.push_back(b);
      triIndices.push_back(d);
      triIndices.push_back(c);
    }
}
