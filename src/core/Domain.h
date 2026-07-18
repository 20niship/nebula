#pragma once
#include <glm/glm.hpp>
#include <cstdint>
#include <algorithm>

// issue #46: ボリュームドメインの立方体制約撤廃(直方体対応)。
// 各エンジンの *Config は domainSize(vec3, ドメイン物理サイズ[m]) + cellSize(float,
// 全軸共通のセルサイズ[m]) を入力として持ち、実セル数・空間ハッシュ用バッファサイズは
// ここで定義するヘルパー経由で導出する。
namespace domain {

// domainSize/cellSize から各軸の実セル数を導出する (ceil、最低1セル/軸を保証)。
// PBF/XPBD/MPM の境界クランプ・近傍探索の範囲判定 (GLSL側 pc.gridRes) に使う。
inline glm::uvec3 gridRes(const glm::vec3& domainSize, float cellSize) {
  const glm::vec3 r = glm::ceil(domainSize / glm::vec3(std::max(cellSize, 1e-6f)));
  return glm::max(glm::uvec3(r), glm::uvec3(1u));
}

// 空間ハッシュの Morton 符号化は各軸を同一ビット数で対称に展開するため、軸ごとの実
// セル数 (nx,ny,nz) が異なっていても、Morton 符号が破綻なく [0, cubeRes^3) に収まる
// ためには全軸を「実セル数の最大軸以上の、最小の2のべき乗」で揃える必要がある。
// (例: nx=40,ny=10,nz=40 → 最大40 → cubeRes=64。このとき有効な Morton 符号は
//  64^3 の中に疎らに分布するが、密ではないため、ハッシュバッファは nx*ny*nz ではなく
//  cubeRes^3 で確保しなければならない)
inline uint32_t mortonCubeRes(const glm::uvec3& res) {
  uint32_t m = std::max({res.x, res.y, res.z, 1u});
  uint32_t p = 1u;
  while(p < m) p <<= 1u;
  return p;
}

// 空間ハッシュ用バッファ(cellCount/cellOffset等)に必要な要素数 = cubeRes^3。
// 各軸の実セル数の積(nx*ny*nz)ではない点に注意 (上記 mortonCubeRes 参照)。
inline uint32_t hashCells(const glm::uvec3& res) {
  const uint32_t c = mortonCubeRes(res);
  return c * c * c;
}

inline uint32_t nGroups(uint32_t cells) { return (cells + 255u) / 256u; }

} // namespace domain
