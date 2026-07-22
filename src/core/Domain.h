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

// n個の値(0..n-1)を表すのに必要なビット数 (n<=1 は 0)。
inline uint32_t bitsNeeded(uint32_t n) {
  uint32_t b = 0;
  while((1u << b) < n) ++b;
  return b;
}

// アダプティブ(直方体)Morton パラメータ。
// 通常のMorton符号化は3軸を常に固定3ビット周期でinterleaveするため、軸ごとの
// 実セル数 (nx,ny,nz) が異なっていても Morton 符号が破綻なく収まるためには、
// 全軸を「実セル数の最大軸のビット幅」で揃えた立方体を確保する必要があり、
// 異方性ドメイン(軸ごとの実セル数の差が大きい場合)では大きな無駄が生じる
// (実測: nx=67,ny=23,nz=12 で立方体209万セル中 実セル18,492のみ使用、
//  約113倍の無駄。MPMは pc.hashCells の数だけグリッドセル単位でスレッドを
//  直接ディスパッチするため、この無駄は実行スレッド数にも直結する)。
//
// アダプティブ版は「全軸が共通して持つ下位ビット (commonBits=min(bx,by,bz))」
// だけ従来通り3軸interleaveし、残りは軸ごとに連結するだけに留める。これにより
// 局所性(27近傍探索でのメモリ距離)は数値実験で通常Mortonと完全に同一と確認
// 済みだが、セル数は bx+by+bz ビット分 (=2^bx*2^by*2^bz、上記の例では
// 65,536 = 従来比1/32)まで縮小できる。
//
// shaders/common.glsl の cellId()/mortonAxisTriples() 実装、および各エンジンが
// ここから生成する ADAPTIVE_MASK/ADAPTIVE_COMMON_BITS/ADAPTIVE_SHIFT_X,Y,Z の
// #define 定数と対になっているため、変更する場合は両方を同時に更新すること。
struct AdaptiveMortonParams {
  uint32_t mask;
  uint32_t commonBits;
  uint32_t shiftX, shiftY, shiftZ;
  uint32_t hashCells; // 空間ハッシュバッファの実要素数 (= 2^(bx+by+bz))
};

inline AdaptiveMortonParams computeAdaptiveMortonParams(const glm::uvec3& res) {
  const uint32_t bx         = bitsNeeded(res.x);
  const uint32_t by         = bitsNeeded(res.y);
  const uint32_t bz         = bitsNeeded(res.z);
  const uint32_t commonBits = std::min({bx, by, bz});

  AdaptiveMortonParams p;
  p.commonBits = commonBits;
  p.mask       = (1u << commonBits) - 1u;
  p.shiftX     = 3u * commonBits;
  p.shiftY     = p.shiftX + (bx - commonBits);
  p.shiftZ     = p.shiftY + (by - commonBits);
  p.hashCells  = 1u << (bx + by + bz);
  return p;
}

// 空間ハッシュ用バッファ(cellCount/cellOffset等)に必要な要素数。
inline uint32_t hashCells(const glm::uvec3& res) {
  return computeAdaptiveMortonParams(res).hashCells;
}

// MPM (mpm_common.glsl) 専用: 従来の固定3ビット周期Morton(立方体)でのセル数。
// MPMのシェーダー側実装(mortonEncodeI/mortonDecodeI)は今回のアダプティブ化の
// スコープ外のため、MPMEngine::totalCells() は必ずこちらを呼ぶこと
// (上の hashCells() をMPM側で使うと、シェーダー側は旧立方体サイズのIDを
// 生成し続けるのにバッファは縮小されてしまい、範囲外アクセスになる)。
inline uint32_t hashCellsCube(const glm::uvec3& res) {
  uint32_t m = std::max({res.x, res.y, res.z, 1u});
  uint32_t p = 1u;
  while(p < m) p <<= 1u;
  return p * p * p;
}

inline uint32_t nGroups(uint32_t cells) { return (cells + 255u) / 256u; }

} // namespace domain
