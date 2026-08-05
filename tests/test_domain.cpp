// src/core/Domain.h の単体テスト (issue #46)。
// domain::gridRes()/hashCells()/hashCellsCube()/bitsNeeded()/
// computeAdaptiveMortonParams() はGPU非依存の純粋関数のため、
// HeadlessCtx を介さずにロジックのみを検証する。
#include "Domain.h"
#include <doctest/doctest.h>
#include <glm/glm.hpp>

TEST_CASE("domain::gridRes - ceil(domainSize/cellSize) を返す") {
  // 10.0 / (10.0/64.0) = 64 (割り切れるケース)
  CHECK(domain::gridRes(glm::vec3(10.0f), 10.0f / 64.0f) == glm::uvec3(64u));

  // 割り切れない場合は切り上げ
  CHECK(domain::gridRes(glm::vec3(10.0f, 10.0f, 10.0f), 3.0f) == glm::uvec3(4u)); // 10/3=3.33→4

  // 軸ごとに異なるサイズ (直方体ドメイン)
  CHECK(domain::gridRes(glm::vec3(40.0f, 10.0f, 10.0f), 5.0f) == glm::uvec3(8u, 2u, 2u));
}

TEST_CASE("domain::gridRes - 最低1セル/軸を保証する") {
  // 極端に小さいdomainSizeでも0セルにはならない
  CHECK(domain::gridRes(glm::vec3(0.01f), 5.0f) == glm::uvec3(1u));
  CHECK(domain::gridRes(glm::vec3(0.0f), 5.0f) == glm::uvec3(1u));
}

TEST_CASE("domain::bitsNeeded - 0..n-1を表すのに必要なビット数") {
  CHECK(domain::bitsNeeded(1u) == 0u);
  CHECK(domain::bitsNeeded(2u) == 1u);
  CHECK(domain::bitsNeeded(64u) == 6u);
  CHECK(domain::bitsNeeded(65u) == 7u);
  CHECK(domain::bitsNeeded(67u) == 7u);
  CHECK(domain::bitsNeeded(23u) == 5u);
  CHECK(domain::bitsNeeded(12u) == 4u);
}

TEST_CASE("domain::hashCellsCube - 実セル数の最大軸以上の最小の2のべき乗の立方体 (MPM専用)") {
  CHECK(domain::hashCellsCube(glm::uvec3(64u, 64u, 64u)) == 64u * 64u * 64u); // ちょうど2のべき乗
  CHECK(domain::hashCellsCube(glm::uvec3(5u, 5u, 5u)) == 8u * 8u * 8u);       // 5→8に切り上げ
  CHECK(domain::hashCellsCube(glm::uvec3(1u, 1u, 1u)) == 1u);                 // 1は既に2の0乗

  // 直方体ドメイン: 各軸の実セル数の積(40*10*10=4000)ではなく、
  // 最大軸を包含する2のべき乗立方体(64^3)のサイズになる
  const glm::uvec3 res(40u, 10u, 10u);
  const uint32_t hc = domain::hashCellsCube(res);
  CHECK(hc == 64u * 64u * 64u);
  CHECK(hc != res.x * res.y * res.z);
  CHECK(hc > res.x * res.y * res.z); // Morton符号空間はより疎なので必ず大きい
}

TEST_CASE("domain::computeAdaptiveMortonParams / hashCells - 軸ごとの必要ビット数の合計で決まる直方体") {
  // 立方体ドメイン (bx=by=bz=6): commonBits=6, shiftはすべてinterleave部のみ
  {
    const auto p = domain::computeAdaptiveMortonParams(glm::uvec3(64u, 64u, 64u));
    CHECK(p.commonBits == 6u);
    CHECK(p.mask == 63u);
    CHECK(p.shiftX == 18u);
    CHECK(p.shiftY == 18u);
    CHECK(p.shiftZ == 18u);
    CHECK(p.hashCells == 1u << 18u); // 2^(6+6+6)、立方体Morton(64^3=2^18)と一致
    CHECK(domain::hashCells(glm::uvec3(64u, 64u, 64u)) == p.hashCells);
  }

  // 異方性ドメイン (fluid_wave_foam実例: nx=67,ny=23,nz=12 → bx=7,by=5,bz=4)
  {
    const glm::uvec3 res(67u, 23u, 12u);
    const auto p = domain::computeAdaptiveMortonParams(res);
    CHECK(p.commonBits == 4u); // min(7,5,4)
    CHECK(p.mask == 15u);
    CHECK(p.shiftX == 12u); // 3*4
    CHECK(p.shiftY == 15u); // 12+(7-4)
    CHECK(p.shiftZ == 16u); // 15+(5-4)
    CHECK(p.hashCells == 1u << 16u); // 2^(7+5+4)=65536

    // 立方体Morton(hashCellsCube)より大幅に小さい (実測1/32)
    CHECK(p.hashCells < domain::hashCellsCube(res));
    CHECK(domain::hashCells(res) == p.hashCells);

    // 実セル数(18,492)は下回らない (バッファが実際の使用範囲を包含する)
    CHECK(p.hashCells >= res.x * res.y * res.z);
  }
}

TEST_CASE("domain::nGroups - 256スレッド単位workgroup数を切り上げで計算する") {
  CHECK(domain::nGroups(256u) == 1u);
  CHECK(domain::nGroups(257u) == 2u);
  CHECK(domain::nGroups(0u) == 0u);
}
