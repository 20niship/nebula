// src/core/Domain.h の単体テスト (issue #46)。
// domain::gridRes()/hashCells()/mortonCubeRes() はGPU非依存の純粋関数のため、
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

TEST_CASE("domain::mortonCubeRes - 実セル数の最大軸以上の最小の2のべき乗を返す") {
  CHECK(domain::mortonCubeRes(glm::uvec3(64u, 64u, 64u)) == 64u); // ちょうど2のべき乗
  CHECK(domain::mortonCubeRes(glm::uvec3(5u, 5u, 5u)) == 8u);     // 5→8に切り上げ
  CHECK(domain::mortonCubeRes(glm::uvec3(40u, 10u, 10u)) == 64u); // 最大軸40→64に切り上げ
  CHECK(domain::mortonCubeRes(glm::uvec3(1u, 1u, 1u)) == 1u);     // 1は既に2の0乗
}

TEST_CASE("domain::hashCells - cubeRes^3を返す (gridRes.x*y*zではない)") {
  // 立方体ドメイン: cubeRes = 64 (既に2のべき乗)
  CHECK(domain::hashCells(glm::uvec3(64u, 64u, 64u)) == 64u * 64u * 64u);

  // 直方体ドメイン: 各軸の実セル数の積(40*10*10=4000)ではなく、
  // 最大軸を包含する2のべき乗立方体(64^3)のサイズになる
  const glm::uvec3 res(40u, 10u, 10u);
  const uint32_t hc = domain::hashCells(res);
  CHECK(hc == 64u * 64u * 64u);
  CHECK(hc != res.x * res.y * res.z);
  CHECK(hc > res.x * res.y * res.z); // Morton符号空間はより疎なので必ず大きい
}

TEST_CASE("domain::nGroups - 256スレッド単位workgroup数を切り上げで計算する") {
  CHECK(domain::nGroups(256u) == 1u);
  CHECK(domain::nGroups(257u) == 2u);
  CHECK(domain::nGroups(0u) == 0u);
}
