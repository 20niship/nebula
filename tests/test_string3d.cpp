// 3D string / rope physics tests
#include "helpers/HeadlessCtx.h"
#include "helpers/PhysicsHarness.h"
#include <cmath>
#include <doctest/doctest.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>

static const std::string SHADERS = SHADER_DIR;

// ── 3D 自由落下 (ピン留めなし) ───────────────────────────────────────────────
// z軸方向に水平配置した糸をピン留めなしで落下させる。
// 1秒後: 全粒子がy方向に2m以上落下。
TEST_CASE("String3D - free fall, no pin") {
  HeadlessCtx ctx;
  ctx.init();

  constexpr uint32_t N    = 8;
  constexpr float SPACING = 0.1f;
  constexpr float INIT_Y  = 6.0f;
  constexpr float PIN_Z   = 3.25f;

  PhysicsHarness::Config cfg;
  cfg.N                = N;
  cfg.worldMin         = 0.0f;
  cfg.worldMax         = 10.0f;
  cfg.radius           = 0.14f;
  cfg.gravity          = -9.8f;
  cfg.gridRes          = 16;
  cfg.numSubsteps      = 4;
  cfg.solverIterations = 4;
  cfg.typeFlag         = 2;
  cfg.selfCollision    = false;
  PhysicsHarness::buildChainEdges(N, SPACING, cfg.edgeData, cfg.edgeColor0End);

  // z軸方向に水平配置
  std::vector<glm::vec4> pos(N), invM(N);
  for(uint32_t i = 0; i < N; ++i) {
    pos[i]  = glm::vec4(5.0f, INIT_Y, PIN_Z + i * SPACING, 1.0f);
    invM[i] = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
  }

  PhysicsHarness sim;
  sim.init(ctx, cfg, SHADERS, pos, invM);

  const float dt = 1.0f / 60.0f;
  for(int f = 0; f < 60; ++f) sim.step(dt);

  for(uint32_t i = 0; i < N; ++i) {
    auto p     = sim.readPos(i);
    float initZ = PIN_Z + i * SPACING; // 各粒子は初期Zがiごとに異なる (水平配置)
    CHECK(p.z < initZ - 2.0f); // 重力はZ軸: 各粒子の初期Zから2m以上落下
    CHECK(p.z > cfg.worldMin);
  }

  sim.cleanup();
  ctx.cleanup();
}

// ── 3D ピン留め・自己衝突なし ─────────────────────────────────────────────────
// 端点をピン留めし水平スタート。自己衝突なし。2秒後に自由端が落下。
TEST_CASE("String3D - pinned, no self-collision") {
  HeadlessCtx ctx;
  ctx.init();

  constexpr uint32_t N    = 16;
  constexpr float SPACING = 0.1f;
  constexpr float INIT_Y  = 7.0f;
  constexpr float PIN_X   = 5.0f;
  constexpr float PIN_Z   = 3.25f;

  PhysicsHarness::Config cfg;
  cfg.N                = N;
  cfg.worldMin         = 0.0f;
  cfg.worldMax         = 10.0f;
  cfg.radius           = 0.14f;
  cfg.gravity          = -9.8f;
  cfg.gridRes          = 16;
  cfg.numSubsteps      = 4;
  cfg.solverIterations = 4;
  cfg.typeFlag         = 2;
  cfg.selfCollision    = false;
  PhysicsHarness::buildChainEdges(N, SPACING, cfg.edgeData, cfg.edgeColor0End);

  // Y方向に垂れ下がる初期配置 (重力Z軸に対して横方向で落下しやすい)
  std::vector<glm::vec4> pos(N), invM(N);
  for(uint32_t i = 0; i < N; ++i) {
    pos[i]  = glm::vec4(PIN_X, INIT_Y - i * SPACING, PIN_Z, 1.0f);
    invM[i] = glm::vec4(i == 0 ? 0.0f : 1.0f, 0.0f, 0.0f, 0.0f);
  }

  PhysicsHarness sim;
  sim.init(ctx, cfg, SHADERS, pos, invM);

  const float dt = 1.0f / 60.0f;
  for(int f = 0; f < 120; ++f) sim.step(dt);

  // ピン点は静止
  auto pin = sim.readPos(0);
  CHECK(pin.x == doctest::Approx(PIN_X).epsilon(0.01f));
  CHECK(pin.y == doctest::Approx(INIT_Y).epsilon(0.01f));

  // 自由端はZ方向に0.8m以上落下 (重力はZ軸。水平配置からの振り子運動のため
  // 2秒でも鉛直静止姿勢まで到達しきらず実測落下量は約0.93m)
  auto tip = sim.readPos(N - 1);
  CHECK(tip.z < PIN_Z - 0.8f);

  sim.cleanup();
  ctx.cleanup();
}

// ── 3D ピン留め・自己衝突あり ─────────────────────────────────────────────────
// 自己衝突を有効にしても振り子の落下動作に大きな差がないこと。
// (typeFlag=2 の colDiam=r*0.5=0.07m < spacing=0.1m なので隣接粒子は衝突しない)
TEST_CASE("String3D - pinned, with self-collision") {
  HeadlessCtx ctx;
  ctx.init();

  constexpr uint32_t N    = 16;
  constexpr float SPACING = 0.1f;
  constexpr float INIT_Y  = 7.0f;
  constexpr float PIN_X   = 5.0f;
  constexpr float PIN_Z   = 3.25f;

  PhysicsHarness::Config cfg;
  cfg.N                = N;
  cfg.worldMin         = 0.0f;
  cfg.worldMax         = 10.0f;
  cfg.radius           = 0.14f;
  cfg.gravity          = -9.8f;
  cfg.gridRes          = 16;
  cfg.numSubsteps      = 4;
  cfg.solverIterations = 4;
  cfg.typeFlag         = 2;
  cfg.selfCollision    = true; // ← 自己衝突あり
  PhysicsHarness::buildChainEdges(N, SPACING, cfg.edgeData, cfg.edgeColor0End);

  // Y方向に垂れ下がる初期配置 (重力Z軸に対して横方向で落下しやすい)
  std::vector<glm::vec4> pos(N), invM(N);
  for(uint32_t i = 0; i < N; ++i) {
    pos[i]  = glm::vec4(PIN_X, INIT_Y - i * SPACING, PIN_Z, 1.0f);
    invM[i] = glm::vec4(i == 0 ? 0.0f : 1.0f, 0.0f, 0.0f, 0.0f);
  }

  PhysicsHarness sim;
  sim.init(ctx, cfg, SHADERS, pos, invM);

  const float dt = 1.0f / 60.0f;
  for(int f = 0; f < 120; ++f) sim.step(dt);

  auto pin = sim.readPos(0);
  CHECK(pin.x == doctest::Approx(PIN_X).epsilon(0.01f));
  CHECK(pin.y == doctest::Approx(INIT_Y).epsilon(0.01f));

  // 自己衝突ありでも自由端はZ方向に0.8m以上落下 (重力はZ軸。振り子運動のため
  // 2秒でも鉛直静止姿勢まで到達しきらず実測落下量は約0.93m)
  auto tip = sim.readPos(N - 1);
  CHECK(tip.z < PIN_Z - 0.8f);

  sim.cleanup();
  ctx.cleanup();
}

// ── 3D ピン留め・自己衝突あり・風あり ────────────────────────────────────────
// 垂直に吊り下がった糸に水平風力を加えた場合、
// 自由端がx方向にピン留め点から0.3m以上なびくこと。
TEST_CASE("String3D - pinned vertical, wind deflects tip") {
  HeadlessCtx ctx;
  ctx.init();

  constexpr uint32_t N    = 16;
  constexpr float SPACING = 0.1f; // 全長 1.5m
  constexpr float PIN_X   = 5.0f;
  constexpr float PIN_Y   = 8.0f;
  constexpr float PIN_Z   = 5.0f;
  constexpr float WIND_X  = 5.0f; // 水平風力

  PhysicsHarness::Config cfg;
  cfg.N                = N;
  cfg.worldMin         = 0.0f;
  cfg.worldMax         = 10.0f;
  cfg.radius           = 0.14f;
  cfg.gravity          = -9.8f;
  cfg.windX            = WIND_X;
  cfg.windZ            = 0.0f;
  cfg.gridRes          = 16;
  cfg.numSubsteps      = 4;
  cfg.solverIterations = 4;
  cfg.typeFlag         = 2;
  cfg.selfCollision    = true;
  PhysicsHarness::buildChainEdges(N, SPACING, cfg.edgeData, cfg.edgeColor0End);

  // 垂直吊り下がり初期配置: 粒子0(ピン)から下方向に
  std::vector<glm::vec4> pos(N), invM(N);
  for(uint32_t i = 0; i < N; ++i) {
    pos[i]  = glm::vec4(PIN_X, PIN_Y - i * SPACING, PIN_Z, 1.0f);
    invM[i] = glm::vec4(i == 0 ? 0.0f : 1.0f, 0.0f, 0.0f, 0.0f);
  }

  PhysicsHarness sim;
  sim.init(ctx, cfg, SHADERS, pos, invM);

  // 3秒間シミュレーション: 風が糸をX方向に押し流す
  const float dt = 1.0f / 60.0f;
  for(int f = 0; f < 180; ++f) sim.step(dt);

  auto pin = sim.readPos(0);
  CHECK(pin.x == doctest::Approx(PIN_X).epsilon(0.01f));

  // 自由端が風でX方向にたわんでいること
  auto tip = sim.readPos(N - 1);
  CHECK(tip.x > PIN_X + 0.3f);
  // 重力(Z軸)によりZがピン留め点のZ(5.0)より下がっている
  CHECK(tip.z < PIN_Z - 0.5f);

  sim.cleanup();
  ctx.cleanup();
}
