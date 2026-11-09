// Pyro (グリッドベース煙・火炎) 単体テスト
// PyroEngine の公開 API (init/step/addEmitter/setColliderSDF/getXxxBuffer) が
// HeadlessCtx 上でそのままテストハーネスとして使えるため、専用 Harness は作らない。
#include "core/MeshSDF.h" 
#include "engine/PyroEngine.h"
#include "helpers/HeadlessCtx.h"
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <string>
#include <vector>

static const std::string SHADERS = SHADER_DIR;

// ═══════════════════════════════════════════════════════════════════════════
// 1: 浮力 — 高温セルの速度が+Y方向に増加する
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Pyro GPU - 1-1: Buoyancy drives net upward velocity") {
  HeadlessCtx ctx;
  ctx.init();

  PyroEngine engine;
  engine.domainSize = glm::vec3(10.0f);
  engine.cellSize   = 10.0f / 16.0f;
  engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS);
  engine.pc_.buoyancyAlpha = 2.0f;
  engine.pc_.buoyancyBeta  = 0.0f;
  engine.pc_.vorticityEps  = 0.0f;
  engine.numPressureIters  = 20;

  auto heat             = std::make_shared<SphereEmitter>();
  heat->center          = {5.0f, 5.0f, 5.0f};
  heat->radius          = 1.5f;
  heat->temperatureRate = 8.0f;
  heat->step_count      = 0;
  engine.addEmitter(heat);

  const float dt = 1.0f / 60.0f;
  for(int i = 0; i < 20; i++) {
    VkCommandBuffer cmd = ctx.beginCmd();
    engine.step(cmd, dt);
    ctx.submitCmd(cmd);
  }

  const uint32_t NC = engine.totalCells();
  std::vector<glm::vec4> vel(NC);
  ctx.readBuffer(engine.getVelocityBuffer(), 0, vel.data(), NC * sizeof(glm::vec4));

  double sumVy = 0.0;
  for(auto& v : vel) {
    CHECK(std::isfinite(v.y));
    sumVy += v.y;
  }
  CHECK(sumVy / double(NC) > 0.0);

  engine.cleanup();
  ctx.cleanup();
}

// ═══════════════════════════════════════════════════════════════════════════
// 2: 圧力投影後、開放領域でdivergenceがほぼ0になる
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Pyro GPU - 2-1: Pressure projection keeps interior divergence small") {
  HeadlessCtx ctx;
  ctx.init();

  PyroEngine engine;
  engine.domainSize = glm::vec3(10.0f);
  engine.cellSize   = 10.0f / 16.0f;
  engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS);
  engine.pc_.buoyancyAlpha = 3.0f;
  engine.numPressureIters  = 30;

  auto src             = std::make_shared<SphereEmitter>();
  src->center          = {5.0f, 2.0f, 5.0f};
  src->radius          = 1.5f;
  src->temperatureRate = 6.0f;
  src->step_count      = 0;
  engine.addEmitter(src);

  const float dt = 1.0f / 60.0f;
  for(int i = 0; i < 30; i++) {
    VkCommandBuffer cmd = ctx.beginCmd();
    engine.step(cmd, dt);
    ctx.submitCmd(cmd);
  }

  const uint32_t G  = engine.gridRes().x;
  const uint32_t NC = engine.totalCells();
  std::vector<glm::vec4> velLinear(NC);
  ctx.readBuffer(engine.getVelocityBuffer(), 0, velLinear.data(), NC * sizeof(glm::vec4));

  auto sampleVel = [&](int x, int y, int z) -> glm::vec3 {
    x = std::clamp(x, 0, int(G) - 1);
    y = std::clamp(y, 0, int(G) - 1);
    z = std::clamp(z, 0, int(G) - 1);
    return glm::vec3(velLinear[engine.cellIndex(uint32_t(x), uint32_t(y), uint32_t(z))]);
  };

  const float h = engine.cellSize;
  float maxDiv  = 0.0f;
  // 境界・source近傍を避けた内部領域のみチェック
  for(int x = 4; x < 12; x++)
    for(int y = 4; y < 12; y++)
      for(int z = 4; z < 12; z++) {
        float div = 0.5f * ((sampleVel(x + 1, y, z).x - sampleVel(x - 1, y, z).x) + (sampleVel(x, y + 1, z).y - sampleVel(x, y - 1, z).y) + (sampleVel(x, y, z + 1).z - sampleVel(x, y, z - 1).z)) / h;
        maxDiv    = std::max(maxDiv, std::abs(div));
      }
  // 投影が機能していれば内部領域の発散はおおむね小さく収まる
  // (Red-Black Gauss-Seidel の粗い近似解のため厳密な0ではなく、桁違いの破綻がないことを確認する)
  CHECK(maxDiv < 1.0f);

  engine.cleanup();
  ctx.cleanup();
}

// ═══════════════════════════════════════════════════════════════════════════
// 3: 障害物SDF内部で速度がゼロ化される
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Pyro GPU - 3-1: Obstacle SDF zeroes velocity inside solid") {
  HeadlessCtx ctx;
  ctx.init();

  PyroEngine engine;
  engine.domainSize = glm::vec3(10.0f);
  engine.cellSize   = 10.0f / 16.0f;
  engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS);
  engine.pc_.buoyancyAlpha = 3.0f;

  const uint32_t G       = engine.gridRes().x;
  const float h          = engine.cellSize;
  const glm::vec3 center = {5.0f, 5.0f, 5.0f};
  const float radius     = 2.0f;

  std::vector<float> sdf(engine.totalCells());
  for(uint32_t z = 0; z < G; z++)
    for(uint32_t y = 0; y < G; y++)
      for(uint32_t x = 0; x < G; x++) {
        glm::vec3 p                       = (glm::vec3(x, y, z) + 0.5f) * h;
        sdf[engine.cellIndex(x, y, z)] = glm::length(p - center) - radius;
      }
  engine.setColliderSDF(sdf);

  // 障害物のすぐ下から強い上向きの流れを注入し、障害物へ向かわせる
  auto src             = std::make_shared<SphereEmitter>();
  src->center          = {5.0f, 2.0f, 5.0f};
  src->radius          = 1.0f;
  src->inflowVelocity  = {0.0f, 6.0f, 0.0f};
  src->temperatureRate = 6.0f;
  src->step_count      = 0;
  engine.addEmitter(src);

  const float dt = 1.0f / 60.0f;
  for(int i = 0; i < 15; i++) {
    VkCommandBuffer cmd = ctx.beginCmd();
    engine.step(cmd, dt);
    ctx.submitCmd(cmd);
  }

  // 障害物中心セルの速度は常にゼロ化されているはず
  glm::vec4 v{};
  ctx.readBuffer(engine.getVelocityBuffer(), engine.cellIndex(G / 2, G / 2, G / 2) * sizeof(glm::vec4), &v, sizeof(v));
  CHECK(glm::length(glm::vec3(v)) == doctest::Approx(0.0f).epsilon(1e-5));

  engine.cleanup();
  ctx.cleanup();
}

// ═══════════════════════════════════════════════════════════════════════════
// 4: Emitter放出でdensity/fuelが対象領域内のみ増加する
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Pyro GPU - 4-1: Emitter emission increases density only near emitter") {
  HeadlessCtx ctx;
  ctx.init();

  PyroEngine engine;
  engine.domainSize = glm::vec3(10.0f);
  engine.cellSize   = 10.0f / 16.0f;
  engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS);
  engine.pc_.buoyancyAlpha = 0.0f;
  engine.pc_.buoyancyBeta  = 0.0f;
  engine.pc_.vorticityEps  = 0.0f;

  auto src         = std::make_shared<SphereEmitter>();
  src->center      = {2.0f, 2.0f, 2.0f};
  src->radius      = 1.0f;
  src->densityRate = 5.0f;
  src->step_count  = 0;
  engine.addEmitter(src);

  VkCommandBuffer cmd = ctx.beginCmd();
  engine.step(cmd, 1.0f / 60.0f);
  ctx.submitCmd(cmd);

  float densNear = 0.0f, densFar = 0.0f;
  ctx.readBuffer(engine.getDensityBuffer(), engine.cellIndex(3, 3, 3) * sizeof(float), &densNear, sizeof(float));
  ctx.readBuffer(engine.getDensityBuffer(), engine.cellIndex(14, 14, 14) * sizeof(float), &densFar, sizeof(float));

  CHECK(densNear > 0.0f);
  CHECK(densFar == doctest::Approx(0.0f).epsilon(1e-6));

  engine.cleanup();
  ctx.cleanup();
}

// ═══════════════════════════════════════════════════════════════════════════
// 5: 燃焼 — fuel減少・temperature上昇・flame加算が発生する
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Pyro GPU - 5-1: Combustion consumes fuel, raises temperature, emits flame") {
  HeadlessCtx ctx;
  ctx.init();

  PyroEngine engine;
  engine.domainSize = glm::vec3(10.0f);
  engine.cellSize   = 10.0f / 16.0f;
  engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS);
  engine.pc_.buoyancyAlpha      = 0.0f;
  engine.pc_.buoyancyBeta       = 0.0f;
  engine.pc_.vorticityEps       = 0.0f;
  engine.pc_.densityDissipation = 0.0f;
  engine.pc_.tempDissipation    = 0.0f;
  engine.pc_.ignitionTemp       = 0.3f;
  engine.pc_.burnRate           = 3.0f;
  engine.pc_.heatRelease        = 5.0f;
  engine.pc_.smokeYieldPerFuel  = 1.0f;
  engine.pc_.flameBrightness    = 2.0f;

  auto src             = std::make_shared<SphereEmitter>();
  src->center          = {5.0f, 5.0f, 5.0f};
  src->radius          = 1.0f;
  src->fuelRate        = 10.0f;
  src->temperatureRate = 10.0f;
  src->step_count      = 3; // 3フレームだけ放出、その後は燃焼のみ進行
  engine.addEmitter(src);

  const uint32_t G    = engine.gridRes().x;
  const uint32_t code = engine.cellIndex(G / 2, G / 2, G / 2);
  const float dt      = 1.0f / 60.0f;

  for(int i = 0; i < 3; i++) {
    VkCommandBuffer cmd = ctx.beginCmd();
    engine.step(cmd, dt);
    ctx.submitCmd(cmd);
  }
  float fuelAfterEmit = 0.0f;
  ctx.readBuffer(engine.getFuelBuffer(), code * sizeof(float), &fuelAfterEmit, sizeof(float));

  for(int i = 0; i < 15; i++) {
    VkCommandBuffer cmd = ctx.beginCmd();
    engine.step(cmd, dt);
    ctx.submitCmd(cmd);
  }
  float fuelLater = 0.0f, temp = 0.0f, flame = 0.0f;
  ctx.readBuffer(engine.getFuelBuffer(), code * sizeof(float), &fuelLater, sizeof(float));
  ctx.readBuffer(engine.getTemperatureBuffer(), code * sizeof(float), &temp, sizeof(float));
  ctx.readBuffer(engine.getFlameBuffer(), code * sizeof(float), &flame, sizeof(float));

  CHECK(fuelAfterEmit > 0.0f);
  CHECK(fuelLater < fuelAfterEmit);
  CHECK(temp > engine.pc_.ignitionTemp);
  CHECK(flame > 0.0f);

  engine.cleanup();
  ctx.cleanup();
}

// ═══════════════════════════════════════════════════════════════════════════
// 6: 直方体(非立方体)ドメイン — 解像度導出とGPUシミュレーションの安定性
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Pyro CPU - 6-1: Non-cubic domain derives per-axis gridRes correctly") {
  // gridRes()/totalCells() はホスト側計算のみで init() 不要
  PyroEngine engine;
  engine.domainSize = {20.0f, 4.0f, 20.0f};
  engine.cellSize   = 20.0f / 32.0f; // = 0.625

  const glm::uvec3 res = engine.gridRes();
  CHECK(res.x == 32);
  CHECK(res.y == 7); // ceil(4/0.625) = ceil(6.4) = 7
  CHECK(res.z == 32);
  // 線形配置: バッファ数 = 実セル数 (穴なし)
  CHECK(engine.totalCells() == res.x * res.y * res.z);
}

TEST_CASE("Pyro GPU - 6-2: Non-cubic domain simulation stays finite inside the real cell bounds") {
  HeadlessCtx ctx;
  ctx.init();

  PyroEngine engine;
  engine.domainSize = {20.0f, 4.0f, 20.0f}; // 薄いY軸ドメイン
  engine.cellSize   = 20.0f / 32.0f;
  engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS);
  engine.pc_.buoyancyAlpha = 2.0f;
  engine.numPressureIters  = 20;

  auto heat             = std::make_shared<SphereEmitter>();
  heat->center          = {10.0f, 2.0f, 10.0f};
  heat->radius          = 1.0f;
  heat->temperatureRate = 8.0f;
  heat->step_count      = 0;
  engine.addEmitter(heat);

  const float dt = 1.0f / 60.0f;
  for(int i = 0; i < 20; i++) {
    VkCommandBuffer cmd = ctx.beginCmd();
    engine.step(cmd, dt);
    ctx.submitCmd(cmd);
  }

  const glm::uvec3 realRes = engine.gridRes();
  const uint32_t NC        = engine.totalCells();
  std::vector<glm::vec4> vel(NC);
  ctx.readBuffer(engine.getVelocityBuffer(), 0, vel.data(), NC * sizeof(glm::vec4));

  // 実セル範囲内(パディング領域を除く)のみ検証: CLAMP_CELL の per-axis 化により、
  // 短い Y 軸の境界(y=realRes.y-1)付近でもパディング領域の未定義値が混入しないはず。
  bool allFinite = true;
  float maxAbsV  = 0.0f;
  for(uint32_t z = 0; z < realRes.z; z++)
    for(uint32_t y = 0; y < realRes.y; y++)
      for(uint32_t x = 0; x < realRes.x; x++) {
        glm::vec4 v = vel[engine.cellIndex(x, y, z)];
        if(!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) allFinite = false;
        maxAbsV = std::max({maxAbsV, std::abs(v.x), std::abs(v.y), std::abs(v.z)});
      }
  CHECK(allFinite);
  CHECK(maxAbsV < 1000.0f);

  engine.cleanup();
  ctx.cleanup();
}

// ═══════════════════════════════════════════════════════════════════════════
// 7: 回転する剛体コライダー (y 軸まわりの一定角速度) — SDF の焼き直しと壁面速度
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Pyro GPU - 7-1: Rotating collider moves with the angle and imposes wall velocity omega x r") {
  HeadlessCtx ctx;
  ctx.init();

  PyroEngine engine;
  engine.domainSize = {16.0f, 8.0f, 16.0f};
  engine.cellSize   = 0.25f;
  engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS);
  engine.numPressureIters = 10;

  // ローカル SDF: 回転中心から (2,3,0) にある半径 1 の球。節点間隔 0.1、範囲 [-4,4]x[0,6]x[-4,4]
  const glm::uvec3 lres(81, 61, 81);
  const float lcell = 0.1f;
  const glm::vec3 lmin(-4.0f, 0.0f, -4.0f);
  std::vector<float> local(size_t(lres.x) * lres.y * lres.z);
  for(uint32_t k = 0; k < lres.z; k++)
    for(uint32_t j = 0; j < lres.y; j++)
      for(uint32_t i = 0; i < lres.x; i++) local[size_t(i) + lres.x * (j + size_t(lres.y) * k)] = glm::length(lmin + glm::vec3(i, j, k) * lcell - glm::vec3(2.0f, 3.0f, 0.0f)) - 1.0f;

  const float omega = 1.0f, dt = 0.1f;
  PyroEngine::RigidBodyState body;
  body.position        = {8.0f, 0.0f, 8.0f};
  body.angularVelocity = {0.0f, omega, 0.0f}; // y 軸まわりの回転
  engine.setMovingCollider(local, lres, lcell, lmin, body);

  const int steps = 5;
  for(int i = 0; i < steps; i++) {
    VkCommandBuffer cmd = ctx.beginCmd();
    engine.step(cmd, dt);
    ctx.submitCmd(cmd);
  }
  CHECK(2.0f * std::atan2(engine.colliderState().rotation.y, engine.colliderState().rotation.w) == doctest::Approx(steps * omega * dt).epsilon(1e-4)); // 0.5 rad

  // 最後のステップは角度 (steps-1)*omega*dt = 0.4 で焼き直されている
  const float th = (steps - 1) * omega * dt;
  const glm::vec3 center(8.0f + 2.0f * std::cos(th), 3.0f, 8.0f - 2.0f * std::sin(th)); // R(th) * (2,0,0)
  const glm::ivec3 c = glm::ivec3(center / engine.cellSize);

  float sdfHere = 0.0f, sdfOld = 0.0f;
  ctx.readBuffer(engine.getColliderSDFBuffer(), engine.cellIndex(c.x, c.y, c.z) * sizeof(float), &sdfHere, sizeof(float));
  // 角度 0 の球 (中心 (10,3,8)) の内側 (中心から 0.9) だが、0.4 rad 回った球 (中心から 1.3) の外側になる点
  const glm::ivec3 c0 = glm::ivec3(glm::vec3(10.9f, 3.0f, 8.0f) / engine.cellSize);
  ctx.readBuffer(engine.getColliderSDFBuffer(), engine.cellIndex(c0.x, c0.y, c0.z) * sizeof(float), &sdfOld, sizeof(float));
  CHECK(sdfHere < -0.5f); // 球の中心付近 (半径 1)
  CHECK(sdfOld > 0.0f);   // 回転前の位置は外側になっている

  glm::vec4 v{};
  ctx.readBuffer(engine.getVelocityBuffer(), engine.cellIndex(c.x, c.y, c.z) * sizeof(glm::vec4), &v, sizeof(v));
  // 壁面速度 omega * (r.z, 0, -r.x)。r は回転軸からの水平変位 (2cos th, -2 sin th)
  CHECK(v.x == doctest::Approx(omega * (-2.0f * std::sin(th))).epsilon(0.2));
  CHECK(v.z == doctest::Approx(omega * (-2.0f * std::cos(th))).epsilon(0.2));
  CHECK(std::abs(v.y) < 0.3f);

  engine.cleanup();
  ctx.cleanup();
}

// ═══════════════════════════════════════════════════════════════════════════
// 8: チャンネルの取捨 (メモリ節約) — 速度場のみで動く
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Pyro GPU - 8-1: velocity-only channels give the same velocity field as all channels (no buoyancy)") {
  auto run = [](uint32_t channels, std::vector<glm::vec4>& vel) {
    HeadlessCtx ctx;
    ctx.init();
    PyroEngine engine;
    engine.domainSize = {8.0f, 4.0f, 4.0f};
    engine.cellSize   = 0.25f;
    engine.channels   = channels;
    engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS);
    engine.pc_.buoyancyAlpha = 0.0f;
    engine.pc_.buoyancyBeta  = 0.0f;
    engine.numPressureIters  = 20;
    auto fan            = std::make_shared<AABBEmitter>();
    fan->center         = {1.0f, 2.0f, 2.0f};
    fan->size           = {1.0f, 3.0f, 3.0f};
    fan->inflowVelocity = {5.0f, 0.0f, 0.0f};
    engine.addEmitter(fan);
    for(int i = 0; i < 10; i++) {
      VkCommandBuffer cmd = ctx.beginCmd();
      engine.step(cmd, 0.05f);
      ctx.submitCmd(cmd);
    }
    vel.resize(engine.totalCells());
    ctx.readBuffer(engine.getVelocityBuffer(), 0, vel.data(), vel.size() * sizeof(glm::vec4));
    engine.cleanup();
    ctx.cleanup();
  };
  std::vector<glm::vec4> a, b;
  run(PyroEngine::kChAll, a);
  run(PyroEngine::kChVelocityOnly, b);
  REQUIRE(a.size() == b.size());
  float maxDiff = 0.0f, maxV = 0.0f;
  for(size_t i = 0; i < a.size(); i++) {
    maxDiff = std::max(maxDiff, glm::length(glm::vec3(a[i]) - glm::vec3(b[i])));
    maxV    = std::max(maxV, glm::length(glm::vec3(a[i])));
  }
  CHECK(maxV > 0.1f); // 風が実際に流れている
  CHECK(maxDiff < 1e-4f);
}

TEST_CASE("Pyro GPU - 7-2: Moving collider with arbitrary rotation axis and translation") {
  HeadlessCtx ctx;
  ctx.init();

  PyroEngine engine;
  engine.domainSize = {16.0f, 12.0f, 16.0f};
  engine.cellSize   = 0.25f;
  engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS);
  engine.numPressureIters = 10;

  // ローカル SDF: 原点から (2,0,0) にある半径 0.8 の球。節点間隔 0.1、範囲 [-4,4]^3
  const glm::uvec3 lres(81, 81, 81);
  const float lcell = 0.1f;
  const glm::vec3 lmin(-4.0f);
  std::vector<float> local(size_t(lres.x) * lres.y * lres.z);
  for(uint32_t k = 0; k < lres.z; k++)
    for(uint32_t j = 0; j < lres.y; j++)
      for(uint32_t i = 0; i < lres.x; i++) local[size_t(i) + lres.x * (j + size_t(lres.y) * k)] = glm::length(lmin + glm::vec3(i, j, k) * lcell - glm::vec3(2.0f, 0.0f, 0.0f)) - 0.8f;

  // 斜めの軸 (1,0,1)/sqrt2 まわりに 1 rad/s で回しつつ、(0.5, 0, 0) m/s で並進する
  PyroEngine::RigidBodyState s;
  s.position        = {6.0f, 5.0f, 8.0f};
  s.angularVelocity = glm::normalize(glm::vec3(1.0f, 0.0f, 1.0f)) * 1.0f;
  s.linearVelocity  = {0.5f, 0.0f, 0.0f};
  engine.setMovingCollider(local, lres, lcell, lmin, s, true);

  const float dt = 0.1f;
  const int steps = 6;
  for(int i = 0; i < steps; i++) {
    VkCommandBuffer cmd = ctx.beginCmd();
    engine.step(cmd, dt);
    ctx.submitCmd(cmd);
  }
  // 最後のステップは (steps-1) 回積分した姿勢・位置で焼き直されている
  const float t = (steps - 1) * dt;
  const glm::vec3 axis = glm::normalize(glm::vec3(1.0f, 0.0f, 1.0f));
  const glm::quat q    = glm::angleAxis(1.0f * t, axis);
  const glm::vec3 pos  = s.position + s.linearVelocity * t;
  const glm::vec3 center = pos + q * glm::vec3(2.0f, 0.0f, 0.0f); // 球の中心の世界座標
  const glm::ivec3 c = glm::ivec3(center / engine.cellSize);

  float sdfHere = 0.0f;
  ctx.readBuffer(engine.getColliderSDFBuffer(), engine.cellIndex(c.x, c.y, c.z) * sizeof(float), &sdfHere, sizeof(float));
  CHECK(sdfHere < -0.4f);

  // 壁面速度 = linVel + omega x (p - pos)。セル中心での値と比べる
  glm::vec4 v{};
  ctx.readBuffer(engine.getVelocityBuffer(), engine.cellIndex(c.x, c.y, c.z) * sizeof(glm::vec4), &v, sizeof(v));
  const glm::vec3 pc = (glm::vec3(c) + 0.5f) * engine.cellSize;
  const glm::vec3 expect = s.linearVelocity + glm::cross(s.angularVelocity, pc - pos);
  CHECK(v.x == doctest::Approx(expect.x).epsilon(0.05).scale(1.0));
  CHECK(v.y == doctest::Approx(expect.y).epsilon(0.05).scale(1.0));
  CHECK(v.z == doctest::Approx(expect.z).epsilon(0.05).scale(1.0));
  CHECK(std::abs(expect.y) > 0.3f); // 斜め軸なので y 成分も出る (y 軸まわりだけの実装では 0)

  // 回転後の姿勢 (クォータニオン) と位置が積分されている
  const auto& st = engine.colliderState();
  CHECK(st.position.x == doctest::Approx(s.position.x + 0.5f * steps * dt).epsilon(1e-4));
  CHECK(glm::length(st.rotation) == doctest::Approx(1.0f).epsilon(1e-4));

  engine.cleanup();
  ctx.cleanup();
}
