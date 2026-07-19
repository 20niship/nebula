// 泡 (spray/foam/bubble) 二次パーティクル GPU テスト (issue #47)
// FluidEngine を直接操作する HeadlessCtx パターン (test_pbf_fluid.cpp の TC9/TC10 と同型)。
#include "FluidEngine.h"
#include "core/Emitter.h"
#include "helpers/HeadlessCtx.h"
#include <doctest/doctest.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>

static const std::string SHADERS = SHADER_DIR;

// 全 kind スロットを読み戻し、非ゼロ(生存)スロット数を数える
static uint32_t countAliveFoam(const HeadlessCtx& ctx, FluidEngine& engine, uint32_t maxDiffuseParticles) {
  std::vector<uint32_t> kinds(maxDiffuseParticles);
  ctx.readBuffer(engine.getFoamKindBuffer(), 0, kinds.data(), sizeof(uint32_t) * maxDiffuseParticles);
  uint32_t alive = 0;
  for(uint32_t k : kinds)
    if(k != 0u) alive++;
  return alive;
}

// ── TC-Foam1: 容量確保 ───────────────────────────────────────────────────────
// maxDiffuseParticles>0 のときのみ foam バッファが確保され、初期状態は全スロット死(kind=0)。
TEST_CASE("PBF Foam - buffer allocation") {
  HeadlessCtx ctx;
  ctx.init();

  FluidConfig cfg;
  cfg.fluid_nx            = 4;
  cfg.fluid_ny             = 1;
  cfg.fluid_nz             = 4;
  cfg.domainSize           = glm::vec3(10.0f);
  cfg.cellSize             = 10.0f / 4.0f;
  cfg.max_boundary         = 0;
  cfg.maxDiffuseParticles  = 1000;

  FluidEngine engine;
  engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS, cfg);

  CHECK(countAliveFoam(ctx, engine, 1000) == 0);

  engine.cleanup();
  ctx.cleanup();
}

// ── TC-Foam2: 生成が発生する ─────────────────────────────────────────────────
// 高速な流体塊を床へ衝突させ、数フレーム後に foamKind に非ゼロスロットが出現すること。
TEST_CASE("PBF Foam - generation occurs on high-speed impact") {
  HeadlessCtx ctx;
  ctx.init();

  FluidConfig cfg;
  cfg.fluid_nx            = 12;
  cfg.fluid_ny             = 6;
  cfg.fluid_nz             = 12;
  cfg.domainSize           = glm::vec3(10.0f);
  cfg.cellSize             = 10.0f / 16.0f;
  cfg.max_boundary         = 0;
  cfg.maxDiffuseParticles  = 20000;

  FluidEngine engine;
  engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS, cfg);
  engine.foamEnabled = true;
  engine.numSubsteps = 2;
  engine.pbfIterations = 2;

  // 既定 FoamParams はそのまま（十分な生成係数）で有効化
  engine.setFoamParams(FluidEngine::FoamParams{});

  auto gravity = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 9.8f);
  engine.addForce(gravity);

  // 塊を床(worldMin付近)へ向けて高速に打ち込む
  auto src                = std::make_shared<AABBEmitter>();
  src->center             = glm::vec3(5.0f, 5.0f, 8.0f);
  src->size               = glm::vec3(3.0f, 3.0f, 1.5f);
  src->vel                = glm::vec3(0.0f, 0.0f, -15.0f); // 高速下向き初速
  src->particles_per_step = cfg.fluidCount();
  src->step_count         = -1;
  engine.addEmitter(src);

  const float dt = 1.0f / 60.0f;
  for(int f = 0; f < 90; ++f) {
    engine.emitFromEmitters(dt);
    VkCommandBuffer cmd = ctx.beginCmd();
    engine.step(cmd, dt);
    ctx.submitCmd(cmd);
  }

  uint32_t alive = countAliveFoam(ctx, engine, cfg.maxDiffuseParticles);
  CHECK(alive > 0u);

  engine.cleanup();
  ctx.cleanup();
}

// ── TC-Foam3: 生成抑制の回帰 (kTa=kWc=0) ─────────────────────────────────────
// 生成係数を0にすると、同じ衝突シナリオでも一切生成されない。
TEST_CASE("PBF Foam - no generation when kTa=kWc=0") {
  HeadlessCtx ctx;
  ctx.init();

  FluidConfig cfg;
  cfg.fluid_nx            = 12;
  cfg.fluid_ny             = 6;
  cfg.fluid_nz             = 12;
  cfg.domainSize           = glm::vec3(10.0f);
  cfg.cellSize             = 10.0f / 16.0f;
  cfg.max_boundary         = 0;
  cfg.maxDiffuseParticles  = 20000;

  FluidEngine engine;
  engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS, cfg);
  engine.foamEnabled = true;

  FluidEngine::FoamParams p{};
  p.kTa = 0.0f;
  p.kWc = 0.0f;
  engine.setFoamParams(p);

  auto gravity = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 9.8f);
  engine.addForce(gravity);

  auto src                = std::make_shared<AABBEmitter>();
  src->center             = glm::vec3(5.0f, 5.0f, 8.0f);
  src->size               = glm::vec3(3.0f, 3.0f, 1.5f);
  src->vel                = glm::vec3(0.0f, 0.0f, -15.0f);
  src->particles_per_step = cfg.fluidCount();
  src->step_count         = -1;
  engine.addEmitter(src);

  const float dt = 1.0f / 60.0f;
  for(int f = 0; f < 90; ++f) {
    engine.emitFromEmitters(dt);
    VkCommandBuffer cmd = ctx.beginCmd();
    engine.step(cmd, dt);
    ctx.submitCmd(cmd);
  }

  CHECK(countAliveFoam(ctx, engine, cfg.maxDiffuseParticles) == 0u);

  engine.cleanup();
  ctx.cleanup();
}

// ── TC-Foam4: 寿命管理 ───────────────────────────────────────────────────────
// 手動で寿命の短い泡粒子を1個仕込み、1step後に kind==0 (死) へ戻ること。
TEST_CASE("PBF Foam - lifetime expiry returns slot to graveyard") {
  HeadlessCtx ctx;
  ctx.init();

  FluidConfig cfg;
  cfg.fluid_nx            = 2;
  cfg.fluid_ny             = 1;
  cfg.fluid_nz             = 2;
  cfg.domainSize           = glm::vec3(10.0f);
  cfg.cellSize             = 10.0f / 4.0f;
  cfg.max_boundary         = 1; // 流体粒子ゼロでも step() 本体を実行させるためのダミー境界粒子用
  cfg.maxDiffuseParticles  = 16;

  FluidEngine engine;
  engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS, cfg);
  engine.foamEnabled = true;
  engine.setFoamParams(FluidEngine::FoamParams{});

  // 流体粒子は無し。slot 0 に寿命 0.01s の spray 粒子を直接仕込む。
  const float shortLife = 0.01f;
  engine.debugSetFoamSlot(0, glm::vec4(5.0f, 5.0f, 5.0f, shortLife), glm::vec4(0.0f, 0.0f, 0.0f, shortLife), 1u);

  // 流体粒子ゼロだと FluidEngine::step() が早期returnするため、境界粒子を1つ置いて
  // step() 本体（advectパスを含む）が実行されるようにする。
  std::vector<glm::vec4> boundaryPts = {glm::vec4(0.5f, 0.5f, 0.5f, 1.0f)};
  engine.loadBoundaryParticles(boundaryPts);

  const float dt = 1.0f / 60.0f; // > shortLife なので1stepで寿命切れになるはず
  VkCommandBuffer cmd = ctx.beginCmd();
  engine.step(cmd, dt);
  ctx.submitCmd(cmd);

  std::vector<uint32_t> kinds(cfg.maxDiffuseParticles);
  ctx.readBuffer(engine.getFoamKindBuffer(), 0, kinds.data(), sizeof(uint32_t) * cfg.maxDiffuseParticles);
  CHECK(kinds[0] == 0u);

  engine.cleanup();
  ctx.cleanup();
}

// ── TC-Foam5: 既存回帰 (foamEnabled=false で影響なし) ────────────────────────
// foamEnabled=false かつ maxDiffuseParticles>0 でも、既存の重力落下・床貫通なし
// という基本挙動 (TC1相当) が変わらないこと。
TEST_CASE("PBF Foam - disabled foam does not affect base fluid behavior") {
  HeadlessCtx ctx;
  ctx.init();

  FluidConfig cfg;
  cfg.fluid_nx            = 4;
  cfg.fluid_ny             = 4;
  cfg.fluid_nz             = 4;
  cfg.domainSize           = glm::vec3(10.0f);
  cfg.cellSize             = 10.0f / 8.0f;
  cfg.max_boundary         = 0;
  cfg.maxDiffuseParticles  = 5000; // バッファは確保されるが foamEnabled=false のまま

  FluidEngine engine;
  engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS, cfg);
  CHECK_FALSE(engine.foamEnabled); // 既定値

  auto gravity = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 9.8f);
  engine.addForce(gravity);

  auto src                = std::make_shared<AABBEmitter>();
  src->center             = glm::vec3(5.0f, 5.0f, 8.0f);
  src->size               = glm::vec3(2.0f, 2.0f, 2.0f);
  src->particles_per_step = cfg.fluidCount();
  src->step_count         = -1;
  engine.addEmitter(src);

  const float dt = 1.0f / 60.0f;
  for(int f = 0; f < 60; ++f) {
    engine.emitFromEmitters(dt);
    VkCommandBuffer cmd = ctx.beginCmd();
    engine.step(cmd, dt);
    ctx.submitCmd(cmd);
  }

  const uint32_t N = engine.nFluid();
  std::vector<glm::vec4> pos(N);
  ctx.readBuffer(engine.getPositionBuffer(), 0, pos.data(), sizeof(glm::vec4) * N);
  for(uint32_t i = 0; i < N; ++i) {
    CHECK(pos[i].z > 0.0f); // 床(worldMin=0)を貫通していない
    CHECK(pos[i].z < 8.0f);                // 落下している
  }
  CHECK(countAliveFoam(ctx, engine, cfg.maxDiffuseParticles) == 0u); // 無効時は生成もされない

  engine.cleanup();
  ctx.cleanup();
}
