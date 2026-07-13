// Pyro (グリッドベース煙・火炎) 単体テスト
// PyroEngine の公開 API (init/step/addEmitter/setColliderSDF/getXxxBuffer) が
// HeadlessCtx 上でそのままテストハーネスとして使えるため、専用 Harness は作らない。
#include <doctest/doctest.h>
#include "helpers/HeadlessCtx.h"
#include "engine/PyroEngine.h"
#include "core/MeshSDF.h" // meshSdfMortonEncode を Morton インデックス計算に再利用

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
    HeadlessCtx ctx; ctx.init();

    PyroConfig cfg; cfg.grid_res = 16; cfg.world_size = 10.0f;
    PyroEngine engine;
    engine.init(ctx.device, ctx.allocator, ctx.descriptorPool,
                ctx.commandPool, ctx.computeQueue, SHADERS, cfg);
    engine.buoyancyAlpha  = 2.0f;
    engine.buoyancyBeta   = 0.0f;
    engine.vorticityEps   = 0.0f;
    engine.numJacobiIters = 20;

    auto heat = std::make_shared<SphereEmitter>();
    heat->center          = {5.0f, 5.0f, 5.0f};
    heat->radius          = 1.5f;
    heat->temperatureRate = 8.0f;
    heat->step_count      = 0;
    engine.addEmitter(heat);

    const float dt = 1.0f / 60.0f;
    for (int i = 0; i < 20; i++) {
        VkCommandBuffer cmd = ctx.beginCmd();
        engine.step(cmd, dt);
        ctx.submitCmd(cmd);
    }

    const uint32_t NC = cfg.totalCells();
    std::vector<glm::vec4> vel(NC);
    ctx.readBuffer(engine.getVelocityBuffer(), 0, vel.data(), NC * sizeof(glm::vec4));

    double sumVy = 0.0;
    for (auto& v : vel) {
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
    HeadlessCtx ctx; ctx.init();

    PyroConfig cfg; cfg.grid_res = 16; cfg.world_size = 10.0f;
    PyroEngine engine;
    engine.init(ctx.device, ctx.allocator, ctx.descriptorPool,
                ctx.commandPool, ctx.computeQueue, SHADERS, cfg);
    engine.buoyancyAlpha  = 3.0f;
    engine.numJacobiIters = 60;

    auto src = std::make_shared<SphereEmitter>();
    src->center          = {5.0f, 2.0f, 5.0f};
    src->radius          = 1.5f;
    src->temperatureRate = 6.0f;
    src->step_count      = 0;
    engine.addEmitter(src);

    const float dt = 1.0f / 60.0f;
    for (int i = 0; i < 30; i++) {
        VkCommandBuffer cmd = ctx.beginCmd();
        engine.step(cmd, dt);
        ctx.submitCmd(cmd);
    }

    const uint32_t G  = cfg.grid_res;
    const uint32_t NC = cfg.totalCells();
    std::vector<glm::vec4> velMorton(NC);
    ctx.readBuffer(engine.getVelocityBuffer(), 0, velMorton.data(), NC * sizeof(glm::vec4));

    auto sampleVel = [&](int x, int y, int z) -> glm::vec3 {
        x = std::clamp(x, 0, int(G) - 1);
        y = std::clamp(y, 0, int(G) - 1);
        z = std::clamp(z, 0, int(G) - 1);
        return glm::vec3(velMorton[meshSdfMortonEncode(uint32_t(x), uint32_t(y), uint32_t(z))]);
    };

    const float h      = cfg.cellSize();
    float       maxDiv = 0.0f;
    // 境界・source近傍を避けた内部領域のみチェック
    for (int x = 4; x < 12; x++)
    for (int y = 4; y < 12; y++)
    for (int z = 4; z < 12; z++) {
        float div = 0.5f * ((sampleVel(x + 1, y, z).x - sampleVel(x - 1, y, z).x) +
                             (sampleVel(x, y + 1, z).y - sampleVel(x, y - 1, z).y) +
                             (sampleVel(x, y, z + 1).z - sampleVel(x, y, z - 1).z)) / h;
        maxDiv = std::max(maxDiv, std::abs(div));
    }
    // 投影が機能していれば内部領域の発散はおおむね小さく収まる
    // (Jacobi 反復の粗い近似解のため厳密な0ではなく、桁違いの破綻がないことを確認する)
    CHECK(maxDiv < 1.0f);

    engine.cleanup();
    ctx.cleanup();
}

// ═══════════════════════════════════════════════════════════════════════════
// 3: 障害物SDF内部で速度がゼロ化される
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Pyro GPU - 3-1: Obstacle SDF zeroes velocity inside solid") {
    HeadlessCtx ctx; ctx.init();

    PyroConfig cfg; cfg.grid_res = 16; cfg.world_size = 10.0f;
    PyroEngine engine;
    engine.init(ctx.device, ctx.allocator, ctx.descriptorPool,
                ctx.commandPool, ctx.computeQueue, SHADERS, cfg);
    engine.buoyancyAlpha = 3.0f;

    const uint32_t G       = cfg.grid_res;
    const float    h       = cfg.cellSize();
    const glm::vec3 center = {5.0f, 5.0f, 5.0f};
    const float     radius = 2.0f;

    std::vector<float> sdf(cfg.totalCells());
    for (uint32_t z = 0; z < G; z++)
    for (uint32_t y = 0; y < G; y++)
    for (uint32_t x = 0; x < G; x++) {
        glm::vec3 p = (glm::vec3(x, y, z) + 0.5f) * h;
        sdf[meshSdfMortonEncode(x, y, z)] = glm::length(p - center) - radius;
    }
    engine.setColliderSDF(sdf);

    // 障害物のすぐ下から強い上向きの流れを注入し、障害物へ向かわせる
    auto src = std::make_shared<SphereEmitter>();
    src->center          = {5.0f, 2.0f, 5.0f};
    src->radius          = 1.0f;
    src->inflowVelocity  = {0.0f, 6.0f, 0.0f};
    src->temperatureRate = 6.0f;
    src->step_count      = 0;
    engine.addEmitter(src);

    const float dt = 1.0f / 60.0f;
    for (int i = 0; i < 15; i++) {
        VkCommandBuffer cmd = ctx.beginCmd();
        engine.step(cmd, dt);
        ctx.submitCmd(cmd);
    }

    // 障害物中心セルの速度は常にゼロ化されているはず
    glm::vec4 v{};
    ctx.readBuffer(engine.getVelocityBuffer(),
                   meshSdfMortonEncode(G / 2, G / 2, G / 2) * sizeof(glm::vec4), &v, sizeof(v));
    CHECK(glm::length(glm::vec3(v)) == doctest::Approx(0.0f).epsilon(1e-5));

    engine.cleanup();
    ctx.cleanup();
}

// ═══════════════════════════════════════════════════════════════════════════
// 4: Emitter放出でdensity/fuelが対象領域内のみ増加する
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Pyro GPU - 4-1: Emitter emission increases density only near emitter") {
    HeadlessCtx ctx; ctx.init();

    PyroConfig cfg; cfg.grid_res = 16; cfg.world_size = 10.0f;
    PyroEngine engine;
    engine.init(ctx.device, ctx.allocator, ctx.descriptorPool,
                ctx.commandPool, ctx.computeQueue, SHADERS, cfg);
    engine.buoyancyAlpha = 0.0f;
    engine.buoyancyBeta  = 0.0f;
    engine.vorticityEps  = 0.0f;

    auto src = std::make_shared<SphereEmitter>();
    src->center      = {2.0f, 2.0f, 2.0f};
    src->radius      = 1.0f;
    src->densityRate = 5.0f;
    src->step_count  = 0;
    engine.addEmitter(src);

    VkCommandBuffer cmd = ctx.beginCmd();
    engine.step(cmd, 1.0f / 60.0f);
    ctx.submitCmd(cmd);

    float densNear = 0.0f, densFar = 0.0f;
    ctx.readBuffer(engine.getDensityBuffer(), meshSdfMortonEncode(3, 3, 3) * sizeof(float),
                   &densNear, sizeof(float));
    ctx.readBuffer(engine.getDensityBuffer(), meshSdfMortonEncode(14, 14, 14) * sizeof(float),
                   &densFar, sizeof(float));

    CHECK(densNear > 0.0f);
    CHECK(densFar == doctest::Approx(0.0f).epsilon(1e-6));

    engine.cleanup();
    ctx.cleanup();
}

// ═══════════════════════════════════════════════════════════════════════════
// 5: 燃焼 — fuel減少・temperature上昇・flame加算が発生する
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("Pyro GPU - 5-1: Combustion consumes fuel, raises temperature, emits flame") {
    HeadlessCtx ctx; ctx.init();

    PyroConfig cfg; cfg.grid_res = 16; cfg.world_size = 10.0f;
    PyroEngine engine;
    engine.init(ctx.device, ctx.allocator, ctx.descriptorPool,
                ctx.commandPool, ctx.computeQueue, SHADERS, cfg);
    engine.buoyancyAlpha      = 0.0f;
    engine.buoyancyBeta       = 0.0f;
    engine.vorticityEps       = 0.0f;
    engine.densityDissipation = 0.0f;
    engine.tempDissipation    = 0.0f;
    engine.ignitionTemp       = 0.3f;
    engine.burnRate           = 3.0f;
    engine.heatRelease        = 5.0f;
    engine.smokeYieldPerFuel  = 1.0f;
    engine.flameBrightness    = 2.0f;

    auto src = std::make_shared<SphereEmitter>();
    src->center          = {5.0f, 5.0f, 5.0f};
    src->radius          = 1.0f;
    src->fuelRate        = 10.0f;
    src->temperatureRate = 10.0f;
    src->step_count      = 3; // 3フレームだけ放出、その後は燃焼のみ進行
    engine.addEmitter(src);

    const uint32_t G    = cfg.grid_res;
    const uint32_t code = meshSdfMortonEncode(G / 2, G / 2, G / 2);
    const float    dt   = 1.0f / 60.0f;

    for (int i = 0; i < 3; i++) {
        VkCommandBuffer cmd = ctx.beginCmd();
        engine.step(cmd, dt);
        ctx.submitCmd(cmd);
    }
    float fuelAfterEmit = 0.0f;
    ctx.readBuffer(engine.getFuelBuffer(), code * sizeof(float), &fuelAfterEmit, sizeof(float));

    for (int i = 0; i < 15; i++) {
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
    CHECK(temp > engine.ignitionTemp);
    CHECK(flame > 0.0f);

    engine.cleanup();
    ctx.cleanup();
}
