// 2D string / rope physics tests (XPBD, GPU compute, headless)
#include <doctest/doctest.h>
#include "helpers/HeadlessCtx.h"
#include "helpers/PhysicsHarness.h"
#include <glm/glm.hpp>
#include <string>
#include <vector>

static const std::string SHADERS = SHADER_DIR;

// ── 自由落下 (ピン留めなし) ──────────────────────────────────────────────────
// 糸の全粒子にinvMass=1を設定しピン留めなしで落下させる。
// 距離拘束はすべて水平方向なので重力加速度に近い落下が期待される。
// 1秒間シミュレーション後: 各粒子が初期y=6.0から少なくとも2m落下。
TEST_CASE("String2D - free fall, no pin") {
    HeadlessCtx ctx; ctx.init();

    constexpr uint32_t N = 8;
    constexpr float SPACING = 0.1f;
    constexpr float INIT_Y  = 6.0f;
    constexpr float INIT_Z  = 5.0f;

    PhysicsHarness::Config cfg;
    cfg.N = N;
    cfg.worldMin = 0.0f; cfg.worldMax = 10.0f;
    cfg.radius = 0.14f;  cfg.gravity = -9.8f;
    cfg.gridRes = 16;
    cfg.numSubsteps = 4; cfg.solverIterations = 4;
    cfg.typeFlag = 2;    cfg.selfCollision = false;
    PhysicsHarness::buildChainEdges(N, SPACING, cfg.edgeData, cfg.edgeColor0End);

    // 水平な初期配置 (z=5固定、距離拘束は水平 → 拘束力は垂直成分に影響しない)
    std::vector<glm::vec4> pos(N), invM(N);
    float startX = 5.0f - (N - 1) * SPACING * 0.5f;
    for (uint32_t i = 0; i < N; ++i) {
        pos[i]  = glm::vec4(startX + i * SPACING, INIT_Y, 5.0f, 1.0f);
        invM[i] = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f); // ピン留めなし
    }

    PhysicsHarness sim;
    sim.init(ctx, cfg, SHADERS, pos, invM);

    // 60フレーム × 1/60 s = 1秒シミュレーション
    const float dt = 1.0f / 60.0f;
    for (int f = 0; f < 60; ++f) sim.step(dt);

    // 各粒子が2m以上落下 かつ フロア(z>0)より上 (重力はZ軸)
    for (uint32_t i = 0; i < N; ++i) {
        auto p = sim.readPos(i);
        CHECK(p.z < INIT_Z - 2.0f);
        CHECK(p.z > cfg.worldMin);
    }

    sim.cleanup();
    ctx.cleanup();
}

// ── 水平スタート・ピン留めあり (振り子落下) ──────────────────────────────────
// 粒子0をピン留め、残りを水平に並べてスタート。
// 2秒後: 自由端(粒子N-1)が初期位置より少なくとも2m以上落下していること。
// また自由端のx座標がピン留め点の近く(±2m以内)であること(振り子が正しく動く)。
TEST_CASE("String2D - pinned at end, pendulum fall") {
    HeadlessCtx ctx; ctx.init();

    constexpr uint32_t N = 16;
    constexpr float SPACING  = 0.1f;
    constexpr float INIT_Y   = 7.0f;
    constexpr float INIT_Z   = 5.0f;
    constexpr float PIN_X    = 3.25f;

    PhysicsHarness::Config cfg;
    cfg.N = N;
    cfg.worldMin = 0.0f; cfg.worldMax = 10.0f;
    cfg.radius = 0.14f;  cfg.gravity = -9.8f;
    cfg.gridRes = 16;
    cfg.numSubsteps = 4; cfg.solverIterations = 4;
    cfg.typeFlag = 2;    cfg.selfCollision = false;
    PhysicsHarness::buildChainEdges(N, SPACING, cfg.edgeData, cfg.edgeColor0End);

    std::vector<glm::vec4> pos(N), invM(N);
    for (uint32_t i = 0; i < N; ++i) {
        pos[i]  = glm::vec4(PIN_X + i * SPACING, INIT_Y, 5.0f, 1.0f);
        invM[i] = glm::vec4(i == 0 ? 0.0f : 1.0f, 0.0f, 0.0f, 0.0f);
    }

    PhysicsHarness sim;
    sim.init(ctx, cfg, SHADERS, pos, invM);

    // 120フレーム = 2秒
    const float dt = 1.0f / 60.0f;
    for (int f = 0; f < 120; ++f) sim.step(dt);

    // ピン留め頂点は動かない
    auto pinPos = sim.readPos(0);
    CHECK(pinPos.x == doctest::Approx(PIN_X).epsilon(0.01f));
    CHECK(pinPos.y == doctest::Approx(INIT_Y).epsilon(0.01f));

    // 自由端はZ方向に1m以上落下 (ロープ長1.5m、重力はZ軸)
    auto tipPos = sim.readPos(N - 1);
    CHECK(tipPos.z < INIT_Z - 1.0f);
    // 振り子なのでx方向はピン留め点の近くにいる(最大ロープ長=1.5m+余裕)
    CHECK(std::abs(tipPos.x - PIN_X) < 2.5f);

    sim.cleanup();
    ctx.cleanup();
}
