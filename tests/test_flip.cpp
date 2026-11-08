// FLIPEngine GPU テスト: 圧力投影による非圧縮性 (静水プールの体積保存)・コライダー貫通なし・泡生成。
#include "FLIPEngine.h"
#include "../examples/FluidStats.h"
#include "core/Collider.h"
#include "core/Emitter.h"
#include "helpers/HeadlessCtx.h"
#include <algorithm>
#include <cmath>
#include <doctest/doctest.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>

static const std::string SHADERS = SHADER_DIR;

namespace {

struct FlipScene {
  HeadlessCtx ctx;
  FLIPEngine engine;
  float dt = 1.0f / 60.0f;

  // size [m] の立方体ドメイン (gridRes=size/cell は 2 のべき乗)
  void init(float size, float cell, uint32_t maxParticles, uint32_t maxDiffuse = 0) {
    ctx.init();
    engine.domainSize           = glm::vec3(size);
    engine.cellSize             = cell;
    engine.maxParticles         = maxParticles;
    engine.maxDiffuseParticles  = maxDiffuse;
    engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS);
    engine.addForce(GravityForce::FromDirection({0.0f, -1.0f, 0.0f}, 9.8f));
  }

  void run(int frames) {
    for(int f = 0; f < frames; ++f) {
      VkCommandBuffer cmd = ctx.beginCmd();
      engine.step(cmd, dt);
      ctx.submitCmd(cmd);
    }
  }

  std::vector<glm::vec4> readPos() const {
    std::vector<glm::vec4> v(engine.liveParticleCount());
    ctx.readBuffer(engine.getPositionBuffer(), 0, v.data(), v.size() * sizeof(glm::vec4));
    return v;
  }
  std::vector<glm::vec4> readVel() const {
    std::vector<glm::vec4> v(engine.liveParticleCount());
    ctx.readBuffer(engine.getVelocityBuffer(), 0, v.data(), v.size() * sizeof(glm::vec4));
    return v;
  }
  uint32_t aliveFoam() const {
    std::vector<uint32_t> k(engine.maxDiffuseParticles);
    ctx.readBuffer(engine.getFoamKindBuffer(), 0, k.data(), k.size() * sizeof(uint32_t));
    return uint32_t(std::count_if(k.begin(), k.end(), [](uint32_t x) { return x != 0u; }));
  }
  void cleanup() {
    engine.cleanup();
    ctx.cleanup();
  }
};

std::shared_ptr<AABBEmitter> block(glm::vec3 lo, glm::vec3 hi, glm::vec3 vel, float cell, float ppc = 8.0f) {
  auto e                = AABBEmitter::FromAABB(lo, hi, vel, 0);
  const glm::vec3 d     = hi - lo;
  e->particles_per_step = int(d.x * d.y * d.z / (cell * cell * cell) * ppc);
  e->step_count         = -1;
  return e;
}

} // namespace

// 静水プール: 非圧縮なら水位が保たれ、速度も落ち着く。弱圧縮/投影なしだと沈む・跳ねる。
TEST_CASE("FLIP - hydrostatic pool keeps its volume and settles") {
  FlipScene s;
  const float cell = 2.0f / 32.0f;
  s.init(2.0f, cell, 200000);
  // 壁クランプ(1.5セル)内側いっぱいに高さ 0.5 m の水
  const float lo = 1.5f * cell, hi = 2.0f - 1.5f * cell;
  s.engine.addEmitter(block({lo, lo, lo}, {hi, lo + 0.5f, hi}, {0, 0, 0}, cell));

  s.run(120);

  const auto pos = s.readPos();
  const auto vel = s.readVel();
  REQUIRE(pos.size() > 10000);
  std::vector<float> ys;
  float vmean = 0.0f;
  for(size_t i = 0; i < pos.size(); ++i) {
    REQUIRE(std::isfinite(pos[i].x + pos[i].y + pos[i].z));
    CHECK(pos[i].y >= 0.0f);
    CHECK(pos[i].y <= 2.0f);
    ys.push_back(pos[i].y);
    vmean += glm::length(glm::vec3(vel[i]));
  }
  vmean /= float(pos.size());
  std::sort(ys.begin(), ys.end());
  const float top = ys[size_t(ys.size() * 0.98)];
  float ymean = 0.0f;
  for(float y : ys) ymean += y;
  ymean /= float(ys.size());
  CHECK(top == doctest::Approx(lo + 0.5f).epsilon(0.1));
  CHECK(ymean == doctest::Approx(lo + 0.25f).epsilon(0.1));
  CHECK(vmean < 0.3f);
  s.cleanup();
}

// 水塊を球コライダーへ落とす: 球内部に粒子が入り込まない
TEST_CASE("FLIP - fluid does not penetrate a sphere collider") {
  FlipScene s;
  const float cell = 2.0f / 32.0f;
  s.init(2.0f, cell, 200000);
  const glm::vec3 c(1.0f, 0.6f, 1.0f);
  const float r = 0.3f;
  ColliderSet cols;
  cols.addSphere(c, r);
  s.engine.setColliders(cols);
  s.engine.addEmitter(block({0.7f, 1.0f, 0.7f}, {1.3f, 1.4f, 1.3f}, {0, 0, 0}, cell));

  s.run(90);

  const auto pos = s.readPos();
  float minPen = 1e9f;
  for(const auto& p : pos) {
    REQUIRE(std::isfinite(p.x + p.y + p.z));
    minPen = std::min(minPen, glm::length(glm::vec3(p) - c) - r);
  }
  CHECK(minPen > -0.5f * cell);
  s.cleanup();
}

// 泡: 高速衝突で生成され、生成係数 0 / 無効では生成されない
TEST_CASE("FLIP - foam is generated on impact and suppressed when disabled") {
  const float cell = 2.0f / 32.0f;
  for(int mode = 0; mode < 2; ++mode) {
    FlipScene s;
    s.init(2.0f, cell, 200000, 50000);
    s.engine.foamEnabled = true;
    FLIPEngine::FoamParams fp;
    fp.gravity = glm::vec3(0.0f, -9.8f, 0.0f); // このテストは Y-up
    if(mode == 1) fp.kTa = fp.kWc = 0.0f;
    s.engine.setFoamParams(fp);
    s.engine.addEmitter(block({0.5f, 0.8f, 0.5f}, {1.5f, 1.4f, 1.5f}, {0, -8.0f, 0}, cell));

    s.run(90);

    if(mode == 0)
      CHECK(s.aliveFoam() > 0u);
    else
      CHECK(s.aliveFoam() == 0u);
    s.cleanup();
  }
}

// ダム崩壊 (examples/fluid_pbf.cpp --flip の縮小版): 水塊が崩れて障害物に当たり、有限・領域内のまま前進して泡が出る
TEST_CASE("FLIP - dam break stays finite, advances and foams on the obstacle") {
  FlipScene s;
  const float cell = 1.0f / 16;
  s.init(4.0f, cell, 300000, 50000);
  s.engine.foamEnabled = true;
  FLIPEngine::FoamParams fp;
  fp.gravity = glm::vec3(0.0f, -9.8f, 0.0f);
  s.engine.setFoamParams(fp);
  ColliderSet cols;
  cols.addBox({2.6f, 0.5f, 2.0f}, {0.2f, 0.4f, 0.9f}, 0.0f, 0.0f);
  s.engine.setColliders(cols);
  s.engine.addEmitter(block({0.2f, 0.2f, 0.2f}, {1.0f, 1.6f, 3.8f}, {0, 0, 0}, cell));

  float frontStart = 0.0f, frontEnd = 0.0f, vmax = 0.0f;
  for(int f = 0; f < 150; ++f) {
    s.run(1);
    if(f == 0 || f == 149) {
      const auto pos = s.readPos();
      float front    = 0.0f;
      for(const auto& p : pos) {
        REQUIRE(std::isfinite(p.x + p.y + p.z));
        CHECK(p.x >= 0.0f);
        CHECK(p.x <= 4.0f);
        front = std::max(front, p.x);
      }
      (f == 0 ? frontStart : frontEnd) = front;
    }
    if(f == 149) {
      for(const auto& v : s.readVel()) vmax = std::max(vmax, glm::length(glm::vec3(v)));
    }
  }
  CHECK(frontEnd > frontStart + 1.0f);
  CHECK(vmax < 40.0f);
  CHECK(s.aliveFoam() > 0u);
  s.cleanup();
}

// 動く箱 (パドル): 水を押しても粒子が箱へ入り込まず、箱が進んだ分だけ水が押し出される
TEST_CASE("FLIP - moving box collider pushes water without penetration") {
  FlipScene s;
  const float cell = 2.0f / 32.0f;
  s.init(2.0f, cell, 200000);
  const float lo = 1.5f * cell, hi = 2.0f - 1.5f * cell;
  s.engine.addEmitter(block({lo + 0.5f, lo, lo}, {hi, lo + 0.5f, hi}, {0, 0, 0}, cell));
  const glm::vec3 half(2.0f * cell, 1.0f, 1.0f);
  const float speed = 0.5f;
  for(int f = 0; f < 120; ++f) {
    const float x = 0.4f + speed * s.dt * float(f);
    ColliderSet cols;
    cols.addBox({x, 1.0f, 1.0f}, half, 0.0f, 0.0f, {speed, 0, 0});
    s.engine.setColliders(cols);
    s.run(1);
  }
  const float boxFront = 0.4f + speed * s.dt * 119.0f + half.x;
  float minX           = 1e9f;
  for(const auto& p : s.readPos()) {
    REQUIRE(std::isfinite(p.x + p.y + p.z));
    if(p.y > 0.05f) minX = std::min(minX, p.x);
  }
  CHECK(minX > boxFront - 0.5f * cell);
  s.cleanup();
}

// fluid 系 example の定期ログ: 間隔 0 なら 2 フレーム目以降は毎回出力され、積算時間が進む
TEST_CASE("FluidStatsLogger prints after the first frame") {
  FluidStatsLogger logger(0.0);
  FluidStatsInfo info;
  info.solver    = "TEST";
  info.particles = 1000;
  info.gridRes   = glm::uvec3(8, 8, 8);
  info.dt        = 1.0f / 60.0f;
  for(int i = 0; i < 3; ++i) logger.frame(info);
  CHECK(fluidstats_detail::rssMB() > 0.0);
  CHECK(fluidstats_detail::processCpuSec() > 0.0);

  // 実 allocator: エンジンが確保したバッファ分の VRAM 確保量が見える
  FlipScene s;
  s.init(2.0f, 2.0f / 32, 1000);
  const auto v = fluidstats_detail::vram(fluidStatsInfo(s.engine, s.dt).allocator);
  CHECK(v.allocMB > 0.0);
  s.cleanup();
}
