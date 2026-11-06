// 気流シミュレーション: ビル群の中を風が通り抜ける。PyroEngine (オイラー格子) で風速場と煙トレーサを解き、.pvox を dump する。
// ビル配置は city.txt (x0 y0 z0 x1 y1 z1 [m]) に書き出し、render/render_wind.py が Blender で読み込む。
#include "core/MeshSDF.h"
#include "engine/PyroEngine.h"
#include "helpers/HeadlessCtx.h"

#include <argparse/argparse.hpp>
#include <glm/glm.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <random>

static const std::string SHADER_DIR_STR = SHADER_DIR;

struct Args : public argparse::Args {
  std::string& out = kwarg("out", "output directory").set_default(std::string("out/wind"));
  int& frames      = kwarg("frames", "simulation steps").set_default(720);
  int& dump_every  = kwarg("dump-every", "dump a .pvox every N steps").set_default(6);
  float& dt        = kwarg("dt", "step [s]").set_default(1.0f / 30.0f);
  float& cell      = kwarg("cell-size", "cell size [m]").set_default(0.75f);
  float& wind      = kwarg("wind", "inflow wind speed [m/s]").set_default(14.0f);
  int& iters       = kwarg("pressure-iters", "pressure solver sweeps").set_default(120);
  int& seed        = kwarg("seed", "city layout seed").set_default(7);
};

struct Box {
  glm::vec3 lo, hi;
};

static float boxSdf(glm::vec3 p, const Box& b) {
  const glm::vec3 c = 0.5f * (b.lo + b.hi), h = 0.5f * (b.hi - b.lo);
  const glm::vec3 q = glm::abs(p - c) - h;
  return glm::length(glm::max(q, glm::vec3(0.0f))) + std::min(std::max(q.x, std::max(q.y, q.z)), 0.0f);
}

int main(int argc, char* argv[]) {
  auto a = argparse::parse<Args>(argc, argv);
  try {
    // 192m x 72m x 96m の街区 (セル 0.75m で 256x96x128)
    const glm::vec3 domain(192.0f, 72.0f, 96.0f);

    // 街区: 道路幅 9m、区画 18m を格子状に並べ、高さを乱数で決める。上流 (x<36m) は助走区間として空ける。
    std::mt19937 rng(a.seed);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    std::vector<Box> city;
    for(float x = 40.0f; x + 18.0f < 180.0f; x += 27.0f)
      for(float z = 6.0f; z + 18.0f < 92.0f; z += 27.0f) {
        const float h = 12.0f + 40.0f * U(rng) * U(rng) + (U(rng) < 0.12f ? 14.0f : 0.0f);
        const float w = 14.0f + 4.0f * U(rng);
        city.push_back({{x, 0.0f, z}, {x + w, h, z + w}});
      }

    std::filesystem::create_directories(a.out);
    if(FILE* f = std::fopen((a.out + "/city.txt").c_str(), "w")) {
      for(auto& b : city) std::fprintf(f, "%g %g %g %g %g %g\n", b.lo.x, b.lo.y, b.lo.z, b.hi.x, b.hi.y, b.hi.z);
      std::fclose(f);
    }

    HeadlessCtx ctx;
    ctx.init();
    PyroEngine e;
    e.domainSize = domain;
    e.cellSize   = a.cell;
    e.maxEmitters = 256;
    e.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADER_DIR_STR);
    e.numPressureIters       = a.iters;
    e.pc_.openBoundaryMask   = 1u | 2u | 8u | 16u | 32u; // 地面以外は開放 (風上 -x はファンが空気を引き込む)
    e.pc_.buoyancyAlpha      = 0.0f;
    e.pc_.buoyancyBeta       = 0.0f;
    e.pc_.vorticityEps       = 1.5f;
    e.pc_.densityDissipation = 0.02f;

    // 障害物 SDF (Morton 順)
    const glm::uvec3 res = e.gridRes();
    std::vector<float> sdf(e.totalCells(), 1e6f);
    for(uint32_t z = 0; z < res.z; z++)
      for(uint32_t y = 0; y < res.y; y++)
        for(uint32_t x = 0; x < res.x; x++) {
          const glm::vec3 p = (glm::vec3(x, y, z) + 0.5f) * a.cell;
          float d           = p.y < a.cell ? -1.0f : p.y;  // 地面 (最下段セルを固体にする)
          for(auto& b : city) d = std::min(d, boxSdf(p, b));
          sdf[meshSdfMortonEncode(x, y, z)] = d;
        }
    e.setColliderSDF(sdf);

    // 圧力ソルバ (Gauss-Seidel) は大域的な圧力場の収束が遅い (N^2 反復) ため、最初の数十ステップだけ領域全体の風速を一様流へ寄せ、
    // 街区の周りの局所的な圧力補正だけを解かせる。
    auto init             = std::make_shared<AABBEmitter>();
    init->center          = domain * 0.5f;
    init->size            = domain;
    init->inflowVelocity  = {a.wind, 0.0f, 0.0f};
    init->step_count      = 45;
    e.addEmitter(init);
    // 上流のファン: 断面全体の速度を風速へ寄せる
    auto fan             = std::make_shared<AABBEmitter>();
    fan->center          = {5.0f, domain.y * 0.5f, domain.z * 0.5f};
    fan->size            = {8.0f, domain.y - 2.0f, domain.z - 2.0f};
    fan->inflowVelocity  = {a.wind, 0.0f, 0.0f};
    e.addEmitter(fan);
    // 煙トレーサ: 高さ 4 段 x 幅方向 12 本の帯
    for(float y : {2.5f, 7.0f, 14.0f, 26.0f})
      for(int k = 0; k < 12; k++) {
        auto s             = std::make_shared<AABBEmitter>();
        s->center          = {9.0f, y, 5.0f + 7.6f * k};
        s->size            = {2.0f, 1.8f, 2.4f};
        s->inflowVelocity  = {a.wind, 0.0f, 0.0f};
        s->densityRate     = 30.0f;
        e.addEmitter(s);
      }

    auto t0 = std::chrono::steady_clock::now();
    for(int f = 0; f < a.frames; f++) {
      VkCommandBuffer cmd = ctx.beginCmd();
      e.step(cmd, a.dt);
      ctx.submitCmd(cmd);
      if(f % a.dump_every == 0) {
        char path[512];
        std::snprintf(path, sizeof path, "%s/frame_%05d.pvox", a.out.c_str(), f / a.dump_every);
        e.dumpFrame(path, f * a.dt);
      }
      if(f % 30 == 0) std::printf("step %d/%d  %.1fs\n", f, a.frames, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()), std::fflush(stdout);
    }
    e.cleanup();
    ctx.cleanup();
  } catch(const std::exception& ex) {
    std::fprintf(stderr, "Fatal: %s\n", ex.what());
    return 1;
  }
  return 0;
}
