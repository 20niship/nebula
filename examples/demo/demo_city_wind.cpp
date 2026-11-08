// 気流シミュレーション: ビル群の中を風が通り抜ける。PyroEngine (オイラー格子) で風速場を解き、毎ステップ流れに乗せたトレーサ粒子の軌跡を tracers.bin に書き出す。
// 流線を各フレームの瞬間速度場から引くと、後流の渦で軌跡が毎フレーム別物になりちらつく。トレーサを時間方向に連続して流す (pathline) ことでちらつきを避ける。
// ビル配置は city.txt (x0 y0 z0 x1 y1 z1 [m])、領域は domain.txt に書き出し、render/render_wind.py が Blender で読み込む。
// 格子は 512x256x256 固定。PyroEngine のバッファは Morton 符号の最大値+1 セル (1 セル約 88 byte) で、x>=y>=z の 2^n 直方体ならちょうど実セル数 (約 3GB)。
// 立方体 512^3 (約 12GB) は載らないので、風下 x を最長・高さ y と幅 z を短くする。--cell-size で領域の実寸が決まる。
#include "core/MeshSDF.h"
#include "engine/PyroEngine.h"
#include "helpers/HeadlessCtx.h"

#include <argparse/argparse.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>

static const std::string SHADER_DIR_STR = SHADER_DIR;

struct Args : public argparse::Args {
  std::string& out = kwarg("out", "output directory").set_default(std::string("out/wind"));
  int& frames      = kwarg("frames", "simulation steps").set_default(720);
  int& dump_every  = kwarg("dump-every", "dump a .pvox (全チャンネル, 1 枚約 150MB) every N steps (0=しない)").set_default(0);
  int& tracers     = kwarg("tracers", "tracer particles (pool size)").set_default(20000);
  int& tracer_life = kwarg("tracer-life", "max tracer age [steps]").set_default(500);
  float& dt        = kwarg("dt", "step [s]").set_default(1.0f / 30.0f);
  float& cell      = kwarg("cell-size", "cell size [m] (格子 512x256x256 なので領域 = 512,256,256 x cell)").set_default(0.4f);
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
    const glm::vec3 domain = glm::vec3(512.0f, 256.0f, 256.0f) * a.cell;

    // 街区: 道路幅 6~10m、区画 14~18m を格子状に並べ、高さを乱数で決める。上流 (x<24m) は助走区間として空ける。
    std::mt19937 rng(a.seed);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    std::vector<Box> city;
    for(float x = 24.0f; x + 18.0f < domain.x - 8.0f; x += 24.0f)
      for(float z = 4.0f; z + 18.0f < domain.z - 2.0f; z += 24.0f) {
        const float h = std::min(12.0f + 26.0f * U(rng) * U(rng) + (U(rng) < 0.2f ? 10.0f : 0.0f), domain.y - 6.0f);
        const float w = 14.0f + 4.0f * U(rng);
        city.push_back({{x, 0.0f, z}, {x + w, h, z + w}});
      }

    std::filesystem::create_directories(a.out);
    if(FILE* f = std::fopen((a.out + "/city.txt").c_str(), "w")) {
      for(auto& b : city) std::fprintf(f, "%g %g %g %g %g %g\n", b.lo.x, b.lo.y, b.lo.z, b.hi.x, b.hi.y, b.hi.z);
      std::fclose(f);
    }
    if(FILE* f = std::fopen((a.out + "/domain.txt").c_str(), "w")) {
      std::fprintf(f, "%g %g %g %g\n", domain.x, domain.y, domain.z, a.cell);
      std::fclose(f);
    }

    HeadlessCtx ctx;
    ctx.init();
    PyroEngine e;
    e.domainSize = domain;
    e.cellSize   = a.cell;
    e.maxEmitters = 8;
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

    // トレーサ粒子: 風上の断面から連続的に湧かせ、格子速度に乗せて流す。街区の外 (風下・上空・側方) か建物の中に入ったら湧き直す。
    // 軌跡は pool の同じ添字に対応し、湧き直すと gen が増える (描画側で軌跡を切る)。
    struct Tracer {
      glm::vec3 p;
      uint32_t gen;
      int age;
    };
    std::vector<Tracer> tr(size_t(a.tracers));
    std::vector<glm::vec4> vel(e.totalCells());
    auto spawn = [&](Tracer& t) {
      t.p = {3.0f, 1.0f + (std::min(domain.y - 8.0f, 56.0f)) * U(rng) * U(rng), 2.0f + (domain.z - 4.0f) * U(rng)};  // 地表近くほど密にして、ビルの周りを濃く見せる
      t.age = 0;
      t.gen++;
    };
    for(auto& t : tr) {
      t.gen = 0;
      spawn(t);
      t.gen = 0;
      t.p.x += (domain.x - 8.0f) * U(rng);  // 最初から領域全体に分布させる (後で建物内に入ったものは湧き直される)
      t.age = int(U(rng) * a.tracer_life);
    }
    // トリリニア補間 (セル中心 = (i+0.5)*cell)
    auto sample = [&](glm::vec3 p) {
      glm::vec3 c       = glm::clamp(p / a.cell - 0.5f, glm::vec3(0.0f), glm::vec3(res) - 1.001f);
      const glm::uvec3 i = glm::uvec3(c);
      const glm::vec3 f  = c - glm::vec3(i);
      glm::vec3 v(0.0f);
      for(int k = 0; k < 8; k++) {
        const uint32_t dx = k & 1, dy = (k >> 1) & 1, dz = (k >> 2) & 1;
        const float w     = (dx ? f.x : 1.0f - f.x) * (dy ? f.y : 1.0f - f.y) * (dz ? f.z : 1.0f - f.z);
        v += w * glm::vec3(vel[meshSdfMortonEncode(i.x + dx, i.y + dy, i.z + dz)]);
      }
      return v;
    };
    auto inSolid = [&](glm::vec3 p) {
      const glm::uvec3 i = glm::uvec3(glm::clamp(p / a.cell, glm::vec3(0.0f), glm::vec3(res) - 1.0f));
      return sdf[meshSdfMortonEncode(i.x, i.y, i.z)] < 0.0f;
    };
    FILE* trf = std::fopen((a.out + "/tracers.bin").c_str(), "wb");  // "TRC1" | u32 steps | u32 M | f32 dt | f32 domain[3] | steps x M x { f32 xyz, u32 gen }
    {
      const uint32_t hdr[3] = {0, uint32_t(tr.size()), 0};
      std::fwrite("TRC1", 1, 4, trf);
      std::fwrite(hdr, 4, 2, trf);
      std::fwrite(&a.dt, 4, 1, trf);
      std::fwrite(&domain, 4, 3, trf);
    }
    uint32_t steps = 0;
    std::vector<float> rec(tr.size() * 4);

    auto t0 = std::chrono::steady_clock::now();
    for(int f = 0; f < a.frames; f++) {
      VkCommandBuffer cmd = ctx.beginCmd();
      e.step(cmd, a.dt);
      ctx.submitCmd(cmd);
      ctx.readBuffer(e.getVelocityBuffer(), 0, vel.data(), vel.size() * sizeof(glm::vec4));
      const float vcap = 2.0f * a.wind;  // 角で出る孤立した異常値を抑える
      auto vlim        = [&](glm::vec3 v) { return glm::length(v) > vcap ? v * (vcap / glm::length(v)) : v; };
      for(size_t i = 0; i < tr.size(); i++) {
        Tracer& t = tr[i];
        if(f >= 1) {  // 最初の数十ステップは一様流への初期化で速度場が荒れるため、流れが落ち着くまで動かさない
          const glm::vec3 v1 = vlim(sample(t.p)), v2 = vlim(sample(t.p + 0.5f * a.dt * v1));
          t.p += a.dt * v2;
          t.age++;
        }
        if(t.p.x > domain.x - 1.5f || t.p.y > domain.y - 1.0f || t.p.y < 0.3f || t.p.z < 1.0f || t.p.z > domain.z - 1.0f || t.age > a.tracer_life || inSolid(t.p)) spawn(t);
        std::memcpy(&rec[i * 4], &t.p, 12);
        std::memcpy(&rec[i * 4 + 3], &t.gen, 4);
      }
      if(f >= 60) {  // 立ち上がり (一様流の初期化 45 ステップ + 余裕) は記録しない
        std::fwrite(rec.data(), 4, rec.size(), trf);
        steps++;
      }
      if(a.dump_every > 0 && f % a.dump_every == 0) {
        char path[512];
        std::snprintf(path, sizeof path, "%s/frame_%05d.pvox", a.out.c_str(), f / a.dump_every);
        e.dumpFrame(path, f * a.dt);
      }
      if(f % 30 == 0) std::printf("step %d/%d  %.1fs\n", f, a.frames, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()), std::fflush(stdout);
    }
    std::fseek(trf, 4, SEEK_SET);
    std::fwrite(&steps, 4, 1, trf);
    std::fclose(trf);
    e.cleanup();
    ctx.cleanup();
  } catch(const std::exception& ex) {
    std::fprintf(stderr, "Fatal: %s\n", ex.what());
    return 1;
  }
  return 0;
}
