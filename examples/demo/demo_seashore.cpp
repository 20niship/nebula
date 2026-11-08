// 海岸の波 (issue #137): 沖のパドルが起こした波が砂浜の斜面と岩場で砕け、泡・飛沫が出る FLIP 流体。
// fluid_wave_foam (単振動パドル + FLIP + 泡) をベースに、地形・岩を SDF コライダーにして headless で .nbp / .nbf を dump する。
// 地形 SDF は seashore/make_scene.py が作る (岩アセットは seashore/fetch_rocks.py で取得。いずれも git 管理外)。
// 確認:  python seashore/preview_points.py out/seashore --frame 100     レンダ: render/render_seashore.py
#include "Collider.h"
#include "Force.h"
#include "NbpDump.h"
#include "engine/FLIPEngine.h"
#include "helpers/HeadlessCtx.h"

#include <argparse/argparse.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>

static const std::string SHADER_DIR_STR = SHADER_DIR;

struct Args : public argparse::Args {
  std::string& out    = kwarg("out", "output directory").set_default(std::string("out/seashore"));
  std::string& sdf    = kwarg("sdf", "scene SDF (seashore/make_scene.py)").set_default(std::string("assets_local/seashore/scene.sdf"));
  float& domain_x     = kwarg("domain-x", "domain X [m] (make_scene.py と一致させる)").set_default(44.0f);
  float& domain_y     = kwarg("domain-y", "domain Y [m]").set_default(12.0f);
  float& domain_z     = kwarg("domain-z", "domain Z [m]").set_default(16.0f);
  float& water        = kwarg("water", "still water level [m]").set_default(3.0f);
  float& spacing      = kwarg("spacing", "particle spacing [m] (cell = 2x). 0.1 ≈ 0.9M, 0.065 ≈ 3.5M, 0.05 ≈ 8M particles").set_default(0.1f);
  int& frames         = kwarg("frames", "output frames").set_default(360);
  float& fps          = kwarg("fps", "output frame rate").set_default(24.0f);
  int& substeps       = kwarg("substeps", "FLIP substeps per output frame").set_default(6);
  int& pressure_iters = kwarg("pressure-iters", "pressure GS iterations").set_default(20);
  float& paddle_amp   = kwarg("paddle-amp", "wave paddle amplitude [m]").set_default(2.5f);
  float& paddle_omega = kwarg("paddle-omega", "wave paddle angular frequency [rad/s]").set_default(1.3f);
  int& max_diffuse    = kwarg("max-diffuse", "max foam particles").set_default(1000000);
  float& foam_rate    = kwarg("foam-rate", "泡の生成係数 (kTa, kWc) [1/s] (0=auto: 100*(spacing/0.2)^2。粒子数が増える分を相殺して泡が飽和しないようにする)").set_default(0.0f);
  float& foam_delay   = kwarg("foam-delay", "この時刻 [s] まで泡を生成しない (初期の水面ノイズでの誤生成を避ける)").set_default(2.0f);
  float& fp_scale     = kwarg("fp-scale", "fixed-point scale (8 粒子/セル固定ならセル質量は解像度によらず一定。1e6 以上は格子で桁あふれして爆発する)").set_default(65536.0f);
};

static float smoothstep01(float t) {
  t = std::clamp(t, 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

int main(int argc, char* argv[]) {
  auto a = argparse::parse<Args>(argc, argv);
  try {
    const glm::vec3 DOMAIN(a.domain_x, a.domain_y, a.domain_z);
    const float d    = a.spacing;
    const float cell = 2.0f * d;

    // ── 地形 SDF ────────────────────────────────────────────────────────────
    LocalMeshSDF grid;
    {
      FILE* f = std::fopen(a.sdf.c_str(), "rb");
      if(!f) throw std::runtime_error("scene SDF not found: " + a.sdf + " (run seashore/make_scene.py)");
      int32_t res;
      float c;
      if(std::fread(&res, 4, 1, f) != 1 || std::fread(&c, 4, 1, f) != 1) throw std::runtime_error("bad SDF header");
      grid.res      = uint32_t(res);
      grid.cellSize = c;
      grid.localMin = glm::vec3(0.0f);
      grid.data.resize(size_t(res) * res * res);
      if(std::fread(grid.data.data(), 4, grid.data.size(), f) != grid.data.size()) throw std::runtime_error("bad SDF body");
      std::fclose(f);
    }
    auto sdfAt = [&](glm::vec3 p) {
      const glm::ivec3 i = glm::clamp(glm::ivec3(p / grid.cellSize), glm::ivec3(0), glm::ivec3(int(grid.res) - 1));
      return grid.data[(size_t(i.z) * grid.res + i.y) * grid.res + i.x];
    };

    // ── 初期の水: パドルから岸まで、静水面以下で地形の外側を満たす ────────────────────
    const float restX  = a.paddle_amp + 1.0f;
    const float xStart = restX + cell + 2.0f * d;
    std::vector<glm::vec4> pos, vel;
    {
      std::mt19937 rng(1);
      std::uniform_real_distribution<float> J(-0.25f * d, 0.25f * d);
      const float Vp = d * d * d;
      for(float z = 0.5f * d; z < DOMAIN.z; z += d)
        for(float x = xStart; x < DOMAIN.x - cell; x += d)
          for(float y = 0.5f * d; y < a.water; y += d) {
            const glm::vec3 p(x + J(rng), y + J(rng), z + J(rng));
            if(sdfAt(p) > 0.75f * d) pos.emplace_back(p, Vp);
          }
    }
    vel.assign(pos.size(), glm::vec4(0.0f));
    std::printf("particles: %zu  cell %.3f  grid %ux%ux%u\n", pos.size(), cell, uint32_t(DOMAIN.x / cell), uint32_t(DOMAIN.y / cell), uint32_t(DOMAIN.z / cell));

    HeadlessCtx ctx;
    ctx.init();
    FLIPEngine e;
    e.domainSize          = DOMAIN;
    e.cellSize            = cell;
    e.maxParticles        = uint32_t(pos.size() + pos.size() / 8); // 余裕 (パドルによる体積増は無いが粒子の押し出しで過密になっても落ちないように)
    e.maxDiffuseParticles = uint32_t(a.max_diffuse);
    e.foamEnabled         = a.max_diffuse > 0;
    e.pressureIterations  = a.pressure_iters;
    e.numSubsteps         = a.substeps;
    e.setFixedPointScale(a.fp_scale);
    e.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADER_DIR_STR);
    e.addForce(GravityForce::FromDirection({0.0f, -1.0f, 0.0f}, 9.8f)); // Y-up

    FLIPEngine::FoamParams fp; // 岩・砕波での生成を見やすくするため、wave_foam より生成しやすく寿命を長めに
    fp.gravity     = {0.0f, -9.8f, 0.0f};
    fp.kTa         = a.foam_rate > 0.0f ? a.foam_rate : 100.0f * (d / 0.2f) * (d / 0.2f);
    fp.kWc         = fp.kTa;
    fp.taLo        = 4.0f;
    fp.taHi        = 20.0f;
    fp.wcLo        = 2.0f;
    fp.wcHi        = 6.0f;
    fp.keLo        = 4.0f;
    fp.lifetimeMin = 1.0f;
    fp.lifetimeMax = 3.5f;
    e.setFoamParams(fp);

    const glm::vec3 paddleHalf(cell, DOMAIN.y * 0.5f, DOMAIN.z * 0.5f), paddleRest(restX, DOMAIN.y * 0.5f, DOMAIN.z * 0.5f);
    const uint32_t sdfIdx = e.uploadColliderMeshSDF(grid);
    auto setColliders = [&](float t) {
      const float ramp = smoothstep01(t / 2.0f); // 2 秒で立ち上げ
      const float dx = a.paddle_amp * ramp * std::sin(a.paddle_omega * t), vx = a.paddle_amp * ramp * a.paddle_omega * std::cos(a.paddle_omega * t);
      ColliderSet cols;
      cols.addMeshSDF(sdfIdx, grid, glm::vec3(0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), {0, 0, 0}, {0, 0, 0}, 0.0f, 0.3f);
      cols.addBox(paddleRest + glm::vec3(dx, 0.0f, 0.0f), paddleHalf, 0.0f, 0.0f, glm::vec3(vx, 0.0f, 0.0f));
      e.setColliders(cols);
    };
    setColliders(0.0f);
    e.appendParticles(pos, vel);

    const uint32_t N = e.liveParticleCount(), M = e.maxDiffuseParticles;
    std::filesystem::create_directories(a.out);
    std::vector<glm::vec4> fpos(M), fvel(M);
    std::vector<uint32_t> fkind(M + 1);
    const float dt = 1.0f / a.fps;
    auto t0        = std::chrono::steady_clock::now();
    for(int f = 0; f <= a.frames; f++) {
      if(f > 0) {
        setColliders(f * dt); // パドルは出力フレーム単位で更新 (サブステップ内は一定速度)
        e.foamEnabled       = a.max_diffuse > 0 && f * dt >= a.foam_delay;
        VkCommandBuffer cmd = ctx.beginCmd();
        e.step(cmd, dt);
        ctx.submitCmd(cmd);
      }
      ctx.readBuffer(e.getPositionBuffer(), 0, pos.data(), size_t(N) * sizeof(glm::vec4));
      ctx.readBuffer(e.getVelocityBuffer(), 0, vel.data(), size_t(N) * sizeof(glm::vec4));
      char path[512];
      std::snprintf(path, sizeof path, "%s/frame_%05d.nbp", a.out.c_str(), f);
      writeNbp(path, pos, vel, glm::vec3(0.0f), DOMAIN, 12.0f);
      size_t nFoam = 0;
      if(M > 0) {
        ctx.readBuffer(e.getFoamPositionBuffer(), 0, fpos.data(), size_t(M) * sizeof(glm::vec4));
        ctx.readBuffer(e.getFoamKindBuffer(), 0, fkind.data(), size_t(M) * sizeof(uint32_t));
        std::vector<glm::vec4> lp;
        std::vector<uint8_t> lk;
        for(uint32_t i = 0; i < M; i++)
          if(fkind[i] != 0 && fpos[i].w > 0.0f) {
            lp.push_back(fpos[i]);
            lk.push_back(uint8_t(fkind[i]));
          }
        nFoam = lp.size();
        std::snprintf(path, sizeof path, "%s/foam_%05d.nbf", a.out.c_str(), f);
        writeNbf(path, lp, lk, glm::vec3(0.0f), DOMAIN);
      }
      std::printf("frame %d/%d  foam %zu  %.1fs elapsed\n", f, a.frames, nFoam, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      std::fflush(stdout);
    }
    e.cleanup();
    ctx.cleanup();
  } catch(const std::exception& ex) {
    std::fprintf(stderr, "Fatal: %s\n", ex.what());
    return 1;
  }
  return 0;
}
