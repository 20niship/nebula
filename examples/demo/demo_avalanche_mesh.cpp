// 数百万粒子の MPM (Drucker-Prager) 雪崩。地形 (ハイトフィールド)・樹木・岩を障害物として配置し、結果を .nbp で dump する。
// Blender 側 (render/render_avalanche.py) が粒子の密度場から数百万三角形の雪面メッシュを再構成し、地形メッシュ・樹木インスタンスと合わせてレンダリングする。
#include "Collider.h"
#include "Force.h"
#include "MaterialParams.h"
#include "NbpDump.h"
#include "engine/MPMEngine.h"
#include "helpers/HeadlessCtx.h"

#include <argparse/argparse.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>

static const std::string SHADER_DIR_STR = SHADER_DIR;

struct Args : public argparse::Args {
  std::string& out = kwarg("out", "output directory").set_default(std::string("out/avalanche"));
  int& frames      = kwarg("frames", "output frames").set_default(288);
  float& fps       = kwarg("fps", "output frame rate").set_default(24.0f);
  int& substeps    = kwarg("substeps", "MPM substeps per output frame").set_default(48);
  float& scale     = kwarg("scale", "particle count multiplier (1 = about 6M)").set_default(1.0f);
  float& friction  = kwarg("friction", "Drucker-Prager M (tan of friction angle)").set_default(0.45f);
  float& cohesion  = kwarg("cohesion", "Drucker-Prager cohesion [Pa]").set_default(300.0f);
  float& ground_friction = kwarg("ground-friction", "terrain collider friction coefficient").set_default(0.1f);
  float& fp_scale  = kwarg("fp-scale", "MPM fixed-point scale").set_default(2e7f);
};

static const glm::vec3 DOMAIN(64.0f, 24.0f, 36.0f);

static float smooth(float t) {
  t = std::clamp(t, 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

// 地形高さ: x 方向に下る斜面 (約 33 度) が x=44..60 でなだらかに接地する。z 方向に沢と尾根のうねり、細かい起伏を足す。
static float terrain(float x, float z) {
  const float slope  = 17.0f - 0.66f * (x - 2.0f);              // 33 度
  const float flat   = 0.6f;
  const float t      = smooth((x - 40.0f) / 22.0f);             // 40..62 で斜面から平地へ
  float h            = (1.0f - t) * std::max(slope, flat) + t * flat;
  h += -2.2f * std::exp(-std::pow((z - 18.0f) / 3.6f, 2.0f)) * (1.0f - t);          // 沢
  h += 1.6f * std::exp(-std::pow((z - 7.0f) / 4.0f, 2.0f)) * (1.0f - t);            // 尾根
  h += 1.2f * std::exp(-std::pow((z - 30.0f) / 4.5f, 2.0f)) * (1.0f - t);
  h += 0.25f * std::sin(x * 0.9f + z * 0.6f) + 0.18f * std::sin(x * 2.1f - z * 1.7f);
  return std::max(h, 0.15f);
}

int main(int argc, char* argv[]) {
  auto a = argparse::parse<Args>(argc, argv);
  try {
    const float spacing = 0.075f / std::cbrt(a.scale);
    const float cell    = spacing * 2.0f;
    const glm::uvec3 G(DOMAIN / cell);
    std::filesystem::create_directories(a.out);

    // ── 地形をファイルに書き出す (Blender 用) と SDF 用高さ ────────────────────
    {
      const int nx = 641, nz = 361;
      std::vector<float> h(size_t(nx) * nz);
      for(int k = 0; k < nz; k++)
        for(int i = 0; i < nx; i++) h[size_t(k) * nx + i] = terrain(i * 0.1f, k * 0.1f);
      FILE* f = std::fopen((a.out + "/terrain.f32").c_str(), "wb");
      const float step = 0.1f;
      std::fwrite(&nx, 4, 1, f);
      std::fwrite(&nz, 4, 1, f);
      std::fwrite(&step, 4, 1, f);
      std::fwrite(h.data(), 4, h.size(), f);
      std::fclose(f);
    }

    // ── 樹木 (円柱) と岩 (球) ───────────────────────────────────────────────
    ColliderSet cols;
    {
      std::mt19937 rng(11);
      std::uniform_real_distribution<float> U(0.0f, 1.0f);
      FILE* tf = std::fopen((a.out + "/obstacles.txt").c_str(), "w");
      int trees = 0;
      for(int tries = 0; tries < 400 && trees < 44; tries++) {
        const float x = 46.0f + 15.0f * U(rng), z = 3.0f + 30.0f * U(rng), r = 0.25f + 0.2f * U(rng), hh = 3.0f + 2.0f * U(rng);
        cols.addCylinder({x, terrain(x, z) + hh - 0.2f, z}, r, hh, 0.0f, 0.3f);
        std::fprintf(tf, "tree %g %g %g %g %g\n", x, terrain(x, z) - 0.2f, z, r, 2.0f * hh);
        trees++;
      }
      for(int i = 0; i < 8; i++) {
        const float x = 30.0f + 28.0f * U(rng), z = 5.0f + 26.0f * U(rng), r = 0.9f + 1.0f * U(rng);
        cols.addSphere({x, terrain(x, z) + 0.3f * r, z}, r, 0.0f, 0.5f);
        std::fprintf(tf, "rock %g %g %g %g\n", x, terrain(x, z) + 0.3f * r, z, r);
      }
      std::fclose(tf);
    }

    // ── 雪の初期配置: 斜面上の層 (厚さ 1.8..3m) ─────────────────────────────
    std::vector<glm::vec4> pos, vel;
    const float Vp = spacing * spacing * spacing;
    for(float z = 7.0f; z < 30.0f; z += spacing)
      for(float x = 6.0f; x < 38.0f; x += spacing) {
        const float depth = 1.8f + 1.2f * (0.5f + 0.5f * std::sin(x * 0.35f) * std::cos(z * 0.5f));
        const float h0    = terrain(x, z);
        for(float y = h0 + 2.2f * cell; y < h0 + 2.2f * cell + depth; y += spacing) pos.emplace_back(x, y, z, Vp);
      }
    vel.assign(pos.size(), glm::vec4(0.0f));
    std::printf("particles: %zu  grid: %ux%ux%u\n", pos.size(), G.x, G.y, G.z);

    HeadlessCtx ctx;
    ctx.init();
    MPMEngine e;
    e.domainSize   = DOMAIN;
    e.cellSize     = cell;
    e.maxParticles = uint32_t(pos.size());
    e.setFixedPointScale(a.fp_scale);
    e.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADER_DIR_STR);
    e.numSubsteps = a.substeps;
    e.flip_ratio  = 0.0f;
    e.addForce(GravityForce::FromDirection({0.0f, -1.0f, 0.0f}, 9.8f));

    MaterialParams snow{};
    snow.mu         = calcMu(1e5f, 0.25f);
    snow.lambda     = calcLambda(1e5f, 0.25f);
    snow.rho0       = 400.0f;
    snow.model      = uint32_t(MaterialModel::DRUCKER_PRAGER);
    snow.M_friction = a.friction;
    snow.q_cohesion = a.cohesion;
    e.setMaterials({snow});

    // 地形 SDF (高さ差を勾配で正規化して距離に近づける)
    LocalMeshSDF grid;
    const uint32_t res = std::max({G.x, G.y, G.z});
    grid.res           = res;
    grid.cellSize      = cell;
    grid.localMin      = glm::vec3(0.0f);
    grid.data.assign(size_t(res) * res * res, 1e9f);
    for(uint32_t iz = 0; iz < G.z; iz++)
      for(uint32_t ix = 0; ix < G.x; ix++) {
        const float x = (ix + 0.5f) * cell, z = (iz + 0.5f) * cell;
        const float h = terrain(x, z);
        const float gx = (terrain(x + 0.1f, z) - terrain(x - 0.1f, z)) / 0.2f, gz = (terrain(x, z + 0.1f) - terrain(x, z - 0.1f)) / 0.2f;
        const float n = 1.0f / std::sqrt(1.0f + gx * gx + gz * gz);
        for(uint32_t iy = 0; iy < G.y; iy++) grid.data[(size_t(iz) * res + iy) * res + ix] = ((iy + 0.5f) * cell - h - 0.5f * cell) * n;
      }
    const uint32_t sdfIdx = e.uploadColliderMeshSDF(grid);
    cols.addMeshSDF(sdfIdx, grid, glm::vec3(0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), {0, 0, 0}, {0, 0, 0}, 0.0f, a.ground_friction);
    cols.addPlane({DOMAIN.x * 0.5f, 0.0f, DOMAIN.z * 0.5f}, {0, 1, 0}, 0.0f, 0.5f);
    cols.addPlane({0.0f, DOMAIN.y * 0.5f, DOMAIN.z * 0.5f}, {1, 0, 0}, 0.0f, 0.1f);
    cols.addPlane({DOMAIN.x, DOMAIN.y * 0.5f, DOMAIN.z * 0.5f}, {-1, 0, 0}, 0.0f, 0.1f);
    cols.addPlane({DOMAIN.x * 0.5f, DOMAIN.y * 0.5f, 0.0f}, {0, 0, 1}, 0.0f, 0.1f);
    cols.addPlane({DOMAIN.x * 0.5f, DOMAIN.y * 0.5f, DOMAIN.z}, {0, 0, -1}, 0.0f, 0.1f);
    e.setColliders(cols);
    e.appendParticles(pos, vel);

    const uint32_t N = e.liveParticleCount();
    const float dt   = 1.0f / a.fps;
    auto t0          = std::chrono::steady_clock::now();
    for(int f = 0; f <= a.frames; f++) {
      if(f > 0) {
        VkCommandBuffer cmd = ctx.beginCmd();
        e.step(cmd, dt);
        ctx.submitCmd(cmd);
      }
      ctx.readBuffer(e.getPositionBuffer(), 0, pos.data(), size_t(N) * sizeof(glm::vec4));
      ctx.readBuffer(e.getVelocityBuffer(), 0, vel.data(), size_t(N) * sizeof(glm::vec4));
      char path[512];
      std::snprintf(path, sizeof path, "%s/frame_%05d.nbp", a.out.c_str(), f);
      writeNbp(path, pos, vel, glm::vec3(0.0f), DOMAIN, 25.0f);
      std::printf("frame %d/%d  %.1fs elapsed\n", f, a.frames, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
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
