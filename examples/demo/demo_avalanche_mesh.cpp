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
  int& frames      = kwarg("frames", "output frames").set_default(400);
  float& fps       = kwarg("fps", "output frame rate").set_default(24.0f);
  int& substeps    = kwarg("substeps", "MPM substeps per output frame").set_default(48);
  float& scale     = kwarg("scale", "particle count multiplier (1 = about 9M)").set_default(1.0f);
  float& friction  = kwarg("friction", "Drucker-Prager M (tan of friction angle)").set_default(0.45f);
  float& cohesion  = kwarg("cohesion", "Drucker-Prager cohesion [Pa]").set_default(300.0f);
  float& ground_friction = kwarg("ground-friction", "terrain collider friction coefficient").set_default(0.1f);
  std::string& dem = kwarg("dem", "DEM patch (assets_local/make_patch.py)").set_default(std::string("assets_local/dem_patch.f32"));
  float& fp_scale  = kwarg("fp-scale", "MPM fixed-point scale").set_default(2e7f);
};

// 地形は DEM 由来のハイトフィールド (assets_local/make_patch.py で作る: int nx, int nz, float step, float h[nz][nx])。
static glm::vec3 DOMAIN;
static int g_nx, g_nz;
static float g_step;
static std::vector<float> g_h;

static void loadDem(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if(!f) throw std::runtime_error("DEM patch not found: " + path + " (run assets_local/make_patch.py)");
  if(std::fread(&g_nx, 4, 1, f) != 1 || std::fread(&g_nz, 4, 1, f) != 1 || std::fread(&g_step, 4, 1, f) != 1) throw std::runtime_error("bad DEM header");
  g_h.resize(size_t(g_nx) * g_nz);
  if(std::fread(g_h.data(), 4, g_h.size(), f) != g_h.size()) throw std::runtime_error("bad DEM body");
  std::fclose(f);
}

static float terrain(float x, float z) {
  const float u = std::clamp(x / g_step, 0.0f, float(g_nx - 1) - 1e-3f), v = std::clamp(z / g_step, 0.0f, float(g_nz - 1) - 1e-3f);
  const int i = int(u), k = int(v);
  const float fu = u - i, fv = v - k;
  const float* r0 = &g_h[size_t(k) * g_nx + i];
  const float* r1 = r0 + g_nx;
  return std::max((r0[0] * (1 - fu) + r0[1] * fu) * (1 - fv) + (r1[0] * (1 - fu) + r1[1] * fu) * fv, 0.15f);
}

int main(int argc, char* argv[]) {
  auto a = argparse::parse<Args>(argc, argv);
  try {
    loadDem(a.dem);
    float hmax = 0.0f;
    for(float h : g_h) hmax = std::max(hmax, h);
    DOMAIN = glm::vec3((g_nx - 1) * g_step, hmax + 4.0f, (g_nz - 1) * g_step);
    const float spacing = 0.07f / std::cbrt(a.scale);
    const float cell    = spacing * 2.0f;
    const glm::uvec3 G(DOMAIN / cell);
    std::filesystem::create_directories(a.out);

    // ── 地形をファイルに書き出す (Blender 用) ──────────────────────────────────
    {
      FILE* f = std::fopen((a.out + "/terrain.f32").c_str(), "wb");
      std::fwrite(&g_nx, 4, 1, f);
      std::fwrite(&g_nz, 4, 1, f);
      std::fwrite(&g_step, 4, 1, f);
      std::fwrite(g_h.data(), 4, g_h.size(), f);
      std::fclose(f);
    }

    // ── 樹木 (幹の円柱) と岩 (球)。雪の通り道 (斜面中腹の岩・下部の林) に置いて雪を砕かせる ──────
    ColliderSet cols;
    struct Obs { bool tree; glm::vec3 c; float r, hh; };   // MPMEngine のコライダーは最大 64 個なので、障害物は地形 SDF に焼き込む
    std::vector<Obs> obs;
    {
      std::mt19937 rng(11);
      std::uniform_real_distribution<float> U(0.0f, 1.0f);
      FILE* tf = std::fopen((a.out + "/obstacles.txt").c_str(), "w");
      struct P { float x, z, r; };
      std::vector<P> placed;
      auto free_ = [&](float x, float z, float r) {
        for(const P& p : placed)
          if(std::hypot(p.x - x, p.z - z) < p.r + r + 0.4f) return false;
        return true;
      };
      for(int tries = 0; tries < 2000 && placed.size() < 16; tries++) {   // 岩
        const float x = 8.0f + 24.0f * U(rng), z = 4.0f + (DOMAIN.z - 8.0f) * U(rng), r = 1.0f + 1.8f * U(rng) * U(rng) * 1.4f;
        if(!free_(x, z, r)) continue;
        placed.push_back({x, z, r});
        obs.push_back({false, {x, terrain(x, z) + 0.25f * r, z}, r, 0.0f});
        std::fprintf(tf, "rock %g %g %g %g\n", x, terrain(x, z) + 0.25f * r, z, r);
      }
      int trees = 0;
      for(int tries = 0; tries < 8000 && trees < 48; tries++) {           // 林は斜面下部〜裾野に。黒松メッシュ (幹の半径は樹高の約 3%) に合わせる
        const float x = 24.0f + 28.0f * U(rng), z = 2.5f + (DOMAIN.z - 5.0f) * U(rng), H = 8.0f + 4.0f * U(rng), r = 0.03f * H + 0.03f, hh = 0.5f * H;
        if(!free_(x, z, r + 1.6f)) continue;                               // 樹冠の間から雪の動きが見えるよう間隔を空ける
        placed.push_back({x, z, r + 1.6f});
        obs.push_back({true, {x, terrain(x, z) + hh - 0.2f, z}, r, hh});
        std::fprintf(tf, "tree %g %g %g %g %g\n", x, terrain(x, z) - 0.2f, z, r, 2.0f * hh);
        trees++;
      }
      std::fclose(tf);
    }

    // ── 雪の初期配置: 斜面上部の厚い層 (3..4.5m) ─────────────────────────────
    std::vector<glm::vec4> pos, vel;
    const float Vp = spacing * spacing * spacing;
    for(float z = 4.0f; z < DOMAIN.z - 4.0f; z += spacing)
      for(float x = 2.5f; x < 24.0f; x += spacing) {
        const float depth = 3.0f + 1.5f * (0.5f + 0.5f * std::sin(x * 0.35f) * std::cos(z * 0.5f));
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
        for(uint32_t iy = 0; iy < G.y; iy++) {
          const glm::vec3 p((ix + 0.5f) * cell, (iy + 0.5f) * cell, (iz + 0.5f) * cell);
          float d = ((iy + 0.5f) * cell - h - 0.5f * cell) * n;
          for(const Obs& o : obs) {
            const glm::vec3 q = p - o.c;
            float od;
            if(o.tree) {
              const glm::vec2 w(std::hypot(q.x, q.z) - o.r, std::fabs(q.y) - o.hh);
              od = std::min(std::max(w.x, w.y), 0.0f) + glm::length(glm::max(w, glm::vec2(0.0f)));
            } else od = glm::length(q) - o.r;
            d = std::min(d, od);
          }
          grid.data[(size_t(iz) * res + iy) * res + ix] = d;
        }
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
