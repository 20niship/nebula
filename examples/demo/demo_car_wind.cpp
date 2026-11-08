// 風洞: 一定角速度で自転するスポーツカーに、固定向き (+x) の一定風速の風を当てる。PyroEngine (オイラー格子) の回転コライダー
// (PyroEngine::setRotatingCollider: 壁面速度 ω×r つき) で車を回し、毎ステップ流れに乗せたトレーサ粒子の軌跡を tracers.bin に書き出す
// (形式は demo_city_wind と同じ)。車 (SDF) は car/make_car.py が手続き生成する。Blender 側は car/render_car.py。
// 格子は既定 1024x256x256 (2^26 セル, 約 6GB)。x>=y>=z の 2^n 直方体なので Morton バッファが実セル数ちょうどで済む (1024x512x256 は約 12GB で 12GB GPU には載らない)。
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
  std::string& out = kwarg("out", "output directory").set_default(std::string("out/car"));
  std::string& sdf = kwarg("sdf", "car SDF (car/make_car.py)").set_default(std::string("assets_local/car/car.sdf"));
  int& nx          = kwarg("nx", "grid cells X (風下方向, 2^n)").set_default(1024);
  int& ny          = kwarg("ny", "grid cells Y (高さ, 2^n, nx 以下)").set_default(256);
  int& nz          = kwarg("nz", "grid cells Z (幅, 2^n, ny 以下)").set_default(256);
  float& cell      = kwarg("cell-size", "cell size [m]").set_default(0.02f);
  int& frames      = kwarg("frames", "simulation steps").set_default(720);
  float& dt        = kwarg("dt", "step [s]").set_default(1.0f / 120.0f);
  float& wind      = kwarg("wind", "inflow wind speed [m/s]").set_default(20.0f);
  float& omega     = kwarg("omega", "car spin [rad/s] (y 軸まわり)").set_default(0.5f);
  int& iters       = kwarg("pressure-iters", "pressure solver sweeps").set_default(120);
  int& tracers     = kwarg("tracers", "tracer particles (pool size)").set_default(40000);
  int& tracer_life = kwarg("tracer-life", "max tracer age [steps]").set_default(400);
  int& warmup      = kwarg("warmup", "この step 数までは記録しない (一様流の初期化 + 後流の立ち上がり)").set_default(120);
  int& seed        = kwarg("seed", "tracer seed").set_default(7);
};

int main(int argc, char* argv[]) {
  auto a = argparse::parse<Args>(argc, argv);
  try {
    const glm::vec3 domain = glm::vec3(float(a.nx), float(a.ny), float(a.nz)) * a.cell;
    const glm::vec2 pivot(0.36f * domain.x, 0.5f * domain.z);  // 車の自転軸 (風上寄りに置いて後流を長く取る)

    // 車のローカル SDF (節点値): "int nx,ny,nz | float cell | float min[3] | float data[]" (x 最速)
    std::vector<float> local;
    glm::uvec3 lres;
    float lcell;
    glm::vec3 lmin;
    {
      FILE* f = std::fopen(a.sdf.c_str(), "rb");
      if(!f) throw std::runtime_error("car SDF not found: " + a.sdf + " (run car/make_car.py)");
      int32_t n[3];
      if(std::fread(n, 4, 3, f) != 3 || std::fread(&lcell, 4, 1, f) != 1 || std::fread(&lmin, 4, 3, f) != 3) throw std::runtime_error("bad car SDF header");
      lres = glm::uvec3(n[0], n[1], n[2]);
      local.resize(size_t(lres.x) * lres.y * lres.z);
      if(std::fread(local.data(), 4, local.size(), f) != local.size()) throw std::runtime_error("bad car SDF body");
      std::fclose(f);
    }

    std::filesystem::create_directories(a.out);
    if(FILE* f = std::fopen((a.out + "/scene.txt").c_str(), "w")) {  // domain xyz, cell, pivot xz, omega, dt, warmup
      std::fprintf(f, "%g %g %g %g %g %g %g %g %d\n", domain.x, domain.y, domain.z, a.cell, pivot.x, pivot.y, a.omega, a.dt, a.warmup);
      std::fclose(f);
    }

    HeadlessCtx ctx;
    ctx.init();
    PyroEngine e;
    e.domainSize  = domain;
    e.cellSize    = a.cell;
    e.maxEmitters = 8;
    e.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADER_DIR_STR);
    e.numPressureIters       = a.iters;
    e.pc_.openBoundaryMask   = 1u | 2u | 8u | 16u | 32u;  // 地面以外は開放
    e.pc_.buoyancyAlpha      = 0.0f;
    e.pc_.buoyancyBeta       = 0.0f;
    e.pc_.vorticityEps       = 0.0f;
    e.pc_.densityDissipation = 0.02f;
    e.setRotatingCollider(local, lres, lcell, lmin, pivot, a.omega);

    // 圧力ソルバ (Gauss-Seidel) は大域的な圧力場の収束が遅いため、最初の数十ステップだけ領域全体を一様流へ寄せ、車の周りの局所補正だけ解かせる。
    auto init            = std::make_shared<AABBEmitter>();
    init->center         = domain * 0.5f;
    init->size           = domain;
    init->inflowVelocity = {a.wind, 0.0f, 0.0f};
    init->step_count     = 45;
    e.addEmitter(init);
    auto fan            = std::make_shared<AABBEmitter>();  // 上流の断面全体を風速に保つ
    fan->center         = {0.15f * a.cell * 20.0f, domain.y * 0.5f, domain.z * 0.5f};
    fan->size           = {0.3f, domain.y - 0.1f, domain.z - 0.1f};
    fan->inflowVelocity = {a.wind, 0.0f, 0.0f};
    e.addEmitter(fan);

    // ── トレーサ (demo_city_wind と同じ方式: 格子速度に乗せて流し、軌跡を記録) ──
    std::mt19937 rng(a.seed);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    struct Tracer {
      glm::vec3 p;
      uint32_t gen;
      int age;
    };
    std::vector<Tracer> tr(size_t(a.tracers));
    const glm::uvec3 res = e.gridRes();
    std::vector<glm::vec4> vel(e.totalCells());
    // 車の SDF をセル中心 (回転後) でトリリニア参照して、トレーサが車体に入ったか判定する (CPU)
    auto carSdf = [&](glm::vec3 p, float angle) {
      const float c = std::cos(angle), s = std::sin(angle);
      const glm::vec3 r(p.x - pivot.x, p.y, p.z - pivot.y);
      const glm::vec3 q(c * r.x - s * r.z, r.y, s * r.x + c * r.z);
      glm::vec3 g = (q - lmin) / lcell;
      if(g.x < 0 || g.y < 0 || g.z < 0 || g.x > lres.x - 1.001f || g.y > lres.y - 1.001f || g.z > lres.z - 1.001f) return 1.0f;
      const glm::uvec3 i = glm::uvec3(g);
      return local[i.x + size_t(lres.x) * (i.y + size_t(lres.y) * i.z)];
    };
    auto spawn = [&](Tracer& t) {
      t.p   = {0.35f, 0.04f + 1.45f * U(rng), pivot.y + 4.4f * (U(rng) - 0.5f)};  // 車の高さ (ウイング上端 約 1.5m) まで、幅は回転した車全体 (±2.2m) を覆う
      t.age = 0;
      t.gen++;
    };
    for(auto& t : tr) {
      t.gen = 0;
      spawn(t);
      t.gen = 0;
      t.p.x += (domain.x - 1.0f) * U(rng);
      t.age = int(U(rng) * a.tracer_life);
    }
    auto sample = [&](glm::vec3 p) {
      glm::vec3 c        = glm::clamp(p / a.cell - 0.5f, glm::vec3(0.0f), glm::vec3(res) - 1.001f);
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
    FILE* trf = std::fopen((a.out + "/tracers.bin").c_str(), "wb");  // "TRC1" | u32 steps | u32 M | f32 dt | f32 domain[3] | steps x M x { f32 xyz, u32 gen }
    {
      const uint32_t hdr[2] = {0, uint32_t(tr.size())};
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
      const float angle = e.rotatingColliderAngle();
      const float vcap  = 2.0f * a.wind;
      auto vlim         = [&](glm::vec3 v) { return glm::length(v) > vcap ? v * (vcap / glm::length(v)) : v; };
      for(size_t i = 0; i < tr.size(); i++) {
        Tracer& t = tr[i];
        const glm::vec3 v1 = vlim(sample(t.p)), v2 = vlim(sample(t.p + 0.5f * a.dt * v1));
        t.p += a.dt * v2;
        t.age++;
        if(t.p.x > domain.x - 0.3f || t.p.y > domain.y - 0.2f || t.p.y < 0.03f || t.p.z < 0.2f || t.p.z > domain.z - 0.2f || t.age > a.tracer_life || carSdf(t.p, angle) < 0.0f) spawn(t);
        std::memcpy(&rec[i * 4], &t.p, 12);
        std::memcpy(&rec[i * 4 + 3], &t.gen, 4);
      }
      if(f >= a.warmup) {
        std::fwrite(rec.data(), 4, rec.size(), trf);
        steps++;
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
