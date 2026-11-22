// .nbp は材料 id を持たないので、Blender 側で材料別に描けるよう milk_*.nbp と cookie_*.nbp に分けて dump する。
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
#include <cstring>
#include <filesystem>
#include <random>

static const std::string SHADER_DIR_STR = SHADER_DIR;

struct Args : public argparse::Args {
  std::string& out       = kwarg("out", "output directory").set_default(std::string("out/milk_cookie"));
  int& frames            = kwarg("frames", "output frames").set_default(288);
  float& fps             = kwarg("fps", "output frame rate").set_default(24.0f);
  float& spacing         = kwarg("spacing", "particle spacing [m] (0.004 = about 0.9M particles, 0.0018 = about 10M)").set_default(0.004f);
  float& cfl             = kwarg("cfl", "CFL number used to pick the substep count").set_default(0.25f);
  float& blade_speed     = kwarg("blade-speed", "blender blade angular speed [rad/s]").set_default(10.0f);
  float& blade_start     = kwarg("blade-start", "time [s] at which the blade starts (ramps up over 1s)").set_default(0.6f);
  float& milk_bulk       = kwarg("milk-bulk", "milk bulk modulus K [Pa] (larger = less compressible but more substeps)").set_default(6e4f);
  float& cookie_E        = kwarg("cookie-E", "cookie Young's modulus [Pa]").set_default(1.5e5f);
  float& cookie_yield    = kwarg("cookie-yield", "cookie Von Mises yield stress [Pa] (smaller = crumbles easier)").set_default(3e3f);
  int& cookies           = kwarg("cookies", "number of cookies").set_default(10);
  float& fp_scale        = kwarg("fp-scale", "MPM fixed-point scale (0 = auto from the particle mass)").set_default(0.0f);
  int& seed              = kwarg("seed", "random seed (cookie placement)").set_default(3);
};

// ジャー (円筒容器) の寸法 [m]。ドメインは 1 x 1 x 1。
static constexpr float kCx = 0.5f, kCz = 0.5f;
static constexpr float kJarR = 0.25f;      // 内半径
static constexpr float kFloorY = 0.05f;    // 底面
static constexpr float kRimY = 0.45f;      // ジャーの縁 (これより上は壁なし)
static constexpr float kMilkTop = 0.33f;   // ミルクの初期液面

static float matW(uint32_t id) {
  float f;
  std::memcpy(&f, &id, 4);
  return f;
}

int main(int argc, char* argv[]) {
  auto a = argparse::parse<Args>(argc, argv);
  try {
    const glm::vec3 DOMAIN(1.0f);
    const float spacing = a.spacing;
    const float cell    = spacing * 2.0f;
    const float Vp      = spacing * spacing * spacing;
    const glm::uvec3 G(DOMAIN / cell);
    std::filesystem::create_directories(a.out);

    // ── 粒子の初期配置 ─────────────────────────────────────────────────────
    std::vector<glm::vec4> pos, vel;
    // ミルク: ジャーの内側を液面まで格子状に満たす (壁・底から半粒子分空ける)
    for(float y = kFloorY + 0.5f * spacing; y < kMilkTop; y += spacing)
      for(float z = kCz - kJarR; z < kCz + kJarR; z += spacing)
        for(float x = kCx - kJarR; x < kCx + kJarR; x += spacing)
          if(std::hypot(x - kCx, z - kCz) < kJarR - 0.5f * spacing) {
            pos.emplace_back(x, y, z, Vp);
            vel.emplace_back(0.0f, 0.0f, 0.0f, matW(0u));
          }
    const size_t nMilk = pos.size();

    // クッキー: 円盤 (半径 5.5cm, 厚さ 1.8cm) を軸まわりの円周上に高さを散らして置き、初速で液面へ突入させる
    std::mt19937 rng(uint32_t(a.seed));
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    const float cr = 0.06f, ch = 0.011f; // 半径, 半厚
    for(int i = 0; i < a.cookies; i++) {
      const float ang = 6.2831853f * (float(i) / float(a.cookies)) + 0.4f;
      const float rad = 0.115f + 0.03f * U(rng);
      const glm::vec3 c(kCx + rad * std::cos(ang), 0.46f + 0.075f * float(i), kCz + rad * std::sin(ang));
      const glm::quat q = glm::angleAxis(0.5f * (U(rng) - 0.5f) * 2.0f, glm::normalize(glm::vec3(U(rng) - 0.5f, 0.2f, U(rng) - 0.5f) + glm::vec3(1e-3f)));
      for(float ly = -ch; ly < ch; ly += spacing)
        for(float lz = -cr; lz < cr; lz += spacing)
          for(float lx = -cr; lx < cr; lx += spacing)
            if(std::hypot(lx, lz) < cr) {
              pos.emplace_back(c + q * glm::vec3(lx, ly + 0.5f * spacing, lz), Vp);
              vel.emplace_back(0.0f, -3.0f, 0.0f, matW(1u));
            }
    }
    std::printf("particles: %zu (milk %zu, cookie %zu)  grid: %ux%ux%u  cell %.4f m\n", pos.size(), nMilk, pos.size() - nMilk, G.x, G.y, G.z, cell);

    // ── 材料とサブステップ ──────────────────────────────────────────────────
    const float rhoMilk = 1030.0f, rhoCookie = 850.0f, nuCookie = 0.25f;
    MaterialParams milk = presetWater(rhoMilk, a.milk_bulk);
    MaterialParams cookie = presetJelly(a.cookie_E, nuCookie, rhoCookie);
    cookie.model  = uint32_t(MaterialModel::VON_MISES);
    cookie.q_max  = a.cookie_yield;
    const float cMilk   = std::sqrt(a.milk_bulk / rhoMilk);
    const float cCookie = std::sqrt((cookie.lambda + 2.0f * cookie.mu) / rhoCookie);
    const float vFlow   = 6.0f; // 想定最大流速 [m/s] (刃の先端速度 + 落下)
    const float dtFrame = 1.0f / a.fps;
    // 刃を 1 フレームで回しすぎない (1 ブロックあたり 0.1rad 以下) よう、フレームをブロックに分けて刃の姿勢を更新する
    const int blocks      = std::max(1, int(std::ceil(a.blade_speed * dtFrame / 0.1f)));
    const int subTotal    = std::max(blocks, int(std::ceil(dtFrame * (std::max(cMilk, cCookie) + vFlow) / (a.cfl * cell))));
    const int subPerBlock = (subTotal + blocks - 1) / blocks;
    std::printf("c_milk %.1f m/s, c_cookie %.1f m/s -> %d substeps/frame (%d blocks x %d)\n", cMilk, cCookie, subPerBlock * blocks, blocks, subPerBlock);

    // 固定小数点スケール: 粒子質量が小さいと P2G の運動量が丸められる。雪崩デモ (質量 0.13kg, 2e7) と同じ比になるよう質量から決める
    const float fp = a.fp_scale > 0.0f ? a.fp_scale : 2e7f * 0.13f / (rhoMilk * Vp);

    HeadlessCtx ctx;
    ctx.init();
    MPMEngine e;
    e.domainSize   = DOMAIN;
    e.cellSize     = cell;
    e.maxParticles = uint32_t(pos.size());
    e.setFixedPointScale(fp);
    e.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADER_DIR_STR);
    e.numSubsteps = subPerBlock;
    e.flip_ratio  = -1.0f; // APIC: 角運動量を保存して撹拌の渦を保つ
    e.addForce(GravityForce::FromDirection({0.0f, -1.0f, 0.0f}, 9.8f));
    e.setMaterials({milk, cookie});

    // ジャーの SDF (内側が自由空間): 底面と円筒壁。縁より上は壁なし
    LocalMeshSDF grid;
    const uint32_t res = std::max({G.x, G.y, G.z});
    grid.res           = res;
    grid.cellSize      = cell;
    grid.localMin      = glm::vec3(0.0f);
    grid.data.assign(size_t(res) * res * res, 1e9f);
    for(uint32_t iz = 0; iz < G.z; iz++)
      for(uint32_t iy = 0; iy < G.y; iy++)
        for(uint32_t ix = 0; ix < G.x; ix++) {
          const float x = (ix + 0.5f) * cell, y = (iy + 0.5f) * cell, z = (iz + 0.5f) * cell;
          float d = y - kFloorY;
          if(y < kRimY) d = std::min(d, kJarR - std::hypot(x - kCx, z - kCz));
          grid.data[(size_t(iz) * res + iy) * res + ix] = d;
        }
    const uint32_t sdfIdx = e.uploadColliderMeshSDF(grid);
    e.appendParticles(pos, vel);

    auto buildColliders = [&](float bladeAngle, float omega) {
      ColliderSet cols;
      cols.addMeshSDF(sdfIdx, grid, glm::vec3(0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), {0, 0, 0}, {0, 0, 0}, 0.0f, 0.2f);
      cols.addPlane({DOMAIN.x * 0.5f, 0.0f, DOMAIN.z * 0.5f}, {0, 1, 0}, 0.0f, 0.5f);
      cols.addPlane({0.0f, DOMAIN.y * 0.5f, DOMAIN.z * 0.5f}, {1, 0, 0}, 0.0f, 0.1f);
      cols.addPlane({DOMAIN.x, DOMAIN.y * 0.5f, DOMAIN.z * 0.5f}, {-1, 0, 0}, 0.0f, 0.1f);
      cols.addPlane({DOMAIN.x * 0.5f, DOMAIN.y * 0.5f, 0.0f}, {0, 0, 1}, 0.0f, 0.1f);
      cols.addPlane({DOMAIN.x * 0.5f, DOMAIN.y * 0.5f, DOMAIN.z}, {0, 0, -1}, 0.0f, 0.1f);
      // ブレンダー: 軸 (円柱) と底近くの十字の刃 (薄い BOX 2 枚)。Y 軸まわりに回転
      const glm::vec3 w(0.0f, omega, 0.0f);
      cols.addCylinder({kCx, 0.5f * (kFloorY + 0.95f), kCz}, 0.014f, 0.5f * (0.95f - kFloorY), 0.0f, 0.4f, {0, 0, 0}, glm::quat(1, 0, 0, 0), w);
      for(int k = 0; k < 2; k++) {
        const glm::quat q = glm::angleAxis(bladeAngle + float(k) * 1.5707963f, glm::vec3(0, 1, 0));
        cols.addBox({kCx, kFloorY + 0.045f, kCz}, {0.17f, 0.03f, 0.008f}, 0.0f, 0.4f, {0, 0, 0}, q, w);
      }
      e.setColliders(cols);
    };

    const uint32_t N = e.liveParticleCount();
    std::vector<glm::vec4> mp, mv, cp, cv;
    float angle = 0.0f;
    FILE* bladeLog = std::fopen((a.out + "/blade.txt").c_str(), "w");
    auto t0     = std::chrono::steady_clock::now();
    for(int f = 0; f <= a.frames; f++) {
      if(f > 0) {
        for(int b = 0; b < blocks; b++) {
          const float t     = (float(f - 1) + float(b) / float(blocks)) * dtFrame;
          const float ramp  = std::clamp((t - a.blade_start) / 1.0f, 0.0f, 1.0f);
          const float omega = a.blade_speed * ramp;
          buildColliders(angle, omega);
          VkCommandBuffer cmd = ctx.beginCmd();
          e.step(cmd, dtFrame / float(blocks));
          ctx.submitCmd(cmd);
          angle += omega * dtFrame / float(blocks);
        }
      }
      ctx.readBuffer(e.getPositionBuffer(), 0, pos.data(), size_t(N) * sizeof(glm::vec4));
      ctx.readBuffer(e.getVelocityBuffer(), 0, vel.data(), size_t(N) * sizeof(glm::vec4));
      mp.clear(); mv.clear(); cp.clear(); cv.clear();
      for(uint32_t i = 0; i < N; i++) {
        uint32_t id;
        std::memcpy(&id, &vel[i].w, 4);
        (id == 0u ? mp : cp).push_back(pos[i]);
        (id == 0u ? mv : cv).push_back(vel[i]);
      }
      char path[512];
      std::snprintf(path, sizeof path, "%s/milk_%05d.nbp", a.out.c_str(), f);
      writeNbp(path, mp, mv, glm::vec3(0.0f), DOMAIN, 4.0f);
      std::snprintf(path, sizeof path, "%s/cookie_%05d.nbp", a.out.c_str(), f);
      writeNbp(path, cp, cv, glm::vec3(0.0f), DOMAIN, 4.0f);
      std::printf("frame %d/%d  %.1fs elapsed\n", f, a.frames, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      std::fprintf(bladeLog, "%d %.6f\n", f, angle);
      std::fflush(bladeLog);
      std::fflush(stdout);
    }
    std::fclose(bladeLog);
    e.cleanup();
    ctx.cleanup();
  } catch(const std::exception& ex) {
    std::fprintf(stderr, "Fatal: %s\n", ex.what());
    return 1;
  }
  return 0;
}
