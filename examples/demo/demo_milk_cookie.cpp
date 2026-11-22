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
  float& cfl             = kwarg("cfl", "CFL number used to pick the substep count (APIC/FLIP aside, PIC is stable for the elastic cookie only below about 0.03 in c*dt/dx)").set_default(0.04f);
  float& blade_speed     = kwarg("blade-speed", "blender blade angular speed [rad/s]").set_default(24.0f);
  float& blade_start     = kwarg("blade-start", "time [s] at which the blade starts (ramps up over 1s)").set_default(0.6f);
  float& milk_bulk       = kwarg("milk-bulk", "milk bulk modulus K [Pa] (larger = less compressible but more substeps)").set_default(4e4f);
  float& cookie_E        = kwarg("cookie-E", "cookie Young's modulus [Pa]").set_default(3e5f);
  float& cookie_cohesion = kwarg("cookie-cohesion", "cookie Drucker-Prager cohesion [Pa] (holds the disc together; smaller = crumbles easier)").set_default(2e4f);
  float& cookie_friction = kwarg("cookie-friction", "cookie Drucker-Prager M (tan of the friction angle)").set_default(0.8f);
  int& cookies           = kwarg("cookies", "number of cookies").set_default(6);
  float& fp_scale        = kwarg("fp-scale", "MPM fixed-point scale (0 = auto from the particle mass)").set_default(0.0f);
  float& drop_slowmo     = kwarg("drop-slowmo", "slow-motion factor while the cookies fall (the first --drop-seconds of simulated time)").set_default(4.0f);
  float& drop_seconds    = kwarg("drop-seconds", "simulated seconds that are played in slow motion").set_default(0.6f);
  float& cookie_radius   = kwarg("cookie-radius", "cookie disc radius [m]").set_default(0.065f);
  float& cookie_thick    = kwarg("cookie-thickness", "cookie disc thickness [m] (needs several cells: a surface layer of about 2 cells is sheared by the shared grid velocity at the milk interface)").set_default(0.04f);
  float& cookie_density  = kwarg("cookie-density", "cookie density [kg/m^3] (denser than the milk so that the cookies sink to the blade)").set_default(2200.0f);
  int& seed              = kwarg("seed", "random seed (cookie placement)").set_default(3);
};

// ジャー (円筒容器) の寸法 [m]。ドメインは 1 x 1 x 1。
static constexpr float kCx = 0.5f, kCz = 0.5f;
static constexpr float kJarR = 0.25f;      // 内半径
static constexpr float kFloorY = 0.05f;    // 底面
static constexpr float kRimY = 0.62f;      // ジャーの縁 (これより上は壁なし)
static constexpr float kMilkTop = 0.33f;   // ミルクの初期液面

static constexpr float kShaftR = 0.014f;
static constexpr float kBladeHalf[3] = {0.17f, 0.17f, 0.008f}; // 底から液面の上まで届く縦長のパドル (クッキーは液面に浮いて底の刃に届かない)
static constexpr float kBladeY = 0.23f;

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
    // ミルク: ジャーの内側を液面まで格子状に満たす (壁・底から半粒子分空ける)。刃と軸の体積には入れない (初期の重なりで粒子が弾き出されるため)
    auto insideBlade = [&](float x, float y, float z) {
      const float m = spacing, dx = x - kCx, dz = z - kCz;
      if(std::hypot(dx, dz) < kShaftR + m) return true;
      if(std::fabs(y - (kBladeY)) > kBladeHalf[1] + m) return false;
      return (std::fabs(dx) < kBladeHalf[0] + m && std::fabs(dz) < kBladeHalf[2] + m) || (std::fabs(dz) < kBladeHalf[0] + m && std::fabs(dx) < kBladeHalf[2] + m);
    };
    for(float y = kFloorY + 0.5f * spacing; y < kMilkTop; y += spacing)
      for(float z = kCz - kJarR; z < kCz + kJarR; z += spacing)
        for(float x = kCx - kJarR; x < kCx + kJarR; x += spacing)
          if(std::hypot(x - kCx, z - kCz) < kJarR - 0.5f * spacing && !insideBlade(x, y, z)) {
            pos.emplace_back(x, y, z, Vp);
            vel.emplace_back(0.0f, 0.0f, 0.0f, matW(0u));
          }
    const size_t nMilk = pos.size();

    // クッキー: 円盤 を軸まわりの円周上に、高さを散らして置く。初速なしで自由落下させ、液面に順に着水する
    std::mt19937 rng(uint32_t(a.seed));
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    const float cr = a.cookie_radius, ch = 0.5f * a.cookie_thick; // 半径, 半厚
    for(int i = 0; i < a.cookies; i++) {
      const float ang = 2.3999632f * float(i); // 黄金角のらせん: 近い高さの隣同士が水平に離れ、初期に重ならない (重なると過圧縮で降伏する)
      const float rad = 0.15f;
      const glm::vec3 c(kCx + rad * std::cos(ang), 0.38f + 0.06f * float(i), kCz + rad * std::sin(ang));
      const glm::quat q = glm::angleAxis(0.6f * (U(rng) - 0.5f), glm::normalize(glm::vec3(U(rng) - 0.5f, 0.2f, U(rng) - 0.5f) + glm::vec3(1e-3f)));
      for(float ly = -ch; ly < ch; ly += spacing)
        for(float lz = -cr; lz < cr; lz += spacing)
          for(float lx = -cr; lx < cr; lx += spacing)
            if(std::hypot(lx, lz) < cr) {
              pos.emplace_back(c + q * glm::vec3(lx, ly + 0.5f * spacing, lz), Vp);
              vel.emplace_back(0.0f, 0.0f, 0.0f, matW(1u));
            }
    }
    std::printf("particles: %zu (milk %zu, cookie %zu)  grid: %ux%ux%u  cell %.4f m\n", pos.size(), nMilk, pos.size() - nMilk, G.x, G.y, G.z, cell);

    // ── 材料とサブステップ ──────────────────────────────────────────────────
    const float rhoMilk = 1030.0f, rhoCookie = a.cookie_density, nuCookie = 0.25f;
    MaterialParams milk = presetWater(rhoMilk, a.milk_bulk);
    MaterialParams cookie = presetJelly(a.cookie_E, nuCookie, rhoCookie);
    // クッキーは Drucker-Prager (粘着力つき): Von Mises の弾性体は空気との境界の軽い格子点で自由落下でも発散した (実測)
    cookie.model      = uint32_t(MaterialModel::DRUCKER_PRAGER);
    cookie.M_friction = a.cookie_friction;
    cookie.q_cohesion = a.cookie_cohesion;
    const float cMilk   = std::sqrt(a.milk_bulk / rhoMilk);
    const float cCookie = std::sqrt((cookie.lambda + 2.0f * cookie.mu) / rhoCookie);
    const float vFlow   = 3.0f; // 想定最大流速 [m/s] (刃の先端速度 + 落下)
    const float dtFrame = 1.0f / a.fps;
    const float cMax    = std::max(cMilk, cCookie) + vFlow;
    // 刃を 1 ブロックあたり 0.1rad 以下しか回さないよう、フレームをブロックに分けて刃の姿勢を更新する (落下区間は dt が小さいのでブロックも少ない)
    auto stepPlan = [&](float dtF, int& blocks, int& subPerBlock) {
      blocks      = std::max(1, int(std::ceil(a.blade_speed * dtF / 0.1f)));
      subPerBlock = (std::max(blocks, int(std::ceil(dtF * cMax / (a.cfl * cell)))) + blocks - 1) / blocks;
    };
    int blocks0, sub0;
    stepPlan(dtFrame, blocks0, sub0);
    std::printf("c_milk %.1f m/s, c_cookie %.1f m/s -> %d substeps/frame (%d blocks x %d)\n", cMilk, cCookie, blocks0 * sub0, blocks0, sub0);

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
    e.numSubsteps = sub0;
    e.flip_ratio  = 0.0f; // PIC: APICは弾性体の自由落下でも発散し (サブステップを増やしても)、FLIPはミルクの体積が潰れる (実測)
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
      cols.addCylinder({kCx, 0.5f * (kFloorY + 0.95f), kCz}, kShaftR, 0.5f * (0.95f - kFloorY), 0.0f, 0.4f, {0, 0, 0}, glm::quat(1, 0, 0, 0), w);
      for(int k = 0; k < 2; k++) {
        const glm::quat q = glm::angleAxis(bladeAngle + float(k) * 1.5707963f, glm::vec3(0, 1, 0));
        cols.addBox({kCx, kBladeY, kCz}, {kBladeHalf[0], kBladeHalf[1], kBladeHalf[2]}, 0.0f, 0.4f, {0, 0, 0}, q, w);
      }
      e.setColliders(cols);
    };

    const uint32_t N = e.liveParticleCount();
    std::vector<glm::vec4> mp, mv, cp, cv;
    float angle = 0.0f, simT = 0.0f;
    FILE* bladeLog = std::fopen((a.out + "/blade.txt").c_str(), "w");
    auto t0     = std::chrono::steady_clock::now();
    for(int f = 0; f <= a.frames; f++) {
      if(f > 0) {
        const float dtF = simT < a.drop_seconds ? dtFrame / a.drop_slowmo : dtFrame;
        int blocks, sub;
        stepPlan(dtF, blocks, sub);
        e.numSubsteps = sub;
        for(int b = 0; b < blocks; b++) {
          const float t     = simT + dtF * float(b) / float(blocks);
          const float ramp  = std::clamp((t - a.blade_start) / 1.0f, 0.0f, 1.0f);
          const float omega = a.blade_speed * ramp;
          buildColliders(angle, omega);
          VkCommandBuffer cmd = ctx.beginCmd();
          e.step(cmd, dtF / float(blocks));
          ctx.submitCmd(cmd);
          angle += omega * dtF / float(blocks);
        }
        simT += dtF;
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
