// .nbp は材料 id を持たないので、Blender 側で材料別に描けるよう milk_*.nbp と cookie_*.nbp に分けて dump する。
#include "Collider.h"
#include "Force.h"
#include "MaterialParams.h"
#include "NbpDump.h"
#include "engine/MPMEngine.h"
#include "helpers/HeadlessCtx.h"

#include <argparse/argparse.hpp>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

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
  float& blade_speed     = kwarg("blade-speed", "mixer angular speed [rad/s]").set_default(12.0f);
  float& blade_start     = kwarg("blade-start", "time [s] at which the mixer starts").set_default(0.0f);
  float& blade_ramp      = kwarg("blade-ramp", "time [s] over which the mixer speeds up (0 = full speed from the first frame)").set_default(0.0f);
  float& tilt_deg        = kwarg("tilt", "mixer axis tilt from vertical [deg] (an inclined axis also mixes up and down)").set_default(20.0f);
  float& milk_flip       = kwarg("milk-flip", "FLIP blend of the milk (0 = pure PIC: thick, larger = less numerical viscosity but noisier)").set_default(0.8f);
  float& milk_visc       = kwarg("milk-visc", "milk viscosity [Pa s]").set_default(0.003f);
  float& milk_bulk       = kwarg("milk-bulk", "milk bulk modulus K [Pa] (larger = less compressible but more substeps)").set_default(4e4f);
  float& cookie_E        = kwarg("cookie-E", "cookie Young's modulus [Pa]").set_default(3e5f);
  float& cookie_cohesion = kwarg("cookie-cohesion", "cookie Drucker-Prager cohesion [Pa] (holds the disc together; smaller = crumbles easier)").set_default(1e5f);
  float& cookie_friction = kwarg("cookie-friction", "cookie Drucker-Prager M (tan of the friction angle)").set_default(1.0f);
  int& cookies           = kwarg("cookies", "number of cookies").set_default(4);
  float& fp_scale        = kwarg("fp-scale", "MPM fixed-point scale (0 = auto from the particle mass)").set_default(0.0f);
  float& drop_slowmo     = kwarg("drop-slowmo", "slow-motion factor while the cookies fall (the first --drop-seconds of simulated time)").set_default(4.0f);
  float& drop_seconds    = kwarg("drop-seconds", "simulated seconds that are played in slow motion").set_default(0.6f);
  float& cookie_radius   = kwarg("cookie-radius", "cookie disc radius [m]").set_default(0.104f);
  float& cookie_thick    = kwarg("cookie-thickness", "cookie disc thickness [m] (needs several cells: a surface layer of about 2 cells is sheared by the shared grid velocity at the milk interface)").set_default(0.04f);
  float& cookie_density  = kwarg("cookie-density", "cookie density [kg/m^3] (denser than the milk so that the cookies sink to the blade)").set_default(2200.0f);
  float& break_strain    = kwarg("break-strain", "plastic deformation per step above which a cookie particle turns into crumb (smaller = crumbles easier, 0 = never)").set_default(0.0006f);
  float& crumb_cohesion  = kwarg("crumb-cohesion", "Drucker-Prager cohesion [Pa] of the crumbs").set_default(500.0f);
  int& cookie_slots      = kwarg("cookie-slots", "1 = give every cookie its own material slot (tests whether separate ids stop cookies from fusing)").set_default(0);
  float& crumb_density   = kwarg("crumb-density", "crumb density [kg/m^3] (0 = same as the cookie; wet biscuit sinks)").set_default(0.0f);
  float& crumb_friction  = kwarg("crumb-friction", "crumb Drucker-Prager M (0 = same as the cookie)").set_default(0.0f);
  float& slat_pitch      = kwarg("slat-pitch", "pitch angle of the mixer slats [rad] (sign flips the pumping direction)").set_default(0.61f);
  float& crumb_settle    = kwarg("crumb-settle", "crumb settling speed relative to the milk flow [m/s] (a single velocity field cannot make heavy crumbs sink)").set_default(0.04f);
  float& mix_y           = kwarg("mix-y", "height of the mixer pivot [m]").set_default(0.21f);
  int& seed              = kwarg("seed", "random seed (cookie placement)").set_default(3);
};

// ジャー (円筒容器) の寸法 [m]。ドメインは 1 x 1 x 1。
static constexpr float kCx = 0.5f, kCz = 0.5f;
static constexpr float kJarR = 0.25f;      // 内半径
static constexpr float kFloorY = 0.05f;    // 底面
static constexpr float kRimY = 0.62f;      // ジャーの縁 (これより上は壁なし)
static constexpr float kMilkTop = 0.33f;   // ミルクの初期液面

static constexpr float kShaftR = 0.014f;
// ミキサーは傾いた軸まわりに回る。ビーターは円周上に並べた薄い板の籠 (中は空洞、板の間は隙間)
static glm::vec3 kMixO(kCx - 0.02f, 0.21f, kCz);
static float kSlatPitch = 0.61f; // 板のピッチ角 [rad]: 符号で液を押し下げる/押し上げる向きが変わる
static constexpr int kSlats = 6;
static constexpr float kCageR = 0.10f, kSlatHalf[3] = {0.008f, 0.11f, 0.04f}, kShaftTop = 0.75f;
struct MixPart { glm::vec3 c, half; glm::quat q; };
static std::vector<MixPart> slats() {
  std::vector<MixPart> v;
  for(int k = 0; k < kSlats; k++) {
    const float phi = 6.2831853f * float(k) / float(kSlats);
    v.push_back({glm::vec3(kCageR * std::cos(phi), 0.0f, kCageR * std::sin(phi)), glm::vec3(kSlatHalf[0], kSlatHalf[1], kSlatHalf[2]), glm::angleAxis(-phi + kSlatPitch, glm::vec3(0, 1, 0))});
  }
  return v;
}

static float matW(uint32_t id) {
  float f;
  std::memcpy(&f, &id, 4);
  return f;
}

// 評価用: 破片の高さ分布と、固体 (クッキー+破片) の最大連結成分の割合 (癒着の指標)
static void printStats(const std::vector<glm::vec4>& cp, const std::vector<glm::vec4>& mp, uint32_t crumbSlot, const std::vector<glm::vec4>& cv, float spacing) {
  std::vector<float> ys;
  for(size_t i = 0; i < cp.size(); i++) {
    uint32_t id;
    std::memcpy(&id, &cv[i].w, 4);
    if(id == crumbSlot) ys.push_back(cp[i].y);
  }
  std::sort(ys.begin(), ys.end());
  auto q = [&](float t) { return ys.empty() ? 0.0f : ys[size_t(t * float(ys.size() - 1))]; };
  float top = 0.0f, maxY = 0.0f; // ミルク液面 (上位 2% の高さ) と比べる
  { std::vector<float> my; for(auto& p : mp) my.push_back(p.y); std::sort(my.begin(), my.end()); top = my.empty() ? 0.0f : my[size_t(0.98f * float(my.size() - 1))]; maxY = my.empty() ? 0.0f : my.back(); }
  // 連結成分: 3 セル幅の格子に粒子を入れ、26 近傍で flood fill
  const float h = 3.0f * spacing;
  const int n = int(1.0f / h) + 1;
  std::unordered_map<int, int> cell;
  auto key = [&](int x, int y, int z) { return (z * n + y) * n + x; };
  for(auto& p : cp) cell[key(int(p.x / h), int(p.y / h), int(p.z / h))]++;
  std::unordered_set<int> seen;
  int best = 0;
  for(auto& kv : cell) {
    if(seen.count(kv.first)) continue;
    int tot = 0;
    std::vector<int> st{kv.first};
    seen.insert(kv.first);
    while(!st.empty()) {
      const int k = st.back(); st.pop_back();
      tot += cell[k];
      const int x = k % n, y = (k / n) % n, z = k / (n * n);
      for(int dz = -1; dz <= 1; dz++) for(int dy = -1; dy <= 1; dy++) for(int dx = -1; dx <= 1; dx++) {
        const int kk = key(x + dx, y + dy, z + dz);
        if(x + dx < 0 || y + dy < 0 || z + dz < 0 || !cell.count(kk) || seen.count(kk)) continue;
        seen.insert(kk); st.push_back(kk);
      }
    }
    best = std::max(best, tot);
  }
  std::printf("STAT crumbY p10 %.3f p50 %.3f p90 %.3f  milkTop %.3f  milkMax %.3f  largestSolid %.2f\n", q(0.1f), q(0.5f), q(0.9f), top, maxY, cp.empty() ? 0.0f : float(best) / float(cp.size()));
}

int main(int argc, char* argv[]) {
  auto a = argparse::parse<Args>(argc, argv);
  kMixO.y = a.mix_y;
  kSlatPitch = a.slat_pitch;
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
    const glm::quat qTilt = glm::angleAxis(glm::radians(a.tilt_deg), glm::vec3(0, 0, 1));
    const auto parts      = slats();
    auto insideBlade = [&](float x, float y, float z) {
      const float m = spacing;
      const glm::vec3 pl = glm::inverse(qTilt) * (glm::vec3(x, y, z) - kMixO);
      if(std::hypot(pl.x, pl.z) < kShaftR + m && pl.y > kSlatHalf[1] - m) return true;
      for(const auto& pt : parts) {
        const glm::vec3 d = glm::inverse(pt.q) * (pl - pt.c);
        if(std::fabs(d.x) < pt.half.x + m && std::fabs(d.y) < pt.half.y + m && std::fabs(d.z) < pt.half.z + m) return true;
      }
      return false;
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
      const float ang = 2.3999632f * float(i) - 0.45f; // 黄金角のらせん: 近い高さの隣同士が水平に離れ、初期に重ならない (重なると過圧縮で降伏する)
      const float rad = std::max(0.0f, 0.235f - cr);
      const glm::vec3 c(kCx + rad * std::cos(ang), 0.38f + 0.06f * float(i), kCz + rad * std::sin(ang));
      const glm::quat q = glm::angleAxis(0.6f * (U(rng) - 0.5f), glm::normalize(glm::vec3(U(rng) - 0.5f, 0.2f, U(rng) - 0.5f) + glm::vec3(1e-3f)));
      for(float ly = -ch; ly < ch; ly += spacing)
        for(float lz = -cr; lz < cr; lz += spacing)
          for(float lx = -cr; lx < cr; lx += spacing)
            if(std::hypot(lx, lz) < cr) {
              pos.emplace_back(c + q * glm::vec3(lx, ly + 0.5f * spacing, lz), Vp);
              vel.emplace_back(0.0f, 0.0f, 0.0f, matW(a.cookie_slots ? 1u + uint32_t(i) : 1u));
            }
    }
    std::printf("particles: %zu (milk %zu, cookie %zu)  grid: %ux%ux%u  cell %.4f m\n", pos.size(), nMilk, pos.size() - nMilk, G.x, G.y, G.z, cell);

    // ── 材料とサブステップ ──────────────────────────────────────────────────
    const uint32_t nCookieSlots = a.cookie_slots ? uint32_t(a.cookies) : 1u, crumbSlot = nCookieSlots + 1u;
    const float rhoMilk = 1030.0f, rhoCookie = a.cookie_density, nuCookie = 0.25f;
    MaterialParams milk = presetWater(rhoMilk, a.milk_bulk);
    milk.viscosity      = a.milk_visc;
    milk.flipOverride   = a.milk_flip;
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
      blocks      = std::max(1, int(std::ceil(a.blade_speed * dtF / 0.05f)));
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
    e.flip_ratio  = a.milk_flip > 0.0f ? 0.5f : 0.0f; // 全体は PIC (APIC は弾性体が発散、FLIP は体積が潰れる (実測))。FLIP 混合はミルクだけ材料ごとに指定する
    e.addForce(GravityForce::FromDirection({0.0f, -1.0f, 0.0f}, 9.8f));
    // 最後の slot: 砕けた破片 (粘着力の低い DP)。cookie が降伏すると遷移する
    MaterialParams crumb = cookie;
    crumb.q_cohesion     = a.crumb_cohesion;
    crumb.settleSpeed = a.crumb_settle;
    if(a.crumb_density > 0.0f) crumb.rho0 = a.crumb_density;
    if(a.crumb_friction > 0.0f) crumb.M_friction = a.crumb_friction;
    if(a.break_strain > 0.0f) {
      cookie.breakMaterial = crumbSlot + 1u;
      cookie.breakStrain   = a.break_strain;
    }
    cookie.flipOverride = 1e-6f;
    crumb.flipOverride  = 1e-6f;
    std::vector<MaterialParams> mats{milk};
    for(uint32_t k = 0; k < nCookieSlots; k++) mats.push_back(cookie);
    mats.push_back(crumb);
    e.setMaterials(mats);

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
      // ミキサー: 傾いた軸 (円柱) と、軸まわりに回る板の籠。各部の線速度は軸上の点 kMixO まわりの剛体回転から求める
      const glm::quat qM = qTilt * glm::angleAxis(bladeAngle, glm::vec3(0, 1, 0));
      const glm::vec3 w  = omega * (qTilt * glm::vec3(0, 1, 0));
      auto vAt = [&](const glm::vec3& pw) { return glm::cross(w, pw - kMixO); };
      const float shaftHh = 0.5f * (kShaftTop - kSlatHalf[1]);
      const glm::vec3 sc  = kMixO + qM * glm::vec3(0.0f, kSlatHalf[1] + shaftHh, 0.0f);
      cols.addCylinder(sc, kShaftR, shaftHh, 0.0f, 0.4f, vAt(sc), qM, w);
      for(const auto& pt : parts) {
        const glm::vec3 cw = kMixO + qM * pt.c;
        cols.addBox(cw, pt.half, 0.0f, 0.4f, vAt(cw), qM * pt.q, w);
      }
      e.setColliders(cols);
    };

    const uint32_t N = e.liveParticleCount();
    std::vector<glm::vec4> mp, mv, cp, cv;
    float angle = 0.0f, simT = 0.0f;
    if(FILE* mx = std::fopen((a.out + "/mixer.txt").c_str(), "w")) {
      std::fprintf(mx, "origin %f %f %f\ntilt %f\nshaft %f %f %f\n", kMixO.x, kMixO.y, kMixO.z, glm::radians(a.tilt_deg), kShaftR, kSlatHalf[1], kShaftTop);
      for(const auto& pt : parts) std::fprintf(mx, "slat %f %f %f %f %f %f %f\n", pt.c.x, pt.c.y, pt.c.z, pt.half.x, pt.half.y, pt.half.z, 2.0f * std::atan2(pt.q.y, pt.q.w));
      std::fclose(mx);
    }
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
          const float ramp  = a.blade_ramp > 0.0f ? std::clamp((t - a.blade_start) / a.blade_ramp, 0.0f, 1.0f) : (t >= a.blade_start ? 1.0f : 0.0f);
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
      uint32_t nCrumb = 0;
      for(uint32_t i = 0; i < N; i++) {
        uint32_t id;
        std::memcpy(&id, &vel[i].w, 4);
        nCrumb += id == crumbSlot;
        (id == 0u ? mp : cp).push_back(pos[i]);
        (id == 0u ? mv : cv).push_back(vel[i]);
      }
      char path[512];
      std::snprintf(path, sizeof path, "%s/milk_%05d.nbp", a.out.c_str(), f);
      writeNbp(path, mp, mv, glm::vec3(0.0f), DOMAIN, 4.0f);
      std::snprintf(path, sizeof path, "%s/cookie_%05d.nbp", a.out.c_str(), f);
      writeNbp(path, cp, cv, glm::vec3(0.0f), DOMAIN, 4.0f);
      if(f % 12 == 0) printStats(cp, mp, crumbSlot, cv, spacing);
      std::printf("frame %d/%d  %.1fs elapsed  crumb %u/%zu\n", f, a.frames, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), nCrumb, cp.size());
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
