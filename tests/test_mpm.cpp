// MPM単体テスト
// セクション1-2: CPU-onlyテスト (形状関数、構成式)
// セクション2-6: GPUテスト (P2G/GridUpdate/G2P/統合)
#include "core/Emitter.h"
#include "engine/MPMEngine.h"
#include "helpers/HeadlessCtx.h"
#include "helpers/MPMHarness.h"
#include <cmath>
#include <doctest/doctest.h>
#include <glm/glm.hpp>
#include <random>
#include <string>
#include <vector>

static const std::string SHADERS = SHADER_DIR;

// ═══════════════════════════════════════════════════════════════════════════
// CPU テスト — GPU不要の純粋な数学関数テスト
// ═══════════════════════════════════════════════════════════════════════════

// ── B-spline 補間関数 (mpm_common.glslと同一実装) ────────────────────────────

static float bspline2(float d) {
  float ad = std::abs(d);
  if(ad < 0.5f) return 0.75f - ad * ad;
  if(ad < 1.5f) {
    float t = 1.5f - ad;
    return 0.5f * t * t;
  }
  return 0.0f;
}

static float bspline2g(float d) {
  float ad = std::abs(d);
  if(ad < 0.5f) return -2.0f * d;
  if(ad < 1.5f) return (d < 0.0f ? 1.0f : -1.0f) * (1.5f - ad);
  return 0.0f;
}

// ── 対角変形勾配 F=diag(a,b,c) に対するHencky応力の解析解 ───────────────────
// SVDが自明 (U=V=I) なので直接計算できる
static glm::mat3 henckyStressDiag(float a, float b, float c, float mu, float lam) {
  float sa = std::max(std::abs(a), 1e-6f);
  float sb = std::max(std::abs(b), 1e-6f);
  float sc = std::max(std::abs(c), 1e-6f);
  float ea = std::log(sa), eb = std::log(sb), ec = std::log(sc);
  float trE = ea + eb + ec;
  float kx  = lam * trE + 2.0f * mu * ea;
  float ky  = lam * trE + 2.0f * mu * eb;
  float kz  = lam * trE + 2.0f * mu * ec;
  // U=I なので τ = diag(kx, ky, kz)
  return glm::mat3(kx, 0.0f, 0.0f, 0.0f, ky, 0.0f, 0.0f, 0.0f, kz);
}

// ─────────────────────────────────────────────────────────────────────────────
// テスト1-1: 形状関数の総和 (Partition of Unity)
// 任意のセル内座標で 27ノードの重み合計が 1.0 になること
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MPM Shape - 1-1: Partition of Unity") {
  const float h = 1.25f;
  // セル内の代表的な位置オフセット (セルサイズh未満)
  const std::vector<float> offsets = {0.1f, 0.3f, 0.6f, 0.9f, 1.1f};

  for(float ox : offsets)
    for(float oy : offsets)
      for(float oz : offsets) {
        // 粒子位置: セル2の内部に配置
        float xp = (2.0f + ox) * h;
        float yp = (2.0f + oy) * h;
        float zp = (2.0f + oz) * h;

        // G2Pシェーダーと同一のステンシルベース計算
        int basex = int(std::floor(xp / h)) - 1;
        int basey = int(std::floor(yp / h)) - 1;
        int basez = int(std::floor(zp / h)) - 1;

        float sumW = 0.0f;
        for(int iz = 0; iz < 3; ++iz)
          for(int iy = 0; iy < 3; ++iy)
            for(int ix = 0; ix < 3; ++ix) {
              // ノードセンター座標: (nijk + 0.5) * h
              float nx = (basex + ix + 0.5f) * h;
              float ny = (basey + iy + 0.5f) * h;
              float nz = (basez + iz + 0.5f) * h;
              float dx = (xp - nx) / h;
              float dy = (yp - ny) / h;
              float dz = (zp - nz) / h;
              sumW += bspline2(dx) * bspline2(dy) * bspline2(dz);
            }
        CHECK(std::abs(sumW - 1.0f) < 1e-5f);
      }
}

// ─────────────────────────────────────────────────────────────────────────────
// テスト1-2: 形状関数の勾配の総和 = 0
// 内力が相殺するために必要な性質
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MPM Shape - 1-2: Gradient Sum = 0") {
  const float h                    = 1.25f;
  const std::vector<float> offsets = {0.2f, 0.5f, 0.8f, 1.2f};

  for(float ox : offsets)
    for(float oy : offsets)
      for(float oz : offsets) {
        float xp = (3.0f + ox) * h;
        float yp = (3.0f + oy) * h;
        float zp = (3.0f + oz) * h;

        int basex = int(std::floor(xp / h)) - 1;
        int basey = int(std::floor(yp / h)) - 1;
        int basez = int(std::floor(zp / h)) - 1;

        float sumDwx = 0.0f, sumDwy = 0.0f, sumDwz = 0.0f;
        for(int iz = 0; iz < 3; ++iz)
          for(int iy = 0; iy < 3; ++iy)
            for(int ix = 0; ix < 3; ++ix) {
              float nx = (basex + ix + 0.5f) * h;
              float ny = (basey + iy + 0.5f) * h;
              float nz = (basez + iz + 0.5f) * h;
              float dx = (xp - nx) / h;
              float dy = (yp - ny) / h;
              float dz = (zp - nz) / h;
              float wx = bspline2(dx), wy = bspline2(dy), wz = bspline2(dz);
              sumDwx += bspline2g(dx) * wy * wz;
              sumDwy += wx * bspline2g(dy) * wz;
              sumDwz += wx * wy * bspline2g(dz);
            }
        CHECK(std::abs(sumDwx) < 1e-5f);
        CHECK(std::abs(sumDwy) < 1e-5f);
        CHECK(std::abs(sumDwz) < 1e-5f);
      }
}

// ─────────────────────────────────────────────────────────────────────────────
// テスト5-1: 構成式 — 変形なし (F=I) → 応力ゼロ
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MPM Constitutive - 5-1: Identity F => zero stress") {
  const float mu  = 5000.0f;
  const float lam = 11538.0f;
  // F = diag(1,1,1): Hencky歪み ε = log(1) = 0 → 応力 = 0
  glm::mat3 tau = henckyStressDiag(1.0f, 1.0f, 1.0f, mu, lam);
  for(int col = 0; col < 3; ++col)
    for(int row = 0; row < 3; ++row) CHECK(std::abs(tau[col][row]) < 1e-4f);
}

// ─────────────────────────────────────────────────────────────────────────────
// テスト5-2: 構成式 — 均一圧縮 (F=0.9*I) → 負の対角応力、零の非対角
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MPM Constitutive - 5-2: Uniform compression => negative diagonal stress") {
  const float mu  = 5000.0f;
  const float lam = 11538.0f;
  glm::mat3 tau   = henckyStressDiag(0.9f, 0.9f, 0.9f, mu, lam);

  // 対角成分が負 (圧縮反力)
  CHECK(tau[0][0] < -1.0f);
  CHECK(tau[1][1] < -1.0f);
  CHECK(tau[2][2] < -1.0f);
  // 等方圧縮なので各成分が等しい
  CHECK(std::abs(tau[0][0] - tau[1][1]) < 1.0f);
  CHECK(std::abs(tau[1][1] - tau[2][2]) < 1.0f);
  // 非対角成分がゼロ
  CHECK(std::abs(tau[0][1]) < 1e-4f);
  CHECK(std::abs(tau[0][2]) < 1e-4f);
  CHECK(std::abs(tau[1][2]) < 1e-4f);
  // 理論値: τ_xx = λ*trε + 2μ*ε_xx = λ*3*log(0.9) + 2μ*log(0.9) = (3λ+2μ)*log(0.9)
  float expected = (3.0f * lam + 2.0f * mu) * std::log(0.9f);
  CHECK(std::abs(tau[0][0] - expected) < std::abs(expected) * 0.01f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Emitter::sample() — GPU不要の純粋なCPUサンプリングテスト (issue #12: MPM/Fluid
// 共通の Emitter クラスに対する初のユニットテスト。従来 Source/AABBSource/
// SphereSource/EllipseSource には一切テストが無かった)
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Emitter - AABBEmitter samples land inside the box bounds") {
  AABBEmitter e;
  e.center = {1.0f, 2.0f, 3.0f};
  e.size   = {2.0f, 4.0f, 6.0f}; // 全辺長

  std::mt19937 rng{42};
  const glm::vec3 half = e.size * 0.5f;
  for(int i = 0; i < 500; i++) {
    glm::vec3 p = e.sample(rng);
    CHECK(std::abs(p.x - e.center.x) <= half.x + 1e-5f);
    CHECK(std::abs(p.y - e.center.y) <= half.y + 1e-5f);
    CHECK(std::abs(p.z - e.center.z) <= half.z + 1e-5f);
  }
}

TEST_CASE("Emitter - SphereEmitter samples land inside the sphere radius") {
  SphereEmitter e;
  e.center = {0.0f, 0.0f, 0.0f};
  e.radius = 2.5f;

  std::mt19937 rng{42};
  double sumR = 0.0;
  const int N = 500;
  for(int i = 0; i < N; i++) {
    glm::vec3 p = e.sample(rng);
    float r     = glm::length(p - e.center);
    CHECK(r <= e.radius + 1e-4f);
    sumR += r;
  }
  // 体積一様分布 (r^3 に比例) であれば平均半径は半径の75%程度になり、
  // 境界付近に偏った実装 (例: r一様分布のバグ) にはならないはず
  double meanR = sumR / N;
  CHECK(meanR > e.radius * 0.5);
  CHECK(meanR < e.radius * 0.95);
}

TEST_CASE("Emitter - EllipseEmitter samples land inside the ellipse and at fixed Z") {
  EllipseEmitter e;
  e.center = {1.0f, -2.0f, 5.0f};
  e.semiA  = 3.0f;
  e.semiB  = 1.5f;

  std::mt19937 rng{42};
  for(int i = 0; i < 500; i++) {
    glm::vec3 p = e.sample(rng);
    float dx    = p.x - e.center.x;
    float dy    = p.y - e.center.y;
    CHECK((dx * dx) / (e.semiA * e.semiA) + (dy * dy) / (e.semiB * e.semiB) <= 1.0f + 1e-5f);
    CHECK(p.z == doctest::Approx(e.center.z));
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU テスト — Vulkan コンピュートシェーダーを使用するテスト
// main.cpp で gpuOk が false の場合はすべてスキップ
// ═══════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// テスト2-1: P2G 質量・運動量保存
// 1粒子でP2Gを実行し、グリッド全体の質量/運動量の合計を確認する
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MPM GPU - 2-1: P2G mass and momentum conservation") {
  HeadlessCtx ctx;
  ctx.init();

  // gridRes=8, worldSize=10.0, cellSize=1.25
  MPMHarness sim;
  sim.init(ctx, 8, 10.0f, SHADERS, 1);

  // 1粒子: pos=(5.0,5.0,5.0), vel=(2.0,-1.0,0.0)
  // rho0=1.0, Vp=1.5 → mass = rho0 * Vp = 1.5
  MPMHarness::Particle p;
  p.pos = {5.0f, 5.0f, 5.0f};
  p.Vp  = 1.5f;
  p.vel = {2.0f, -1.0f, 0.0f};
  p.F   = glm::mat3(1.0f); // F=I → tau=0 (応力項の寄与なし)
  p.tau = glm::mat3(0.0f);
  sim.uploadParticles({p});
  sim.buildHashGridCPU({p});

  auto pc = sim.makePC(0.001f, 1.0f /*rho0*/, 5000.0f, 11538.0f, 0.0f);
  sim.runZeroGrid(pc);
  sim.runP2G(pc);

  // グリッド全体の質量合計 = m_p = rho0 * Vp = 1.5
  float totalMass = sim.sumGridMass();
  CHECK(std::abs(totalMass - 1.5f) < 1e-3f);

  // グリッド全体の運動量合計 = m_p * v_p = (3.0, -1.5, 0.0)
  glm::vec3 totalMom = sim.sumGridMom();
  CHECK(std::abs(totalMom.x - 3.0f) < 1e-2f);
  CHECK(std::abs(totalMom.y - -1.5f) < 1e-2f);
  CHECK(std::abs(totalMom.z - 0.0f) < 1e-2f);

  sim.cleanup();
  ctx.cleanup();
}

// ─────────────────────────────────────────────────────────────────────────────
// テスト3-1: GridUpdate — 重力による一様加速
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MPM GPU - 3-1: Grid update with gravity") {
  HeadlessCtx ctx;
  ctx.init();

  MPMHarness sim;
  sim.init(ctx, 8, 10.0f, SHADERS, 1);

  const uint32_t NC = sim.totalCells();
  // Morton(4,4,4): 境界から離れた内部ノード
  uint32_t targetMorton = mpmMortonEncode(4, 4, 4);

  std::vector<glm::vec4> gridMom(NC, glm::vec4(0.0f));
  std::vector<float> gridMass(NC, 0.0f);
  gridMass[targetMorton] = 1.0f; // mass=1.0, mom=(0,0,0)
  sim.uploadGrid(gridMom, gridMass);

  // gravity=-9.8, dt=0.01 → v.y = 0/1.0 + (-9.8)*0.01 = -0.098
  auto pc = sim.makePC(0.01f, 1000.0f, 5000.0f, 11538.0f, -9.8f);
  sim.runGridUpdate(pc);

  glm::vec3 v = sim.readGridVel(targetMorton);
  CHECK(std::abs(v.x - 0.0f) < 1e-4f);
  CHECK(std::abs(v.y - -0.098f) < 1e-4f);
  CHECK(std::abs(v.z - 0.0f) < 1e-4f);

  sim.cleanup();
  ctx.cleanup();
}

// ─────────────────────────────────────────────────────────────────────────────
// テスト3-2: GridUpdate — 壁境界条件 (法線速度のゼロ化)
// シェーダー: ijk.x <= 1 && v.x < 0 → v.x = 0 (自由滑り境界)
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MPM GPU - 3-2: Grid update wall boundary condition") {
  HeadlessCtx ctx;
  ctx.init();

  MPMHarness sim;
  sim.init(ctx, 8, 10.0f, SHADERS, 1);

  const uint32_t NC = sim.totalCells();
  // Morton(1,4,4): x-左壁 (ijk.x=1, 境界条件が適用)
  uint32_t wallMorton = mpmMortonEncode(1, 4, 4);

  std::vector<glm::vec4> gridMom(NC, glm::vec4(0.0f));
  std::vector<float> gridMass(NC, 0.0f);
  gridMass[wallMorton] = 1.0f;
  gridMom[wallMorton]  = glm::vec4(-5.0f, 2.0f, 0.0f, 0.0f); // v=(-5,2,0)
  sim.uploadGrid(gridMom, gridMass);

  // gravity=0 で境界条件のみを確認
  auto pc = sim.makePC(0.01f, 1000.0f, 5000.0f, 11538.0f, 0.0f);
  sim.runGridUpdate(pc);

  glm::vec3 v = sim.readGridVel(wallMorton);
  // 法線方向 (x<0 → 壁への向き) はゼロ化される
  CHECK(std::abs(v.x - 0.0f) < 1e-4f);
  // 接線方向は保持される
  CHECK(std::abs(v.y - 2.0f) < 1e-4f);
  CHECK(std::abs(v.z - 0.0f) < 1e-4f);

  sim.cleanup();
  ctx.cleanup();
}

// ─────────────────────────────────────────────────────────────────────────────
// テスト4-1: G2P — 一様速度場 → 粒子速度が格子速度と一致
// すべての近傍ノードが同じ速度を持つ場合、重み付き平均 = その速度
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MPM GPU - 4-1: G2P uniform velocity field") {
  HeadlessCtx ctx;
  ctx.init();

  MPMHarness sim;
  sim.init(ctx, 8, 10.0f, SHADERS, 1);

  const uint32_t NC = sim.totalCells();
  const float h     = sim.cellSize(); // 1.25

  // 粒子: pos=(5.1,5.1,5.1) (セル境界を避けた内部点)
  MPMHarness::Particle p;
  p.pos = {5.1f, 5.1f, 5.1f};
  p.Vp  = 1.0f;
  p.vel = {0.0f, 0.0f, 0.0f};
  p.F   = glm::mat3(1.0f);
  p.tau = glm::mat3(0.0f);
  sim.uploadParticles({p});

  // G2Pステンシルベース: floor(5.1/1.25) - 1 = 4 - 1 = 3
  int basex = int(std::floor(5.1f / h)) - 1; // = 3
  int basey = int(std::floor(5.1f / h)) - 1;
  int basez = int(std::floor(5.1f / h)) - 1;

  // 27個のステンシルノードに一様速度を設定
  const glm::vec3 UNIFORM_VEL = {1.5f, 3.0f, 0.0f};
  std::vector<glm::vec4> gridMom(NC, glm::vec4(0.0f));
  std::vector<float> gridMass(NC, 0.0f);
  for(int iz = 0; iz < 3; ++iz)
    for(int iy = 0; iy < 3; ++iy)
      for(int ix = 0; ix < 3; ++ix) {
        int nx = basex + ix, ny = basey + iy, nz = basez + iz;
        if(nx < 0 || ny < 0 || nz < 0 || nx >= int(sim.gridRes()) || ny >= int(sim.gridRes()) || nz >= int(sim.gridRes())) continue;
        uint32_t m  = mpmMortonEncode(uint32_t(nx), uint32_t(ny), uint32_t(nz));
        gridMass[m] = 1.0f;
        // GridUpdate後に速度がgridMomに格納される仕様のため直接速度を設定
        gridMom[m] = glm::vec4(UNIFORM_VEL, 0.0f);
      }
  sim.uploadGrid(gridMom, gridMass);

  auto pc = sim.makePC(0.001f, 1000.0f, 5000.0f, 11538.0f, 0.0f);
  sim.runG2P(pc);

  // 一様速度場なので粒子速度 = 格子速度
  glm::vec4 vel = sim.readParticleVel(0);
  CHECK(std::abs(vel.x - UNIFORM_VEL.x) < 1e-3f);
  CHECK(std::abs(vel.y - UNIFORM_VEL.y) < 1e-3f);
  CHECK(std::abs(vel.z - UNIFORM_VEL.z) < 1e-3f);

  sim.cleanup();
  ctx.cleanup();
}

// ─────────────────────────────────────────────────────────────────────────────
// テスト6-1: 統合テスト — フルステップ50回でNaNなし・境界内保持
// 8粒子を中央に配置し重力下でシミュレーション
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MPM GPU - 6-1: Full step 50 iterations, no NaN, in bounds") {
  HeadlessCtx ctx;
  ctx.init();

  MPMHarness sim;
  sim.init(ctx, 16, 10.0f, SHADERS, 8);

  // 2×2×2 = 8粒子を中央上部に配置
  const float spacing = 0.3f;
  const float cx = 5.0f, cy = 7.0f, cz = 5.0f;
  std::vector<MPMHarness::Particle> particles(8);
  int idx = 0;
  for(int iz = 0; iz < 2; ++iz)
    for(int iy = 0; iy < 2; ++iy)
      for(int ix = 0; ix < 2; ++ix, ++idx) {
        particles[idx].pos = {cx + (ix - 0.5f) * spacing, cy + (iy - 0.5f) * spacing, cz + (iz - 0.5f) * spacing};
        particles[idx].Vp  = spacing * spacing * spacing;
        particles[idx].vel = {0.0f, 0.0f, 0.0f};
        particles[idx].F   = glm::mat3(1.0f);
        particles[idx].tau = glm::mat3(0.0f);
      }
  sim.uploadParticles(particles);

  auto pc          = sim.makePC(1.0f / (60.0f * 4.0f), 1000.0f, 1e4f, 2.3e4f, -9.8f);
  pc.particleCount = 8;

  const float margin = sim.cellSize() * 1.5f;
  const float lo     = 0.0f + margin;
  const float hi     = 10.0f - margin;

  for(int step = 0; step < 50; ++step) {
    // 毎ステップCPU側でハッシュグリッドを再構築
    std::vector<MPMHarness::Particle> cur(8);
    for(int i = 0; i < 8; ++i) {
      glm::vec4 pos = sim.readParticlePos(i);
      cur[i].pos    = glm::vec3(pos);
      cur[i].Vp     = pos.w;
    }
    sim.buildHashGridCPU(cur);
    sim.runFullStep(pc);
  }

  bool anyNaN       = false;
  bool anyOutBounds = false;
  for(int i = 0; i < 8; ++i) {
    glm::vec4 pos = sim.readParticlePos(i);
    glm::vec4 vel = sim.readParticleVel(i);
    if(std::isnan(pos.x) || std::isnan(pos.y) || std::isnan(pos.z) || std::isnan(vel.x) || std::isnan(vel.y) || std::isnan(vel.z)) anyNaN = true;
    if(pos.x < lo || pos.x > hi || pos.y < lo || pos.y > hi || pos.z < lo || pos.z > hi) anyOutBounds = true;
  }
  CHECK(!anyNaN);
  CHECK(!anyOutBounds);

  sim.cleanup();
  ctx.cleanup();
}

// ─────────────────────────────────────────────────────────────────────────────
// テスト6-2: 統合テスト — 1粒子の等速直線運動 (グリッド境界をまたぐ)
// 1粒子PICは速度を完全に保存するため、グリッド境界をまたいでも速度が変わらない
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MPM GPU - 6-2: Single particle constant velocity (grid crossing)") {
  HeadlessCtx ctx;
  ctx.init();

  MPMHarness sim;
  sim.init(ctx, 8, 10.0f, SHADERS, 1);

  // 1粒子: x=4.2 から速度(1.0,0,0)で移動
  // セル境界は 5.0*1.25=6.25 付近を通過
  MPMHarness::Particle p;
  p.pos = {4.2f, 5.0f, 5.0f};
  p.Vp  = 1.0f;
  p.vel = {1.0f, 0.0f, 0.0f};
  p.F   = glm::mat3(1.0f);
  p.tau = glm::mat3(0.0f);
  sim.uploadParticles({p});

  const float dt    = 0.01f;
  const float initX = 4.2f;
  const float velX  = 1.0f;
  auto pc           = sim.makePC(dt, 1.0f, 5000.0f, 11538.0f, 0.0f);
  pc.particleCount  = 1;

  for(int step = 0; step < 20; ++step) {
    glm::vec4 curPos = sim.readParticlePos(0);
    MPMHarness::Particle cur;
    cur.pos = glm::vec3(curPos);
    cur.Vp  = curPos.w;
    sim.buildHashGridCPU({cur});
    sim.runFullStep(pc);

    // 各ステップで速度が保持されていること (1粒子PICは正確)
    glm::vec4 vel = sim.readParticleVel(0);
    CHECK(std::abs(vel.x - velX) < 1e-3f);
    CHECK(std::abs(vel.y - 0.0f) < 1e-3f);
  }

  // 最終位置の確認: x ≈ 4.2 + 20 * 0.01 * 1.0 = 4.4
  glm::vec4 finalPos = sim.readParticlePos(0);
  float expectedX    = initX + 20.0f * dt * velX;
  CHECK(std::abs(finalPos.x - expectedX) < 0.05f);

  sim.cleanup();
  ctx.cleanup();
}

// ═══════════════════════════════════════════════════════════════════════════
// より複雑なテスト (7-1〜7-4) — 物理的不変量・速度勾配・境界条件
// ═══════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// テスト7-1: フルパイプライン重力 — 1ステップ後に粒子速度 = g*dt
//
// 根拠: 1粒子(v=0, tau=0) → P2G後グリッド速度=0 → GridUpdate後v_g.y=g*dt
//       → G2P(PIC)で全ノード同じ速度 → v_particle.y = g*dt (誤差ゼロ)
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MPM GPU Advanced - 7-1: Full pipeline gravity acceleration (1 step)") {
  HeadlessCtx ctx;
  ctx.init();
  MPMHarness sim;
  sim.init(ctx, 8, 10.0f, SHADERS, 1);

  const float h = sim.cellSize();
  MPMHarness::Particle p;
  p.pos = {5.0f, 5.0f, 5.0f};
  p.Vp  = h * h * h;
  p.vel = {0.0f, 0.0f, 0.0f};
  p.F   = glm::mat3(1.0f);
  p.tau = glm::mat3(0.0f);
  sim.uploadParticles({p});

  const float dt      = 0.01f;
  const float gravity = -9.8f;
  auto pc             = sim.makePC(dt, 1000.0f, 5000.0f, 11538.0f, gravity);
  pc.particleCount    = 1;

  sim.buildHashGridCPU({p});
  sim.runFullStep(pc);

  // v_p.y = g*dt (PICなので全近傍ノードが同じ速度を持ち誤差なし)
  const float expected_vy = gravity * dt;
  glm::vec4 vel           = sim.readParticleVel(0);
  CHECK(std::abs(vel.x) < 1e-4f);
  CHECK(std::abs(vel.y - expected_vy) < 1e-4f);
  CHECK(std::abs(vel.z) < 1e-4f);

  // 位置: x_new = x_old + dt * v_new
  glm::vec4 pos = sim.readParticlePos(0);
  CHECK(std::abs(pos.x - 5.0f) < 1e-4f);
  CHECK(std::abs(pos.y - (5.0f + dt * expected_vy)) < 1e-4f);
  CHECK(std::abs(pos.z - 5.0f) < 1e-4f);

  sim.cleanup();
  ctx.cleanup();
}

// ─────────────────────────────────────────────────────────────────────────────
// テスト7-2: G2P速度勾配によるF行列更新
//
// 設定: v_x = A*(xi.x - xp.x), v_y=v_z=0 をグリッドに設定
// B-splineの線形完全性 sum_i (xi-xp)*dw_i.x = 1 より:
//   速度勾配 L_xx = A, その他 = 0
//   F_new = (I + dt*L)*I = diag(1+A*dt, 1, 1)
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MPM GPU Advanced - 7-2: G2P velocity gradient updates F matrix") {
  HeadlessCtx ctx;
  ctx.init();
  MPMHarness sim;
  sim.init(ctx, 8, 10.0f, SHADERS, 1);

  const float h = sim.cellSize();

  MPMHarness::Particle p;
  p.pos = {5.0f, 5.0f, 5.0f};
  p.Vp  = 1.0f;
  p.vel = {0.0f, 0.0f, 0.0f};
  p.F   = glm::mat3(1.0f);
  p.tau = glm::mat3(0.0f);
  sim.uploadParticles({p});

  // 速度場 v_x = A*(xi.x - xp.x) を27ノードに設定 (xp.x=5.0)
  // v_y = v_z = 0
  const float A  = 10.0f;
  const float dt = 0.01f;
  const int base = int(std::floor(5.0f / h)) - 1; // = 3

  const uint32_t NC = sim.totalCells();
  std::vector<glm::vec4> gridMom(NC, glm::vec4(0.0f));
  std::vector<float> gridMass(NC, 0.0f);

  for(int iz = 0; iz < 3; ++iz)
    for(int iy = 0; iy < 3; ++iy)
      for(int ix = 0; ix < 3; ++ix) {
        int nx = base + ix, ny = base + iy, nz = base + iz;
        if(nx < 0 || ny < 0 || nz < 0 || nx >= int(sim.gridRes()) || ny >= int(sim.gridRes()) || nz >= int(sim.gridRes())) continue;
        uint32_t m  = mpmMortonEncode(uint32_t(nx), uint32_t(ny), uint32_t(nz));
        float xi_x  = (nx + 0.5f) * h;
        gridMom[m]  = glm::vec4(A * (xi_x - 5.0f), 0.0f, 0.0f, 0.0f);
        gridMass[m] = 1.0f;
      }
  sim.uploadGrid(gridMom, gridMass);

  auto pc          = sim.makePC(dt, 1000.0f, 5000.0f, 11538.0f, 0.0f);
  pc.particleCount = 1;
  sim.runG2P(pc);

  // F[0][0] (col=0, row=0) = 1 + A*dt = 1.1
  glm::mat3 F = sim.readParticleF(0);
  CHECK(std::abs(F[0][0] - (1.0f + A * dt)) < 2e-3f);
  CHECK(std::abs(F[1][1] - 1.0f) < 1e-4f);
  CHECK(std::abs(F[2][2] - 1.0f) < 1e-4f);
  // 非対角 = 0 (速度場がx方向にのみ依存)
  CHECK(std::abs(F[0][1]) < 1e-4f);
  CHECK(std::abs(F[1][0]) < 1e-4f);
  CHECK(std::abs(F[0][2]) < 1e-4f);
  CHECK(std::abs(F[2][0]) < 1e-4f);

  // F_new=diag(1.1,1,1): x方向に伸び→ trε>0 → 体積膨張→ 全方向正の応力
  // τ_xx = λ*trε + 2μ*log(1.1) > τ_yy = λ*trε (x方向の追加引張り分だけ大きい)
  glm::mat3 tau = sim.readParticleStress(0);
  CHECK(tau[0][0] > 0.0f);               // 引張り応力 (正)
  CHECK(tau[1][1] > 0.0f);               // 体積膨張で全方向正
  CHECK(tau[0][0] > tau[1][1] + 100.0f); // x方向がy,zより顕著に大きい (差≈953)
  CHECK(std::abs(tau[0][1]) < 1.0f);     // 非対角は小さい

  sim.cleanup();
  ctx.cleanup();
}

// ─────────────────────────────────────────────────────────────────────────────
// テスト7-3: 複数粒子フルステップ — 運動量保存 (外力なし, tau=0)
//
// 根拠 (PIC): sum_p(m_p*v_p_new) = sum_g(m_g*v_g) = sum_g(P2G_mom_g) = sum_p(m_p*v_p_old)
//             ただし tau=0 (応力項なし), 境界なし, 重力なし が前提
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MPM GPU Advanced - 7-3: Multi-particle momentum conservation (no gravity, tau=0)") {
  HeadlessCtx ctx;
  ctx.init();
  MPMHarness sim;
  sim.init(ctx, 8, 10.0f, SHADERS, 4);

  // 4粒子: 境界から離れた位置、スケールアップ後も境界フリー
  // 各粒子が異なる軸方向に運動し、正反対の組で合計ゼロ
  const float rho0 = 1.0f;
  std::vector<MPMHarness::Particle> particles(4);
  // p0: x正向き, p1: x負向き (対称)
  particles[0].pos = {4.0f, 5.0f, 5.0f};
  particles[0].vel = {2.0f, 0.0f, 0.0f};
  particles[1].pos = {6.0f, 5.0f, 5.0f};
  particles[1].vel = {-2.0f, 0.0f, 0.0f};
  // p2: y正向き, p3: y負向き (対称)
  particles[2].pos = {5.0f, 4.0f, 5.0f};
  particles[2].vel = {0.0f, 2.0f, 0.0f};
  particles[3].pos = {5.0f, 6.0f, 5.0f};
  particles[3].vel = {0.0f, -2.0f, 0.0f};
  for(auto& pt : particles) {
    pt.Vp  = 0.3f;
    pt.F   = glm::mat3(1.0f);
    pt.tau = glm::mat3(0.0f);
  }
  sim.uploadParticles(particles);

  // 初期総運動量 p0+p1のx: 0, p2+p3のy: 0, z: 0 → 合計(0,0,0)
  const float mass        = rho0 * 0.3f;
  const glm::vec3 initMom = glm::vec3(0.0f); // 対称配置なので

  auto pc          = sim.makePC(0.01f, rho0, 5000.0f, 11538.0f, 0.0f);
  pc.particleCount = 4;

  sim.buildHashGridCPU(particles);
  sim.runFullStep(pc);

  // 各粒子の速度を読み取り総運動量を計算
  glm::vec3 totalMom(0.0f);
  for(uint32_t i = 0; i < 4; ++i) {
    glm::vec4 vel = sim.readParticleVel(i);
    totalMom += mass * glm::vec3(vel);
  }
  // tau=0・重力なし・境界なしのPICは運動量保存 → 誤差 < 1e-3
  CHECK(std::abs(totalMom.x - initMom.x) < 1e-3f);
  CHECK(std::abs(totalMom.y - initMom.y) < 1e-3f);
  CHECK(std::abs(totalMom.z - initMom.z) < 1e-3f);

  sim.cleanup();
  ctx.cleanup();
}

// ─────────────────────────────────────────────────────────────────────────────
// テスト7-4: 壁境界衝突 — 粒子が境界を超えず速度が反転またはゼロに
//
// 1粒子を左壁に向けて高速で発射。グリッドのfree-slip境界でv.xが減衰し、
// G2Pの位置クランプで粒子が境界外に出ないことを検証する。
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("MPM GPU Advanced - 7-4: Wall boundary clamps particle position and velocity") {
  HeadlessCtx ctx;
  ctx.init();
  MPMHarness sim;
  sim.init(ctx, 8, 10.0f, SHADERS, 1);

  const float h      = sim.cellSize(); // 1.25
  const float margin = h * 1.5f;       // 1.875
  const float lo_x   = 0.0f + margin;

  // 左壁方向に高速発射
  MPMHarness::Particle p;
  p.pos = {2.5f, 5.0f, 5.0f};
  p.Vp  = h * h * h;
  p.vel = {-5.0f, 0.0f, 0.0f};
  p.F   = glm::mat3(1.0f);
  p.tau = glm::mat3(0.0f);
  sim.uploadParticles({p});

  auto pc          = sim.makePC(0.01f, 1000.0f, 1e4f, 2.3e4f, 0.0f);
  pc.particleCount = 1;

  // 30ステップで左壁に当たるはず (2.5 - 1.875) / 5 ≈ 0.125s = 12.5 steps)
  for(int step = 0; step < 30; ++step) {
    glm::vec4 curPos = sim.readParticlePos(0);
    MPMHarness::Particle cur;
    cur.pos = glm::vec3(curPos);
    cur.Vp  = curPos.w;
    cur.F   = sim.readParticleF(0);
    cur.tau = sim.readParticleStress(0);
    sim.buildHashGridCPU({cur});
    sim.runFullStep(pc);

    // 毎ステップ位置が境界内であること
    glm::vec4 pos = sim.readParticlePos(0);
    CHECK(pos.x >= lo_x - 1e-4f); // クランプされているはず
  }

  // 最終的に速度xは非負 (壁でゼロまたは反転)
  glm::vec4 finalVel = sim.readParticleVel(0);
  CHECK(finalVel.x >= -0.5f); // 大きな負のx速度は持たないはず
  CHECK(!std::isnan(finalVel.x));
  CHECK(!std::isnan(finalVel.y));
  CHECK(!std::isnan(finalVel.z));

  sim.cleanup();
  ctx.cleanup();
}

// ── 直方体(非立方体)ドメイン — 短辺軸の境界クランプ (issue #46) ────────────────
// MPMHarness は簡略化のため常に立方体ドメイン (gridRes/worldSize スカラー) を
// 前提とするため、このテストは MPMEngine/MPMConfig を直接使い、domainSize を
// Y 軸だけ短い偏平ドメインにして G2P の境界クランプ (mpm_g2p.comp) を検証する。
TEST_CASE("MPM GPU - 8-1: rectangular (non-cube) domain clamps particle to the short axis boundary") {
  HeadlessCtx ctx;
  ctx.init();

  MPMConfig cfg;
  cfg.nx           = 0;
  cfg.ny           = 0;
  cfg.nz           = 0;
  cfg.maxParticles = 4;
  cfg.domainSize   = glm::vec3(20.0f, 5.0f, 20.0f); // Y 軸だけ短い偏平ドメイン
  cfg.cellSize     = 1.25f;

  MPMEngine engine;
  engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADERS, cfg);

  const float Vp = cfg.cellSize * cfg.cellSize * cfg.cellSize;
  // ドメイン中央付近から Y 方向へ高速発射 (X/Z 方向には力を加えない)
  std::vector<glm::vec4> pos = {glm::vec4(10.0f, 2.5f, 10.0f, Vp)};
  std::vector<glm::vec4> vel = {glm::vec4(0.0f, 50.0f, 0.0f, 0.0f)};
  engine.appendParticles(pos, vel);

  const float dt = 1.0f / 60.0f;
  for(int f = 0; f < 30; ++f) {
    VkCommandBuffer cmd = ctx.beginCmd();
    engine.step(cmd, dt);
    ctx.submitCmd(cmd);
  }

  glm::vec4 p{};
  ctx.readBuffer(engine.getPositionBuffer(), 0, &p, sizeof(p));

  // Y 軸は短いドメイン (5m) の境界でクランプされているはず。
  // もし誤って X/Z 側の 20m がクランプ境界に使われていれば p.y はここまで到達しない。
  CHECK(!std::isnan(p.y));
  CHECK(p.y <= cfg.domainSize.y);
  CHECK(p.y > cfg.domainSize.y * 0.3f);
  // X/Z には力を加えていないため、ほぼ初期位置に留まっているはず。
  CHECK(std::abs(p.x - 10.0f) < 2.0f);
  CHECK(std::abs(p.z - 10.0f) < 2.0f);

  engine.cleanup();
  ctx.cleanup();
}
