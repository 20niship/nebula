/**
 * nebula_bench — Warp 比較用の headless ベンチマーク実行ファイル
 *
 * scenes.py が決めたパラメータを CLI で受け取り、MPMEngine (fluid / sand) または
 * SoftBodyEngine (softbody) を headless で駆動して 1 フレームあたりの実時間を計測する。
 * 各フレームは submit + 完了待ちまで含めて計測する (GPU 同期込みの wall-clock)。
 * 結果は stdout に `BENCH_JSON {...}` の 1 行で出力する。
 */

#include "Force.h"
#include "MaterialParams.h"
#include "engine/MPMEngine.h"
#include "engine/SoftBodyEngine.h"
#include "helpers/HeadlessCtx.h"

#include <argparse/argparse.hpp>
#include <glm/glm.hpp>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

static const std::string SHADER_DIR_STR = SHADER_DIR;

struct BenchArgs : public argparse::Args {
  std::string& scene       = kwarg("scene", "fluid | sand | softbody").set_default(std::string("fluid"));
  float& dt                = kwarg("dt", "frame timestep [s]").set_default(1.0f / 60.0f);
  int& substeps            = kwarg("substeps", "substeps per frame").set_default(20);
  int& warmup              = kwarg("warmup", "warmup frames (not timed)").set_default(10);
  int& frames              = kwarg("frames", "timed frames").set_default(100);
  float& domain            = kwarg("domain", "cubic domain size [m]").set_default(10.0f);
  float& cell_size         = kwarg("cell-size", "grid cell size [m]").set_default(10.0f / 64.0f);
  float& gravity           = kwarg("gravity", "gravity magnitude (-Y) [m/s^2]").set_default(9.8f);
  std::string& block_min   = kwarg("block-min", "MPM particle lattice min corner 'x,y,z'").set_default(std::string("0.5,0.5,0.5"));
  std::string& block_n     = kwarg("block-n", "MPM particle lattice counts 'nx,ny,nz'").set_default(std::string("10,10,10"));
  float& spacing           = kwarg("spacing", "MPM particle lattice spacing [m]").set_default(0.078125f);
  float& flip_ratio        = kwarg("flip-ratio", "MPM transfer: 0=PIC, -1=APIC, (0,1]=FLIP blend").set_default(0.0f);
  float& bulk_k            = kwarg("bulk-k", "fluid bulk modulus K [Pa] (fluid scene)").set_default(5e3f);
  std::string& sb          = kwarg("sb", "softbody .sb file").set_default(std::string(""));
  std::string& sb_offset   = kwarg("sb-offset", "softbody offset 'x,y,z'").set_default(std::string("0,0,0"));
  int& iters               = kwarg("iters", "XPBD solver iterations").set_default(5);
};

static glm::vec3 parseVec3(const std::string& s) {
  glm::vec3 v(0.0f);
  std::sscanf(s.c_str(), "%f,%f,%f", &v.x, &v.y, &v.z);
  return v;
}

struct Result {
  uint32_t n = 0;
  double msPerFrame = 0.0;
  glm::vec3 centroid{0.0f};
};

// warmup 後に frames フレーム回して計測する。stepFn(cmd) が 1 フレーム分のコマンドを記録する。
template <class StepFn>
static double timeFrames(HeadlessCtx& ctx, int warmup, int frames, StepFn stepFn) {
  for(int i = 0; i < warmup; i++) {
    VkCommandBuffer cmd = ctx.beginCmd();
    stepFn(cmd);
    ctx.submitCmd(cmd);
  }
  auto t0 = std::chrono::steady_clock::now();
  for(int i = 0; i < frames; i++) {
    VkCommandBuffer cmd = ctx.beginCmd();
    stepFn(cmd);
    ctx.submitCmd(cmd);
  }
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return s * 1000.0 / double(frames);
}

static glm::vec3 centroidOf(HeadlessCtx& ctx, VkBuffer buf, uint32_t n) {
  std::vector<glm::vec4> p(n);
  ctx.readBuffer(buf, 0, p.data(), n * sizeof(glm::vec4));
  glm::dvec3 c(0.0);
  for(const auto& q : p) c += glm::dvec3(q.x, q.y, q.z);
  return glm::vec3(c / double(n));
}

static Result runMPM(HeadlessCtx& ctx, const BenchArgs& a, const MaterialParams& mat) {
  const glm::vec3 bmin = parseVec3(a.block_min);
  const glm::vec3 bn   = parseVec3(a.block_n);

  std::vector<glm::vec4> pos, vel;
  const float Vp = a.spacing * a.spacing * a.spacing;
  for(int k = 0; k < int(bn.z); k++)
    for(int j = 0; j < int(bn.y); j++)
      for(int i = 0; i < int(bn.x); i++) {
        pos.emplace_back(bmin + (glm::vec3(i, j, k) + 0.5f) * a.spacing, Vp);
        vel.emplace_back(0.0f); // w = material id 0
      }

  MPMEngine e;
  e.domainSize   = glm::vec3(a.domain);
  e.cellSize     = a.cell_size;
  e.maxParticles = uint32_t(pos.size()); // != 0 で自動シードを無効化
  e.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADER_DIR_STR);
  e.numSubsteps = a.substeps;
  e.flip_ratio  = a.flip_ratio;
  e.addForce(GravityForce::FromDirection({0.0f, -1.0f, 0.0f}, a.gravity));
  e.setMaterials({mat});
  e.appendParticles(pos, vel);

  Result r;
  r.n          = e.liveParticleCount();
  r.msPerFrame = timeFrames(ctx, a.warmup, a.frames, [&](VkCommandBuffer cmd) { e.step(cmd, a.dt); });
  r.centroid   = centroidOf(ctx, e.getPositionBuffer(), r.n);
  e.cleanup();
  return r;
}

static Result runSoftBody(HeadlessCtx& ctx, const BenchArgs& a) {
  SoftBodyEngine e;
  e.addInstance({a.sb, parseVec3(a.sb_offset), 1.0f});
  e.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADER_DIR_STR, glm::vec3(a.domain), a.cell_size);
  e.numSubsteps            = a.substeps;
  e.solverIterations       = a.iters;
  e.particleCollisionRadius = 0.0f; // ボディ内の粒子間衝突は無効化 (Warp 側も同じ O(N^2) ループを回す)
  e.addForce(GravityForce::FromDirection({0.0f, -1.0f, 0.0f}, a.gravity));

  Result r;
  r.n          = e.totalParticleCount();
  r.msPerFrame = timeFrames(ctx, a.warmup, a.frames, [&](VkCommandBuffer cmd) { e.step(cmd, a.dt); });
  r.centroid   = centroidOf(ctx, e.getPositionBuffer(), r.n);
  e.cleanup();
  return r;
}

int main(int argc, char* argv[]) {
  auto args = argparse::parse<BenchArgs>(argc, argv);
  try {
    HeadlessCtx ctx;
    ctx.init();

    Result r;
    if(args.scene == "fluid") {
      r = runMPM(ctx, args, presetWater(1000.0f, args.bulk_k));
    } else if(args.scene == "sand") {
      r = runMPM(ctx, args, presetSand());
    } else if(args.scene == "softbody") {
      r = runSoftBody(ctx, args);
    } else {
      std::fprintf(stderr, "unknown scene: %s\n", args.scene.c_str());
      return 2;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physicalDevice, &props);
    std::printf("BENCH_JSON {\"engine\":\"nebula\",\"device\":\"%s\",\"scene\":\"%s\",\"n\":%u,\"frames\":%d,\"ms_per_frame\":%.6f,\"centroid\":[%.6f,%.6f,%.6f]}\n",
                props.deviceName, args.scene.c_str(), r.n, args.frames, r.msPerFrame, r.centroid.x, r.centroid.y, r.centroid.z);
    std::fflush(stdout);
    ctx.cleanup();
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
