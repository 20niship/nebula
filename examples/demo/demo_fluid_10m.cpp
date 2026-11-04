// 1000 万粒子の MPM 流体 (ダムブレイク + 円柱障害物)。結果は .nbp で dump し、render/render_fluid.py で Blender レンダリングする。
#include "Collider.h"
#include "Force.h"
#include "MaterialParams.h"
#include "NbpDump.h"
#include "engine/MPMEngine.h"
#include "helpers/HeadlessCtx.h"

#include <argparse/argparse.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>

static const std::string SHADER_DIR_STR = SHADER_DIR;

struct Args : public argparse::Args {
  std::string& out = kwarg("out", "output directory for .nbp frames").set_default(std::string("out/fluid"));
  int& frames      = kwarg("frames", "output frames").set_default(240);
  float& fps       = kwarg("fps", "output frame rate").set_default(24.0f);
  int& substeps    = kwarg("substeps", "MPM substeps per output frame").set_default(192);
  float& scale     = kwarg("scale", "particle count multiplier (1 = about 10M, 0.1 = 1M)").set_default(1.0f);
  float& bulk_k    = kwarg("bulk-k", "bulk modulus K [Pa] (sound speed ~ sqrt(K/rho) must be well above the flow speed)").set_default(1e6f);
  float& fp_scale  = kwarg("fp-scale", "MPM fixed-point scale (default 65536 quantizes the tiny per-particle pressure impulses of >1M particles to zero)").set_default(1e8f);
  float& flip      = kwarg("flip-ratio", "FLIP ratio (0.95 clumps and collapses dam breaks; 0.5 is stable)").set_default(0.5f);
};

int main(int argc, char* argv[]) {
  auto a = argparse::parse<Args>(argc, argv);
  try {
    // 4m x 2m x 2m の水槽。セル 1/64m、粒子間隔はセルの半分 (8 粒子/セル)。--scale は粒子数の倍率 (間隔とセルを一緒に変え、常に 8 粒子/セル)
    const glm::vec3 domain(4.0f, 2.0f, 2.0f);
    const float spacing = 4.0f / 512.0f / std::cbrt(a.scale);
    const float cell    = spacing * 2.0f;
    const glm::vec3 bmin(0.03f, 0.03f, 0.03f), bsize(2.46f, 1.0f, 1.95f);

    std::vector<glm::vec4> pos, vel;
    const glm::ivec3 n = glm::ivec3(bsize / spacing);
    pos.reserve(size_t(n.x) * n.y * n.z);
    const float Vp = spacing * spacing * spacing;
    for(int k = 0; k < n.z; k++)
      for(int j = 0; j < n.y; j++)
        for(int i = 0; i < n.x; i++) pos.emplace_back(bmin + (glm::vec3(i, j, k) + 0.5f) * spacing, Vp);
    vel.assign(pos.size(), glm::vec4(0.0f));
    std::printf("particles: %zu  grid: %ux%ux%u\n", pos.size(), unsigned(domain.x / cell), unsigned(domain.y / cell), unsigned(domain.z / cell));

    HeadlessCtx ctx;
    ctx.init();
    MPMEngine e;
    e.domainSize   = domain;
    e.cellSize     = cell;
    e.maxParticles = uint32_t(pos.size());
    e.setFixedPointScale(a.fp_scale);
    e.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADER_DIR_STR);
    e.numSubsteps = a.substeps;
    e.flip_ratio  = a.flip;
    e.addForce(GravityForce::FromDirection({0.0f, -1.0f, 0.0f}, 9.8f));
    e.setMaterials({presetWater(1000.0f, a.bulk_k)});
    e.appendParticles(pos, vel);

    ColliderSet cols;
    cols.addCylinder({2.6f, 0.5f, 0.55f}, 0.14f, 0.5f, 0.0f, 0.0f);
    cols.addCylinder({2.6f, 0.5f, 1.45f}, 0.14f, 0.5f, 0.0f, 0.0f);
    cols.addBox({3.3f, 0.2f, 1.0f}, {0.12f, 0.2f, 0.35f}, 0.0f, 0.0f);
    e.setColliders(cols);

    std::filesystem::create_directories(a.out);
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
      writeNbp(path, pos, vel, glm::vec3(0.0f), domain, 8.0f);
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
