/**
 * mpm_bench — GPU MPM 転写モード(PIC/APIC/FLIP)別・粒子数別ヘッドレスベンチマーク
 *
 * ウィンドウなし (HeadlessCtx) で MPMEngine::step() を n-frames 回実行し、
 * 1フレームあたりの平均処理時間を計測する。NEBULA_GPU_PROFILING ビルド時は
 * Vulkan タイムスタンプクエリによるパス単位 (P2G/G2P/ハッシュ構築 等) の
 * GPU 時間内訳も出力する。
 */

#include "engine/MPMEngine.h"
#include "helpers/HeadlessCtx.h"

#include <argparse/argparse.hpp>
#include <glm/glm.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

static const std::string SHADER_DIR_STR = SHADER_DIR;

struct BenchArgs : public argparse::Args {
  int& particles        = kwarg("particles", "目標粒子数 (nx=ny=nz=cbrt(particles) の立方体ブロック)").set_default(80000);
  float& domain_size     = kwarg("domain-size", "ドメイン物理サイズ (立方体) [m]").set_default(10.0f);
  float& cell_size       = kwarg("cell-size", "MPM グリッドセルサイズ [m]").set_default(10.0f / 64.0f);
  float& flip_ratio       = kwarg("flip-ratio", "転写モード: 0=PIC -1=APIC 0~1=FLIP").set_default(1.0f);
  int& n_frames          = kwarg("n-frames", "計測フレーム数").set_default(60);
  int& warmup_frames     = kwarg("warmup-frames", "計測前のウォームアップフレーム数").set_default(5);
  int& substeps          = kwarg("substeps", "フレームあたりサブステップ数").set_default(20);
  float& dt              = kwarg("dt", "フレームタイムステップ [s]").set_default(1.0f / 60.0f);
};

int main(int argc, char* argv[]) {
  auto args = argparse::parse<BenchArgs>(argc, argv);

  try {
    HeadlessCtx ctx;
    ctx.init();

    MPMConfig cfg;
    uint32_t side = uint32_t(std::ceil(std::cbrt(double(args.particles))));
    cfg.nx = cfg.ny = cfg.nz = side;
    cfg.domainSize            = glm::vec3(args.domain_size);
    cfg.cellSize              = args.cell_size;

    MPMEngine engine;
    engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADER_DIR_STR, cfg);
#ifdef NEBULA_GPU_PROFILING
    engine.enableGpuProfiling(ctx.physicalDevice);
#endif

    auto gravity = GravityForce::FromDirection({0.0f, -1.0f, 0.0f}, 9.8f);
    engine.addForce(gravity);
    engine.numSubsteps = args.substeps;
    engine.flip_ratio  = args.flip_ratio;

    std::printf("particles=%u (side=%u) totalCells=%u flip_ratio=%.2f substeps=%d\n", cfg.particleCount(), side, cfg.totalCells(), args.flip_ratio, args.substeps);
    std::fflush(stdout);

    for(int frame = 0; frame < args.warmup_frames; frame++) {
      VkCommandBuffer cmd = ctx.beginCmd();
      engine.step(cmd, args.dt);
      ctx.submitCmd(cmd);
    }

    auto perfStart = std::chrono::steady_clock::now();
    for(int frame = 0; frame < args.n_frames; frame++) {
      VkCommandBuffer cmd = ctx.beginCmd();
      engine.step(cmd, args.dt);
      ctx.submitCmd(cmd);
    }
    double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - perfStart).count();
    std::printf("PERF_RESULT particles=%u frames=%d elapsed_s=%.6f ms_per_frame=%.6f fps=%.2f\n", cfg.particleCount(), args.n_frames, elapsed_s, elapsed_s * 1000.0 / double(args.n_frames), double(args.n_frames) / elapsed_s);
    std::fflush(stdout);

#ifdef NEBULA_GPU_PROFILING
    engine.printGpuProfile();
#endif

    engine.cleanup();
    ctx.cleanup();
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
