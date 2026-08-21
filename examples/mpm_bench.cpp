/**
 * mpm_bench — GPU MPM 転写モード(PIC/APIC/FLIP)別・粒子数別ヘッドレスベンチマーク
 *
 * ウィンドウなし (HeadlessCtx) で MPMEngine::step() を n-frames 回実行し、
 * 1フレームあたりの平均処理時間を計測する。パス単位の内訳が要る場合は
 * NEBULA_TRACY ビルドでTracyプロファイラを接続すること (MPMEngine::step 内の
 * ZoneScopedN が FluidEngine 等と同じ形式でP2G/G2P/ZeroGrid/GridUpdateを計測する)。
 */

#include "engine/MPMEngine.h"
#include "helpers/HeadlessCtx.h"

#include <argparse/argparse.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

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

    // 半精度パック(pos/gridVel)導入後の健全性チェック: NaN/Infなし、粒子がドメイン内に収まっているか
    {
      uint32_t n = cfg.particleCount();
      std::vector<uint32_t> raw(n * 3);
      ctx.readBuffer(engine.getPositionBuffer(), 0, raw.data(), raw.size() * sizeof(uint32_t));
      float margin = args.domain_size * 0.5f; // クランプ境界(1.5*cellSize)より緩めに、発散のみ検出
      bool ok       = true;
      for(uint32_t i = 0; i < n; i++) {
        glm::vec2 xy = glm::unpackHalf2x16(raw[i * 3]);
        glm::vec2 z0 = glm::unpackHalf2x16(raw[i * 3 + 1]);
        glm::vec3 p(xy.x, xy.y, z0.x);
        bool finite = std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        if(!finite || glm::any(glm::lessThan(p, glm::vec3(-margin))) || glm::any(glm::greaterThan(p, glm::vec3(args.domain_size + margin)))) {
          ok = false;
          std::printf("SANITY_CHECK FAIL particle=%u pos=(%f,%f,%f)\n", i, p.x, p.y, p.z);
          break;
        }
      }
      std::printf("SANITY_CHECK %s (n=%u, finite + in-bounds check on final positions)\n", ok ? "PASS" : "FAIL", n);
    }

    engine.cleanup();
    ctx.cleanup();
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
