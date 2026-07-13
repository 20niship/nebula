/**
 * pyro_explosion — 地表付近での爆発燃焼シミュレーション (キノコ雲を狙う)
 *
 * ヘッドレス実行。地表近くで fuel+temperature+density を短時間だけ一気に注入し
 * 燃焼させ、強い浮力で立ち上らせる。渦度閉じ込め (vorticityEps) を強めに設定して
 * 上昇柱の縁が巻き上がり、上部で横に広がる「キノコ雲」的な形状形成を狙う。
 * ドームは高さ方向を広めに取り、傘の広がりが見えるようにする。
 */

#include "engine/PyroEngine.h"
#include "helpers/HeadlessCtx.h"

#include <argparse/argparse.hpp>
#include <glm/glm.hpp>

#include <cstdio>
#include <filesystem>
#include <string>

static const std::string SHADER_DIR_STR = SHADER_DIR;

struct ExplosionArgs : public argparse::Args {
  // Morton 符号化のため 2 のべき乗であること (PyroEngine::init() が検証する)
  int&   grid_res      = kwarg("grid-res",     "Pyro グリッド解像度 (2のべき乗)").set_default(64);
  float& world_size    = kwarg("world-size",   "world size [m] (高さ方向に余裕を持たせる)").set_default(14.0f);
  int&   n_frames      = kwarg("n-frames",     "実行フレーム数").set_default(240);
  float& dt            = kwarg("dt",           "フレームタイムステップ [s]").set_default(1.0f / 60.0f);
  int&   substeps      = kwarg("substeps",     "1フレームあたりのサブステップ数").set_default(1);
  int&   jacobi_iters  = kwarg("jacobi-iters", "圧力投影 Jacobi 反復回数").set_default(50);
  float& vorticity_eps = kwarg("vorticity-eps","渦度閉じ込め強度 (傘の巻き上がりを強調)").set_default(6.0f);
  std::string& out_dir = kwarg("out",          "ボクセルダンプ出力先ディレクトリ")
                                .set_default(std::string("sim_captures/pyro_explosion"));
  int&   dump_every    = kwarg("dump-every",   "何フレームごとに .pvox をダンプするか").set_default(1);
  int&   burst_frames  = kwarg("burst-frames", "爆発バーストの継続フレーム数").set_default(6);
};

int main(int argc, char* argv[]) {
  auto args = argparse::parse<ExplosionArgs>(argc, argv);

  try {
    HeadlessCtx ctx;
    ctx.init();

    PyroConfig cfg;
    cfg.grid_res   = uint32_t(args.grid_res);
    cfg.world_size = args.world_size;

    PyroEngine engine;
    engine.init(ctx.device, ctx.allocator, ctx.descriptorPool,
                ctx.commandPool, ctx.computeQueue, SHADER_DIR_STR, cfg);

    engine.numSubsteps        = args.substeps;
    engine.numJacobiIters     = args.jacobi_iters;
    engine.vorticityEps       = args.vorticity_eps;
    engine.buoyancyAlpha      = 5.0f;  // 強い浮力で急速に立ち上らせる
    engine.buoyancyBeta       = 0.5f;
    engine.ambientTemp        = 0.0f;
    engine.densityDissipation = 0.02f; // 傘の煙が長く残るよう減衰を抑える
    engine.tempDissipation    = 0.3f;
    // 燃焼 (fire) パラメータ: 短時間の一気燃焼
    engine.ignitionTemp       = 1.0f;
    engine.burnRate           = 4.0f;
    engine.heatRelease        = 6.0f;
    engine.smokeYieldPerFuel  = 2.5f;
    engine.flameBrightness    = 4.0f;

    const float W = cfg.world_size;

    // ── 地表付近での爆発源 (短時間バースト後、燃焼と浮力のみで自律的に発達) ──
    auto blast = std::make_shared<SphereEmitter>();
    blast->center          = {W * 0.5f, W * 0.06f, W * 0.5f};
    blast->radius          = W * 0.07f;
    blast->inflowVelocity  = {0.0f, 4.0f, 0.0f}; // 上向きの初速キック
    blast->densityRate     = 30.0f;
    blast->temperatureRate = 40.0f;
    blast->fuelRate        = 25.0f;
    blast->step_count      = args.burst_frames;
    engine.addEmitter(blast);

    std::filesystem::create_directories(args.out_dir);

    float simTime = 0.0f;
    for (int frame = 0; frame < args.n_frames; frame++) {
      VkCommandBuffer cmd = ctx.beginCmd();
      engine.step(cmd, args.dt);
      ctx.submitCmd(cmd);

      simTime += args.dt;

      if (args.dump_every > 0 && frame % args.dump_every == 0) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/frame_%04d.pvox", args.out_dir.c_str(), frame);
        engine.dumpFrame(path, simTime);
        std::printf("frame %4d / %4d  t=%.3fs  -> %s\n", frame, args.n_frames, simTime, path);
      }
    }

    engine.cleanup();
    ctx.cleanup();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
