/**
 * pyro_cow_blast — 超高密度の爆風煙を静的な牛オブジェクトに浴びせるシミュレーション
 *
 * ヘッドレス実行 (可視化は tools/pyro_raymarch.py --mode heatmap 側で行う想定)。
 * assets/cow_obstacle.stl (tools/gen_cow_stl.py で生成した低ポリ牛) を静的障害物
 * として一度だけ SDF 化し、-X 側から超高密度・高速の smoke バースト (fuel/fire なし、
 * 純粋な運動量による "爆風") を短時間だけ放出して牛にぶつける。
 * 可視化は velocity magnitude のヒートマップ表示を想定しているため、density の
 * 減衰は抑えめにして煙の広がり (乱流構造) が長く残るようにする。
 */

#include "engine/PyroEngine.h"
#include "core/MeshSDF.h"
#include "helpers/HeadlessCtx.h"

#include <argparse/argparse.hpp>
#include <glm/glm.hpp>

#include <cstdio>
#include <filesystem>
#include <string>

static const std::string SHADER_DIR_STR = SHADER_DIR;
static const std::string ASSET_DIR_STR  = ASSET_DIR;

struct CowBlastArgs : public argparse::Args {
  // Morton 符号化のため 2 のべき乗であること (PyroEngine::init() が検証する)
  int&   grid_res      = kwarg("grid-res",     "Pyro グリッド解像度 (2のべき乗)").set_default(64);
  float& world_size    = kwarg("world-size",   "world size [m]").set_default(10.0f);
  int&   n_frames      = kwarg("n-frames",     "実行フレーム数").set_default(180);
  float& dt            = kwarg("dt",           "フレームタイムステップ [s]").set_default(1.0f / 60.0f);
  int&   substeps      = kwarg("substeps",     "1フレームあたりのサブステップ数").set_default(1);
  int&   jacobi_iters  = kwarg("jacobi-iters", "圧力投影 Jacobi 反復回数").set_default(40);
  float& vorticity_eps = kwarg("vorticity-eps","渦度閉じ込め強度 (乱流構造を強調)").set_default(4.0f);
  std::string& out_dir = kwarg("out",          "ボクセルダンプ出力先ディレクトリ")
                                .set_default(std::string("sim_captures/pyro_cow"));
  int&   dump_every    = kwarg("dump-every",   "何フレームごとに .pvox をダンプするか").set_default(1);
  std::string& cow_stl = kwarg("cow-stl",      "牛 STL パス")
                                .set_default(ASSET_DIR_STR + "/cow_obstacle.stl");
  int&   blast_frames  = kwarg("blast-frames", "爆風バーストの継続フレーム数").set_default(6);
  float& blast_speed   = kwarg("blast-speed",  "爆風の初速 [m/s]").set_default(18.0f);
  float& blast_density = kwarg("blast-density","爆風の密度注入速度 [1/s] (超高密度)").set_default(60.0f);
};

int main(int argc, char* argv[]) {
  auto args = argparse::parse<CowBlastArgs>(argc, argv);

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
    // 爆風は運動量が主体で熱源ではないため浮力はごく弱めに (吹き飛ばされた後にわずかに立ち上る程度)
    engine.buoyancyAlpha      = 0.3f;
    engine.buoyancyBeta       = 0.0f;
    engine.ambientTemp        = 0.0f;
    // 乱流構造が長く見えるよう減衰を抑える (fire/fuelは使わない)
    engine.densityDissipation = 0.01f;
    engine.tempDissipation    = 0.2f;

    const float W = cfg.world_size;

    // ── 静的な牛障害物 (毎フレーム再構築せず一度だけ SDF 化) ────────────────
    std::printf("牛 STL 読み込み: %s\n", args.cow_stl.c_str());
    auto cowTris = loadBinarySTL(args.cow_stl);
    std::printf("  三角形数: %zu\n", cowTris.size());
    engine.setColliderSDF(buildMeshSDF(cowTris, cfg.grid_res, cfg.world_size));

    // ── 超高密度・高速の爆風バースト (-X 側から牛へ向けて) ──────────────────
    PyroSource blast;
    blast.shape           = PyroSourceShape::AABB;
    blast.center          = {W * 0.08f, W * 0.20f, W * 0.5f};
    blast.size            = glm::vec3(W * 0.05f, W * 0.15f, W * 0.15f);
    blast.inflowVelocity  = {args.blast_speed, 0.0f, 0.0f};
    blast.densityRate     = args.blast_density;
    blast.temperatureRate = 0.0f; // fire なし
    blast.fuelRate        = 0.0f; // fire なし
    blast.step_count      = args.blast_frames; // 短時間バーストのみ
    engine.addSource(blast);

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
