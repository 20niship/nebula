/**
 * pyro_basic — グリッドベース煙・火炎 (Pyro) シミュレーション、ヘッドレス実行
 *
 * 可視化は不要 (ウィンドウ/ImGui なし)。複数の Emitter (炎/煙の発生源、位置違い・
 * 一部は移動) + 1個の移動 STL 障害物 (--obstacle-rebuild-interval フレームごとに
 * SDF 再構築) を設定し、--n-frames 回 step() を回して --dump-every フレームごとに
 * ボクセルを .pvox としてダンプする。可視化は tools/pyro_raymarch.py (Python) 側で行う。
 */

#include "core/MeshSDF.h"
#include "engine/PyroEngine.h"
#include "helpers/HeadlessCtx.h"

#include <argparse/argparse.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdio>
#include <filesystem>
#include <string>

static const std::string SHADER_DIR_STR = SHADER_DIR;
static const std::string ASSET_DIR_STR  = ASSET_DIR;

struct PyroBasicArgs : public argparse::Args {
  // issue #46フォローアップ: ドメインは domainSize(XYZ物理サイズ)+cellSize(全軸共通セルサイズ)
  // で指定する。Morton dispatch用の立方体解像度(2^n)は自動導出される (PyroConfig::cubeRes())。
  float& domain_size_x = kwarg("domain-size-x", "ドメイン物理サイズ X [m]").set_default(10.0f);
  float& domain_size_y = kwarg("domain-size-y", "ドメイン物理サイズ Y [m]").set_default(10.0f);
  float& domain_size_z = kwarg("domain-size-z", "ドメイン物理サイズ Z [m]").set_default(10.0f);
  float& cell_size      = kwarg("cell-size", "セルサイズ [m] (小さいほど高解像度)").set_default(10.0f / 32.0f);
  int& n_frames                  = kwarg("n-frames", "実行フレーム数").set_default(120);
  float& dt                      = kwarg("dt", "フレームタイムステップ [s]").set_default(1.0f / 60.0f);
  int& substeps                  = kwarg("substeps", "1フレームあたりのサブステップ数").set_default(1);
  int& pressure_iters            = kwarg("pressure-iters", "圧力投影 Red-Black Gauss-Seidel sweep回数").set_default(10);
  float& vorticity_eps           = kwarg("vorticity-eps", "渦度閉じ込め強度").set_default(2.0f);
  std::string& out_dir           = kwarg("out", "ボクセルダンプ出力先ディレクトリ").set_default(std::string("sim_captures/pyro"));
  int& dump_every                = kwarg("dump-every", "何フレームごとに .pvox をダンプするか (0=無効)").set_default(10);
  std::string& stl_path          = kwarg("obstacle-stl", "移動障害物 STL パス").set_default(ASSET_DIR_STR + "/sphere_obstacle.stl");
  int& obstacle_rebuild_interval = kwarg("obstacle-rebuild-interval", "障害物 SDF 再構築間隔 (フレーム)").set_default(4);
  // issue #30 デモ: Pyroは従来wind非対応だったが、addForce()で任意方向の風を追加できる
  float& wind_x = kwarg("wind-x", "X方向の風の強さ [m/s^2] (issue #30 Forceデモ)").set_default(0.0f);
  float& wind_z = kwarg("wind-z", "Z方向の風の強さ [m/s^2] (issue #30 Forceデモ)").set_default(0.0f);
};

int main(int argc, char* argv[]) {
  auto args = argparse::parse<PyroBasicArgs>(argc, argv);

  try {
    HeadlessCtx ctx;
    ctx.init();

    PyroConfig cfg;
    cfg.domainSize = {args.domain_size_x, args.domain_size_y, args.domain_size_z};
    cfg.cellSize   = args.cell_size;

    PyroEngine engine;
    engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADER_DIR_STR, cfg);

    engine.numSubsteps        = args.substeps;
    engine.numPressureIters   = args.pressure_iters;
    engine.vorticityEps       = args.vorticity_eps;
    engine.buoyancyAlpha      = 1.2f;
    engine.buoyancyBeta       = 0.4f;
    engine.ambientTemp        = 0.0f;
    engine.densityDissipation = 0.05f;
    engine.tempDissipation    = 0.2f;
    // 燃焼 (fire) パラメータ
    engine.ignitionTemp      = 0.5f;
    engine.burnRate          = 1.5f;
    engine.heatRelease       = 4.0f;
    engine.smokeYieldPerFuel = 2.0f;
    engine.flameBrightness   = 3.0f;

    // issue #30 デモ: Pyroは従来wind非対応だったが、addForce()で任意方向の風を
    // 浮力(buoyancyAlpha/Beta)と独立に追加できる。--wind-x/--wind-z で指定。
    if(args.wind_x != 0.0f || args.wind_z != 0.0f) {
      engine.addForce(ConstantWindForce::FromDirection({args.wind_x, 0.0f, args.wind_z}, 1.0f));
      std::printf("風を追加: wind=(%.2f, 0, %.2f)\n", args.wind_x, args.wind_z);
    }

    const glm::vec3 W = cfg.domainSize;

    // ── 複数 Emitter (位置違い、一部は移動) ─────────────────────────────────
    {
      auto fire1             = std::make_shared<SphereEmitter>();
      fire1->center          = {W.x * 0.3f, W.y * 0.08f, W.z * 0.5f};
      fire1->radius          = W.x * 0.06f;
      fire1->inflowVelocity  = {0.0f, 2.0f, 0.0f};
      fire1->densityRate     = 0.5f;
      fire1->temperatureRate = 3.0f;
      fire1->fuelRate        = 2.0f;
      fire1->step_count      = 0; // 無限
      engine.addEmitter(fire1);

      auto fire2             = std::make_shared<SphereEmitter>();
      fire2->center          = {W.x * 0.7f, W.y * 0.08f, W.z * 0.5f};
      fire2->radius          = W.x * 0.05f;
      fire2->center_vel      = {0.0f, 0.0f, 0.3f}; // Z方向へゆっくり移動
      fire2->inflowVelocity  = {0.0f, 1.8f, 0.0f};
      fire2->densityRate     = 0.4f;
      fire2->temperatureRate = 2.5f;
      fire2->fuelRate        = 1.6f;
      fire2->step_count      = 0;
      engine.addEmitter(fire2);

      auto smoke3             = std::make_shared<EllipseEmitter>();
      smoke3->center          = {W.x * 0.5f, W.y * 0.15f, W.z * 0.25f};
      smoke3->semiA           = W.x * 0.08f;
      smoke3->halfHeightY     = W.y * 0.03f;
      smoke3->semiB           = W.z * 0.08f;
      smoke3->center_vel      = {0.15f, 0.0f, 0.0f}; // X方向へ移動 (燃料なし、煙のみ)
      smoke3->inflowVelocity  = {0.0f, 1.0f, 0.0f};
      smoke3->densityRate     = 0.6f;
      smoke3->temperatureRate = 0.8f;
      smoke3->fuelRate        = 0.0f;
      smoke3->step_count      = 0;
      engine.addEmitter(smoke3);
    }

    // ── 移動 STL 障害物 (--obstacle-rebuild-interval フレームごとに SDF 再構築) ──
    std::printf("障害物 STL 読み込み: %s\n", args.stl_path.c_str());
    auto baseTris = loadBinarySTL(args.stl_path);
    std::printf("  三角形数: %zu\n", baseTris.size());

    std::filesystem::create_directories(args.out_dir);

    float simTime = 0.0f;
    for(int frame = 0; frame < args.n_frames; frame++) {
      if(args.obstacle_rebuild_interval > 0 && frame % args.obstacle_rebuild_interval == 0) {
        glm::vec3 translation(W.x * 0.5f, W.y * 0.35f + 0.15f * W.y * std::sin(simTime * 0.8f), W.z * 0.5f);
        glm::quat rotation = glm::angleAxis(simTime * 0.5f, glm::vec3(0, 1, 0));
        auto moved         = transformTriangles(baseTris, translation, rotation);
        engine.setColliderSDF(buildMeshSDF(moved, cfg.gridRes(), cfg.totalCells(), cfg.cellSize, /*verbose=*/false));
      }

      VkCommandBuffer cmd = ctx.beginCmd();
      engine.step(cmd, args.dt);
      ctx.submitCmd(cmd);

      simTime += args.dt;

      if(args.dump_every > 0 && frame % args.dump_every == 0) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/frame_%04d.pvox", args.out_dir.c_str(), frame);
        engine.dumpFrame(path, simTime);
        std::printf("frame %4d / %4d  t=%.3fs  -> %s\n", frame, args.n_frames, simTime, path);
      }
    }

    engine.cleanup();
    ctx.cleanup();
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
