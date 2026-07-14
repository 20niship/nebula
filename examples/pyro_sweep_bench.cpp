/**
 * pyro_sweep_bench — 圧力投影 Red-Black Gauss-Seidel の sweep 回数を振って
 * 大規模グリッド (デフォルト 128^3) で発散しないか・実行時間がどう変わるかを検証する
 * ベンチマークツール。examples/pyro_explosion.cpp 相当の強い浮力+渦度+高速注入という
 * 発散が起きやすい条件を使い、各 sweep 数ごとに新規 PyroEngine で n-frames 実行した後、
 * (1) 速度場に NaN/Inf が出ていないか (2) 内部領域の速度発散(divergence)残差の最大値
 * (3) 実行時間 (壁時計) を CSV に1行ずつ追記する (中断しても途中結果が残るようflushする)。
 *
 * 使い方:
 *   ./pyro_sweep_bench --grid-res 128 --sweeps 5,10,15,20,25,30,40,50,60,80 \
 *       --out sweep_bench_results.csv
 */

#include "core/MeshSDF.h" // meshSdfMortonEncode
#include "engine/PyroEngine.h"
#include "helpers/HeadlessCtx.h"

#include <argparse/argparse.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

static const std::string SHADER_DIR_STR = SHADER_DIR;

struct SweepBenchArgs : public argparse::Args {
  int& grid_res           = kwarg("grid-res", "Pyro グリッド解像度 (2のべき乗)").set_default(128);
  float& world_size       = kwarg("world-size", "world size [m]").set_default(14.0f);
  int& n_frames           = kwarg("n-frames", "各sweep設定で実行するフレーム数").set_default(90);
  float& dt                = kwarg("dt", "フレームタイムステップ [s]").set_default(1.0f / 60.0f);
  std::string& sweeps_csv = kwarg("sweeps", "検証する sweep 回数のカンマ区切りリスト").set_default(std::string("5,10,15,20,25,30,40,50,60,80"));
  std::string& out_path   = kwarg("out", "結果 CSV の出力先パス").set_default(std::string("sweep_bench_results.csv"));
};

static std::vector<int> parseIntList(const std::string& csv) {
  std::vector<int> out;
  std::stringstream ss(csv);
  std::string tok;
  while(std::getline(ss, tok, ',')) {
    if(!tok.empty()) out.push_back(std::stoi(tok));
  }
  return out;
}

int main(int argc, char* argv[]) {
  auto args = argparse::parse<SweepBenchArgs>(argc, argv);

  try {
    const std::vector<int> sweeps = parseIntList(args.sweeps_csv);
    if(sweeps.empty()) {
      std::fprintf(stderr, "Fatal: --sweeps が空です\n");
      return 1;
    }

    FILE* csv = std::fopen(args.out_path.c_str(), "w");
    if(!csv) {
      std::fprintf(stderr, "Fatal: 出力ファイルを開けません: %s\n", args.out_path.c_str());
      return 1;
    }
    std::fprintf(csv, "sweep,dispatches_per_iter,grid_res,n_frames,elapsed_s,ms_per_frame,has_nan,has_inf,max_divergence\n");
    std::fflush(csv);

    std::printf("=== Pyro sweep bench: grid_res=%d world_size=%.1f n_frames=%d ===\n", args.grid_res, args.world_size, args.n_frames);

    for(int sweep : sweeps) {
      HeadlessCtx ctx;
      ctx.init();

      PyroConfig cfg;
      cfg.grid_res   = uint32_t(args.grid_res);
      cfg.world_size = args.world_size;

      PyroEngine engine;
      engine.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADER_DIR_STR, cfg);

      engine.numPressureIters = sweep;
      // pyro_explosion.cpp 相当の「発散しやすい」条件 (強い浮力+渦度閉じ込め)
      engine.buoyancyAlpha      = 5.0f;
      engine.buoyancyBeta       = 0.0f;
      engine.ambientTemp        = 0.0f;
      engine.vorticityEps       = 6.0f;
      engine.densityDissipation = 0.03f;
      engine.tempDissipation    = 0.15f;

      // pyro_cow_blast.cpp 相当の高速運動量注入 (最も発散を誘発しやすい入力)
      auto blast             = std::make_shared<AABBEmitter>();
      const float W          = cfg.world_size;
      blast->center          = {W * 0.08f, W * 0.20f, W * 0.5f};
      blast->size            = glm::vec3(W * 0.10f, W * 0.30f, W * 0.30f);
      blast->inflowVelocity  = {18.0f, 0.0f, 0.0f};
      blast->densityRate     = 60.0f;
      blast->temperatureRate = 20.0f;
      blast->fuelRate        = 0.0f;
      blast->step_count      = 8;
      engine.addEmitter(blast);

      const auto t0 = std::chrono::steady_clock::now();
      for(int frame = 0; frame < args.n_frames; frame++) {
        VkCommandBuffer cmd = ctx.beginCmd();
        engine.step(cmd, args.dt);
        ctx.submitCmd(cmd);
      }
      const auto t1     = std::chrono::steady_clock::now();
      const double elapsedS = std::chrono::duration<double>(t1 - t0).count();

      const uint32_t G  = cfg.grid_res;
      const uint32_t NC = cfg.totalCells();
      std::vector<glm::vec4> vel(NC);
      ctx.readBuffer(engine.getVelocityBuffer(), 0, vel.data(), NC * sizeof(glm::vec4));

      bool hasNan = false, hasInf = false;
      for(const auto& v : vel) {
        if(std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z)) hasNan = true;
        if(std::isinf(v.x) || std::isinf(v.y) || std::isinf(v.z)) hasInf = true;
      }

      auto sampleVel = [&](int x, int y, int z) -> glm::vec3 {
        x = std::clamp(x, 0, int(G) - 1);
        y = std::clamp(y, 0, int(G) - 1);
        z = std::clamp(z, 0, int(G) - 1);
        return glm::vec3(vel[meshSdfMortonEncode(uint32_t(x), uint32_t(y), uint32_t(z))]);
      };

      const float h = cfg.cellSize();
      float maxDiv  = 0.0f;
      // 境界を避けた内部領域のみチェック (全体の中央8割)
      const int lo = int(G) / 10 + 1;
      const int hi = int(G) - lo;
      for(int x = lo; x < hi; x++)
        for(int y = lo; y < hi; y++)
          for(int z = lo; z < hi; z++) {
            float div = 0.5f * ((sampleVel(x + 1, y, z).x - sampleVel(x - 1, y, z).x) + (sampleVel(x, y + 1, z).y - sampleVel(x, y - 1, z).y) + (sampleVel(x, y, z + 1).z - sampleVel(x, y, z - 1).z)) / h;
            if(!std::isfinite(div)) continue; // NaN/Infは上のhasNan/hasInfで別途記録済み
            maxDiv = std::max(maxDiv, std::abs(div));
          }

      const double msPerFrame = elapsedS / args.n_frames * 1000.0;
      std::printf("sweep=%3d  elapsed=%.3fs  ms/frame=%.2f  NaN=%d  Inf=%d  maxDiv=%.4f\n", sweep, elapsedS, msPerFrame, int(hasNan), int(hasInf), double(maxDiv));

      std::fprintf(csv, "%d,%d,%d,%d,%.6f,%.4f,%d,%d,%.6f\n", sweep, sweep * 2, args.grid_res, args.n_frames, elapsedS, msPerFrame, int(hasNan), int(hasInf), double(maxDiv));
      std::fflush(csv);

      engine.cleanup();
      ctx.cleanup();
    }

    std::fclose(csv);
    std::printf("=== 完了: 結果を %s に保存しました ===\n", args.out_path.c_str());
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
