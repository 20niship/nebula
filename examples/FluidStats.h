#pragma once

#include "core/Domain.h"
#include "engine/FLIPEngine.h"
#include "engine/FluidEngine.h"
#include "engine/ParticleGridEngine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <glm/glm.hpp>

// fluid 系 example 共通の定期ログ。性能の考察に使う条件 (粒子数・グリッド・サブステップ) と、
// 区間内の実時間 (ms/frame, real-time factor) を一定間隔で printf する。
// 実時間は描画ループの CPU 壁時計で、vsync 待ち・描画時間を含む (GPU 単体の時間ではない)。
struct FluidStatsInfo {
  const char* solver = "";
  uint32_t particles = 0;   // 流体粒子数
  uint32_t boundary  = 0;   // 境界粒子数 (PBF のみ)
  uint32_t foamCap   = 0;   // 泡粒子の上限 (0=泡なし)
  glm::uvec3 gridRes{0};
  float cellSize = 0.0f;
  glm::vec3 domain{0.0f};
  int substeps   = 0;
  int iterations = 0; // PBF: pbfIterations, FLIP: pressureIterations, MPM: 0
  float dt       = 0.0f;
};

inline FluidStatsInfo fluidStatsInfo(const FluidEngine& e, float dt) {
  const FluidConfig& c = e.config();
  return {"PBF", e.nFluid(), e.nBoundary, c.maxDiffuseParticles, c.gridRes(), c.cellSize, c.domainSize, e.numSubsteps, e.pbfIterations, dt};
}

inline FluidStatsInfo fluidStatsInfo(const ParticleGridEngine& e, float dt) {
  const auto* flip = dynamic_cast<const FLIPEngine*>(&e);
  return {flip ? "FLIP" : "MPM", e.liveParticleCount(), 0, flip ? flip->maxDiffuseParticles : 0u, domain::gridRes(e.domainSize, e.cellSize),
          e.cellSize, e.domainSize, e.numSubsteps, flip ? flip->pressureIterations : 0, dt};
}

class FluidStatsLogger {
public:
  explicit FluidStatsLogger(double intervalSec = 2.0) : interval_(intervalSec) {}

  // 1 フレーム (= dt 進めた) ごとに呼ぶ。前回呼び出しからの実時間を積算し、interval_ 経過ごとに出力する。
  void frame(const FluidStatsInfo& info) {
    const auto now = std::chrono::steady_clock::now();
    if(frames_ == 0 && !started_) {
      started_ = true;
      last_ = intervalStart_ = now;
      return;
    }
    const double ms = std::chrono::duration<double, std::milli>(now - last_).count();
    last_           = now;
    ++frames_;
    ++intervalFrames_;
    simTime_ += info.dt;
    intervalMs_ += ms;
    maxMs_ = std::max(maxMs_, ms);

    if(std::chrono::duration<double>(now - intervalStart_).count() < interval_) return;
    const double avg = intervalMs_ / intervalFrames_;
    const uint64_t cells = uint64_t(info.gridRes.x) * info.gridRes.y * info.gridRes.z;
    std::printf("[stats] %s t=%.2fs frame=%u | %.2f ms/frame (max %.2f, %.1f fps) rtf=%.2fx | particles=%u boundary=%u foam_cap=%u (%.1f/cell) | grid=%ux%ux%u (%llu cells) h=%.4fm domain=%.1fx%.1fx%.1fm | substeps=%d iters=%d dt=%.4fs\n",
                info.solver, simTime_, frames_, avg, maxMs_, 1000.0 / avg, (info.dt * intervalFrames_) / (intervalMs_ * 1e-3),
                info.particles, info.boundary, info.foamCap, cells ? double(info.particles) / double(cells) : 0.0,
                info.gridRes.x, info.gridRes.y, info.gridRes.z, (unsigned long long)cells, info.cellSize,
                info.domain.x, info.domain.y, info.domain.z, info.substeps, info.iterations, info.dt);
    std::fflush(stdout);
    intervalStart_ = now;
    intervalFrames_ = 0;
    intervalMs_     = 0.0;
    maxMs_          = 0.0;
  }

private:
  double interval_;
  bool started_ = false;
  std::chrono::steady_clock::time_point last_{}, intervalStart_{};
  uint32_t frames_ = 0, intervalFrames_ = 0;
  double simTime_ = 0.0, intervalMs_ = 0.0, maxMs_ = 0.0;
};
