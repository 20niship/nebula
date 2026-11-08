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
#include <vk_mem_alloc.h>
// CPU 時間・常駐メモリはプラットフォームごとに取得方法が違う。未対応環境では 0 を返す。
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#ifdef _MSC_VER
#pragma comment(lib, "psapi.lib")
#endif
#else
#include <sys/resource.h>
#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <fstream>
#include <unistd.h>
#endif
#endif

// fluid 系 example 共通の定期ログ。性能の考察に使う条件 (粒子数・グリッド・サブステップ) と、
// 区間内の実時間 (ms/frame, real-time factor) を一定間隔で printf する。
// 実時間は描画ループの CPU 壁時計で、vsync 待ち・描画時間を含む (GPU 単体の時間ではない)。
// cpu% は区間内のプロセス CPU 時間 / 壁時間 (100% = 1 コア、マルチスレッドなら超える)。rss は常駐メモリ。
// vram は VMA の確保量 (このアプリの allocation 合計) と OS 側の使用量/予算 (device-local ヒープ、Apple は統合メモリ)。
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
  VmaAllocator allocator = VK_NULL_HANDLE; // VRAM 使用量の取得用 (null なら vram は 0)
};

inline FluidStatsInfo fluidStatsInfo(const FluidEngine& e, float dt) {
  const FluidConfig& c = e.config();
  return {"PBF", e.nFluid(), e.nBoundary, c.maxDiffuseParticles, c.gridRes(), c.cellSize, c.domainSize, e.numSubsteps, e.pbfIterations, dt, e.allocator()};
}

inline FluidStatsInfo fluidStatsInfo(const ParticleGridEngine& e, float dt) {
  const auto* flip = dynamic_cast<const FLIPEngine*>(&e);
  return {flip ? "FLIP" : "MPM", e.liveParticleCount(), 0, flip ? flip->maxDiffuseParticles : 0u, domain::gridRes(e.domainSize, e.cellSize),
          e.cellSize, e.domainSize, e.numSubsteps, flip ? flip->pressureIterations : 0, dt, e.allocator()};
}

namespace fluidstats_detail {
inline double processCpuSec() {
#if defined(_WIN32)
  FILETIME creation, exit, kernel, user;
  if(!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) return 0.0;
  auto sec = [](const FILETIME& t) { return double((uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime) * 1e-7; }; // 100ns 単位
  return sec(kernel) + sec(user);
#else
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return double(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) + double(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) * 1e-6;
#endif
}

inline double rssMB() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc{};
  if(GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return double(pmc.WorkingSetSize) / (1024.0 * 1024.0);
#elif defined(__APPLE__)
  mach_task_basic_info info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if(task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) return double(info.resident_size) / (1024.0 * 1024.0);
#elif defined(__linux__)
  std::ifstream f("/proc/self/statm");
  uint64_t size = 0, resident = 0;
  if(f >> size >> resident) return double(resident) * double(sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
#endif
  return 0.0;
}

struct Vram {
  double allocMB = 0.0, usageMB = 0.0, budgetMB = 0.0;
};
inline Vram vram(VmaAllocator a) {
  Vram v;
  if(a == VK_NULL_HANDLE) return v;
  const VkPhysicalDeviceMemoryProperties* props = nullptr;
  vmaGetMemoryProperties(a, &props);
  VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
  vmaGetHeapBudgets(a, budgets);
  for(uint32_t i = 0; i < props->memoryHeapCount; ++i) {
    if(!(props->memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
    v.allocMB += double(budgets[i].statistics.allocationBytes) / (1024.0 * 1024.0);
    v.usageMB += double(budgets[i].usage) / (1024.0 * 1024.0);
    v.budgetMB += double(budgets[i].budget) / (1024.0 * 1024.0);
  }
  return v;
}
} // namespace fluidstats_detail

class FluidStatsLogger {
public:
  explicit FluidStatsLogger(double intervalSec = 2.0) : interval_(intervalSec) {}

  // 1 フレーム (= dt 進めた) ごとに呼ぶ。前回呼び出しからの実時間を積算し、interval_ 経過ごとに出力する。
  void frame(const FluidStatsInfo& info) {
    const auto now = std::chrono::steady_clock::now();
    if(frames_ == 0 && !started_) {
      started_ = true;
      last_ = intervalStart_ = now;
      intervalCpu_           = fluidstats_detail::processCpuSec();
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
    const double wallSec = std::chrono::duration<double>(now - intervalStart_).count();
    const double cpuNow  = fluidstats_detail::processCpuSec();
    const double cpuPct  = (cpuNow - intervalCpu_) / wallSec * 100.0;
    const auto vram      = fluidstats_detail::vram(info.allocator);
    std::printf("[stats] %s t=%.2fs frame=%u | %.2f ms/frame (max %.2f, %.1f fps) rtf=%.2fx | particles=%u boundary=%u foam_cap=%u (%.1f/cell) | grid=%ux%ux%u (%llu cells) h=%.4fm domain=%.1fx%.1fx%.1fm | substeps=%d iters=%d dt=%.4fs | cpu=%.0f%% rss=%.0fMB vram_alloc=%.0fMB (os %.0f / budget %.0f MB)\n",
                info.solver, simTime_, frames_, avg, maxMs_, 1000.0 / avg, (info.dt * intervalFrames_) / (intervalMs_ * 1e-3),
                info.particles, info.boundary, info.foamCap, cells ? double(info.particles) / double(cells) : 0.0,
                info.gridRes.x, info.gridRes.y, info.gridRes.z, (unsigned long long)cells, info.cellSize,
                info.domain.x, info.domain.y, info.domain.z, info.substeps, info.iterations, info.dt, cpuPct, fluidstats_detail::rssMB(), vram.allocMB, vram.usageMB, vram.budgetMB);
    std::fflush(stdout);
    intervalStart_ = now;
    intervalCpu_   = cpuNow;
    intervalFrames_ = 0;
    intervalMs_     = 0.0;
    maxMs_          = 0.0;
  }

private:
  double interval_;
  bool started_ = false;
  std::chrono::steady_clock::time_point last_{}, intervalStart_{};
  uint32_t frames_ = 0, intervalFrames_ = 0;
  double intervalCpu_ = 0.0;
  double simTime_ = 0.0, intervalMs_ = 0.0, maxMs_ = 0.0;
};
