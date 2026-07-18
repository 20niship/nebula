#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "../core/Emitter.h"
#include "../core/PyroSimPC.h"
#include "ComputePipeline.h"
#include "EngineBase.h"

struct PyroConfig {
  // Morton (Z-order) 符号化を使うため 2 のべき乗であること (例: 16, 32, 64)。
  // べき乗以外を指定すると PyroEngine::init() が例外を投げる
  // (べき乗でない場合 decode(gid) が gridRes 範囲外の (ix,iy,iz) を生成し、
  //  GPU バッファ境界外アクセスを起こすため)。
  uint32_t grid_res    = 64;
  float world_size     = 10.0f;
  uint32_t maxEmitters = 32; // 同時に登録できる Emitter の上限 (SSBO 固定容量)

  uint32_t totalCells() const { return grid_res * grid_res * grid_res; }
  float cellSize() const { return world_size / float(grid_res); }
  uint32_t nGroups() const { return (totalCells() + 255u) / 256u; }
};

// Houdini Pyro 的なグリッド(オイラー)ソルバー。MPMEngine と異なりパーティクルを
// 持たず、Morton順の Dense セル中心グリッド上で density/temperature/fuel/velocity
// を直接解く (semi-Lagrangian 移流 + 圧力投影 + 燃焼反応)。
class PyroEngine : public EngineBase {
public:
  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, const PyroConfig& cfg = {});
  void cleanup();

  void step(VkCommandBuffer cmd, float dt);

#ifdef NEBULA_GPU_PROFILING
  // ── GPUパス単位プロファイリング(診断用) ──────────────────────────────
  void enableGpuProfiling(VkPhysicalDevice physicalDevice);
  void printGpuProfile();
#endif

  const PyroConfig& config() const { return cfg_; }

  // ── 物理パラメータ (ImGui/CLI から調整可能) ─────────────────────────────
  float buoyancyAlpha      = 1.2f;  // 温度浮力係数
  float buoyancyBeta       = 0.4f;  // 密度による重さ (下降) 係数
  float ambientTemp        = 0.0f;  // 環境温度
  float vorticityEps       = 0.0f;  // 渦度閉じ込め強度 (Phase2)
  float densityDissipation = 0.05f; // 密度減衰係数 [1/s]
  float tempDissipation    = 0.2f;  // 温度減衰係数 [1/s] (環境温度への復帰)
  float ignitionTemp       = 0.0f;  // 発火温度 (Phase3)
  float burnRate           = 0.0f;  // 燃料消費速度 [1/s] (Phase3)
  float heatRelease        = 0.0f;  // 燃焼による温度上昇量 (Phase3)
  float smokeYieldPerFuel  = 0.0f;  // 燃焼による密度生成量 (Phase3)
  float flameBrightness    = 0.0f;  // 燃焼による発光量 (Phase3)
  // 圧力投影 Red-Black Gauss-Seidel sweep回数 (1 sweepにつきred/black 2ディスパッチ)。
  int numPressureIters     = 10;
  int numSubsteps          = 1;

  // ── 障害物 SDF (任意形状、mpm_stl_drop.cpp 由来の MeshSDF.h で構築) ──────
  // Morton 順に並んだ float SDF 配列 (totalCells() 要素)。負値=障害物内部。
  void setColliderSDF(const std::vector<float>& mortonSDF);
  void clearCollider();

  // ── 複数 Emitter (連続的な density/temperature/fuel/velocity 注入) ──────
  void addEmitter(std::shared_ptr<Emitter> emitter);
  void clearEmitters();

  // 任意方向の風・Turbulence・Noise は addForce() で登録する (issue #30)。
  // 浮力(buoyancyAlpha/Beta)はPyro固有の温度連成物理でありForce化しない。

  // ── 読み取り (テスト/ダンプ用) ────────────────────────────────────────
  VkBuffer getDensityBuffer() const;
  VkBuffer getTemperatureBuffer() const;
  VkBuffer getFuelBuffer() const;
  VkBuffer getFlameBuffer() const;
  VkBuffer getVelocityBuffer() const;
  bool hasCollider() const { return colliderSDFIdx_ != 0; }

  // ── ボクセルダンプ (.pvox) ───────────────────────────────────────────────
  // density/temperature/fuel/flame/velocity/sdf を GPU→CPU readback し、Morton→線形
  // (x + y*nx + z*nx*ny) に並べ替えてから独自バイナリ形式で書き出す。
  // sdf チャンネルは障害物未設定時、全セル背景値 (1e6, 障害物なし) で埋める。
  void dumpFrame(const std::string& path, float simTime) const;

  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet             = VK_NULL_HANDLE;

protected:
  ComputePipeline& forceTargetPipeline() override { return kForces_; }
  const char* forceShaderName() const override { return "pyro_forces.comp"; }

private:
  PyroConfig cfg_;

  // ダブルバッファ (A/B): cur_==0 なら [0]が現在値/[1]が次フレーム書き込み先
  uint32_t velIdx_[2]         = {0, 0};
  uint32_t densityIdx_[2]     = {0, 0};
  uint32_t temperatureIdx_[2] = {0, 0};
  uint32_t fuelIdx_[2]        = {0, 0};
  int cur_                    = 0;

  uint32_t flameIdx_       = 0;
  uint32_t pressureIdx_    = 0; // Red-Black Gauss-Seidel、in-place更新のため単一バッファ
  uint32_t divergenceIdx_  = 0;
  uint32_t curlIdx_        = 0; // 渦度閉じ込め用スクラッチ (vec4)

  uint32_t colliderSDFIdx_ = 0; // 0 = 無効

  // Emitter
  uint32_t emittersIdx_         = 0; // init() で cfg_.maxEmitters 分を固定確保
  uint32_t emittersActiveCount_ = 0; // 直近 updateEmitters() でアップロードした有効数
  std::vector<std::shared_ptr<Emitter>> emitters_;
  std::vector<int> emitterStepsDone_;

  ComputePipeline kEmit_;
  ComputePipeline kCombustion_;
  ComputePipeline kForces_;
  ComputePipeline kObstacleBC_;
  ComputePipeline kAdvect_;
  ComputePipeline kCurl_;
  ComputePipeline kVorticityForce_;
  ComputePipeline kDivergence_;
  ComputePipeline kPressureGS_;
  ComputePipeline kProject_;

  PyroSimPC buildPC(float dt) const;
  void dispatchPyro(VkCommandBuffer cmd, ComputePipeline& k, const PyroSimPC& pc);
  void computeBarrier(VkCommandBuffer cmd);
  void dispatchAndBarrier(VkCommandBuffer cmd, ComputePipeline& k, const PyroSimPC& pc, const char* label);
  void updateEmitters(float dt); // CPU側でアクティブな Emitter を選別・アップロード

  // ステージングバッファ経由の GPU→CPU 同期読み戻し (dumpFrame 専用)
  void readBufferToCPU(VkBuffer src, void* dst, size_t bytes) const;

#ifdef NEBULA_GPU_PROFILING
  // プロファイリング用
  VkQueryPool profPool_ = VK_NULL_HANDLE;
  bool profEnabled_     = false;
  double profTsPeriodNs_ = 1.0;
  static constexpr uint32_t kProfMaxQueries = 256;
  std::vector<std::string> profLabels_;
  uint32_t profQueryIndex_ = 0;
#endif
};
