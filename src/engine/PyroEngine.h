#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "../core/Domain.h"
#include "../core/Emitter.h"
#include "../core/PyroSimPC.h"
#include "ComputePipeline.h"
#include "EngineBase.h"

// Houdini Pyro 的なグリッド(オイラー)ソルバー。PyroConfigは廃止済み(issue #98相当): 設定値はinit()前にpublicメンバへ直接設定し、GPU側状態はPyroSimPC pc_一箇所で管理する。
class PyroEngine : public EngineBase {
public:
  // Pyroは「スレッドID=Morton符号」の稠密グリッド設計のため、GPU dispatch/バッファ確保はcubeRes(2のべき乗立方体)単位で行う(domain::mortonCubeRes()が自動丸め)。
  glm::vec3 domainSize{10.0f, 10.0f, 10.0f}; // ドメイン物理サイズ [m] (旧 world_size)
  float cellSize       = 10.0f / 64.0f;      // 全軸共通のセルサイズ [m] (旧 grid_res の逆算値)
  uint32_t maxEmitters = 32;                 // 同時に登録できる Emitter の上限 (SSBO 固定容量)

  glm::uvec3 gridRes() const { return domain::gridRes(domainSize, cellSize); } // 各軸の実セル数 (nx,ny,nz)。2^n不要
  uint32_t cubeRes() const { return domain::mortonCubeRes(gridRes()); }        // Morton dispatch用の立方体解像度 (自動的に2^n、CPU限定)
  uint32_t totalCells() const { return domain::hashCellsCube(gridRes()); }     // = cubeRes()^3。GPUバッファ確保数/dispatch数
  uint32_t nGroups() const { return domain::nGroups(totalCells()); }

  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir);
  void cleanup();

  void step(VkCommandBuffer cmd, float dt);

#ifdef NEBULA_GPU_PROFILING
  // ── GPUパス単位プロファイリング(診断用) ──────────────────────────────
  void enableGpuProfiling(VkPhysicalDevice physicalDevice);
  void printGpuProfile();
#endif

  // pc_と同名同義の物理パラメータ(buoyancyAlpha等)は個別メンバを持たずpc_を直接書き換える。numPressureItersはRed-Black Gauss-Seidel sweep回数(1 sweepにつきred/black 2ディスパッチ)。
  int numPressureIters = 10; // CPU側ループ回数(PyroSimPCに対応フィールドなし)
  int numSubsteps      = 1;  // CPU側ループ回数(PyroSimPCに対応フィールドなし)

  // GPU側状態の唯一の格納先。バッファindex類はinit()で一度だけ設定される。
  PyroSimPC pc_{};

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
  uint32_t emittersIdx_         = 0; // init() で maxEmitters 分を固定確保
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
