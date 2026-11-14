#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cmath>
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
  // Pyro のバッファは線形配置 (x 最速, 添字 = x + nx*(y + ny*z))。確保セル数 = nx*ny*nz で穴が無く、軸あたり 1024 の上限も無い。
  glm::vec3 domainSize{10.0f, 10.0f, 10.0f}; // ドメイン物理サイズ [m] (旧 world_size)
  float cellSize       = 10.0f / 64.0f;      // 全軸共通のセルサイズ [m] (旧 grid_res の逆算値)
  uint32_t maxEmitters = 32;                 // 同時に登録できる Emitter の上限 (SSBO 固定容量)

  glm::uvec3 gridRes() const { return domain::gridRes(domainSize, cellSize); } // 各軸の実セル数 (nx,ny,nz)。2^n不要
  uint32_t totalCells() const { return domain::linearCells(gridRes()); }  // GPU バッファ確保数/dispatch 数 (= nx*ny*nz)
  uint32_t cellIndex(uint32_t x, uint32_t y, uint32_t z) const { return domain::linearIndex(gridRes(), x, y, z); }
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
  // 有効なチャンネル (init() 前に設定)。無効なチャンネルは 1 要素のダミーバッファになり、その分のメモリ (1 セルあたり density 8B, temperature 8B, fuel 8B, flame 4B, curl 16B) を節約できる。
  // 風洞のように速度場しか使わないなら kChVelocityOnly で 1 セル約 44 byte (既定 88 byte)。燃焼 (fuel/flame) には fuel+temperature+flame+density、浮力には temperature+density、渦度閉じ込めには curl が必要。
  static constexpr uint32_t kChDensity = 1u, kChTemperature = 2u, kChFuel = 4u, kChFlame = 8u, kChCurl = 16u;
  static constexpr uint32_t kChAll = 31u, kChVelocityOnly = 0u;
  uint32_t channels = kChAll;

  int numPressureIters = 10; // CPU側ループ回数(PyroSimPCに対応フィールドなし)
  int numSubsteps      = 1;  // CPU側ループ回数(PyroSimPCに対応フィールドなし)

  // GPU側状態の唯一の格納先。バッファindex類はinit()で一度だけ設定される。
  PyroSimPC pc_{};

  // ── 障害物 SDF (任意形状、mpm_stl_drop.cpp 由来の MeshSDF.h で構築) ──────
  // 線形順 (x 最速) に並んだ float SDF 配列 (totalCells() 要素)。負値=障害物内部。
  void setColliderSDF(const std::vector<float>& linearSDF);
  void clearCollider();

  // ── 移動する剛体コライダー (任意の位置・姿勢・線速度・角速度。回転する車の風洞、動く物体など) ──────────
  // 剛体の状態。角速度は剛体原点まわり、すべて世界座標系。表面の速さは v = linearVelocity + angularVelocity x (p - position)。
  struct RigidBodyState {
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f}; // 剛体ローカル -> 世界
    glm::vec3 linearVelocity{0.0f};
    glm::vec3 angularVelocity{0.0f};
  };
  // localSDF: 剛体ローカル座標の SDF (節点値, x 最速の線形配置 res.x*res.y*res.z, 各軸 1..1023)。節点 (i,j,k) の位置は localMin + (i,j,k)*localCell (剛体ローカル座標)。
  // step() は毎サブステップ colliderSDF をこの SDF から現在の姿勢で焼き直し (GPU)、障害物 BC と発散に壁面速度を課す。地面 (最下段セル) も固体になる。
  // autoAdvance=true: 一定の線速度・角速度で状態を積分する (位置 += v dt、姿勢 = exp(ω dt) * 姿勢)。false: 呼び出し側が毎ステップ setColliderState() で与える (任意の運動)。
  void setMovingCollider(const std::vector<float>& localSDF, glm::uvec3 res, float localCell, glm::vec3 localMin, const RigidBodyState& initial, bool autoAdvance = true);
  void setColliderState(const RigidBodyState& s) { body_ = s; }
  const RigidBodyState& colliderState() const { return body_; }
  VkBuffer getColliderSDFBuffer() const { return colliderSDFIdx_ ? attrBuf_.getBuffer("colliderSDF") : VK_NULL_HANDLE; }

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
  // 圧力・発散 (色優先レイアウト: shaders/pyro_common.glsl の pyroRBIndex。テスト・デバッグ用)
  VkBuffer getPressureBuffer() const { return attrBuf_.getBuffer("pres"); }
  VkBuffer getDivergenceBuffer() const { return attrBuf_.getBuffer("div"); }
  bool hasCollider() const { return colliderSDFIdx_ != 0; }

  // ── ボクセルダンプ (.pvox) ───────────────────────────────────────────────
  // density/temperature/fuel/flame/velocity/sdf を GPU→CPU readback し、線形
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
  ComputePipeline kRotateSDF_;
  RigidBodyState body_;
  bool bodyAutoAdvance_ = false;
  ComputePipeline kAdvect_;
  ComputePipeline kCurl_;
  ComputePipeline kVorticityForce_;
  ComputePipeline kDivergence_;
  ComputePipeline kPressureGS_;
  ComputePipeline kProject_;

  PyroSimPC buildPC(float dt) const;
  void dispatchPyro(VkCommandBuffer cmd, ComputePipeline& k, const PyroSimPC& pc);
  void dispatchPressureGS(VkCommandBuffer cmd, const PyroSimPC& pc);
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
