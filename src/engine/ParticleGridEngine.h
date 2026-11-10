#pragma once

#include <glm/glm.hpp>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "../core/Domain.h"
#include "../core/Emitter.h"
#include "Collider.h"
#include "ComputePipeline.h"
#include "EngineBase.h"
#include "MPMSimPC.h"
#include "MaterialParams.h"

// 粒子-格子転写 (P2G→グリッド更新→G2P) の共通基盤。MPM(F+構成則)と FLIP(圧力投影)は数理モデルが別物なので is-a にせず、共通部のみ持つ。
class ParticleGridEngine : public EngineBase {
public:
  // 各軸の実セル数は gridRes() (=domainSize/cellSize) から導出する。
  glm::vec3 domainSize{10.0f, 10.0f, 10.0f}; // ドメイン物理サイズ [m]
  float cellSize = 10.0f / 64.0f;            // 全軸共通のセルサイズ [m]

  // バッファ上限 (0 = 派生クラスの既定シード数。0 以外なら init() 時の自動シードは行わない)
  uint32_t maxParticles = 0;

  float rho0 = 1000.0f; // 基準密度 [kg/m^3] (既定マテリアルの密度)

  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir);
  void cleanup();

  void step(VkCommandBuffer cmd, float dt);
  void setFixedPointScale(float scale);

  uint32_t liveParticleCount() const { return nParticles_; }
  VkBuffer getPositionBuffer() const;
  VkBuffer getVelocityBuffer() const;
  // コライダーが今フレーム受け取った力積/トルク積(fixed-point, vec4×64, 未使用スロットは0)。CPU側でdt割りして反力[N]/[N・m]に変換する
  VkBuffer getColliderForceBuffer() const;
  VkBuffer getColliderTorqueBuffer() const;
  uint32_t colliderForceCount() const { return pc_.colliderCount; }
  // trueならmpm_grid_update.compのatomicAdd集計とstep()冒頭のゼロクリアdispatchを毎フレーム発行する。falseなら両方とも一切発行されずGPU負荷は増えない(既定false)。
  bool enableColliderForceFeedback = false;
  // true なら粒子が触れる格子ノードの AABB (アクティブ領域) だけをゼロクリア/グリッド更新する (結果は全セル処理と同一)。
  // FLIP は圧力投影が全セルを走査するので使えない (FLIPEngine が false にする)。
  bool activeGridBox = true;
  // P2G の local_size_x (共有メモリタイル集約の粒度)。粒子の並びが空間的にまとまっているほどタイルが効く。
  // 小さいほどタイルが小さくて済むが、グループ数が増える。MPM(雪崩) は 64、FLIP は 256 が最速だった。init() 前に設定する
  uint32_t p2gGroupSize = 64;

  float restitution   = 0.3f;
  float wall_friction = 0.0f;
  int numSubsteps     = 20;
  // 転写モード: 0=PIC (散逸大), -1=APIC (散逸小; MPM のみ), (0,1]=FLIP ブレンド (1=純 FLIP)
  float flip_ratio = 0.0f;

  void setMaterials(const std::vector<MaterialParams>& mats);
  // 各パーティクルの vel.w に floatBitsToUint(matIds[i]) をセットする (初期化直後(vel=0)にのみ呼ぶこと)
  void setParticleMaterialIds(const std::vector<uint32_t>& matIds);

  // ── 解析コライダー ───────────────────────────────────────────────────
  void setColliders(const ColliderSet& cols);
  void clearAnalyticColliders();
  // メッシュSDFコライダー: OBJを読みローカルSDFを焼いてbindlessバッファへアップロード、そのindexを返す。
  uint32_t loadColliderMesh(const std::string& objPath, LocalMeshSDF& gridOut, uint32_t res = 48, float scale = 1.0f);
  uint32_t uploadColliderMeshSDF(const LocalMeshSDF& grid);

  // Emitter::particleType を material id として解釈する
  void addEmitter(std::shared_ptr<Emitter> emitter);
  void clearEmitters();
  virtual void resetParticles(); // ライブ粒子数を 0 にリセット (バッファは再生成しない)

  // pos.w = 粒子体積 Vp, vel.w = floatBitsToUint(material_id)。maxParticles まで追加
  void appendParticles(const std::vector<glm::vec4>& pos, const std::vector<glm::vec4>& vel);

  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet             = VK_NULL_HANDLE;
  uint32_t posIdx                           = 0;
  uint32_t velIdx                           = 0;

protected:
  ComputePipeline& forceTargetPipeline() override { return kGridUpdate_; }
  const char* forceShaderName() const override { return "mpm_grid_update.comp"; }

  // maxParticles==0 のときのシード粒子数。0 を返す派生は maxParticles の明示指定が必須
  virtual uint32_t defaultSeedCount(const glm::uvec3& /*gridRes*/) const { return 0; }
  // 粒子ごとの追加属性バッファ (F/B 等) の確保と pc_ へのインデックス/既定値設定 (P/v/grid 確保後に呼ばれる)
  virtual void allocParticleExtras(uint32_t /*N*/, const glm::uvec3& /*gridRes*/) {}
  // 既定マテリアル (slot 0)。呼び出し時点で rho0 メンバは有効
  virtual MaterialParams defaultMaterial() const;
  // init() 末尾: シード粒子 (nParticles_>0) の投入や派生パイプラインの生成
  virtual void onInitParticles(const std::string& /*shaderDir*/) {}
  // 新規粒子 [first, first+count) の追加属性を初期化 (P/v は基底がアップロード済み)
  virtual void initNewParticleExtras(uint32_t /*first*/, uint32_t /*count*/) {}
  // P2G/G2P/grid_update に注入する #define (FIXED_POINT_SCALE は setFixedPointScale が別途追加)
  virtual std::vector<std::pair<std::string, std::string>> shaderDefines() const { return {}; }
  // buildPC() で毎フレーム変わりうる派生固有の値を pc に書き込む
  virtual void fillPC(MPMSimPC& /*pc*/) const {}
  // GridUpdate の後・G2P の前 (圧力投影など)
  virtual void projectGrid(VkCommandBuffer /*cmd*/, const MPMSimPC& /*pc*/) {}
  // G2P の後 (二次粒子など)
  virtual void afterG2P(VkCommandBuffer /*cmd*/, const MPMSimPC& /*pc*/) {}
  virtual void cleanupDerived() {}
  // Emitter::particleVolume 未指定時の粒子体積
  virtual float defaultParticleVolume() const { return cellSize * cellSize * cellSize; }

  void uploadNewParticles_(const glm::vec4* pos, const glm::vec4* vel, uint32_t count, uint32_t first);
  MPMSimPC buildPC(float subDt) const;
  // groupSize: シェーダの local_size_x (既定 256)
  void dispatchMPM(VkCommandBuffer cmd, ComputePipeline& k, const MPMSimPC& pc, uint32_t count, uint32_t groupSize = 256u);
  void computeBarrier(VkCommandBuffer cmd);
  // アクティブ領域 (activeBox) を記録した後の indirect dispatch 用: compute 書き込み -> DRAW_INDIRECT 読み取り
  void indirectBarrier(VkCommandBuffer cmd);
  void dispatchMPMIndirect(VkCommandBuffer cmd, ComputePipeline& k, const MPMSimPC& pc);
  // NEBULA_TRACY時のみ有効: submit+queueWaitIdleしてZoneScopedN区間に実GPU時間を含める計測専用パス(通常ビルドではno-op)
  void syncGpuForProfiling(VkCommandBuffer cmd);

  // GPU 側状態の唯一の格納先
  MPMSimPC pc_{};
  std::vector<ColliderPrimitive> pendingColliders_;

  uint32_t nParticles_   = 0; // ライブパーティクル数（dispatch ルーピング用）
  uint32_t maxParticles_ = 0; // バッファ確保数 (init() 時に確定)

  uint32_t nextMeshSDFId_ = 0; // メッシュSDF: アップロードごとに一意なバッファ名を振るためのカウンタ

  std::vector<std::pair<std::string, std::string>> userDefines_; // setFixedPointScale 等、利用者指定の define

  ComputePipeline kZeroGrid_;
  ComputePipeline kP2G_;
  ComputePipeline kGridUpdate_;
  ComputePipeline kG2P_;
  ComputePipeline kBounds_;       // 粒子が触れるノードの AABB (mpm_bounds.comp)
  ComputePipeline kBoundsFinish_; // AABB から範囲と indirect 引数を確定 (mpm_bounds_finish.comp)

private:
  std::vector<std::shared_ptr<Emitter>> emitters_;
  std::vector<int> emitterStepsDone_;
  std::mt19937 emitterRng_{12345};
  void emitFromEmitters(float dt);
};
