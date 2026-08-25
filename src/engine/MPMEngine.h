#pragma once

#include <glm/glm.hpp>
#include <string>
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

#include <memory>
#include <random>

// 重力は addForce() で GravityForce を登録すること (issue #30 レビュー対応: gravity の public メンバは廃止)。MPMConfig は廃止済み(issue #98): 設定値は init() 前に public メンバへ直接設定し、GPU 側状態は MPMSimPC pc_ 一箇所で管理する。
class MPMEngine : public EngineBase {
public:
  // issue #46: 各軸の実セル数は gridRes() (=domainSize/cellSize) から導出する。
  glm::vec3 domainSize{10.0f, 10.0f, 10.0f}; // ドメイン物理サイズ [m]
  float cellSize = 10.0f / 64.0f;            // 全軸共通のセルサイズ [m]

  // バッファ上限（0 = 初期パーティクル数(gridRes立方体の40%ブロック)と同じ。0以外を指定するとinit()時の自動シードは行わない）
  uint32_t maxParticles = 0;

  // 初期化時のデフォルトマテリアル(弾性体): Young率 E, ポアソン比 nu, 密度 rho0
  float E    = 1e4f;
  float nu   = 0.3f;
  float rho0 = 1000.0f;

  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir);
  void cleanup();

  void step(VkCommandBuffer cmd, float dt);

  uint32_t liveParticleCount() const { return nParticles_; }
  VkBuffer getPositionBuffer() const;
  VkBuffer getVelocityBuffer() const;
  // コライダーが今フレーム受け取った力積/トルク積(fixed-point, vec4×64, 未使用スロットは0)。CPU側でdt割りして反力[N]/[N・m]に変換する
  VkBuffer getColliderForceBuffer() const;
  VkBuffer getColliderTorqueBuffer() const;
  uint32_t colliderForceCount() const { return pc_.colliderCount; }

  // 外部から調整可能
  float restitution     = 0.3f;
  float wall_friction   = 0.0f;
  int numSubsteps       = 20;
  uint32_t plasticModel = 0;      // 0=弾性, 1=VM, 2=DP（Phase 1 まではグローバル）
  float q_max           = 1e5f;   // VM 降伏応力
  float M_friction      = 0.577f; // DP: tan(30°)
  float q_cohesion      = 0.0f;
  // 転写モード: 0=PIC (散逸大), -1=APIC (散逸小), 1=FLIP (将来実装)
  float flip_ratio = 0.0f;

  // ── マテリアルテーブル設定 (Phase 1) ───────────────────────────────────
  // mats.size() 個のマテリアルを GPU にアップロードし materialCount を更新
  void setMaterials(const std::vector<MaterialParams>& mats);
  // 各パーティクルの vel.w に floatBitsToUint(matIds[i]) をセットする
  // matIds.size() == particleCount() であること
  void setParticleMaterialIds(const std::vector<uint32_t>& matIds);

  // ── 解析コライダー設定 (Phase 3) ───────────────────────────────────────
  // ColliderSet に登録したプリミティブを GPU にアップロードして BC を有効化
  void setColliders(const ColliderSet& cols);
  // アップロード済みの解析コライダーを無効化 (colliderCount = 0)
  void clearAnalyticColliders();

  // ── Emitter (Phase 4) ────────────────────────────────────────────────
  // particleType フィールドを MPM の material id として解釈する
  // step_count=-1: 最初の 1 フレームのみ, 0: 無限, >0: 指定フレーム数
  void addEmitter(std::shared_ptr<Emitter> emitter);
  void clearEmitters();
  void resetParticles(); // ライブ粒子数を 0 にリセット (バッファは再生成しない)

  // ── 粒子の直接追加 ────────────────────────────────────────────────────
  // pos.w = 粒子体積 Vp, vel.w = floatBitsToUint(material_id)
  // F=単位行列, B=0, stress=0 で初期化して maxParticleCount() まで追加
  void appendParticles(const std::vector<glm::vec4>& pos, const std::vector<glm::vec4>& vel);

  // メッシュSDFコライダー: OBJを読みローカルSDFを焼いてbindlessバッファへアップロード、そのindexを返す。gridOutはColliderSet::addMeshSDF用のローカル空間パラメータ出力。scaleはOBJ読み込み時の等方拡大率。
  uint32_t loadColliderMesh(const std::string& objPath, LocalMeshSDF& gridOut, uint32_t res = 48, float scale = 1.0f);
  // 既に焼き済みのLocalMeshSDFをbindlessバッファへアップロードするだけ(loadColliderMeshの後半部分を単独利用したい場合用)。
  uint32_t uploadColliderMeshSDF(const LocalMeshSDF& grid);

  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet             = VK_NULL_HANDLE;
  uint32_t posIdx                           = 0;
  uint32_t velIdx                           = 0;

protected:
  ComputePipeline& forceTargetPipeline() override { return kGridUpdate_; }
  const char* forceShaderName() const override { return "mpm_grid_update.comp"; }

private:
  // GPU 側状態の唯一の格納先 (旧cfg_ + 個別indexメンバとの二重管理を廃止)。
  MPMSimPC pc_{};

  // ライブパーティクル数（dispatch ルーピング用）
  uint32_t nParticles_   = 0;
  uint32_t maxParticles_ = 0; // バッファ確保数 (init() 時に確定)

  // メッシュSDFコライダー: アップロードごとに一意なバッファ名を振るためのカウンタ
  uint32_t nextMeshSDFId_ = 0;

  // Emitter (Phase 4)
  std::vector<std::shared_ptr<Emitter>> emitters_;
  std::vector<int> emitterStepsDone_;
  std::mt19937 emitterRng_{12345};
  void emitFromEmitters(float dt);

  // コンピュートパイプライン (MLS-MPM scatter化により空間ハッシュ系パイプラインは廃止)
  ComputePipeline kZeroGrid_;
  ComputePipeline kP2G_;
  ComputePipeline kGridUpdate_;
  ComputePipeline kG2P_;

  MPMSimPC buildPC(float subDt) const;
  void dispatchMPM(VkCommandBuffer cmd, ComputePipeline& k, const MPMSimPC& pc, uint32_t count);
  void computeBarrier(VkCommandBuffer cmd);
  // NEBULA_TRACY時のみ有効: submit+queueWaitIdleしてZoneScopedN区間に実GPU時間を含める計測専用パス(通常ビルドではno-op)
  void syncGpuForProfiling(VkCommandBuffer cmd);
};
