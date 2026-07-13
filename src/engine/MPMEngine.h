#pragma once

#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include "AttributeBuffer.h"
#include "ComputePipeline.h"
#include "MPMSimPC.h"
#include "MaterialParams.h"
#include "Collider.h"
#include "../core/Emitter.h"

#include <memory>
#include <random>

struct MPMConfig {
  // パーティクル配置: nx × ny × nz の格子
  uint32_t nx          = 32;
  uint32_t ny          = 32;
  uint32_t nz          = 32;
  float    world_size  = 10.0f;
  uint32_t grid_res    = 64;

  // バッファ上限（0 = nx*ny*nz と同じ）
  // Phase 4 のソースエミッタで動的追加する場合に設定
  uint32_t maxParticles = 0;

  // 材料: Young率 E, ポアソン比 nu, 密度 rho0
  float E    = 1e4f;
  float nu   = 0.3f;
  float rho0 = 1000.0f;

  uint32_t particleCount()    const { return nx * ny * nz; }
  uint32_t maxParticleCount() const { return maxParticles > 0 ? maxParticles : particleCount(); }
  uint32_t totalCells()       const { return grid_res * grid_res * grid_res; }
  float    cellSize()         const { return world_size / float(grid_res); }
  float    spacing()          const { return world_size / float(nx); }
  float    particleVolume()   const { float s = spacing(); return s * s * s; }
  float    mu()   const { return E / (2.0f * (1.0f + nu)); }
  float    lame() const { return E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu)); }

  uint32_t nGroups() const { return (totalCells() + 255u) / 256u; }
};

class MPMEngine {
public:
  void init(VkDevice device, VmaAllocator allocator,
            VkDescriptorPool descriptorPool,
            VkCommandPool cmdPool, VkQueue queue,
            const std::string& shaderDir,
            const MPMConfig& cfg = {});
  void cleanup();

  void step(VkCommandBuffer cmd, float dt);

  const MPMConfig& config() const { return cfg_; }
  uint32_t liveParticleCount() const { return nParticles_; }
  VkBuffer getPositionBuffer() const;
  VkBuffer getVelocityBuffer() const;

  // ImGui から調整可能
  float gravity        = -9.8f;
  float restitution    = 0.3f;
  float wall_friction  = 0.0f;
  int   numSubsteps    = 20;
  uint32_t plasticModel = 0;    // 0=弾性, 1=VM, 2=DP（Phase 1 まではグローバル）
  float q_max          = 1e5f; // VM 降伏応力
  float M_friction     = 0.577f; // DP: tan(30°)
  float q_cohesion     = 0.0f;
  // 転写モード: 0=PIC (散逸大), -1=APIC (散逸小), 1=FLIP (将来実装)
  float flip_ratio     = 0.0f;

  // NanoVDB SDF コライダー設定
  // radius, cx, cy, cz は世界座標 (worldMin=0, worldMax=world_size)
  void setColliderSphere(float radius, float cx, float cy, float cz);
  void clearCollider();

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
  void appendParticles(const std::vector<glm::vec4>& pos,
                        const std::vector<glm::vec4>& vel);

  // ── 任意形状 SDF コライダー ────────────────────────────────────────────
  // Morton 順に並んだ float SDF 配列 (totalCells() 要素) を地形コライダーとして設定
  // sdf[mortonEncode(ix,iy,iz)] = 符号付き距離 [m]  負値=障害物内部
  void setColliderSDF(const std::vector<float>& mortonSDF);

  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet       descriptorSet       = VK_NULL_HANDLE;
  uint32_t              posIdx              = 0;
  uint32_t              velIdx              = 0;

private:
  MPMConfig    cfg_;
  VkDevice     device_    = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkCommandPool cmdPool_  = VK_NULL_HANDLE;
  VkQueue       queue_    = VK_NULL_HANDLE;

  // ライブパーティクル数（dispatch ルーピング用）
  uint32_t nParticles_ = 0;

  AttributeBuffer attrBuf_;

  // パーティクルバッファ
  // F0-2: xyz = F列, w = 対角応力 (σ_xx, σ_yy, σ_zz)
  // B0-2: xyz = APIC B列 (Phase 2), w = 非対角応力 (σ_xy, σ_xz, σ_yz)
  uint32_t F0Idx_ = 0;
  uint32_t F1Idx_ = 0;
  uint32_t F2Idx_ = 0;
  uint32_t B0Idx_ = 0;
  uint32_t B1Idx_ = 0;
  uint32_t B2Idx_ = 0;

  // ハッシュグリッドバッファ
  uint32_t cellCountIdx_  = 0;
  uint32_t cellOffsetIdx_ = 0;
  uint32_t sortedIdxIdx_  = 0;

  // MPM グリッドバッファ
  uint32_t gridMomIdx_  = 0;
  uint32_t gridMassIdx_ = 0;

  // マテリアルテーブル SSBO
  uint32_t materialsIdx_   = 0;
  uint32_t materialCount_  = 1;   // 現在の有効エントリ数

  // 解析コライダー SSBO (Phase 3)
  uint32_t collidersIdx_   = 0;
  uint32_t colliderCount_  = 0;   // 0 = 無効

  // NanoVDB SDF コライダー
  uint32_t nanoVDBIdx_  = 0;  // 0 = 未設定 (シェーダー内でスキップ)

  // Emitter (Phase 4)
  std::vector<std::shared_ptr<Emitter>> emitters_;
  std::vector<int>                      emitterStepsDone_;
  std::mt19937                          emitterRng_{12345};
  void emitFromEmitters(float dt);

  // コンピュートパイプライン
  // ハッシュ系 (MPM 版: posIdx を使う)
  ComputePipeline kMpmZeroCells_;
  ComputePipeline kMpmHashCount_;
  ComputePipeline kHashScanLocal_;
  ComputePipeline kHashScanGlobal_;
  ComputePipeline kHashAddBase_;
  ComputePipeline kMpmHashSort_;
  // MPM メインパイプライン
  ComputePipeline kZeroGrid_;
  ComputePipeline kP2G_;
  ComputePipeline kGridUpdate_;
  ComputePipeline kNanoVDBBC_;  // NanoVDB SDF 境界条件 (kGridUpdate_ の後)
  ComputePipeline kG2P_;

  MPMSimPC buildPC(float subDt) const;
  void dispatchMPM(VkCommandBuffer cmd, ComputePipeline& k, const MPMSimPC& pc, uint32_t count);
  void computeBarrier(VkCommandBuffer cmd);
};
