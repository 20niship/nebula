#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "../core/Domain.h"
#include "../core/Emitter.h"
#include "ComputePipeline.h"
#include "EngineBase.h"
#include "SimPC.h"

struct FluidConfig {
  float particleRadius = 20.0f / 192.0f / 2.0f; // 唯一のサイズ知識源 (spacing = 2*radius)

  glm::vec3 domainSize{20.0f, 20.0f, 20.0f}; // ドメイン物理サイズ [m] (旧 world_size)
  float cellSize = 20.0f / 64.0f;            // 全軸共通のセルサイズ [m] (旧 grid_res の逆算値)
  uint32_t max_boundary = 50000;

  uint32_t maxDiffuseParticles = 0; // 0=無効 (バッファ・dispatch 完全スキップ)

  // fluidCount() の理論値は大ドメインで実使用量を大幅に超えるため、実量に近い値を指定して初期確保を抑える。
  uint32_t initialCapacityHint = 0;

  bool halfVec4    = false; // 全 vec4 half (小ドメイン専用; 大ドメインでは位置精度不足で発散)
  bool halfVec4Vel = false; // v/omega のみ half (大ドメインでも安定; P/predP は FP32 維持)

  float particleSpacing() const { return particleRadius * 2.0f; }
  // domain体積を粒子スペーシングの立方体で埋め尽くした場合の粒子数 (GPUバッファ確保上限)
  uint32_t fluidCount() const {
    const float d = particleSpacing();
    return (uint32_t)(domainSize.x * domainSize.y * domainSize.z / (d * d * d));
  }
  glm::uvec3 gridRes() const { return domain::gridRes(domainSize, cellSize); }
  // 空間ハッシュバッファの実要素数 (= cubeRes^3。gridRes.x*y*zではない点に注意)
  uint32_t totalCells() const { return domain::hashCells(gridRes()); }
  uint32_t nTotalMax() const { return fluidCount() + max_boundary; }

  // 粒子間距離 d と平滑化長 h から静止密度 ρ₀ を数値計算
  // 均一格子上での Poly6 カーネル和 = シェーダー側の pbf_density と一致する値
  float computeRestDensity() const {
    const float h     = cellSize;
    const float d     = particleSpacing();
    const float h2    = h * h;
    const float h9    = h2 * h2 * h2 * h2 * h;
    const float poly6 = 315.0f / (64.0f * 3.14159265f * h9);
    const int R       = static_cast<int>(h / d) + 1;
    float rho         = 0.0f;
    for(int iz = -R; iz <= R; ++iz)
      for(int iy = -R; iy <= R; ++iy)
        for(int ix = -R; ix <= R; ++ix) {
          float r2 = float(ix * ix + iy * iy + iz * iz) * d * d;
          if(r2 >= h2) continue;
          float hr2 = h2 - r2;
          rho += poly6 * hr2 * hr2 * hr2;
        }
    return rho;
  }
};

class FluidEngine : public EngineBase {
public:
  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, const FluidConfig& cfg = {});
  void cleanup();

  void emitFromEmitters(float dt);

  void step(VkCommandBuffer cmd, float dt);
  void resetParticles(); // 粒子を初期位置・速度にリセット（バッファ/パイプラインは再生成しない）

  void addEmitter(std::shared_ptr<Emitter> emitter);
  void clearEmitters();
  uint32_t nFluid() const { return nFluid_; }

  // growFluidCapacity() で増加するため呼び出し側は毎フレーム参照すること。
  uint32_t totalParticleCapacity() const { return totalBufferCapacity(); }

  void loadBoundary(const std::string& objPath, float spacing);
  void loadBoundary(const std::string& objPath, float spacing, float scale, glm::vec3 offset, bool yup_to_zup);
  // OBJ を経由せず位置ベクタから直接境界粒子を登録する (TC8 スクリュー用)
  void loadBoundaryParticles(const std::vector<glm::vec4>& pts);
  void clearBoundary();

  const std::vector<glm::vec3>& getBoundaryTriVerts() const { return boundaryTriVerts_; }
  const FluidConfig& config() const { return cfg_; }

protected:
  ComputePipeline& forceTargetPipeline() override { return kPredictSdf_; }
  const char* forceShaderName() const override { return "predict_sdf.comp"; }

public:
  // 外部から調整可能なパラメータ
  float restitution = 0.1f;  // ※PBF流体では未使用（衝突は位置投影のみ）
  float friction    = 0.05f; // ※PBF流体では未使用
  float rho0        = 35.0f;
  float viscosityC  = 0.01f;
  int pbfIterations = 2;
  int numSubsteps   = 2;

  float cfmEpsilon       = 3000.0f; // CFM 緩和 ε (式11)。元のハードコード値
  float scorrK           = 0.001f;  // 人工圧力 k (式13; 0=無効)
  float surfaceTension   = 0.0f;    // 表面張力係数 σ (Akinci 2013 cohesion; 0=無効)
  float vorticityEpsilon = 0.1f;    // 渦度閉じ込め ε (式16)
  float linearDamping    = 0.02f;   // 速度減衰 [1/s]。元のハードコード値
  bool vorticityEnabled  = false;   // 渦度閉じ込めの ON/OFF

  // ── 煙・粉体パラメータ ──────────────────────────────────────────────────────
  float smokeRiseAccel = 8.0f; // 煙の浮力加速度 [m/s²] (typeFlag==4)
  float smokeDamping   = 0.5f; // 煙の速度減衰係数 [1/s] (typeFlag==4)
  float powderFriction = 0.0f; // 粉体摩擦係数 [1/s] (typeFlag==5; 将来拡張用)

  uint32_t nBoundary = 0;

  // ── 吸収形状ディスクリプタ ───────────────────────────────────────────────────
  // type: 0=Sphere, 1=CylinderZ, 2=Box, 3=CapsuleZ
  // 各フィールドは float (type も float として格納; シェーダーで uint() キャスト)
  struct AbsorberDesc {
    float type;       // 0=Sphere, 1=CylinderZ, 2=Box, 3=CapsuleZ
    float cx, cy, cz; // 中心座標 [m]
    float p0, p1, p2; // 形状パラメータ: Sphere→(r), CylZ→(r,halfH), Box→(hx,hy,hz), Capsule→(r,halfL)
    float rate;       // 吸収確率 per substep [0.0, 1.0]

    static AbsorberDesc Sphere(float cx_, float cy_, float cz_, float r, float rate_ = 1.0f) { return {0.0f, cx_, cy_, cz_, r, 0.0f, 0.0f, rate_}; }
    static AbsorberDesc CylinderZ(float cx_, float cy_, float cz_, float r, float halfH, float rate_ = 1.0f) { return {1.0f, cx_, cy_, cz_, r, halfH, 0.0f, rate_}; }
    static AbsorberDesc Box(float cx_, float cy_, float cz_, float hx, float hy, float hz, float rate_ = 1.0f) { return {2.0f, cx_, cy_, cz_, hx, hy, hz, rate_}; }
    static AbsorberDesc CapsuleZ(float cx_, float cy_, float cz_, float r, float halfL, float rate_ = 1.0f) { return {3.0f, cx_, cy_, cz_, r, halfL, 0.0f, rate_}; }
  };
  static constexpr uint32_t MAX_ABSORBERS = 32;

  // 吸収形状を登録（毎フレーム step() の前に呼ぶ; absorbers が空なら吸収パスをスキップ）
  void setAbsorbers(const std::vector<AbsorberDesc>& absorbers);

  struct FoamParams { // Ihmsen 2012 ポテンシャル関数ベース
    float kTa = 4000.0f, kWc = 4000.0f;                // 生成係数 (trapped-air / wave-crest)
    float taLo = 5.0f, taHi = 20.0f;                    // trapped-air 正規化範囲
    float wcLo = 1.0f, wcHi = 5.0f;                     // wave-crest 正規化範囲
    float keLo = 5.0f, keHi = 50.0f;                    // kinetic-energy ゲート範囲
    float lifetimeMin = 1.0f, lifetimeMax = 3.0f;       // 寿命 [s]
    float bubbleBuoyancy = 4.0f;                        // bubble の浮力加速度 [m/s²]
    float dragCoeff      = 0.4f;                        // foam/bubble が周囲流体速度へ追従する割合 [0,1]
    float gravityAccel   = -9.8f;                       // spray に適用する重力加速度 (Z-up, 負値)
    float neighborLo = 6.0f, neighborHi = 20.0f;        // 近傍数による分類閾値: spray<Lo<=foam<Hi<=bubble
    float surfaceDensityRatio = 0.95f;                  // 生成ゲート: rho_i/rho0 がこれ未満の粒子のみ対象
  };
  static constexpr uint32_t MAX_DIFFUSE_PARTICLES_HARD_CAP = 2'000'000u; // 暴走設定に対する安全上限

  // 泡パラメータを登録（毎フレーム呼ぶ必要はない。setAbsorbers と同様 upload のみ）
  void setFoamParams(const FoamParams& params);
  bool foamEnabled = false; // false のとき dispatch スキップ (バッファは保持)

  uint32_t foamPosIdx() const { return pc_.foamPosIdx; }   // 描画側から参照 (vec4: xyz=pos, w=残り寿命)
  uint32_t foamVelIdx() const { return pc_.foamVelIdx; }   // vec4: xyz=vel, w=初期寿命
  uint32_t foamKindIdx() const { return pc_.foamKindIdx; } // uint: 0=死/未使用,1=spray,2=foam,3=bubble

  void debugSetFoamSlot(uint32_t slot, glm::vec4 pos, glm::vec4 vel, uint32_t kind);

  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet             = VK_NULL_HANDLE;
  uint32_t posIdx() const { return pc_.posIdx; }
  uint32_t velIdx() const { return pc_.velIdx; }
  // 外部シェーダー（fluid_absorb.comp 等）からも参照されるバッファインデックス
  uint32_t predPIdx() const { return pc_.predPIdx; }
  uint32_t invMassIdx() const { return pc_.invMassIdx; }
  uint32_t typeFlagIdx() const { return pc_.typeFlagIdx; }

  VkBuffer getPositionBuffer() const;
  // 泡バッファの直接読み戻し用 (テスト/デバッグ; issue #47)
  VkBuffer getFoamPositionBuffer() const;
  VkBuffer getFoamVelocityBuffer() const;
  VkBuffer getFoamKindBuffer() const;
  VkBuffer getTypeFlagBuffer() const; // 描画側の生存判定用(typeFlag==0=墓場送り済み/死)
  VkBuffer getLifeBuffer() const;     // 残り寿命[s](<0=無限)
  VkBuffer getEmitterIndexBuffer() const; // 各粒子の放出元エミッタindex(uint×N)。描画側でindex→マテリアル(色)に変換する

  // ── TC8: 運動学的境界粒子 (回転スクリュー等) の per-frame GPU 更新 ────────────
  // maxBoundaryCount: 毎フレーム更新する境界粒子の最大数
  void initKinematicBoundaryStaging(uint32_t maxBoundaryCount);
  // cmd の中に vkCmdCopyBuffer × 3 + barrier を記録する。engine_.step() の前に呼ぶ。
  void recordKinematicBoundaryUpdate(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t boundaryOffset, uint32_t count,
                                     const glm::vec4* positions,   // world-space 位置
                                     const glm::vec4* velocities); // ω × (pos - pivot)
  void cleanupKinematicBoundaryStaging();

private:
  FluidConfig cfg_;

  uint32_t fluidCapacity_                = 0;   // 現在確保済みの流体パーティクル容量 (>= nFluid_)
  static constexpr uint32_t kDispatchPad = 256; // ローカルワークグループサイズと同じ
  uint32_t totalBufferCapacity() const { return cfg_.max_boundary + fluidCapacity_ + kDispatchPad; }
  void growFluidCapacity(uint32_t minRequired);

  // 近傍リストキャッシュ (issue #87 perf実験): 48だと高密度シーン(TC13, h=2*spacing)で近傍が切り捨てられ発散するため96
  static constexpr uint32_t kMaxNeighbors = 96;
  // lifeIdx_ は廃止: 寿命はv.w(velIdx)に格納 (task2: バッファ統合)
  uint32_t emitterIdxIdx_ = 0; // 放出元エミッタindex (uint × N)
  bool lifetimeEnabled_   = false; // lifetime>0のEmitterが登録されたら有効(lifetimeパスを実行)

  ComputePipeline kPredictSdf_;
  ComputePipeline kSdfVelocity_; // issue #66: SDF境界再衝突+速度更新を統合(旧kSdfCollision_+kUpdateVelocity_)
  ComputePipeline kHashCount_;
  ComputePipeline kHashScanLocal_;
  ComputePipeline kHashScanGlobal_;
  ComputePipeline kHashSort_;
  ComputePipeline kPbfDensity_;
  ComputePipeline kPbfDeltaP_;
  ComputePipeline kPbfViscosity_;
  ComputePipeline kZeroCells_;
  ComputePipeline kHashAddBase_;
  ComputePipeline kVorticityOmega_;
  ComputePipeline kVorticityForce_;
  ComputePipeline kAbsorb_;       // 吸収パス（fluid_absorb.comp; pc_.absorberCount>0 のときのみ使用）
  ComputePipeline kFoamGenerate_; // 泡生成パス（pbf_foam_generate.comp; foamEnabled かつ maxDiffuseParticles>0 のときのみ使用）
  ComputePipeline kFoamAdvect_;   // 泡移流・分類パス（pbf_foam_advect.comp; 同上）
  ComputePipeline kLifetime_;     // 寿命パス（fluid_lifetime.comp; lifetimeEnabled_ のときのみ使用）

  // ── kinematic staging (TC8) ──────────────────────────────────────────────
  static constexpr uint32_t MAX_CONCURRENT_FRAMES       = 2;
  VkBuffer kinStagingBuf_[MAX_CONCURRENT_FRAMES]        = {};
  VmaAllocation kinStagingAlloc_[MAX_CONCURRENT_FRAMES] = {};
  void* kinStagingMapped_[MAX_CONCURRENT_FRAMES]        = {};
  uint32_t kinStagingMaxCount_                          = 0;

  std::vector<glm::vec3> boundaryTriVerts_;

  std::vector<std::shared_ptr<Emitter>> emitters_;
  std::vector<int> emitterStepsDone_;
  uint32_t nFluid_ = 0;
  std::mt19937 emitterRng_{12345};

  // スロット再利用: 死亡予定時刻(sample_lifetime既知)をCPUで持ちreadback無しで寿命切れの穴を新規放出で埋め、バッファを有界化する。
  float simTime_ = 0.0f;             // 累積シミュレーション時刻 [s] (emitFromEmitters で dt 加算)
  std::vector<float> slotDeath_;     // fluidスロットの死亡予定sim時刻 (無限寿命=+inf)
  std::vector<uint8_t> slotAlive_;   // 1=生存(再利用不可) 0=空き(再利用可)
  std::vector<uint32_t> freeSlots_;  // 再利用可能な空きスロットindex
  void reclaimDeadSlots_();          // slotDeath_<=simTime_ の生存スロットを空きへ回収する

  void computeBarrier(VkCommandBuffer cmd);

  // バッファindex類はinit()/setAbsorbers()で一度だけ設定され、step()内で毎substep書き換わる値のみ更新される
  SimPC pc_{};
};
