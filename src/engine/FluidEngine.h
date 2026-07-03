#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "../core/source.h"
#include "AttributeBuffer.h"
#include "ComputePipeline.h"
#include "SimPC.h"

// h = cellSize = world_size/grid_res, 粒子間隔 d = world_size/fluid_nx
// 推奨: h >= 2d → grid_res <= world_size / (2 * world_size/fluid_nx) = fluid_nx/2
struct FluidConfig {
  uint32_t fluid_nx     = 192;
  uint32_t fluid_ny     = 3;
  uint32_t fluid_nz     = 192;
  float world_size      = 20.0f;
  uint32_t grid_res     = 64;
  uint32_t max_boundary = 50000;

  uint32_t fluidCount() const { return fluid_nx * fluid_ny * fluid_nz; }
  uint32_t totalCells() const { return grid_res * grid_res * grid_res; }
  float cellSize() const { return world_size / float(grid_res); }
  float particleSpacing() const { return world_size / float(fluid_nx); }
  uint32_t nTotalMax() const { return fluidCount() + max_boundary; }

  // 粒子間距離 d と平滑化長 h から静止密度 ρ₀ を数値計算
  // 均一格子上での Poly6 カーネル和 = シェーダー側の pbf_density と一致する値
  float computeRestDensity() const {
    const float h     = cellSize();
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

class FluidEngine {
public:
  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, const FluidConfig& cfg = {});
  void cleanup();

  void step(VkCommandBuffer cmd, float dt);
  void resetParticles(); // 粒子を初期位置・速度にリセット（バッファ/パイプラインは再生成しない）

  void addSource(std::shared_ptr<Source> src);
  void clearSources();
  uint32_t nFluid() const { return nFluid_; }

  void loadBoundary(const std::string& objPath, float spacing);
  void loadBoundary(const std::string& objPath, float spacing, float scale, glm::vec3 offset, bool yup_to_zup);
  // OBJ を経由せず位置ベクタから直接境界粒子を登録する (TC8 スクリュー用)
  void loadBoundaryParticles(const std::vector<glm::vec4>& pts);
  void clearBoundary();

  const std::vector<glm::vec3>& getBoundaryTriVerts() const { return boundaryTriVerts_; }
  const FluidConfig& config() const { return cfg_; }

  // ImGui から調整可能なパラメータ
  float gravity     = -9.8f;
  float restitution = 0.1f;  // ※PBF流体では未使用（衝突は位置投影のみ）
  float friction    = 0.05f; // ※PBF流体では未使用
  float rho0        = 35.0f;
  float viscosityC  = 0.01f;
  int pbfIterations = 2;
  int numSubsteps   = 2;

  // ── PBF 論文準拠の追加パラメータ ──────────────────────────────────────────
  // ε=3000 / damping=0.6 は元のハードコード値。論文忠実化(ε↓・人工圧力有効化)は
  // 発散したため一旦この既定に戻している。チューニングは ImGui または CLI 引数で行う。
  float cfmEpsilon       = 3000.0f; // CFM 緩和 ε (式11)。元のハードコード値
  float scorrK           = 0.001f;    // 人工圧力 k (式13; 0=無効)
  float vorticityEpsilon = 0.1f;    // 渦度閉じ込め ε (式16)
  float linearDamping    = 0.02f;   // 速度減衰 [1/s]。元のハードコード値
  bool vorticityEnabled  = false;   // 渦度閉じ込めの ON/OFF
  // IPBF (Implicit Position-Based Fluids, SIGGRAPH Asia 2025) 風の under-relaxation。
  // pbf_delta_p.comp の ΔP に乗じる固定緩和係数 (1.0=無効、標準PBF互換)。
  // これにより cfmEpsilon をより小さく (拘束をより硬く) しても、同じ pbfIterations の
  // まま発散しにくくなり、過圧縮 (excessive compression) を抑えられる。
  // 検証: rho0=35 (fluid_pbf既定と同スケール) で cfmEpsilon=300 + relaxOmega=0.7 の
  // 組み合わせは、平均密度誤差を既定 (ε=3000, ω=1.0) 比で数十%改善しつつ NaN/発散なし
  // (headless実験、tests/helpers/PBFHarness 経由)。ただし過圧縮シナリオでは初期配置に
  // 対して結果がやや敏感 (GPU側の総和順序に起因すると見られる非決定性あり) なため、
  // 既定値は安全側の 1.0 のまま据え置き、fluid_pbf の --relax-omega で個別に検証・
  // チューニングすることを推奨する。
  float relaxOmega       = 1.0f;

  // ── 煙・粉体パラメータ ──────────────────────────────────────────────────────
  float smokeRiseAccel = 8.0f;  // 煙の浮力加速度 [m/s²] (typeFlag==4)
  float smokeDamping   = 0.5f;  // 煙の速度減衰係数 [1/s] (typeFlag==4)
  float powderFriction = 0.0f;  // 粉体摩擦係数 [1/s] (typeFlag==5; 将来拡張用)

  uint32_t nBoundary = 0;

  // ── 吸収形状ディスクリプタ ───────────────────────────────────────────────────
  // type: 0=Sphere, 1=CylinderZ, 2=Box, 3=CapsuleZ
  // 各フィールドは float (type も float として格納; シェーダーで uint() キャスト)
  struct AbsorberDesc {
    float type;        // 0=Sphere, 1=CylinderZ, 2=Box, 3=CapsuleZ
    float cx, cy, cz;  // 中心座標 [m]
    float p0, p1, p2;  // 形状パラメータ: Sphere→(r), CylZ→(r,halfH), Box→(hx,hy,hz), Capsule→(r,halfL)
    float rate;        // 吸収確率 per substep [0.0, 1.0]

    static AbsorberDesc Sphere(float cx_, float cy_, float cz_, float r, float rate_ = 1.0f) {
      return {0.0f, cx_, cy_, cz_, r, 0.0f, 0.0f, rate_};
    }
    static AbsorberDesc CylinderZ(float cx_, float cy_, float cz_, float r, float halfH, float rate_ = 1.0f) {
      return {1.0f, cx_, cy_, cz_, r, halfH, 0.0f, rate_};
    }
    static AbsorberDesc Box(float cx_, float cy_, float cz_, float hx, float hy, float hz, float rate_ = 1.0f) {
      return {2.0f, cx_, cy_, cz_, hx, hy, hz, rate_};
    }
    static AbsorberDesc CapsuleZ(float cx_, float cy_, float cz_, float r, float halfL, float rate_ = 1.0f) {
      return {3.0f, cx_, cy_, cz_, r, halfL, 0.0f, rate_};
    }
  };
  static constexpr uint32_t MAX_ABSORBERS = 32;

  // 吸収形状を登録（毎フレーム step() の前に呼ぶ; absorbers が空なら吸収パスをスキップ）
  void setAbsorbers(const std::vector<AbsorberDesc>& absorbers);

  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet             = VK_NULL_HANDLE;
  uint32_t posIdx                           = 0;
  uint32_t velIdx                           = 0;
  // 外部シェーダー（fluid_absorb.comp 等）からも参照されるバッファインデックス
  uint32_t predPIdx    = 0;
  uint32_t invMassIdx  = 0;
  uint32_t typeFlagIdx = 0;

  VkBuffer getPositionBuffer() const;

  // ── TC8: 運動学的境界粒子 (回転スクリュー等) の per-frame GPU 更新 ────────────
  // maxBoundaryCount: 毎フレーム更新する境界粒子の最大数
  void initKinematicBoundaryStaging(uint32_t maxBoundaryCount);
  // cmd の中に vkCmdCopyBuffer × 3 + barrier を記録する。engine_.step() の前に呼ぶ。
  void recordKinematicBoundaryUpdate(
      VkCommandBuffer cmd, uint32_t frameIndex,
      uint32_t boundaryOffset, uint32_t count,
      const glm::vec4* positions,   // world-space 位置
      const glm::vec4* velocities); // ω × (pos - pivot)
  void cleanupKinematicBoundaryStaging();

private:
  FluidConfig cfg_;
  VkDevice device_        = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkCommandPool cmdPool_  = VK_NULL_HANDLE;
  VkQueue queue_          = VK_NULL_HANDLE;

  AttributeBuffer attrBuf_;

  uint32_t cellCountIdx_  = 0;
  uint32_t cellOffsetIdx_ = 0;
  uint32_t sortedIdxIdx_  = 0;
  uint32_t densityIdx_    = 0;
  uint32_t lambdaPbfIdx_  = 0;
  uint32_t omegaIdx_      = 0; // 渦度 ω バッファ (vec4 × N)

  // 吸収パス用プライベートメンバー
  uint32_t absorberBufIdx_ = 0; // absorbers バッファの bindless index
  uint32_t absorberCount_  = 0; // 現フレームの有効吸収形状数

  ComputePipeline kPredictSdf_;
  ComputePipeline kSdfCollision_;
  ComputePipeline kHashCount_;
  ComputePipeline kHashScanLocal_;
  ComputePipeline kHashScanGlobal_;
  ComputePipeline kHashSort_;
  ComputePipeline kPbfDensity_;
  ComputePipeline kPbfDeltaP_;
  ComputePipeline kPbfViscosity_;
  ComputePipeline kUpdateVelocity_;
  ComputePipeline kZeroCells_;
  ComputePipeline kHashAddBase_;
  ComputePipeline kVorticityOmega_;
  ComputePipeline kVorticityForce_;
  ComputePipeline kAbsorb_; // 吸収パス（fluid_absorb.comp; absorberCount_>0 のときのみ使用）

  // ── kinematic staging (TC8) ──────────────────────────────────────────────
  static constexpr uint32_t MAX_CONCURRENT_FRAMES = 2;
  VkBuffer      kinStagingBuf_[MAX_CONCURRENT_FRAMES]    = {};
  VmaAllocation kinStagingAlloc_[MAX_CONCURRENT_FRAMES]  = {};
  void*         kinStagingMapped_[MAX_CONCURRENT_FRAMES] = {};
  uint32_t      kinStagingMaxCount_                      = 0;

  std::vector<glm::vec3> boundaryTriVerts_;

  std::vector<std::shared_ptr<Source>> sources_;
  std::vector<int> sourceStepsDone_;
  uint32_t nFluid_ = 0;
  std::mt19937 sourceRng_{12345};

  void emitSources(float dt);
  void computeBarrier(VkCommandBuffer cmd);
};
