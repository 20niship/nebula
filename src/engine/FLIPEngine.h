#pragma once

#include "ParticleGridEngine.h"

// 非圧縮 FLIP 流体。粒子-格子転写は ParticleGridEngine を共有し、GridUpdate 後に圧力投影
// (flip_classify → flip_divergence → flip_pressure_rb × 反復 → flip_project) で非圧縮性を課す。
// 泡 (spray/foam/bubble) はグリッドの密度・速度勾配から生成するため近傍探索を持たない。
// 粒子体積は Emitter::particleVolume 未指定なら cellSize³ / particlesPerCell。maxParticles の明示指定が必須。
class FLIPEngine : public ParticleGridEngine {
public:
  FLIPEngine() {
    numSubsteps = 4;
    flip_ratio  = 0.95f;
  }

  int pressureIterations  = 6;     // 赤黒 Gauss-Seidel の反復回数 (1 反復 = 赤+黒)
  float densityCorrection = 0.05f; // 過密ノードを膨張させる強さ (0 で無効)。GS が未収束で沈む体積ドリフトの防止。大きいと膨張・振動
  float particlesPerCell  = 8.0f;  // 1 セルあたりの目標粒子数 (既定粒子体積の算出用)
  float maxGridSpeed      = 40.0f; // グリッド速度上限 [m/s] (init() 前に設定)

  uint32_t maxDiffuseParticles = 0; // 泡粒子の上限。0 なら泡パスとバッファを持たない (init() 前に設定)
  bool foamEnabled             = false;

  struct FoamParams {                             // Ihmsen 2012 ポテンシャル関数ベース
    float kTa = 500.0f, kWc = 500.0f;             // 生成係数 (trapped-air / wave-crest) [1/s]
    float taLo = 1.0f, taHi = 6.0f;               // trapped-air (セル間の相対速度 [m/s]) の正規化範囲
    float wcLo = 1.0f, wcHi = 5.0f;               // wave-crest の正規化範囲
    float keLo = 2.0f, keHi = 20.0f;              // kinetic-energy ゲート範囲
    float lifetimeMin = 1.0f, lifetimeMax = 3.0f; // 寿命 [s]
    float bubbleBuoyancy = 4.0f;                  // bubble の浮力加速度 [m/s²]
    float dragCoeff      = 0.4f;                  // foam/bubble が格子速度へ追従する割合 [0,1]
    glm::vec3 gravity{0.0f, 0.0f, -9.8f};         // spray に適用する重力 (Z-up: 他エンジンと同じ)
    float densityLo = 0.05f, densityHi = 0.6f;    // 密度比による分類: spray<Lo<=foam<Hi<=bubble
    float surfaceDensityRatio = 0.8f;             // 生成ゲート: 粒子位置の密度比がこれ未満の粒子のみ対象
  };
  void setFoamParams(const FoamParams& params);

  uint32_t foamPosIdx() const { return foamPosIdx_; }   // vec4: xyz=pos, w=残り寿命
  uint32_t foamVelIdx() const { return foamVelIdx_; }   // vec4: xyz=vel, w=初期寿命
  uint32_t foamKindIdx() const { return foamKindIdx_; } // uint: 0=死/未使用,1=spray,2=foam,3=bubble
  VkBuffer getFoamPositionBuffer() const { return attrBuf_.getBuffer("foamPos"); }
  VkBuffer getFoamVelocityBuffer() const { return attrBuf_.getBuffer("foamVel"); }
  VkBuffer getFoamKindBuffer() const { return attrBuf_.getBuffer("foamKind"); }

  void resetParticles() override;

protected:
  std::vector<std::pair<std::string, std::string>> shaderDefines() const override;
  void onInitParticles(const std::string& shaderDir) override;
  void fillPC(MPMSimPC& pc) const override;
  void projectGrid(VkCommandBuffer cmd, const MPMSimPC& pc) override;
  void afterG2P(VkCommandBuffer cmd, const MPMSimPC& pc) override;
  void cleanupDerived() override;
  float defaultParticleVolume() const override { return cellSize * cellSize * cellSize / particlesPerCell; }

private:
  ComputePipeline kClassify_;
  ComputePipeline kDivergence_;
  ComputePipeline kPressure_;
  ComputePipeline kProject_;
  ComputePipeline kFoamGenerate_;
  ComputePipeline kFoamAdvect_;

  uint32_t foamPosIdx_    = 0;
  uint32_t foamVelIdx_    = 0;
  uint32_t foamKindIdx_   = 0;
  uint32_t foamParamsIdx_ = 0;
  uint32_t foamSalt_      = 0; // 泡生成の乱数ソルト (サブステップごとに進める)
  FoamParams foamParams_;
};
