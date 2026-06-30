#pragma once
#include "HeadlessCtx.h"
#include "AttributeBuffer.h"
#include "ComputePipeline.h"
#include "MPMSimPC.h"
#include "MaterialParams.h"
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <cstdint>

// CPU側Mortonエンコード (mpm_common.glslと同一ロジック)
uint32_t mpmMortonExpand(uint32_t v);
uint32_t mpmMortonEncode(uint32_t x, uint32_t y, uint32_t z);

// MPMシェーダー単体テスト用ハーネス
// PhysicsHarnessと同様にHeadlessCtx上で動作する
class MPMHarness {
public:
    struct Particle {
        glm::vec3 pos   = {5.0f, 5.0f, 5.0f};
        float     Vp    = 1.0f;
        glm::vec3 vel   = {0.0f, 0.0f, 0.0f};
        glm::mat3 F     = glm::mat3(1.0f);   // 変形勾配 (初期=単位行列)
        glm::mat3 tau   = glm::mat3(0.0f);   // Kirchhoff応力 (初期=零)
    };

    // gridRes×gridRes×gridRes のMorton符号グリッド
    // worldSize×worldSize×worldSize [m] のワールド
    void init(const HeadlessCtx& ctx,
              uint32_t gridRes,
              float worldSize,
              const std::string& shaderDir,
              uint32_t maxParticles = 64);

    // 粒子データをGPUへアップロード
    // 応力は F/B バッファの w レーンにパックされる（Phase 0 以降）
    void uploadParticles(const std::vector<Particle>& particles);

    // CPU側でハッシュグリッドを手動構築してアップロード (P2Gテスト用)
    // P2Gシェーダーが使う cellCount/cellOffset/sortedIdx を設定する
    void buildHashGridCPU(const std::vector<Particle>& particles);

    // グリッドMom/Massを直接アップロード (GridUpdate/G2Pテスト用)
    // gridMom[i] = (velocity_x, velocity_y, velocity_z, 0)
    // gridMass[i] = mass
    void uploadGrid(const std::vector<glm::vec4>& gridMom,
                    const std::vector<float>& gridMass);

    // Push Constants を構築するヘルパー
    // 同時にデフォルト弾性マテリアル (slot 0) を GPU にアップロードする
    MPMSimPC makePC(float dt,
                    float rho0    = 1000.0f,
                    float mu      = 5000.0f,
                    float lam     = 11538.0f,
                    float gravity = 0.0f);

    // マテリアルテーブルを GPU にアップロードする (Phase 1 以降)
    // uploadParticles() の vel.w に floatBitsToUint(matIdx) をセットして使う
    void uploadMaterials(const std::vector<MaterialParams>& mats);

    // 個別パス実行 (各関数内でbeginCmd/submitCmdして同期)
    void runZeroGrid(const MPMSimPC& pc);
    void runP2G(const MPMSimPC& pc);
    void runGridUpdate(const MPMSimPC& pc);
    void runG2P(const MPMSimPC& pc);

    // フルステップ実行 (ZeroGrid + ハッシュビルド + P2G + GridUpdate + G2P)
    // NanoVDB BCは含まない (nanoVDBIdx=0でスキップ)
    void runFullStep(const MPMSimPC& pc);

    // 結果の読み取り
    glm::vec4 readParticlePos(uint32_t i) const;
    glm::vec4 readParticleVel(uint32_t i) const;
    float     readGridMass(uint32_t mortonIdx) const;
    glm::vec3 readGridVel(uint32_t mortonIdx) const;  // GridUpdate後の速度を読む
    glm::mat3 readParticleF(uint32_t i) const;        // G2P後の変形勾配F
    glm::mat3 readParticleStress(uint32_t i) const;   // G2P後のKirchhoff応力τ（w レーンから再構成）

    // グリッド全体の合計を計算 (P2G保存則テスト用)
    float     sumGridMass() const;
    glm::vec3 sumGridMom() const;

    void cleanup();

    uint32_t gridRes()     const { return gridRes_; }
    uint32_t totalCells()  const { return gridRes_ * gridRes_ * gridRes_; }
    uint32_t nGroups()     const { return (totalCells() + 255u) / 256u; }
    float    cellSize()    const { return worldSize_ / float(gridRes_); }
    float    worldSize()   const { return worldSize_; }

private:
    const HeadlessCtx* ctx_         = nullptr;
    uint32_t           gridRes_     = 8;
    float              worldSize_   = 10.0f;
    uint32_t           maxParticles_= 64;

    AttributeBuffer attrBuf_;

    // バッファインデックス (Bindless)
    // F0-2: xyz = F 列, w = 対角応力 σ_xx/σ_yy/σ_zz
    // B0-2: xyz = APIC B 列, w = 非対角応力 σ_xy/σ_xz/σ_yz
    uint32_t posIdx_      = 0;
    uint32_t velIdx_      = 0;
    uint32_t F0Idx_       = 0;
    uint32_t F1Idx_       = 0;
    uint32_t F2Idx_       = 0;
    uint32_t B0Idx_       = 0;
    uint32_t B1Idx_       = 0;
    uint32_t B2Idx_       = 0;
    uint32_t cellCntIdx_  = 0;
    uint32_t cellOffIdx_  = 0;
    uint32_t sortedIdx_   = 0;
    uint32_t gridMomIdx_  = 0;
    uint32_t gridMassIdx_ = 0;
    uint32_t materialsIdx_= 0;

    // コンピュートパイプライン
    ComputePipeline kZeroGrid_;    // mpm_zero_grid.comp
    ComputePipeline kZeroCells_;   // zero_cells.comp (SimPC互換)
    ComputePipeline kHashCount_;   // mpm_hash_count.comp
    ComputePipeline kScanLocal_;   // hash_scan_local.comp (SimPC互換)
    ComputePipeline kScanGlobal_;  // hash_scan_global.comp (SimPC互換)
    ComputePipeline kHashAddBase_; // hash_add_base.comp (SimPC互換)
    ComputePipeline kHashSort_;    // mpm_hash_sort.comp
    ComputePipeline kP2G_;         // mpm_p2g.comp
    ComputePipeline kGridUpdate_;  // mpm_grid_update.comp
    ComputePipeline kG2P_;         // mpm_g2p.comp

    // MPMSimPCを使って直接dispatch (ComputePipeline::dispatchはSimPCを取るため使わない)
    void dispatchMPM(VkCommandBuffer cmd, ComputePipeline& k,
                     const MPMSimPC& pc, uint32_t count);
    void barrier(VkCommandBuffer cmd);
};
