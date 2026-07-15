#include "MPMHarness.h"
#include "ForceShaderCompiler.h"
#include "MaterialParams.h"
#include <cstring>
#include <glm/glm.hpp>
#include <vector>

// ── Morton符号 (mpm_common.glslと同一ロジック) ───────────────────────────────

uint32_t mpmMortonExpand(uint32_t v) {
  v &= 0x000003ffu;
  v = (v ^ (v << 16u)) & 0xff0000ffu;
  v = (v ^ (v << 8u)) & 0x0300f00fu;
  v = (v ^ (v << 4u)) & 0x030c30c3u;
  v = (v ^ (v << 2u)) & 0x09249249u;
  return v;
}

uint32_t mpmMortonEncode(uint32_t x, uint32_t y, uint32_t z) { return mpmMortonExpand(x) | (mpmMortonExpand(y) << 1u) | (mpmMortonExpand(z) << 2u); }

// ── 内部ヘルパー ──────────────────────────────────────────────────────────────

void MPMHarness::barrier(VkCommandBuffer cmd) {
  VkMemoryBarrier b{};
  b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void MPMHarness::dispatchMPM(VkCommandBuffer cmd, ComputePipeline& k, const MPMSimPC& pc, uint32_t count) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipelineLayout, 0, 1, &attrBuf_.descriptorSet, 0, nullptr);
  vkCmdPushConstants(cmd, k.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MPMSimPC), &pc);
  uint32_t groups = (count + 255u) / 256u;
  vkCmdDispatch(cmd, groups, 1, 1);
}

// ── 初期化 ────────────────────────────────────────────────────────────────────

void MPMHarness::init(const HeadlessCtx& ctx, uint32_t gridRes, float worldSize, const std::string& shaderDir, uint32_t maxParticles) {
  ctx_          = &ctx;
  gridRes_      = gridRes;
  worldSize_    = worldSize;
  maxParticles_ = maxParticles;

  const uint32_t NC = totalCells();
  const uint32_t NG = nGroups();
  const uint32_t N  = maxParticles;

  attrBuf_.init(ctx.device, ctx.allocator, ctx.descriptorPool);

  // F0-2: xyz = F列, w = 対角応力 (σ_xx, σ_yy, σ_zz)
  // B0-2: xyz = APIC B列 (Phase 2), w = 非対角応力 (σ_xy, σ_xz, σ_yz)
  posIdx_     = attrBuf_.addAttribute("P", sizeof(glm::vec4), N);
  velIdx_     = attrBuf_.addAttribute("v", sizeof(glm::vec4), N);
  F0Idx_      = attrBuf_.addAttribute("F0", sizeof(glm::vec4), N);
  F1Idx_      = attrBuf_.addAttribute("F1", sizeof(glm::vec4), N);
  F2Idx_      = attrBuf_.addAttribute("F2", sizeof(glm::vec4), N);
  B0Idx_      = attrBuf_.addAttribute("B0", sizeof(glm::vec4), N);
  B1Idx_      = attrBuf_.addAttribute("B1", sizeof(glm::vec4), N);
  B2Idx_      = attrBuf_.addAttribute("B2", sizeof(glm::vec4), N);
  cellCntIdx_ = attrBuf_.addAttribute("cellCnt", sizeof(uint32_t), NC);
  cellOffIdx_ = attrBuf_.addAttribute("cellOff", sizeof(uint32_t), NC + NG);
  sortedIdx_  = attrBuf_.addAttribute("sorted", sizeof(uint32_t), N);
  // gridMom: 2×NC — [0,NC) = v_new, [NC,2*NC) = v_old (FLIP 用)
  gridMomIdx_  = attrBuf_.addAttribute("gridMom", sizeof(glm::vec4), NC * 2);
  gridMassIdx_ = attrBuf_.addAttribute("gridMass", sizeof(float), NC);
  // マテリアルテーブル: 最大 16 エントリ (Phase 1 以降)
  materialsIdx_ = attrBuf_.addAttribute("materials", sizeof(MaterialParams), 16);

  // Force (issue #30): makePC() の gravity 引数互換の GravityForce を1個登録する
  forcesIdx_     = attrBuf_.addAttribute("forces", sizeof(ForceGPU), 1);
  legacyGravity_ = GravityForce::FromDirection({0.0f, 1.0f, 0.0f}, 0.0f); // Y-up; strengthはmakePC()で都度更新

  auto load = [&](ComputePipeline& k, const std::string& name) { k.init(ctx.device, attrBuf_.descriptorSetLayout, shaderDir + "/" + name + ".spv"); };
  load(kZeroGrid_, "mpm_zero_grid.comp");
  load(kZeroCells_, "zero_cells.comp");
  load(kHashCount_, "mpm_hash_count.comp");
  load(kScanLocal_, "hash_scan_local.comp");
  load(kScanGlobal_, "hash_scan_global.comp");
  load(kHashAddBase_, "hash_add_base.comp");
  load(kHashSort_, "mpm_hash_sort.comp");
  load(kP2G_, "mpm_p2g.comp");
  load(kG2P_, "mpm_g2p.comp");
  {
    std::vector<uint32_t> spirv = ForceShaderCompiler::compile({legacyGravity_}, "mpm_grid_update.comp");
    kGridUpdate_.initFromSpirv(ctx.device, attrBuf_.descriptorSetLayout, spirv);
  }
}

// ── 粒子データアップロード ─────────────────────────────────────────────────────
// 応力 τ は w レーンにパック:
//   F0.w = τ[0][0] (σ_xx)
//   F1.w = τ[1][1] (σ_yy)
//   F2.w = τ[2][2] (σ_zz)
//   B0.w = τ[0][1] (σ_xy)
//   B1.w = τ[0][2] (σ_xz)
//   B2.w = τ[1][2] (σ_yz)

void MPMHarness::uploadParticles(const std::vector<Particle>& particles) {
  const uint32_t N = static_cast<uint32_t>(particles.size());
  auto pool        = ctx_->commandPool;
  auto queue       = ctx_->computeQueue;

  std::vector<glm::vec4> pos(N), vel(N), F0(N), F1(N), F2(N), B0(N), B1(N), B2(N);
  for(uint32_t i = 0; i < N; ++i) {
    const auto& p = particles[i];
    pos[i]        = glm::vec4(p.pos, p.Vp);
    vel[i]        = glm::vec4(p.vel, 0.0f); // w = material id (初期は 0)
    // F 列 xyz + 対角応力 w
    F0[i] = glm::vec4(p.F[0], p.tau[0][0]); // w = σ_xx
    F1[i] = glm::vec4(p.F[1], p.tau[1][1]); // w = σ_yy
    F2[i] = glm::vec4(p.F[2], p.tau[2][2]); // w = σ_zz
    // APIC B 列 xyz (初期=0) + 非対角応力 w
    B0[i] = glm::vec4(0.0f, 0.0f, 0.0f, p.tau[0][1]); // w = σ_xy
    B1[i] = glm::vec4(0.0f, 0.0f, 0.0f, p.tau[0][2]); // w = σ_xz
    B2[i] = glm::vec4(0.0f, 0.0f, 0.0f, p.tau[1][2]); // w = σ_yz
  }
  attrBuf_.upload("P", pos.data(), N * sizeof(glm::vec4), pool, queue);
  attrBuf_.upload("v", vel.data(), N * sizeof(glm::vec4), pool, queue);
  attrBuf_.upload("F0", F0.data(), N * sizeof(glm::vec4), pool, queue);
  attrBuf_.upload("F1", F1.data(), N * sizeof(glm::vec4), pool, queue);
  attrBuf_.upload("F2", F2.data(), N * sizeof(glm::vec4), pool, queue);
  attrBuf_.upload("B0", B0.data(), N * sizeof(glm::vec4), pool, queue);
  attrBuf_.upload("B1", B1.data(), N * sizeof(glm::vec4), pool, queue);
  attrBuf_.upload("B2", B2.data(), N * sizeof(glm::vec4), pool, queue);
}

// ── CPU側ハッシュグリッド構築 ─────────────────────────────────────────────────

void MPMHarness::buildHashGridCPU(const std::vector<Particle>& particles) {
  const uint32_t N  = static_cast<uint32_t>(particles.size());
  const uint32_t NC = totalCells();
  const uint32_t NG = nGroups();
  const float h     = cellSize();

  std::vector<uint32_t> cellCnt(NC, 0u);
  std::vector<uint32_t> cellOff(NC + NG, 0u);
  std::vector<uint32_t> sorted(N, 0u);

  // 各粒子のMorton cellを計算
  std::vector<uint32_t> particleCellMorton(N);
  for(uint32_t i = 0; i < N; ++i) {
    glm::vec3 local       = particles[i].pos / h;
    local                 = glm::clamp(local, glm::vec3(0.0f), glm::vec3(float(gridRes_ - 1)));
    uint32_t gx           = uint32_t(local.x);
    uint32_t gy           = uint32_t(local.y);
    uint32_t gz           = uint32_t(local.z);
    uint32_t morton       = mpmMortonEncode(gx, gy, gz);
    particleCellMorton[i] = morton;
    cellCnt[morton]++;
  }

  // P2Gシェーダー仕様: start = cellOff[cid] - cellCnt[cid]
  // → cellOff[c] = 範囲の末尾インデックス+1 (end-exclusive scan)
  uint32_t acc = 0;
  for(uint32_t c = 0; c < NC; ++c) {
    acc += cellCnt[c];
    cellOff[c] = acc;
  }

  // sorted配列を各cellのstart位置から書き込む
  std::vector<uint32_t> writePtr(NC);
  for(uint32_t c = 0; c < NC; ++c) {
    writePtr[c] = cellOff[c] - cellCnt[c];
  }
  for(uint32_t i = 0; i < N; ++i) {
    uint32_t m            = particleCellMorton[i];
    sorted[writePtr[m]++] = i;
  }

  auto pool  = ctx_->commandPool;
  auto queue = ctx_->computeQueue;
  attrBuf_.upload("cellCnt", cellCnt.data(), NC * sizeof(uint32_t), pool, queue);
  attrBuf_.upload("cellOff", cellOff.data(), (NC + NG) * sizeof(uint32_t), pool, queue);
  attrBuf_.upload("sorted", sorted.data(), N * sizeof(uint32_t), pool, queue);
}

// ── グリッドデータ直接アップロード ────────────────────────────────────────────

void MPMHarness::uploadGrid(const std::vector<glm::vec4>& gridMom, const std::vector<float>& gridMass) {
  const uint32_t NC = totalCells();
  auto pool         = ctx_->commandPool;
  auto queue        = ctx_->computeQueue;
  attrBuf_.upload("gridMom", gridMom.data(), NC * sizeof(glm::vec4), pool, queue);
  attrBuf_.upload("gridMass", gridMass.data(), NC * sizeof(float), pool, queue);
}

// ── Push Constants 構築 ───────────────────────────────────────────────────────

MPMSimPC MPMHarness::makePC(float dt, float rho0, float mu, float lam, float gravity) {
  // デフォルト弾性マテリアル (slot 0) を GPU にアップロード
  // 既存テストが makePC パラメータで材料を指定するパターンに対応するため
  MaterialParams defMat{};
  defMat.mu     = mu;
  defMat.lambda = lam;
  defMat.rho0   = rho0;
  defMat.model  = uint32_t(MaterialModel::ELASTIC);
  defMat.q_max  = 1e5f;
  attrBuf_.upload("materials", &defMat, sizeof(MaterialParams), ctx_->commandPool, ctx_->computeQueue);

  // Force (issue #30): gravity 引数を GravityForce として都度アップロード
  legacyGravity_->strength = gravity;
  ForceGPU forceGpu        = legacyGravity_->pack();
  attrBuf_.upload("forces", &forceGpu, sizeof(ForceGPU), ctx_->commandPool, ctx_->computeQueue);

  MPMSimPC pc{};
  pc.posIdx           = posIdx_;
  pc.velIdx           = velIdx_;
  pc.F0Idx            = F0Idx_;
  pc.F1Idx            = F1Idx_;
  pc.typeFlagIdx      = 0;
  pc.cellCountIdx     = cellCntIdx_;
  pc.cellOffsetIdx    = cellOffIdx_;
  pc.sortedIdxIdx     = sortedIdx_;
  pc.particleCount    = maxParticles_;
  pc.gridRes          = gridRes_;
  pc.F2Idx            = F2Idx_;
  pc.materialsIdx     = materialsIdx_;
  pc.dt               = dt;
  pc.cellSize         = cellSize();
  pc.worldMin         = 0.0f;
  pc.worldMax         = worldSize_;
  pc.forceBufIdx      = forcesIdx_;
  pc.mu_lame          = mu;
  pc.lambda_lame      = lam;
  pc.particleVolume   = cellSize() * cellSize() * cellSize();
  pc.M_friction       = 0.577f;
  pc.q_cohesion       = 0.0f;
  pc.q_max            = 1e5f;
  pc.flip_ratio       = 0.0f; // PICモード
  pc.colliderIdx      = 0;    // Phase 3 まで未使用
  pc.colliderCount    = 0;
  pc.B0Idx            = B0Idx_;
  pc.B1Idx            = B1Idx_;
  pc.B2Idx            = B2Idx_;
  pc.nanoVDBIdx       = 0; // 無効
  pc.gridMomIdx       = gridMomIdx_;
  pc.gridMassIdx      = gridMassIdx_;
  pc.restitution      = 0.3f;
  pc.wall_friction    = 0.0f;
  pc.plasticModel     = 0;
  pc.materialCount    = 1; // デフォルト弾性マテリアル 1 エントリ
  pc.rho0             = rho0;
  pc.p0_mcc           = 0.0f;
  pc.xi_hard          = 0.0f;
  pc.forceCount       = 1;
  return pc;
}

void MPMHarness::uploadMaterials(const std::vector<MaterialParams>& mats) {
  if(mats.empty()) return;
  attrBuf_.upload("materials", mats.data(), mats.size() * sizeof(MaterialParams), ctx_->commandPool, ctx_->computeQueue);
}

// ── 個別パス実行 ──────────────────────────────────────────────────────────────

void MPMHarness::runZeroGrid(const MPMSimPC& pc) {
  VkCommandBuffer cmd = ctx_->beginCmd();
  dispatchMPM(cmd, kZeroGrid_, pc, totalCells());
  ctx_->submitCmd(cmd);
}

void MPMHarness::runP2G(const MPMSimPC& pc) {
  VkCommandBuffer cmd = ctx_->beginCmd();
  dispatchMPM(cmd, kP2G_, pc, totalCells());
  ctx_->submitCmd(cmd);
}

void MPMHarness::runGridUpdate(const MPMSimPC& pc) {
  VkCommandBuffer cmd = ctx_->beginCmd();
  dispatchMPM(cmd, kGridUpdate_, pc, totalCells());
  ctx_->submitCmd(cmd);
}

void MPMHarness::runG2P(const MPMSimPC& pc) {
  VkCommandBuffer cmd = ctx_->beginCmd();
  dispatchMPM(cmd, kG2P_, pc, maxParticles_);
  ctx_->submitCmd(cmd);
}

void MPMHarness::runFullStep(const MPMSimPC& pc) {
  const uint32_t N  = maxParticles_;
  const uint32_t NC = totalCells();
  const uint32_t NG = nGroups();

  VkCommandBuffer cmd = ctx_->beginCmd();

  // ① グリッドをゼロクリア
  dispatchMPM(cmd, kZeroGrid_, pc, NC);
  barrier(cmd);

  // ② cellCountをゼロクリア
  dispatchMPM(cmd, kZeroCells_, pc, NC);
  barrier(cmd);

  // ③ ハッシュグリッド構築
  dispatchMPM(cmd, kHashCount_, pc, N);
  barrier(cmd);

  // hash_scan_local: NG workgroups
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kScanLocal_.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kScanLocal_.pipelineLayout, 0, 1, &attrBuf_.descriptorSet, 0, nullptr);
  vkCmdPushConstants(cmd, kScanLocal_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MPMSimPC), &pc);
  vkCmdDispatch(cmd, NG, 1, 1);
  barrier(cmd);

  // hash_scan_global: 1 workgroup
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kScanGlobal_.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kScanGlobal_.pipelineLayout, 0, 1, &attrBuf_.descriptorSet, 0, nullptr);
  vkCmdPushConstants(cmd, kScanGlobal_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MPMSimPC), &pc);
  vkCmdDispatch(cmd, 1, 1, 1);
  barrier(cmd);

  // hash_add_base: NG workgroups
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashAddBase_.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashAddBase_.pipelineLayout, 0, 1, &attrBuf_.descriptorSet, 0, nullptr);
  vkCmdPushConstants(cmd, kHashAddBase_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MPMSimPC), &pc);
  vkCmdDispatch(cmd, NG, 1, 1);
  barrier(cmd);

  dispatchMPM(cmd, kHashSort_, pc, N);
  barrier(cmd);

  // ④ P2G
  dispatchMPM(cmd, kP2G_, pc, NC);
  barrier(cmd);

  // ⑤ GridUpdate
  dispatchMPM(cmd, kGridUpdate_, pc, NC);
  barrier(cmd);

  // ⑥ G2P (NanoVDB BCはスキップ)
  dispatchMPM(cmd, kG2P_, pc, N);

  ctx_->submitCmd(cmd);
}

// ── 結果の読み取り ────────────────────────────────────────────────────────────

glm::vec4 MPMHarness::readParticlePos(uint32_t i) const {
  glm::vec4 r{};
  ctx_->readBuffer(attrBuf_.getBuffer("P"), i * sizeof(glm::vec4), &r, sizeof(r));
  return r;
}

glm::vec4 MPMHarness::readParticleVel(uint32_t i) const {
  glm::vec4 r{};
  ctx_->readBuffer(attrBuf_.getBuffer("v"), i * sizeof(glm::vec4), &r, sizeof(r));
  return r;
}

float MPMHarness::readGridMass(uint32_t mortonIdx) const {
  float r = 0.0f;
  ctx_->readBuffer(attrBuf_.getBuffer("gridMass"), mortonIdx * sizeof(float), &r, sizeof(r));
  return r;
}

glm::vec3 MPMHarness::readGridVel(uint32_t mortonIdx) const {
  glm::vec4 r{};
  ctx_->readBuffer(attrBuf_.getBuffer("gridMom"), mortonIdx * sizeof(glm::vec4), &r, sizeof(r));
  return glm::vec3(r);
}

glm::mat3 MPMHarness::readParticleF(uint32_t i) const {
  glm::vec4 c0{}, c1{}, c2{};
  ctx_->readBuffer(attrBuf_.getBuffer("F0"), i * sizeof(glm::vec4), &c0, sizeof(c0));
  ctx_->readBuffer(attrBuf_.getBuffer("F1"), i * sizeof(glm::vec4), &c1, sizeof(c1));
  ctx_->readBuffer(attrBuf_.getBuffer("F2"), i * sizeof(glm::vec4), &c2, sizeof(c2));
  return glm::mat3(glm::vec3(c0), glm::vec3(c1), glm::vec3(c2));
}

glm::mat3 MPMHarness::readParticleStress(uint32_t i) const {
  // w レーンから対称 Kirchhoff 応力を再構成
  // F0.w=σ_xx, F1.w=σ_yy, F2.w=σ_zz, B0.w=σ_xy, B1.w=σ_xz, B2.w=σ_yz
  glm::vec4 f0{}, f1{}, f2{}, b0{}, b1{}, b2{};
  ctx_->readBuffer(attrBuf_.getBuffer("F0"), i * sizeof(glm::vec4), &f0, sizeof(f0));
  ctx_->readBuffer(attrBuf_.getBuffer("F1"), i * sizeof(glm::vec4), &f1, sizeof(f1));
  ctx_->readBuffer(attrBuf_.getBuffer("F2"), i * sizeof(glm::vec4), &f2, sizeof(f2));
  ctx_->readBuffer(attrBuf_.getBuffer("B0"), i * sizeof(glm::vec4), &b0, sizeof(b0));
  ctx_->readBuffer(attrBuf_.getBuffer("B1"), i * sizeof(glm::vec4), &b1, sizeof(b1));
  ctx_->readBuffer(attrBuf_.getBuffer("B2"), i * sizeof(glm::vec4), &b2, sizeof(b2));
  float txx = f0.w, tyy = f1.w, tzz = f2.w;
  float txy = b0.w, txz = b1.w, tyz = b2.w;
  // 対称 mat3: col0=(σ_xx, σ_xy, σ_xz), col1=(σ_xy, σ_yy, σ_yz), col2=(σ_xz, σ_yz, σ_zz)
  return glm::mat3(glm::vec3(txx, txy, txz),  // col0
                   glm::vec3(txy, tyy, tyz),  // col1
                   glm::vec3(txz, tyz, tzz)); // col2
}

float MPMHarness::sumGridMass() const {
  const uint32_t NC = totalCells();
  std::vector<float> buf(NC);
  ctx_->readBuffer(attrBuf_.getBuffer("gridMass"), 0, buf.data(), NC * sizeof(float));
  float s = 0.0f;
  for(float v : buf) s += v;
  return s;
}

glm::vec3 MPMHarness::sumGridMom() const {
  const uint32_t NC = totalCells();
  std::vector<glm::vec4> buf(NC);
  ctx_->readBuffer(attrBuf_.getBuffer("gridMom"), 0, buf.data(), NC * sizeof(glm::vec4));
  glm::vec3 s(0.0f);
  for(auto& v : buf) s += glm::vec3(v);
  return s;
}

// ── クリーンアップ ────────────────────────────────────────────────────────────

void MPMHarness::cleanup() {
  for(auto* k : {&kZeroGrid_, &kZeroCells_, &kHashCount_, &kScanLocal_, &kScanGlobal_, &kHashAddBase_, &kHashSort_, &kP2G_, &kGridUpdate_, &kG2P_}) k->cleanup();
  attrBuf_.cleanup();
}
