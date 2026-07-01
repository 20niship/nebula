#include "MPMEngine.h"

#include <vector>
#include <cstring>
#include <cmath>
#include <random>
#include <glm/glm.hpp>

// ── バリア ────────────────────────────────────────────────────────────────

void MPMEngine::computeBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier b{};
  b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &b, 0, nullptr, 0, nullptr);
}

// ── MPMSimPC を用いた dispatch ヘルパー ───────────────────────────────────
// ComputePipeline::dispatch は SimPC を取るため、生バイトで渡す

void MPMEngine::dispatchMPM(VkCommandBuffer cmd, ComputePipeline& k,
                             const MPMSimPC& pc, uint32_t count) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          k.pipelineLayout, 0, 1, &attrBuf_.descriptorSet, 0, nullptr);
  vkCmdPushConstants(cmd, k.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(MPMSimPC), &pc);
  uint32_t groups = (count + 255u) / 256u;
  vkCmdDispatch(cmd, groups, 1, 1);
}

// ── 初期化 ────────────────────────────────────────────────────────────────

void MPMEngine::init(VkDevice device, VmaAllocator allocator,
                     VkDescriptorPool descriptorPool,
                     VkCommandPool cmdPool, VkQueue queue,
                     const std::string& shaderDir, const MPMConfig& cfg) {
  cfg_       = cfg;
  device_    = device;
  allocator_ = allocator;
  cmdPool_   = cmdPool;
  queue_     = queue;

  attrBuf_.init(device, allocator, descriptorPool);

  const uint32_t N  = cfg_.maxParticleCount();   // バッファ上限
  nParticles_       = cfg_.particleCount();       // ライブパーティクル数
  const uint32_t NC = cfg_.totalCells();
  const uint32_t NG = cfg_.nGroups();

  // ── パーティクルバッファ ───────────────────────────────────────────────
  // F0-2: xyz = 変形勾配 F の列, w = 対角応力 (σ_xx / σ_yy / σ_zz)
  // B0-2: xyz = APIC アフィン行列 B の列 (Phase 2), w = 非対角応力 (σ_xy / σ_xz / σ_yz)
  posIdx       = attrBuf_.addAttribute("P",      sizeof(glm::vec4), N);
  velIdx       = attrBuf_.addAttribute("v",      sizeof(glm::vec4), N);
  F0Idx_       = attrBuf_.addAttribute("F0",     sizeof(glm::vec4), N);
  F1Idx_       = attrBuf_.addAttribute("F1",     sizeof(glm::vec4), N);
  F2Idx_       = attrBuf_.addAttribute("F2",     sizeof(glm::vec4), N);
  B0Idx_       = attrBuf_.addAttribute("B0",     sizeof(glm::vec4), N);
  B1Idx_       = attrBuf_.addAttribute("B1",     sizeof(glm::vec4), N);
  B2Idx_       = attrBuf_.addAttribute("B2",     sizeof(glm::vec4), N);

  // ── ハッシュグリッドバッファ ───────────────────────────────────────────
  cellCountIdx_  = attrBuf_.addAttribute("cellCnt", sizeof(uint32_t), NC);
  cellOffsetIdx_ = attrBuf_.addAttribute("cellOff", sizeof(uint32_t), NC + NG);
  sortedIdxIdx_  = attrBuf_.addAttribute("sorted",  sizeof(uint32_t), N);

  // ── MPM グリッドバッファ ───────────────────────────────────────────────
  // gridMom は 2 × NC を確保: [0, NC) = v_new, [NC, 2*NC) = v_old (FLIP 用)
  gridMomIdx_  = attrBuf_.addAttribute("gridMom",  sizeof(glm::vec4), NC * 2);
  gridMassIdx_ = attrBuf_.addAttribute("gridMass", sizeof(float),     NC);

  // ── マテリアルテーブル SSBO ────────────────────────────────────────────
  // cfg_ のグローバルパラメータからデフォルトマテリアル（弾性体）を生成
  {
    std::vector<MaterialParams> mats(1);
    mats[0] = presetJelly(cfg_.E, cfg_.nu, cfg_.rho0);
    // plasticModel に合わせてモデルを上書き
    mats[0].model      = 0;  // ELASTIC をデフォルト
    mats[0].M_friction = M_friction;
    mats[0].q_cohesion = q_cohesion;
    mats[0].q_max      = q_max;
    materialsIdx_  = attrBuf_.addAttribute("materials", sizeof(MaterialParams), 16);
    materialCount_ = 1;
    attrBuf_.upload("materials", mats.data(), sizeof(MaterialParams), cmdPool_, queue_);
  }

  // ── 解析コライダー SSBO ────────────────────────────────────────────────
  // 最大 64 個のプリミティブを事前確保 (colliderCount_ == 0 で無効)
  collidersIdx_  = attrBuf_.addAttribute("colliders", sizeof(ColliderPrimitive), 64);
  colliderCount_ = 0;

  // ── 初期パーティクルデータをアップロード ──────────────────────────────
  {
    const uint32_t Nlive = nParticles_;
    const float blockFrac = 0.40f;
    const float sp = cfg_.world_size * blockFrac
                     / float(std::max({cfg_.nx, cfg_.ny, cfg_.nz}));
    const float Vp = sp * sp * sp;

    const float cx = cfg_.world_size * 0.5f;
    const float cz = cfg_.world_size * 0.5f;
    const float cy = cfg_.world_size * 0.70f;

    const float halfX = sp * float(cfg_.nx - 1) / 2.0f;
    const float halfY = sp * float(cfg_.ny - 1) / 2.0f;
    const float halfZ = sp * float(cfg_.nz - 1) / 2.0f;

    std::vector<glm::vec4> pos(N, glm::vec4(0.0f));
    std::vector<glm::vec4> vel(N, glm::vec4(0.0f));
    // F 列 (初期=単位行列), w=0 (初期応力=0)
    std::vector<glm::vec4> F0(N, glm::vec4(0.0f));
    std::vector<glm::vec4> F1(N, glm::vec4(0.0f));
    std::vector<glm::vec4> F2(N, glm::vec4(0.0f));
    // APIC B 列 (初期=0), w=0 (初期非対角応力=0)
    std::vector<glm::vec4> B0(N, glm::vec4(0.0f));
    std::vector<glm::vec4> B1(N, glm::vec4(0.0f));
    std::vector<glm::vec4> B2(N, glm::vec4(0.0f));

    uint32_t idx = 0;
    for (uint32_t iz = 0; iz < cfg_.nz; iz++)
    for (uint32_t iy = 0; iy < cfg_.ny; iy++)
    for (uint32_t ix = 0; ix < cfg_.nx; ix++, idx++) {
      pos[idx] = glm::vec4(
          cx - halfX + ix * sp,
          cy - halfY + iy * sp,
          cz - halfZ + iz * sp,
          Vp);
      // F = I: F0=(1,0,0), F1=(0,1,0), F2=(0,0,1), w=0 (応力=0)
      F0[idx] = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
      F1[idx] = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
      F2[idx] = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
      // B = 0: w=0 (非対角応力=0)
      B0[idx] = glm::vec4(0.0f);
      B1[idx] = glm::vec4(0.0f);
      B2[idx] = glm::vec4(0.0f);
    }

    auto up = [&](const std::string& name, const void* data, size_t bytes) {
      attrBuf_.upload(name, data, bytes, cmdPool_, queue_);
    };
    up("P",  pos.data(), N * sizeof(glm::vec4));
    up("v",  vel.data(), N * sizeof(glm::vec4));
    up("F0", F0.data(),  N * sizeof(glm::vec4));
    up("F1", F1.data(),  N * sizeof(glm::vec4));
    up("F2", F2.data(),  N * sizeof(glm::vec4));
    up("B0", B0.data(),  N * sizeof(glm::vec4));
    up("B1", B1.data(),  N * sizeof(glm::vec4));
    up("B2", B2.data(),  N * sizeof(glm::vec4));
  }

  // ── シェーダーパイプライン ─────────────────────────────────────────────
  auto load = [&](ComputePipeline& k, const char* name) {
    k.init(device, attrBuf_.descriptorSetLayout,
           shaderDir + "/" + name + ".spv");
  };
  load(kMpmZeroCells_,  "zero_cells.comp");
  load(kMpmHashCount_,  "mpm_hash_count.comp");
  load(kHashScanLocal_, "hash_scan_local.comp");
  load(kHashScanGlobal_,"hash_scan_global.comp");
  load(kHashAddBase_,   "hash_add_base.comp");
  load(kMpmHashSort_,   "mpm_hash_sort.comp");
  load(kZeroGrid_,      "mpm_zero_grid.comp");
  load(kP2G_,           "mpm_p2g.comp");
  load(kGridUpdate_,    "mpm_grid_update.comp");
  load(kNanoVDBBC_,     "mpm_nanovdb_bc.comp");
  load(kG2P_,           "mpm_g2p.comp");

  descriptorSetLayout = attrBuf_.descriptorSetLayout;
  descriptorSet       = attrBuf_.descriptorSet;
}

// ── クリーンアップ ────────────────────────────────────────────────────────

void MPMEngine::cleanup() {
  for (auto* k : {&kMpmZeroCells_, &kMpmHashCount_, &kHashScanLocal_,
                   &kHashScanGlobal_, &kHashAddBase_, &kMpmHashSort_,
                   &kZeroGrid_, &kP2G_, &kGridUpdate_, &kNanoVDBBC_, &kG2P_})
    k->cleanup();
  attrBuf_.cleanup();
}

VkBuffer MPMEngine::getPositionBuffer() const { return attrBuf_.getBuffer("P"); }

// ── NanoVDB SDF コライダー ────────────────────────────────────────────────

static uint32_t mortonExpand(uint32_t v) {
  v &= 0x000003ffu;
  v = (v ^ (v << 16u)) & 0xff0000ffu;
  v = (v ^ (v <<  8u)) & 0x0300f00fu;
  v = (v ^ (v <<  4u)) & 0x030c30c3u;
  v = (v ^ (v <<  2u)) & 0x09249249u;
  return v;
}
static uint32_t mortonEncode(uint32_t x, uint32_t y, uint32_t z) {
  return mortonExpand(x) | (mortonExpand(y) << 1u) | (mortonExpand(z) << 2u);
}

void MPMEngine::setColliderSphere(float radius, float cx, float cy, float cz) {
  const uint32_t GR  = cfg_.grid_res;
  const float    h   = cfg_.cellSize();
  const uint32_t NC  = GR * GR * GR;

  std::vector<float> sdf(NC);
  for (uint32_t iz = 0; iz < GR; iz++)
  for (uint32_t iy = 0; iy < GR; iy++)
  for (uint32_t ix = 0; ix < GR; ix++) {
    float wx = (ix + 0.5f) * h;
    float wy = (iy + 0.5f) * h;
    float wz = (iz + 0.5f) * h;
    float d  = std::sqrt((wx-cx)*(wx-cx) + (wy-cy)*(wy-cy) + (wz-cz)*(wz-cz));
    sdf[mortonEncode(ix, iy, iz)] = d - radius;
  }

  if (nanoVDBIdx_ == 0) {
    nanoVDBIdx_ = attrBuf_.addAttribute("nanoVDB", sizeof(float), NC);
  }
  attrBuf_.upload("nanoVDB", sdf.data(), NC * sizeof(float), cmdPool_, queue_);
}

void MPMEngine::clearCollider() {
  nanoVDBIdx_ = 0;
}

void MPMEngine::setColliderSDF(const std::vector<float>& mortonSDF) {
  if (nanoVDBIdx_ == 0) {
    nanoVDBIdx_ = attrBuf_.addAttribute("nanoVDB", sizeof(float), cfg_.totalCells());
  }
  attrBuf_.upload("nanoVDB", mortonSDF.data(),
                  mortonSDF.size() * sizeof(float), cmdPool_, queue_);
}

void MPMEngine::appendParticles(const std::vector<glm::vec4>& pos,
                                  const std::vector<glm::vec4>& vel) {
  const uint32_t maxN = cfg_.maxParticleCount();
  const uint32_t nNew = std::min(uint32_t(pos.size()), maxN - nParticles_);
  if (nNew == 0) return;

  std::vector<glm::vec4> F0(nNew, glm::vec4(1,0,0,0));
  std::vector<glm::vec4> F1(nNew, glm::vec4(0,1,0,0));
  std::vector<glm::vec4> F2(nNew, glm::vec4(0,0,1,0));
  std::vector<glm::vec4> B0(nNew, glm::vec4(0));
  std::vector<glm::vec4> B1(nNew, glm::vec4(0));
  std::vector<glm::vec4> B2(nNew, glm::vec4(0));

  VkDeviceSize byteOff = VkDeviceSize(nParticles_) * sizeof(glm::vec4);
  auto up = [&](const std::string& name, const void* data) {
    attrBuf_.uploadAt(name, data, VkDeviceSize(nNew) * sizeof(glm::vec4), byteOff, cmdPool_, queue_);
  };
  up("P",  pos.data());
  up("v",  vel.data());
  up("F0", F0.data()); up("F1", F1.data()); up("F2", F2.data());
  up("B0", B0.data()); up("B1", B1.data()); up("B2", B2.data());

  nParticles_ += nNew;
}

// ── マテリアルテーブル設定 ────────────────────────────────────────────────

void MPMEngine::setMaterials(const std::vector<MaterialParams>& mats) {
  if (mats.empty()) return;
  materialCount_ = uint32_t(mats.size());
  attrBuf_.upload("materials", mats.data(),
                  mats.size() * sizeof(MaterialParams), cmdPool_, queue_);
}

void MPMEngine::setParticleMaterialIds(const std::vector<uint32_t>& matIds) {
  const uint32_t N = uint32_t(matIds.size());
  // vel バッファ全体を (0, 0, 0, floatBitsToUint(matId)) で上書き
  // 初期化直後(vel=0)にのみ呼ぶこと
  std::vector<glm::vec4> vel(N);
  for (uint32_t i = 0; i < N; i++) {
    float wf;
    std::memcpy(&wf, &matIds[i], sizeof(float));
    vel[i] = glm::vec4(0.0f, 0.0f, 0.0f, wf);
  }
  attrBuf_.upload("v", vel.data(), N * sizeof(glm::vec4), cmdPool_, queue_);
}

// ── 解析コライダー ────────────────────────────────────────────────────────

void MPMEngine::setColliders(const ColliderSet& cols) {
  colliderCount_ = cols.count();
  if (colliderCount_ > 0) {
    attrBuf_.upload("colliders", cols.data().data(),
                    cols.count() * sizeof(ColliderPrimitive), cmdPool_, queue_);
  }
}

void MPMEngine::clearAnalyticColliders() {
  colliderCount_ = 0;
}

// ── Source エミッタ ───────────────────────────────────────────────────────

void MPMEngine::addSource(std::shared_ptr<Source> src) {
  sources_.push_back(std::move(src));
  sourceStepsDone_.push_back(0);
}

void MPMEngine::clearSources() {
  sources_.clear();
  sourceStepsDone_.clear();
}

void MPMEngine::resetParticles() {
  if (device_ == VK_NULL_HANDLE) return;
  vkDeviceWaitIdle(device_);
  nParticles_ = 0;
  std::fill(sourceStepsDone_.begin(), sourceStepsDone_.end(), 0);
  sourceRng_.seed(12345);
}

void MPMEngine::emitSources(float dt) {
  if (sources_.empty()) return;
  const uint32_t maxN = cfg_.maxParticleCount();

  for (size_t si = 0; si < sources_.size(); si++) {
    Source& src = *sources_[si];
    int&    done = sourceStepsDone_[si];

    bool shouldEmit = false;
    if      (src.step_count == -1) shouldEmit = (done == 0);
    else if (src.step_count ==  0) shouldEmit = true;
    else                           shouldEmit = (done < src.step_count);
    if (!shouldEmit) continue;

    int available = int(maxN) - int(nParticles_);
    int nNew      = std::min(src.particles_per_step, available);
    if (nNew <= 0) { done++; continue; }

    // material id を vel.w に格納するためビット再解釈
    uint32_t matId = src.particleType;
    float    matIdF;
    std::memcpy(&matIdF, &matId, sizeof(float));

    const float Vp = cfg_.particleVolume();
    std::vector<glm::vec4> pos(nNew), vel(nNew);
    std::vector<glm::vec4> F0v(nNew), F1v(nNew), F2v(nNew);
    std::vector<glm::vec4> B0v(nNew), B1v(nNew), B2v(nNew);

    // 位置生成
    if (auto* aabb = dynamic_cast<AABBSource*>(&src)) {
      glm::vec3 half = aabb->size * 0.5f;
      std::uniform_real_distribution<float> dx(-half.x, half.x);
      std::uniform_real_distribution<float> dy(-half.y, half.y);
      std::uniform_real_distribution<float> dz(-half.z, half.z);
      for (int j = 0; j < nNew; j++)
        pos[j] = glm::vec4(aabb->center + glm::vec3(dx(sourceRng_), dy(sourceRng_), dz(sourceRng_)), Vp);
    } else if (auto* sphere = dynamic_cast<SphereSource*>(&src)) {
      std::uniform_real_distribution<float> dr(0.0f, 1.0f);
      std::uniform_real_distribution<float> dtheta(0.0f, 3.14159265f);
      std::uniform_real_distribution<float> dphi(0.0f, 6.28318530f);
      for (int j = 0; j < nNew; j++) {
        float r     = sphere->radius * std::cbrt(dr(sourceRng_));
        float theta = dtheta(sourceRng_);
        float phi   = dphi(sourceRng_);
        pos[j] = glm::vec4(
            sphere->center.x + r * std::sin(theta) * std::cos(phi),
            sphere->center.y + r * std::sin(theta) * std::sin(phi),
            sphere->center.z + r * std::cos(theta), Vp);
      }
    } else {
      for (int j = 0; j < nNew; j++)
        pos[j] = glm::vec4(src.center, Vp);
    }

    // 速度 (vel.w = material id)、F = I、B = 0、stress = 0
    for (int j = 0; j < nNew; j++) {
      vel[j] = glm::vec4(src.vel, matIdF);
      F0v[j] = glm::vec4(1, 0, 0, 0);
      F1v[j] = glm::vec4(0, 1, 0, 0);
      F2v[j] = glm::vec4(0, 0, 1, 0);
      B0v[j] = glm::vec4(0);
      B1v[j] = glm::vec4(0);
      B2v[j] = glm::vec4(0);
    }

    VkDeviceSize byteOff = VkDeviceSize(nParticles_) * sizeof(glm::vec4);
    auto up = [&](const std::string& name, const void* data) {
      attrBuf_.uploadAt(name, data, VkDeviceSize(nNew) * sizeof(glm::vec4), byteOff, cmdPool_, queue_);
    };
    up("P",  pos.data());
    up("v",  vel.data());
    up("F0", F0v.data());
    up("F1", F1v.data());
    up("F2", F2v.data());
    up("B0", B0v.data());
    up("B1", B1v.data());
    up("B2", B2v.data());

    nParticles_ += uint32_t(nNew);
    done++;
    src.center += src.center_vel * dt;
  }
}

// ── Push Constants 構築 ───────────────────────────────────────────────────

MPMSimPC MPMEngine::buildPC(float subDt) const {
  MPMSimPC pc{};
  pc.posIdx        = posIdx;
  pc.velIdx        = velIdx;
  pc.F0Idx         = F0Idx_;
  pc.F1Idx         = F1Idx_;
  pc.typeFlagIdx   = 0;
  pc.cellCountIdx  = cellCountIdx_;
  pc.cellOffsetIdx = cellOffsetIdx_;
  pc.sortedIdxIdx  = sortedIdxIdx_;
  pc.particleCount = nParticles_;          // ライブパーティクル数
  pc.gridRes       = cfg_.grid_res;
  pc.F2Idx         = F2Idx_;
  pc.materialsIdx  = materialsIdx_;        // マテリアルテーブル SSBO
  pc.dt            = subDt;
  pc.cellSize      = cfg_.cellSize();
  pc.worldMin      = 0.0f;
  pc.worldMax      = cfg_.world_size;
  pc.gravity       = gravity;
  pc.mu_lame       = cfg_.mu();
  pc.lambda_lame   = cfg_.lame();
  pc.particleVolume= cfg_.particleVolume();
  pc.M_friction    = M_friction;
  pc.q_cohesion    = q_cohesion;
  pc.q_max         = q_max;
  pc.flip_ratio    = flip_ratio;
  pc.colliderIdx   = collidersIdx_;
  pc.colliderCount = colliderCount_;
  pc.B0Idx         = B0Idx_;
  pc.B1Idx         = B1Idx_;
  pc.B2Idx         = B2Idx_;
  pc.nanoVDBIdx    = nanoVDBIdx_;
  pc.gridMomIdx    = gridMomIdx_;
  pc.gridMassIdx   = gridMassIdx_;
  pc.restitution   = restitution;
  pc.wall_friction = wall_friction;
  pc.plasticModel  = plasticModel;
  pc.materialCount = materialCount_;
  pc.rho0          = cfg_.rho0;
  pc.p0_mcc        = 0.0f;
  pc.xi_hard       = 0.0f;
  pc.maxParticlesFrac = 0.0f;
  return pc;
}

// ── ステップ ──────────────────────────────────────────────────────────────

void MPMEngine::step(VkCommandBuffer cmd, float dt) {
  // Source エミッタ (GPU upload は compute dispatch の前に完結)
  emitSources(dt);

  const uint32_t N  = nParticles_;         // ライブパーティクル数
  const uint32_t NC = cfg_.totalCells();
  const uint32_t NG = cfg_.nGroups();
  float subDt = dt / float(std::max(1, numSubsteps));

  for (int sub = 0; sub < numSubsteps; ++sub) {
    MPMSimPC pc = buildPC(subDt);
    auto ds = attrBuf_.descriptorSet;

    // ① グリッドバッファをゼロクリア
    dispatchMPM(cmd, kZeroGrid_, pc, NC);
    computeBarrier(cmd);

    // ② cellCount バッファをゼロクリア
    dispatchMPM(cmd, kMpmZeroCells_, pc, NC);
    computeBarrier(cmd);

    // ③ 空間ハッシュ構築 (5パス)
    dispatchMPM(cmd, kMpmHashCount_, pc, N);
    computeBarrier(cmd);

    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanLocal_.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              kHashScanLocal_.pipelineLayout, 0, 1, &ds, 0, nullptr);
      vkCmdPushConstants(cmd, kHashScanLocal_.pipelineLayout,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MPMSimPC), &pc);
      vkCmdDispatch(cmd, NG, 1, 1);
    }
    computeBarrier(cmd);

    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanGlobal_.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              kHashScanGlobal_.pipelineLayout, 0, 1, &ds, 0, nullptr);
      vkCmdPushConstants(cmd, kHashScanGlobal_.pipelineLayout,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MPMSimPC), &pc);
      vkCmdDispatch(cmd, 1, 1, 1);
    }
    computeBarrier(cmd);

    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashAddBase_.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              kHashAddBase_.pipelineLayout, 0, 1, &ds, 0, nullptr);
      vkCmdPushConstants(cmd, kHashAddBase_.pipelineLayout,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MPMSimPC), &pc);
      vkCmdDispatch(cmd, NG, 1, 1);
    }
    computeBarrier(cmd);

    dispatchMPM(cmd, kMpmHashSort_, pc, N);
    computeBarrier(cmd);

    // ④ P2G (グリッドノード単位 gather)
    dispatchMPM(cmd, kP2G_, pc, NC);
    computeBarrier(cmd);

    // ⑤ グリッド速度更新 (正規化 + 重力 + 壁BC)
    dispatchMPM(cmd, kGridUpdate_, pc, NC);
    computeBarrier(cmd);

    // ⑥ NanoVDB SDF 境界条件
    dispatchMPM(cmd, kNanoVDBBC_, pc, NC);
    computeBarrier(cmd);

    // ⑦ G2P + F 更新 + 応力 + 位置更新
    dispatchMPM(cmd, kG2P_, pc, N);
    if (sub < numSubsteps - 1) computeBarrier(cmd);
  }
}
