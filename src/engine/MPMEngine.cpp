#include "MPMEngine.h"
#include "../core/Profiling.h"
#include "BoundaryParticles.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>
#include <map>
#include <random>
#include <vector>

// pos/vel half-float パック(12B/要素、実験用): xyzをpackHalf2x16、wはfloatBitsToUint往復(mpm_common.glslのwritePackedVec4と同一仕様)
static std::vector<uint32_t> packVec4ToHalf(const std::vector<glm::vec4>& src) {
  std::vector<uint32_t> dst(src.size() * 3);
  for(size_t i = 0; i < src.size(); ++i) {
    dst[i * 3]     = glm::packHalf2x16(glm::vec2(src[i].x, src[i].y));
    dst[i * 3 + 1] = glm::packHalf2x16(glm::vec2(src[i].z, 0.0f));
    dst[i * 3 + 2] = glm::floatBitsToUint(src[i].w);
  }
  return dst;
}

// ── バリア ────────────────────────────────────────────────────────────────

void MPMEngine::syncGpuForProfiling(VkCommandBuffer cmd) {
#ifdef NEBULA_TRACY
  vkEndCommandBuffer(cmd);
  VkSubmitInfo si{};
  si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers    = &cmd;
  vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue_);
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);
#else
  (void)cmd;
#endif
}

void MPMEngine::computeBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier b{};
  b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

// ── MPMSimPC を用いた dispatch ヘルパー ───────────────────────────────────
// ComputePipeline::dispatch は SimPC を取るため、生バイトで渡す

void MPMEngine::dispatchMPM(VkCommandBuffer cmd, ComputePipeline& k, const MPMSimPC& pc, uint32_t count) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipelineLayout, 0, 1, &attrBuf_.descriptorSet, 0, nullptr);
  vkCmdPushConstants(cmd, k.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MPMSimPC), &pc);
  uint32_t groups = (count + 255u) / 256u;
  vkCmdDispatch(cmd, groups, 1, 1);
}

// ── GPUパス単位プロファイリング (PyroEngine と同一パターン) ──────────────
#ifdef NEBULA_GPU_PROFILING

void MPMEngine::profBegin(VkCommandBuffer cmd) {
  if(profEnabled_ && profQueryIndex_ + 1 < kProfMaxQueries) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, profPool_, profQueryIndex_);
}

void MPMEngine::profEnd(VkCommandBuffer cmd, const char* label) {
  if(profEnabled_ && profQueryIndex_ + 1 < kProfMaxQueries) {
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, profPool_, profQueryIndex_ + 1);
    profLabels_.push_back(label);
    profQueryIndex_ += 2;
  }
}

void MPMEngine::enableGpuProfiling(VkPhysicalDevice physicalDevice) {
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physicalDevice, &props);
  profTsPeriodNs_ = props.limits.timestampPeriod;
  VkQueryPoolCreateInfo qpci{};
  qpci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
  qpci.queryCount = kProfMaxQueries;
  vkCreateQueryPool(device_, &qpci, nullptr, &profPool_);
  profEnabled_ = true;
}

void MPMEngine::printGpuProfile() {
  if(!profEnabled_ || profLabels_.empty()) return;
  uint32_t n = uint32_t(profLabels_.size()) * 2;
  std::vector<uint64_t> ts(n);
  vkGetQueryPoolResults(device_, profPool_, 0, n, ts.size() * sizeof(uint64_t), ts.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

  std::map<std::string, double> sumMs;
  std::map<std::string, int> counts;
  double total = 0.0;
  for(size_t i = 0; i < profLabels_.size(); i++) {
    double ms = double(ts[i * 2 + 1] - ts[i * 2]) * profTsPeriodNs_ / 1e6;
    sumMs[profLabels_[i]] += ms;
    counts[profLabels_[i]] += 1;
    total += ms;
  }
  std::vector<std::pair<std::string, double>> sorted(sumMs.begin(), sumMs.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  std::fprintf(stderr, "=== [MPMEngine GPU profile] total=%.4f ms ===\n", total);
  for(const auto& [label, ms] : sorted) {
    std::fprintf(stderr, "  %-26s %9.4f ms  (%5.1f%%, x%d)\n", label.c_str(), ms, ms / total * 100.0, counts[label]);
  }
}

#endif // NEBULA_GPU_PROFILING

// ── 初期化 ────────────────────────────────────────────────────────────────

void MPMEngine::init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, const MPMConfig& cfg) {
  cfg_ = cfg;
  initEngineBase(device, allocator, descriptorPool, cmdPool, queue);

  const uint32_t N  = cfg_.maxParticleCount(); // バッファ上限
  nParticles_       = cfg_.particleCount();    // ライブパーティクル数
  const uint32_t NC = cfg_.totalCells();

  // ── パーティクルバッファ ───────────────────────────────────────────────
  // F0-2: xyz = 変形勾配 F の列, w = 対角応力 (σ_xx / σ_yy / σ_zz)
  // B0-2: xyz = APIC アフィン行列 B の列 (Phase 2), w = 非対角応力 (σ_xy / σ_xz / σ_yz)
  posIdx = attrBuf_.addAttribute("P", sizeof(uint32_t) * 3, N); // half-floatパック実験(12B/要素)
  velIdx = attrBuf_.addAttribute("v", sizeof(uint32_t) * 3, N);
  F0Idx_ = attrBuf_.addAttribute("F0", sizeof(uint32_t) * 3, N); // half-floatパック実験(12B/要素)
  F1Idx_ = attrBuf_.addAttribute("F1", sizeof(uint32_t) * 3, N);
  F2Idx_ = attrBuf_.addAttribute("F2", sizeof(uint32_t) * 3, N);
  B0Idx_ = attrBuf_.addAttribute("B0", sizeof(uint32_t) * 3, N);
  B1Idx_ = attrBuf_.addAttribute("B1", sizeof(uint32_t) * 3, N);
  B2Idx_ = attrBuf_.addAttribute("B2", sizeof(uint32_t) * 3, N);

  // gridMom/gridMass: P2G(scatter)固定小数点atomicAdd蓄積専用、G2Pは読まない
  gridMomIdx_  = attrBuf_.addAttribute("gridMom", sizeof(glm::vec4), NC);
  gridMassIdx_ = attrBuf_.addAttribute("gridMass", sizeof(float), NC);
  // gridVel/gridVelOld: GridUpdate出力をhalf-floatパック(8B/セル)で保持、G2P帯域を半減
  gridVelIdx_    = attrBuf_.addAttribute("gridVel", sizeof(uint32_t) * 2, NC);
  gridVelOldIdx_ = attrBuf_.addAttribute("gridVelOld", sizeof(uint32_t) * 2, NC);

  // ── マテリアルテーブル SSBO ────────────────────────────────────────────
  // cfg_ のグローバルパラメータからデフォルトマテリアル（弾性体）を生成
  {
    std::vector<MaterialParams> mats(1);
    mats[0] = presetJelly(cfg_.E, cfg_.nu, cfg_.rho0);
    // plasticModel に合わせてモデルを上書き
    mats[0].model      = 0; // ELASTIC をデフォルト
    mats[0].M_friction = M_friction;
    mats[0].q_cohesion = q_cohesion;
    mats[0].q_max      = q_max;
    materialsIdx_      = attrBuf_.addAttribute("materials", sizeof(MaterialParams), 16);
    materialCount_     = 1;
    attrBuf_.upload("materials", mats.data(), sizeof(MaterialParams), cmdPool_, queue_);
  }

  // ── 解析コライダー SSBO ────────────────────────────────────────────────
  // 最大 64 個のプリミティブを事前確保 (colliderCount_ == 0 で無効)
  collidersIdx_  = attrBuf_.addAttribute("colliders", sizeof(ColliderPrimitive), 64);
  colliderCount_ = 0;

  // ── 初期パーティクルデータをアップロード ──────────────────────────────
  {
    const uint32_t Nlive  = nParticles_;
    const float blockFrac = 0.40f;
    // 最短軸を基準に spacing を決め、非立方体ドメインでもブロックが短辺からはみ出さないようにする
    const float minDomain = std::min({cfg_.domainSize.x, cfg_.domainSize.y, cfg_.domainSize.z});
    const float sp        = minDomain * blockFrac / float(std::max({cfg_.nx, cfg_.ny, cfg_.nz}));
    const float Vp        = sp * sp * sp;

    const float cx = cfg_.domainSize.x * 0.5f;
    const float cz = cfg_.domainSize.z * 0.5f;
    const float cy = cfg_.domainSize.y * 0.70f;

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
    for(uint32_t iz = 0; iz < cfg_.nz; iz++)
      for(uint32_t iy = 0; iy < cfg_.ny; iy++)
        for(uint32_t ix = 0; ix < cfg_.nx; ix++, idx++) {
          pos[idx] = glm::vec4(cx - halfX + ix * sp, cy - halfY + iy * sp, cz - halfZ + iz * sp, Vp);
          // F = I: F0=(1,0,0), F1=(0,1,0), F2=(0,0,1), w=0 (応力=0)
          F0[idx] = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
          F1[idx] = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
          F2[idx] = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
          // B = 0: w=0 (非対角応力=0)
          B0[idx] = glm::vec4(0.0f);
          B1[idx] = glm::vec4(0.0f);
          B2[idx] = glm::vec4(0.0f);
        }

    auto up = [&](const std::string& name, const void* data, size_t bytes) { attrBuf_.upload(name, data, bytes, cmdPool_, queue_); };
    std::vector<uint32_t> posPacked = packVec4ToHalf(pos);
    std::vector<uint32_t> velPacked = packVec4ToHalf(vel);
    up("P", posPacked.data(), posPacked.size() * sizeof(uint32_t));
    up("v", velPacked.data(), velPacked.size() * sizeof(uint32_t));
    auto upPacked = [&](const std::string& name, const std::vector<glm::vec4>& v) {
      std::vector<uint32_t> packed = packVec4ToHalf(v);
      up(name, packed.data(), packed.size() * sizeof(uint32_t));
    };
    upPacked("F0", F0);
    upPacked("F1", F1);
    upPacked("F2", F2);
    upPacked("B0", B0);
    upPacked("B1", B1);
    upPacked("B2", B2);
  }

  initForces();

  // ── シェーダーパイプライン ─────────────────────────────────────────────
  auto load = [&](ComputePipeline& k, const char* name) { k.init(device, attrBuf_.descriptorSetLayout, shaderDir + "/" + name + ".spv"); };
  load(kZeroGrid_, "mpm_zero_grid.comp");
  load(kP2G_, "mpm_p2g.comp");
  load(kG2P_, "mpm_g2p.comp");

  descriptorSetLayout = attrBuf_.descriptorSetLayout;
  descriptorSet       = attrBuf_.descriptorSet;
}

// ── クリーンアップ ────────────────────────────────────────────────────────

void MPMEngine::cleanup() {
  for(auto* k : {&kZeroGrid_, &kP2G_, &kGridUpdate_, &kG2P_}) k->cleanup();
#ifdef NEBULA_GPU_PROFILING
  if(profPool_ != VK_NULL_HANDLE) vkDestroyQueryPool(device_, profPool_, nullptr);
#endif
  cleanupEngineBase();
}

VkBuffer MPMEngine::getPositionBuffer() const { return attrBuf_.getBuffer("P"); }
VkBuffer MPMEngine::getVelocityBuffer() const { return attrBuf_.getBuffer("v"); }

// ── メッシュSDFコライダー ────────────────────────────────────────────────

uint32_t MPMEngine::loadColliderMesh(const std::string& objPath, LocalMeshSDF& gridOut, uint32_t res, float scale) {
  BoundaryParticles bp;
  BoundaryMesh mesh = bp.loadOBJ(objPath, 1e6f, scale, glm::vec3(0.0f), false);

  std::vector<MeshTriangle> tris;
  tris.reserve(mesh.triVerts.size() / 3);
  for(size_t i = 0; i + 2 < mesh.triVerts.size(); i += 3) {
    MeshTriangle t;
    t.v[0] = mesh.triVerts[i];
    t.v[1] = mesh.triVerts[i + 1];
    t.v[2] = mesh.triVerts[i + 2];
    t.n    = glm::normalize(glm::cross(t.v[1] - t.v[0], t.v[2] - t.v[0]));
    tris.push_back(t);
  }

  gridOut = bakeLocalMeshSDF(tris, res);
  return uploadColliderMeshSDF(gridOut);
}

uint32_t MPMEngine::uploadColliderMeshSDF(const LocalMeshSDF& grid) {
  std::string name = "meshSDF_" + std::to_string(nextMeshSDFId_++);
  uint32_t idx      = attrBuf_.addAttribute(name, sizeof(float), grid.data.size());
  attrBuf_.upload(name, grid.data.data(), grid.data.size() * sizeof(float), cmdPool_, queue_);
  return idx;
}

void MPMEngine::appendParticles(const std::vector<glm::vec4>& pos, const std::vector<glm::vec4>& vel) {
  const uint32_t maxN = cfg_.maxParticleCount();
  const uint32_t nNew = std::min(uint32_t(pos.size()), maxN - nParticles_);
  if(nNew == 0) return;

  std::vector<glm::vec4> F0(nNew, glm::vec4(1, 0, 0, 0));
  std::vector<glm::vec4> F1(nNew, glm::vec4(0, 1, 0, 0));
  std::vector<glm::vec4> F2(nNew, glm::vec4(0, 0, 1, 0));
  std::vector<glm::vec4> B0(nNew, glm::vec4(0));
  std::vector<glm::vec4> B1(nNew, glm::vec4(0));
  std::vector<glm::vec4> B2(nNew, glm::vec4(0));

  VkDeviceSize packedOff = VkDeviceSize(nParticles_) * sizeof(uint32_t) * 3;
  auto upPacked = [&](const std::string& name, const std::vector<glm::vec4>& v) {
    std::vector<uint32_t> packed = packVec4ToHalf(v);
    attrBuf_.uploadAt(name, packed.data(), VkDeviceSize(nNew) * sizeof(uint32_t) * 3, packedOff, cmdPool_, queue_);
  };
  upPacked("P", pos);
  upPacked("v", vel);
  upPacked("F0", F0);
  upPacked("F1", F1);
  upPacked("F2", F2);
  upPacked("B0", B0);
  upPacked("B1", B1);
  upPacked("B2", B2);

  nParticles_ += nNew;
}

// ── マテリアルテーブル設定 ────────────────────────────────────────────────

void MPMEngine::setMaterials(const std::vector<MaterialParams>& mats) {
  if(mats.empty()) return;
  materialCount_ = uint32_t(mats.size());
  attrBuf_.upload("materials", mats.data(), mats.size() * sizeof(MaterialParams), cmdPool_, queue_);
}

void MPMEngine::setParticleMaterialIds(const std::vector<uint32_t>& matIds) {
  const uint32_t N = uint32_t(matIds.size());
  // vel バッファ全体を (0, 0, 0, floatBitsToUint(matId)) で上書き
  // 初期化直後(vel=0)にのみ呼ぶこと
  std::vector<glm::vec4> vel(N);
  for(uint32_t i = 0; i < N; i++) {
    float wf;
    std::memcpy(&wf, &matIds[i], sizeof(float));
    vel[i] = glm::vec4(0.0f, 0.0f, 0.0f, wf);
  }
  attrBuf_.upload("v", vel.data(), N * sizeof(glm::vec4), cmdPool_, queue_);
}

// ── 解析コライダー ────────────────────────────────────────────────────────

void MPMEngine::setColliders(const ColliderSet& cols) {
  ZoneScoped;
  colliderCount_ = cols.count();
  if(colliderCount_ > 0) {
    attrBuf_.upload("colliders", cols.data().data(), cols.count() * sizeof(ColliderPrimitive), cmdPool_, queue_);
  }
}

void MPMEngine::clearAnalyticColliders() { colliderCount_ = 0; }

// ── Emitter ───────────────────────────────────────────────────────────────

void MPMEngine::addEmitter(std::shared_ptr<Emitter> emitter) {
  emitters_.push_back(std::move(emitter));
  emitterStepsDone_.push_back(0);
}

void MPMEngine::clearEmitters() {
  emitters_.clear();
  emitterStepsDone_.clear();
}

void MPMEngine::resetParticles() {
  if(device_ == VK_NULL_HANDLE) return;
  vkDeviceWaitIdle(device_);
  nParticles_ = 0;
  std::fill(emitterStepsDone_.begin(), emitterStepsDone_.end(), 0);
  emitterRng_.seed(12345);
}

void MPMEngine::emitFromEmitters(float dt) {
  if(emitters_.empty()) return;
  const uint32_t maxN = cfg_.maxParticleCount();

  for(size_t si = 0; si < emitters_.size(); si++) {
    Emitter& emitter = *emitters_[si];
    int& done        = emitterStepsDone_[si];

    bool shouldEmit = false;
    if(emitter.step_count == -1)
      shouldEmit = (done == 0);
    else if(emitter.step_count == 0)
      shouldEmit = true;
    else
      shouldEmit = (done < emitter.step_count);
    if(!shouldEmit) continue;

    int available = int(maxN) - int(nParticles_);
    int nNew      = std::min(emitter.particles_per_step, available);
    if(nNew <= 0) {
      done++;
      continue;
    }

    // material id を vel.w に格納するためビット再解釈
    uint32_t matId = emitter.particleType;
    float matIdF;
    std::memcpy(&matIdF, &matId, sizeof(float));

    const float Vp = cfg_.particleVolume();
    std::vector<glm::vec4> pos(nNew), vel(nNew);
    std::vector<glm::vec4> F0v(nNew), F1v(nNew), F2v(nNew);
    std::vector<glm::vec4> B0v(nNew), B1v(nNew), B2v(nNew);

    // 位置生成 (形状ごとの分岐は Emitter::sample() に委譲)
    for(int j = 0; j < nNew; j++) pos[j] = glm::vec4(emitter.sample(emitterRng_), Vp);

    // 速度 (vel.w = material id)、F = I、B = 0、stress = 0
    for(int j = 0; j < nNew; j++) {
      vel[j] = glm::vec4(emitter.sample_velocity(glm::vec3(pos[j]), emitterRng_), matIdF);
      F0v[j] = glm::vec4(1, 0, 0, 0);
      F1v[j] = glm::vec4(0, 1, 0, 0);
      F2v[j] = glm::vec4(0, 0, 1, 0);
      B0v[j] = glm::vec4(0);
      B1v[j] = glm::vec4(0);
      B2v[j] = glm::vec4(0);
    }

    VkDeviceSize packedOff = VkDeviceSize(nParticles_) * sizeof(uint32_t) * 3;
    auto upPacked = [&](const std::string& name, const std::vector<glm::vec4>& v) {
      std::vector<uint32_t> packed = packVec4ToHalf(v);
      attrBuf_.uploadAt(name, packed.data(), VkDeviceSize(nNew) * sizeof(uint32_t) * 3, packedOff, cmdPool_, queue_);
    };
    upPacked("P", pos);
    upPacked("v", vel);
    upPacked("F0", F0v);
    upPacked("F1", F1v);
    upPacked("F2", F2v);
    upPacked("B0", B0v);
    upPacked("B1", B1v);
    upPacked("B2", B2v);

    nParticles_ += uint32_t(nNew);
    done++;
    emitter.center += emitter.center_vel * dt;
  }
}

// ── Push Constants 構築 ───────────────────────────────────────────────────

MPMSimPC MPMEngine::buildPC(float subDt) const {
  MPMSimPC pc{};
  pc.posIdx           = posIdx;
  pc.velIdx           = velIdx;
  pc.F0Idx            = F0Idx_;
  pc.F1Idx            = F1Idx_;
  pc.typeFlagIdx      = 0;
  pc.particleCount    = nParticles_; // ライブパーティクル数
  pc.hashCells        = cfg_.totalCells();
  pc.F2Idx            = F2Idx_;
  pc.materialsIdx     = materialsIdx_; // マテリアルテーブル SSBO
  pc.dt               = subDt;
  pc.cellSize         = cfg_.cellSize;
  pc.gridRes          = cfg_.gridRes();
  pc.worldMin         = glm::vec3(0.0f);
  pc.worldMax         = cfg_.domainSize;
  pc.forceBufIdx      = forcesIdx_;
  pc.mu_lame          = cfg_.mu();
  pc.lambda_lame      = cfg_.lame();
  pc.particleVolume   = cfg_.particleVolume();
  pc.M_friction       = M_friction;
  pc.q_cohesion       = q_cohesion;
  pc.q_max            = q_max;
  pc.flip_ratio       = flip_ratio;
  pc.colliderIdx      = collidersIdx_;
  pc.colliderCount    = colliderCount_;
  pc.B0Idx            = B0Idx_;
  pc.B1Idx            = B1Idx_;
  pc.B2Idx            = B2Idx_;
  pc.gridMomIdx       = gridMomIdx_;
  pc.gridMassIdx      = gridMassIdx_;
  pc.gridVelIdx       = gridVelIdx_;
  pc.gridVelOldIdx    = gridVelOldIdx_;
  pc.restitution      = restitution;
  pc.wall_friction    = wall_friction;
  pc.plasticModel     = plasticModel;
  pc.materialCount    = materialCount_;
  pc.rho0             = cfg_.rho0;
  pc.p0_mcc           = 0.0f;
  pc.xi_hard          = 0.0f;
  pc.forceCount       = (uint32_t)forces_.size();
  return pc;
}

// ── ステップ ──────────────────────────────────────────────────────────────

void MPMEngine::step(VkCommandBuffer cmd, float dt) {
  ZoneScoped;
  FrameMark;
  // Emitter (GPU upload は compute dispatch の前に完結)
  {
    ZoneScopedN("EmitFromEmitters");
    emitFromEmitters(dt);
  }

  {
    ZoneScopedN("UploadForces");
    uploadForces(dt);
  }

  const uint32_t N  = nParticles_; // ライブパーティクル数
  const uint32_t NC = cfg_.totalCells();
  float subDt       = dt / float(std::max(1, numSubsteps));

#ifdef NEBULA_GPU_PROFILING
  if(profEnabled_) {
    vkCmdResetQueryPool(cmd, profPool_, 0, kProfMaxQueries);
    profLabels_.clear();
  }
  profQueryIndex_ = 0;
#endif

  for(int sub = 0; sub < numSubsteps; ++sub) {
    MPMSimPC pc = buildPC(subDt);

    // ① グリッドバッファをゼロクリア
    {
      ZoneScopedN("ZeroGrid");
#ifdef NEBULA_GPU_PROFILING
      profBegin(cmd);
#endif
      dispatchMPM(cmd, kZeroGrid_, pc, NC);
      computeBarrier(cmd);
#ifdef NEBULA_GPU_PROFILING
      profEnd(cmd, "ZeroGrid");
#endif
      syncGpuForProfiling(cmd);
    }

    // ② P2G (MLS-MPM: パーティクル並列scatter、固定小数点atomicAdd。空間ハッシュ不要)
    {
      ZoneScopedN("P2G");
#ifdef NEBULA_GPU_PROFILING
      profBegin(cmd);
#endif
      dispatchMPM(cmd, kP2G_, pc, N);
      computeBarrier(cmd);
#ifdef NEBULA_GPU_PROFILING
      profEnd(cmd, "P2G");
#endif
      syncGpuForProfiling(cmd);
    }

    // ③ グリッド速度更新 (正規化 + 重力 + 壁BC)
    {
      ZoneScopedN("GridUpdate");
#ifdef NEBULA_GPU_PROFILING
      profBegin(cmd);
#endif
      dispatchMPM(cmd, kGridUpdate_, pc, NC);
      computeBarrier(cmd);
#ifdef NEBULA_GPU_PROFILING
      profEnd(cmd, "GridUpdate");
#endif
      syncGpuForProfiling(cmd);
    }

    // ④ G2P + F 更新 + 応力 + 位置更新
    {
      ZoneScopedN("G2P");
#ifdef NEBULA_GPU_PROFILING
      profBegin(cmd);
#endif
      dispatchMPM(cmd, kG2P_, pc, N);
      if(sub < numSubsteps - 1) computeBarrier(cmd);
#ifdef NEBULA_GPU_PROFILING
      profEnd(cmd, "G2P");
#endif
      syncGpuForProfiling(cmd);
    }
  }
}
