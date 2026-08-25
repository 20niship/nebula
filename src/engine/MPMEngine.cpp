#include "MPMEngine.h"
#include "../core/Profiling.h"
#include "BoundaryParticles.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <glm/glm.hpp>
#include <random>
#include <vector>

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

// ── 初期化 ────────────────────────────────────────────────────────────────

void MPMEngine::init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir) {
  initEngineBase(device, allocator, descriptorPool, cmdPool, queue);

  // issue #98: 旧MPMConfig::nx/ny/nzを廃止し、gridRes()をデフォルト粒子シード格子解像度として流用する。
  const glm::uvec3 gr             = domain::gridRes(domainSize, cellSize);
  const uint32_t NC               = domain::hashCellsCube(gr);
  const uint32_t defaultSeedCount = gr.x * gr.y * gr.z;

  // maxParticles を明示指定した場合は init() 時の自動シードを行わない (appendParticles/emitter で後から追加する用途)
  maxParticles_    = maxParticles > 0 ? maxParticles : defaultSeedCount;
  const uint32_t N = maxParticles_; // バッファ上限
  nParticles_      = maxParticles > 0 ? 0 : defaultSeedCount;

  // ── パーティクルバッファ ───────────────────────────────────────────────
  // F0-2: xyz = 変形勾配 F の列, w = 対角応力 (σ_xx / σ_yy / σ_zz)
  // B0-2: xyz = APIC アフィン行列 B の列 (Phase 2), w = 非対角応力 (σ_xy / σ_xz / σ_yz)
  posIdx          = attrBuf_.addAttribute("P", sizeof(glm::vec4), N);
  velIdx          = attrBuf_.addAttribute("v", sizeof(glm::vec4), N);
  pc_.posIdx      = posIdx;
  pc_.velIdx      = velIdx;
  pc_.F0Idx       = attrBuf_.addAttribute("F0", sizeof(glm::vec4), N);
  pc_.F1Idx       = attrBuf_.addAttribute("F1", sizeof(glm::vec4), N);
  pc_.F2Idx       = attrBuf_.addAttribute("F2", sizeof(glm::vec4), N);
  pc_.B0Idx       = attrBuf_.addAttribute("B0", sizeof(glm::vec4), N);
  pc_.B1Idx       = attrBuf_.addAttribute("B1", sizeof(glm::vec4), N);
  pc_.B2Idx       = attrBuf_.addAttribute("B2", sizeof(glm::vec4), N);
  pc_.typeFlagIdx = 0;

  // gridMom は 2 × NC を確保: [0, NC) = v_new, [NC, 2*NC) = v_old (FLIP 用)
  pc_.gridMomIdx  = attrBuf_.addAttribute("gridMom", sizeof(glm::vec4), NC * 2);
  pc_.gridMassIdx = attrBuf_.addAttribute("gridMass", sizeof(float), NC);
  pc_.hashCells   = NC;
  pc_.gridRes     = gr;
  pc_.cellSize    = cellSize;
  pc_.worldMax    = domainSize; // ドメイン下限は常に原点固定 (worldMin は廃止)
  pc_.mu_lame     = E / (2.0f * (1.0f + nu));
  pc_.lambda_lame = E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));
  pc_.rho0        = rho0;
  pc_.p0_mcc      = 0.0f;
  pc_.xi_hard     = 0.0f;

  // ── マテリアルテーブル SSBO ────────────────────────────────────────────
  {
    std::vector<MaterialParams> mats(1);
    mats[0]            = presetJelly(E, nu, rho0);
    mats[0].model      = 0; // ELASTIC をデフォルト
    mats[0].M_friction = M_friction;
    mats[0].q_cohesion = q_cohesion;
    mats[0].q_max      = q_max;
    pc_.materialsIdx   = attrBuf_.addAttribute("materials", sizeof(MaterialParams), 16);
    pc_.materialCount  = 1;
    attrBuf_.upload("materials", mats.data(), sizeof(MaterialParams), cmdPool_, queue_);
  }

  // ── 解析コライダー SSBO: 最大64個のプリミティブを事前確保 (colliderCount == 0 で無効) ──
  pc_.colliderIdx       = attrBuf_.addAttribute("colliders", sizeof(ColliderPrimitive), 64);
  pc_.colliderCount     = 0;
  pc_.colliderForceIdx  = attrBuf_.addAttribute("colliderForce", sizeof(glm::vec4), 64);
  pc_.colliderTorqueIdx = attrBuf_.addAttribute("colliderTorque", sizeof(glm::vec4), 64);

  // ── 初期パーティクルデータをアップロード (ドメイン中央付近にgridRes解像度のブロックをシード) ──
  {
    const float blockFrac = 0.40f;
    const float minDomain = std::min({domainSize.x, domainSize.y, domainSize.z});
    const float sp        = minDomain * blockFrac / float(std::max({gr.x, gr.y, gr.z}));
    const float Vp        = sp * sp * sp;

    const float cx = domainSize.x * 0.5f;
    const float cz = domainSize.z * 0.5f;
    const float cy = domainSize.y * 0.70f;

    const float halfX = sp * float(gr.x - 1) / 2.0f;
    const float halfY = sp * float(gr.y - 1) / 2.0f;
    const float halfZ = sp * float(gr.z - 1) / 2.0f;

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
    if(nParticles_ > 0) {
      for(uint32_t iz = 0; iz < gr.z; iz++)
        for(uint32_t iy = 0; iy < gr.y; iy++)
          for(uint32_t ix = 0; ix < gr.x; ix++, idx++) {
            pos[idx] = glm::vec4(cx - halfX + ix * sp, cy - halfY + iy * sp, cz - halfZ + iz * sp, Vp);
            F0[idx]  = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
            F1[idx]  = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
            F2[idx]  = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
          }
    }

    auto up = [&](const std::string& name, const void* data, size_t bytes) { attrBuf_.upload(name, data, bytes, cmdPool_, queue_); };
    up("P", pos.data(), N * sizeof(glm::vec4));
    up("v", vel.data(), N * sizeof(glm::vec4));
    up("F0", F0.data(), N * sizeof(glm::vec4));
    up("F1", F1.data(), N * sizeof(glm::vec4));
    up("F2", F2.data(), N * sizeof(glm::vec4));
    up("B0", B0.data(), N * sizeof(glm::vec4));
    up("B1", B1.data(), N * sizeof(glm::vec4));
    up("B2", B2.data(), N * sizeof(glm::vec4));
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
  cleanupEngineBase();
}

VkBuffer MPMEngine::getPositionBuffer() const { return attrBuf_.getBuffer("P"); }
VkBuffer MPMEngine::getVelocityBuffer() const { return attrBuf_.getBuffer("v"); }
VkBuffer MPMEngine::getColliderForceBuffer() const { return attrBuf_.getBuffer("colliderForce"); }
VkBuffer MPMEngine::getColliderTorqueBuffer() const { return attrBuf_.getBuffer("colliderTorque"); }

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
  uint32_t idx     = attrBuf_.addAttribute(name, sizeof(float), grid.data.size());
  attrBuf_.upload(name, grid.data.data(), grid.data.size() * sizeof(float), cmdPool_, queue_);
  return idx;
}

void MPMEngine::appendParticles(const std::vector<glm::vec4>& pos, const std::vector<glm::vec4>& vel) {
  const uint32_t nNew = std::min(uint32_t(pos.size()), maxParticles_ - nParticles_);
  if(nNew == 0) return;

  std::vector<glm::vec4> F0(nNew, glm::vec4(1, 0, 0, 0));
  std::vector<glm::vec4> F1(nNew, glm::vec4(0, 1, 0, 0));
  std::vector<glm::vec4> F2(nNew, glm::vec4(0, 0, 1, 0));
  std::vector<glm::vec4> B0(nNew, glm::vec4(0));
  std::vector<glm::vec4> B1(nNew, glm::vec4(0));
  std::vector<glm::vec4> B2(nNew, glm::vec4(0));

  VkDeviceSize byteOff = VkDeviceSize(nParticles_) * sizeof(glm::vec4);
  auto up              = [&](const std::string& name, const void* data) { attrBuf_.uploadAt(name, data, VkDeviceSize(nNew) * sizeof(glm::vec4), byteOff, cmdPool_, queue_); };
  up("P", pos.data());
  up("v", vel.data());
  up("F0", F0.data());
  up("F1", F1.data());
  up("F2", F2.data());
  up("B0", B0.data());
  up("B1", B1.data());
  up("B2", B2.data());

  nParticles_ += nNew;
}

// ── マテリアルテーブル設定 ────────────────────────────────────────────────

void MPMEngine::setMaterials(const std::vector<MaterialParams>& mats) {
  if(mats.empty()) return;
  pc_.materialCount = uint32_t(mats.size());
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
  pc_.colliderCount = cols.count();
  if(pc_.colliderCount > 0) {
    attrBuf_.upload("colliders", cols.data().data(), cols.count() * sizeof(ColliderPrimitive), cmdPool_, queue_);
  }
}

void MPMEngine::clearAnalyticColliders() { pc_.colliderCount = 0; }

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
  const uint32_t maxN = maxParticles_;

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

    const float Vp = cellSize * cellSize * cellSize;
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

    VkDeviceSize byteOff = VkDeviceSize(nParticles_) * sizeof(glm::vec4);
    auto up              = [&](const std::string& name, const void* data) { attrBuf_.uploadAt(name, data, VkDeviceSize(nNew) * sizeof(glm::vec4), byteOff, cmdPool_, queue_); };
    up("P", pos.data());
    up("v", vel.data());
    up("F0", F0v.data());
    up("F1", F1v.data());
    up("F2", F2v.data());
    up("B0", B0v.data());
    up("B1", B1v.data());
    up("B2", B2v.data());

    nParticles_ += uint32_t(nNew);
    done++;
    emitter.center += emitter.center_vel * dt;
  }
}

// ── Push Constants 構築 ───────────────────────────────────────────────────

// init() 時に確定済みの値は pc_ が保持しているため、ここでは毎フレーム変わりうる値だけを上書きする。
MPMSimPC MPMEngine::buildPC(float subDt) const {
  MPMSimPC pc      = pc_;
  pc.particleCount = nParticles_;
  pc.dt            = subDt;
  pc.forceBufIdx   = forcesIdx_;
  pc.forceCount    = (uint32_t)forces_.size();
  pc.M_friction    = M_friction;
  pc.q_cohesion    = q_cohesion;
  pc.q_max         = q_max;
  pc.flip_ratio    = flip_ratio;
  pc.restitution   = restitution;
  pc.wall_friction = wall_friction;
  pc.plasticModel  = plasticModel;
  if(!enableColliderForceFeedback) {
    pc.colliderForceIdx  = 0;
    pc.colliderTorqueIdx = 0;
  }
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
  const uint32_t NC = pc_.hashCells;
  float subDt       = dt / float(std::max(1, numSubsteps));

  // コライダー反力/反トルクは全サブステップ分を積算してから1回だけ読み戻すため、フレーム先頭で一度だけゼロクリアする(enableColliderForceFeedback=false時はdispatch自体を発行せずGPU負荷を増やさない)
  if(enableColliderForceFeedback) {
    ZoneScopedN("ZeroColliderForce");
    VkDeviceSize forceBytes = VkDeviceSize(64) * sizeof(glm::vec4);
    vkCmdFillBuffer(cmd, attrBuf_.getBuffer("colliderForce"), 0, forceBytes, 0u);
    vkCmdFillBuffer(cmd, attrBuf_.getBuffer("colliderTorque"), 0, forceBytes, 0u);
    VkMemoryBarrier b{};
    b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
  }

  for(int sub = 0; sub < numSubsteps; ++sub) {
    MPMSimPC pc = buildPC(subDt);

    // ① グリッドバッファをゼロクリア
    {
      ZoneScopedN("ZeroGrid");
      dispatchMPM(cmd, kZeroGrid_, pc, NC);
      computeBarrier(cmd);
      syncGpuForProfiling(cmd);
    }

    // ② P2G (MLS-MPM: パーティクル並列scatter、固定小数点atomicAdd。空間ハッシュ不要)
    {
      ZoneScopedN("P2G");
      dispatchMPM(cmd, kP2G_, pc, N);
      computeBarrier(cmd);
      syncGpuForProfiling(cmd);
    }

    // ③ グリッド速度更新 (正規化 + 重力 + 壁BC)
    {
      ZoneScopedN("GridUpdate");
      dispatchMPM(cmd, kGridUpdate_, pc, NC);
      computeBarrier(cmd);
      syncGpuForProfiling(cmd);
    }

    // ④ G2P + F 更新 + 応力 + 位置更新
    {
      ZoneScopedN("G2P");
      dispatchMPM(cmd, kG2P_, pc, N);
      if(sub < numSubsteps - 1) computeBarrier(cmd);
      syncGpuForProfiling(cmd);
    }
  }
}
