#include "ParticleGridEngine.h"
#include "../core/DefineShaderCompiler.h"
#include "../core/Profiling.h"
#include "BoundaryParticles.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <glm/glm.hpp>
#include <random>
#include <stdexcept>
#include <vector>

void ParticleGridEngine::syncGpuForProfiling(VkCommandBuffer cmd) {
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

void ParticleGridEngine::computeBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier b{};
  b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

// ComputePipeline::dispatch は SimPC を取るため、MPMSimPC は生バイトで渡す
void ParticleGridEngine::dispatchMPM(VkCommandBuffer cmd, ComputePipeline& k, const MPMSimPC& pc, uint32_t count) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipelineLayout, 0, 1, &attrBuf_.descriptorSet, 0, nullptr);
  vkCmdPushConstants(cmd, k.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MPMSimPC), &pc);
  uint32_t groups = (count + 255u) / 256u;
  vkCmdDispatch(cmd, groups, 1, 1);
}

void ParticleGridEngine::setFixedPointScale(float scale) {
  if(!std::isfinite(scale) || scale <= 0.0f)
    throw std::invalid_argument("fixed-point scale must be finite and positive");
  if(kP2G_.pipeline != VK_NULL_HANDLE)
    throw std::runtime_error("fixed-point scale must be set before init()");
  userDefines_.erase(std::remove_if(userDefines_.begin(), userDefines_.end(), [](const auto& d) { return d.first == "FIXED_POINT_SCALE"; }), userDefines_.end());
  userDefines_.push_back({"FIXED_POINT_SCALE", std::to_string(scale)});
}

MaterialParams ParticleGridEngine::defaultMaterial() const {
  MaterialParams m{};
  m.rho0 = rho0;
  return m;
}

// ── 初期化 ────────────────────────────────────────────────────────────────

void ParticleGridEngine::init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir) {
  initEngineBase(device, allocator, descriptorPool, cmdPool, queue);

  const glm::uvec3 gr = domain::gridRes(domainSize, cellSize);
  const uint32_t NC   = domain::hashCellsCube(gr);

  // maxParticles を明示指定した場合は自動シードを行わない (appendParticles/emitter で後から追加する用途)
  const uint32_t seedCount = maxParticles > 0 ? 0 : defaultSeedCount(gr);
  maxParticles_            = maxParticles > 0 ? maxParticles : seedCount;
  if(maxParticles_ == 0) throw std::invalid_argument("maxParticles must be set");
  nParticles_      = seedCount;
  const uint32_t N = maxParticles_;

  posIdx          = attrBuf_.addAttribute("P", sizeof(glm::vec4), N);
  velIdx          = attrBuf_.addAttribute("v", sizeof(glm::vec4), N);
  pc_.posIdx      = posIdx;
  pc_.velIdx      = velIdx;
  pc_.typeFlagIdx = 0;

  // gridMom は 2 × NC を確保: [0, NC) = v_new, [NC, 2*NC) = v_old (FLIP 用)
  pc_.gridMomIdx  = attrBuf_.addAttribute("gridMom", sizeof(glm::vec4), NC * 2);
  pc_.gridMassIdx = attrBuf_.addAttribute("gridMass", sizeof(float), NC);
  pc_.hashCells   = NC;
  pc_.gridRes     = gr;
  pc_.cellSize    = cellSize;
  pc_.worldMax    = domainSize; // ドメイン下限は常に原点固定 (worldMin は廃止)
  pc_.rho0        = rho0;
  pc_.p0_mcc      = 0.0f;
  pc_.xi_hard     = 0.0f;

  {
    std::vector<MaterialParams> mats(1, defaultMaterial());
    pc_.materialsIdx  = attrBuf_.addAttribute("materials", sizeof(MaterialParams), 16);
    pc_.materialCount = 1;
    attrBuf_.upload("materials", mats.data(), sizeof(MaterialParams), cmdPool_, queue_);
  }

  // 解析コライダー SSBO: 最大64個のプリミティブを事前確保 (colliderCount == 0 で無効)
  pc_.colliderIdx       = attrBuf_.addAttribute("colliders", sizeof(ColliderPrimitive), 64);
  pc_.colliderCount     = 0;
  pc_.colliderForceIdx  = attrBuf_.addAttribute("colliderForce", sizeof(glm::vec4), 64);
  pc_.colliderTorqueIdx = attrBuf_.addAttribute("colliderTorque", sizeof(glm::vec4), 64);

  allocParticleExtras(N, gr);

  extraForceDefines_ = userDefines_;
  for(auto& d : shaderDefines()) extraForceDefines_.push_back(d);
  initForces();

  auto load = [&](ComputePipeline& k, const char* name) { k.init(device, attrBuf_.descriptorSetLayout, shaderDir + "/" + name + ".spv"); };
  load(kZeroGrid_, "mpm_zero_grid.comp");
  kP2G_.initFromSpirv(device_, attrBuf_.descriptorSetLayout, DefineShaderCompiler::compile("mpm_p2g.comp", extraForceDefines_));
  kG2P_.initFromSpirv(device_, attrBuf_.descriptorSetLayout, DefineShaderCompiler::compile("mpm_g2p.comp", extraForceDefines_));

  descriptorSetLayout = attrBuf_.descriptorSetLayout;
  descriptorSet       = attrBuf_.descriptorSet;

  onInitParticles(shaderDir);
}

void ParticleGridEngine::cleanup() {
  cleanupDerived();
  for(auto* k : {&kZeroGrid_, &kP2G_, &kGridUpdate_, &kG2P_}) k->cleanup();
  cleanupEngineBase();
}

VkBuffer ParticleGridEngine::getPositionBuffer() const { return attrBuf_.getBuffer("P"); }
VkBuffer ParticleGridEngine::getVelocityBuffer() const { return attrBuf_.getBuffer("v"); }
VkBuffer ParticleGridEngine::getColliderForceBuffer() const { return attrBuf_.getBuffer("colliderForce"); }
VkBuffer ParticleGridEngine::getColliderTorqueBuffer() const { return attrBuf_.getBuffer("colliderTorque"); }

// ── メッシュSDFコライダー ────────────────────────────────────────────────

uint32_t ParticleGridEngine::loadColliderMesh(const std::string& objPath, LocalMeshSDF& gridOut, uint32_t res, float scale) {
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

uint32_t ParticleGridEngine::uploadColliderMeshSDF(const LocalMeshSDF& grid) {
  std::string name = "meshSDF_" + std::to_string(nextMeshSDFId_++);
  uint32_t idx     = attrBuf_.addAttribute(name, sizeof(float), grid.data.size());
  attrBuf_.upload(name, grid.data.data(), grid.data.size() * sizeof(float), cmdPool_, queue_);
  return idx;
}

// ── 粒子追加 ─────────────────────────────────────────────────────────────

void ParticleGridEngine::uploadNewParticles_(const glm::vec4* pos, const glm::vec4* vel, uint32_t count, uint32_t first) {
  VkDeviceSize byteOff = VkDeviceSize(first) * sizeof(glm::vec4);
  attrBuf_.uploadAt("P", pos, VkDeviceSize(count) * sizeof(glm::vec4), byteOff, cmdPool_, queue_);
  attrBuf_.uploadAt("v", vel, VkDeviceSize(count) * sizeof(glm::vec4), byteOff, cmdPool_, queue_);
  initNewParticleExtras(first, count);
}

void ParticleGridEngine::appendParticles(const std::vector<glm::vec4>& pos, const std::vector<glm::vec4>& vel) {
  const uint32_t nNew = std::min(uint32_t(pos.size()), maxParticles_ - nParticles_);
  if(nNew == 0) return;
  uploadNewParticles_(pos.data(), vel.data(), nNew, nParticles_);
  nParticles_ += nNew;
}

// ── マテリアルテーブル ────────────────────────────────────────────────────

void ParticleGridEngine::setMaterials(const std::vector<MaterialParams>& mats) {
  if(mats.empty()) return;
  pc_.materialCount = uint32_t(mats.size());
  attrBuf_.upload("materials", mats.data(), mats.size() * sizeof(MaterialParams), cmdPool_, queue_);
}

void ParticleGridEngine::setParticleMaterialIds(const std::vector<uint32_t>& matIds) {
  const uint32_t N = uint32_t(matIds.size());
  // vel バッファ全体を (0, 0, 0, floatBitsToUint(matId)) で上書き
  std::vector<glm::vec4> vel(N);
  for(uint32_t i = 0; i < N; i++) {
    float wf;
    std::memcpy(&wf, &matIds[i], sizeof(float));
    vel[i] = glm::vec4(0.0f, 0.0f, 0.0f, wf);
  }
  attrBuf_.upload("v", vel.data(), N * sizeof(glm::vec4), cmdPool_, queue_);
}

// ── 解析コライダー ────────────────────────────────────────────────────────

void ParticleGridEngine::setColliders(const ColliderSet& cols) {
  ZoneScoped;
  pc_.colliderCount = cols.count();
  pendingColliders_ = cols.data();
}

void ParticleGridEngine::clearAnalyticColliders() {
  pc_.colliderCount = 0;
  pendingColliders_.clear();
}

// ── Emitter ───────────────────────────────────────────────────────────────

void ParticleGridEngine::addEmitter(std::shared_ptr<Emitter> emitter) {
  emitters_.push_back(std::move(emitter));
  emitterStepsDone_.push_back(0);
}

void ParticleGridEngine::clearEmitters() {
  emitters_.clear();
  emitterStepsDone_.clear();
}

void ParticleGridEngine::resetParticles() {
  if(device_ == VK_NULL_HANDLE) return;
  vkDeviceWaitIdle(device_);
  nParticles_ = 0;
  std::fill(emitterStepsDone_.begin(), emitterStepsDone_.end(), 0);
  emitterRng_.seed(12345);
}

void ParticleGridEngine::emitFromEmitters(float dt) {
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

    const float Vp = emitter.particleVolume > 0.0f ? emitter.particleVolume : cellSize * cellSize * cellSize;
    std::vector<glm::vec4> pos(nNew), vel(nNew);

    for(int j = 0; j < nNew; j++) pos[j] = glm::vec4(emitter.sample(emitterRng_), Vp);

    // 同じ格子セルの粒子を近くに配置し、P2G のアクセスを局所化する。
    if(nNew > 1024) {
      struct KeyedPosition { uint32_t key; glm::vec4 position; };
      const glm::uvec3 res = domain::gridRes(domainSize, cellSize);
      std::vector<KeyedPosition> keyed;
      keyed.reserve(nNew);
      for(const auto& p : pos) {
        const uint32_t x = uint32_t(std::clamp(int(p.x / cellSize), 0, int(res.x) - 1));
        const uint32_t y = uint32_t(std::clamp(int(p.y / cellSize), 0, int(res.y) - 1));
        const uint32_t z = uint32_t(std::clamp(int(p.z / cellSize), 0, int(res.z) - 1));
        keyed.push_back({(z * res.y + y) * res.x + x, p});
      }
      std::sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
      for(int j = 0; j < nNew; ++j) pos[j] = keyed[j].position;
    }

    for(int j = 0; j < nNew; j++) vel[j] = glm::vec4(emitter.sample_velocity(glm::vec3(pos[j]), emitterRng_), matIdF);

    uploadNewParticles_(pos.data(), vel.data(), uint32_t(nNew), nParticles_);
    nParticles_ += uint32_t(nNew);
    done++;
    emitter.center += emitter.center_vel * dt;
  }
}

// ── Push Constants 構築 ───────────────────────────────────────────────────

// init() 時に確定済みの値は pc_ が保持しているため、ここでは毎フレーム変わりうる値だけを上書きする。
MPMSimPC ParticleGridEngine::buildPC(float subDt) const {
  MPMSimPC pc      = pc_;
  pc.particleCount = nParticles_;
  pc.dt            = subDt;
  pc.forceBufIdx   = forcesIdx_;
  pc.forceCount    = (uint32_t)forces_.size();
  pc.flip_ratio    = flip_ratio;
  pc.restitution   = restitution;
  pc.wall_friction = wall_friction;
  if(!enableColliderForceFeedback) {
    pc.colliderForceIdx  = 0;
    pc.colliderTorqueIdx = 0;
  }
  fillPC(pc);
  return pc;
}

// ── ステップ ──────────────────────────────────────────────────────────────

void ParticleGridEngine::step(VkCommandBuffer cmd, float dt) {
  ZoneScoped;
  FrameMark;
  {
    ZoneScopedN("EmitFromEmitters");
    emitFromEmitters(dt);
  }

  // 前ステップの計算が終わってから転送と次の計算を始める。
  VkMemoryBarrier stepBarrier{};
  stepBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  stepBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  stepBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 1, &stepBarrier, 0, nullptr, 0, nullptr);

  {
    ZoneScopedN("UploadForces");
    std::vector<ForceGPU> packed;
    packed.reserve(forces_.size());
    for(const auto& force : forces_) {
      force->advanceTime(dt);
      packed.push_back(force->pack());
    }
    const bool uploaded = !packed.empty() || !pendingColliders_.empty();
    if(!packed.empty())
      vkCmdUpdateBuffer(cmd, attrBuf_.getBuffer("forces"), 0, sizeof(ForceGPU) * packed.size(), packed.data());
    if(!pendingColliders_.empty()) {
      vkCmdUpdateBuffer(cmd, attrBuf_.getBuffer("colliders"), 0,
                        sizeof(ColliderPrimitive) * pendingColliders_.size(), pendingColliders_.data());
      pendingColliders_.clear();
    }
    if(uploaded) {
      VkMemoryBarrier b{};
      b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 1, &b, 0, nullptr, 0, nullptr);
    }
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

    {
      ZoneScopedN("ZeroGrid");
      dispatchMPM(cmd, kZeroGrid_, pc, NC);
      computeBarrier(cmd);
      syncGpuForProfiling(cmd);
    }

    // P2G (MLS-MPM: パーティクル並列scatter、固定小数点atomicAdd。空間ハッシュ不要)
    {
      ZoneScopedN("P2G");
      dispatchMPM(cmd, kP2G_, pc, N);
      computeBarrier(cmd);
      syncGpuForProfiling(cmd);
    }

    // グリッド速度更新 (正規化 + Force + 壁BC + コライダーBC)
    {
      ZoneScopedN("GridUpdate");
      dispatchMPM(cmd, kGridUpdate_, pc, NC);
      computeBarrier(cmd);
      syncGpuForProfiling(cmd);
    }

    {
      ZoneScopedN("ProjectGrid");
      projectGrid(cmd, pc);
    }

    {
      ZoneScopedN("G2P");
      dispatchMPM(cmd, kG2P_, pc, N);
      computeBarrier(cmd);
      syncGpuForProfiling(cmd);
    }

    {
      ZoneScopedN("AfterG2P");
      afterG2P(cmd, pc);
    }
  }
}
