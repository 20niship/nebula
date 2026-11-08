#include "PyroEngine.h"
#include "../core/Profiling.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

// ── バリア ────────────────────────────────────────────────────────────────

void PyroEngine::computeBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier b{};
  b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

// ── PyroSimPC を用いた dispatch ヘルパー ──────────────────────────────────

void PyroEngine::dispatchPyro(VkCommandBuffer cmd, ComputePipeline& k, const PyroSimPC& pc) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipelineLayout, 0, 1, &attrBuf_.descriptorSet, 0, nullptr);
  vkCmdPushConstants(cmd, k.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PyroSimPC), &pc);
  vkCmdDispatch(cmd, nGroups(), 1, 1);
}

void PyroEngine::dispatchAndBarrier(VkCommandBuffer cmd, ComputePipeline& k, const PyroSimPC& pc, const char* label) {
#ifdef NEBULA_GPU_PROFILING
  if(profEnabled_ && profQueryIndex_ + 1 < kProfMaxQueries) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, profPool_, profQueryIndex_);
#else
  (void)label;
#endif
  dispatchPyro(cmd, k, pc);
  computeBarrier(cmd);
#ifdef NEBULA_GPU_PROFILING
  if(profEnabled_ && profQueryIndex_ + 1 < kProfMaxQueries) {
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, profPool_, profQueryIndex_ + 1);
    profLabels_.push_back(label);
    profQueryIndex_ += 2;
  }
#endif
}

// ── 初期化 ────────────────────────────────────────────────────────────────

void PyroEngine::init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir) {
  // Morton符号化は10bit幅(軸あたり最大1024)のため、domainSize/cellSizeから導出されるcubeResがこれを超える場合のみ例外を投げる。
  if(cubeRes() > 1024) throw std::runtime_error("PyroEngine: domainSize/cellSize から導出される Morton cube 解像度が上限を超えています (cubeRes=" + std::to_string(cubeRes()) + " > 1024)");

  initEngineBase(device, allocator, descriptorPool, cmdPool, queue);

  const uint32_t NC = totalCells();
  if((channels & kChCurl) == 0 && pc_.vorticityEps > 0.0f) throw std::runtime_error("PyroEngine: vorticityEps > 0 requires kChCurl");
  // 無効なチャンネルは 1 要素のダミー (シェーダは channelMask で読み書きしない)
  const uint32_t nDens = (channels & kChDensity) ? NC : 1u, nTemp = (channels & kChTemperature) ? NC : 1u, nFuel = (channels & kChFuel) ? NC : 1u, nFlame = (channels & kChFlame) ? NC : 1u, nCurl = (channels & kChCurl) ? NC : 1u;

  velIdx_[0]         = attrBuf_.addAttribute("velA", sizeof(glm::vec4), NC);
  velIdx_[1]         = attrBuf_.addAttribute("velB", sizeof(glm::vec4), NC);
  densityIdx_[0]     = attrBuf_.addAttribute("densA", sizeof(float), nDens);
  densityIdx_[1]     = attrBuf_.addAttribute("densB", sizeof(float), nDens);
  temperatureIdx_[0] = attrBuf_.addAttribute("tempA", sizeof(float), nTemp);
  temperatureIdx_[1] = attrBuf_.addAttribute("tempB", sizeof(float), nTemp);
  fuelIdx_[0]        = attrBuf_.addAttribute("fuelA", sizeof(float), nFuel);
  fuelIdx_[1]        = attrBuf_.addAttribute("fuelB", sizeof(float), nFuel);
  flameIdx_          = attrBuf_.addAttribute("flame", sizeof(float), nFlame);
  pressureIdx_       = attrBuf_.addAttribute("pres", sizeof(float), NC);
  divergenceIdx_     = attrBuf_.addAttribute("div", sizeof(float), NC);
  curlIdx_           = attrBuf_.addAttribute("curl", sizeof(glm::vec4), nCurl);
  emittersIdx_       = attrBuf_.addAttribute("pyroEmitters", sizeof(EmitterGPU), maxEmitters);
  cur_               = 0;

  // pc_ の既定値 (呼び出し側は init() 後に pc_.xxx を直接上書きすればよい)
  pc_.buoyancyAlpha      = 1.2f;  // 温度浮力係数
  pc_.buoyancyBeta       = 0.4f;  // 密度による重さ (下降) 係数
  pc_.ambientTemp        = 0.0f;  // 環境温度
  pc_.vorticityEps       = 0.0f;  // 渦度閉じ込め強度
  pc_.densityDissipation = 0.05f; // 密度減衰係数 [1/s]
  pc_.tempDissipation    = 0.2f;  // 温度減衰係数 [1/s] (環境温度への復帰)
  pc_.ignitionTemp       = 0.0f;  // 発火温度
  pc_.burnRate           = 0.0f;  // 燃料消費速度 [1/s]
  pc_.heatRelease        = 0.0f;  // 燃焼による温度上昇量
  pc_.smokeYieldPerFuel  = 0.0f;  // 燃焼による密度生成量
  pc_.flameBrightness    = 0.0f;  // 燃焼による発光量

  // GPU バッファは未初期化のため、全フィールドを明示的にゼロクリアする
  {
    std::vector<glm::vec4> zeroVec4(NC, glm::vec4(0.0f));
    std::vector<float> zeroFloat((channels & (kChDensity | kChTemperature | kChFuel | kChFlame)) ? NC : 1u, 0.0f);
    auto up = [&](const std::string& name, const void* data, size_t bytes) { attrBuf_.upload(name, data, bytes, cmdPool_, queue_); };
    up("velA", zeroVec4.data(), NC * sizeof(glm::vec4));
    up("velB", zeroVec4.data(), NC * sizeof(glm::vec4));
    if(channels & kChDensity) {
      up("densA", zeroFloat.data(), NC * sizeof(float));
      up("densB", zeroFloat.data(), NC * sizeof(float));
    }
    if(channels & kChTemperature) {
      up("tempA", zeroFloat.data(), NC * sizeof(float));
      up("tempB", zeroFloat.data(), NC * sizeof(float));
    }
    if(channels & kChFuel) {
      up("fuelA", zeroFloat.data(), NC * sizeof(float));
      up("fuelB", zeroFloat.data(), NC * sizeof(float));
    }
    if(channels & kChFlame) up("flame", zeroFloat.data(), NC * sizeof(float));
    up("pres", std::vector<float>(NC, 0.0f).data(), NC * sizeof(float));
    up("div", std::vector<float>(NC, 0.0f).data(), NC * sizeof(float));
    if(channels & kChCurl) up("curl", zeroVec4.data(), NC * sizeof(glm::vec4));
  }

  initForces();

  auto load = [&](ComputePipeline& k, const char* name) { k.init(device, attrBuf_.descriptorSetLayout, shaderDir + "/" + name + ".spv"); };
  load(kEmit_, "pyro_emit.comp");
  load(kCombustion_, "pyro_combustion.comp");
  load(kObstacleBC_, "pyro_obstacle_bc.comp");
  load(kRotateSDF_, "pyro_rotate_sdf.comp");
  load(kAdvect_, "pyro_advect.comp");
  load(kCurl_, "pyro_curl.comp");
  load(kVorticityForce_, "pyro_vorticity_force.comp");
  load(kDivergence_, "pyro_divergence.comp");
  load(kPressureGS_, "pyro_pressure_gs.comp");
  load(kProject_, "pyro_project.comp");

  descriptorSetLayout = attrBuf_.descriptorSetLayout;
  descriptorSet       = attrBuf_.descriptorSet;
}

void PyroEngine::cleanup() {
  for(auto* k : {&kEmit_, &kCombustion_, &kForces_, &kObstacleBC_, &kRotateSDF_, &kAdvect_, &kCurl_, &kVorticityForce_, &kDivergence_, &kPressureGS_, &kProject_}) k->cleanup();
#ifdef NEBULA_GPU_PROFILING
  if(profPool_ != VK_NULL_HANDLE) vkDestroyQueryPool(device_, profPool_, nullptr);
#endif
  cleanupEngineBase();
}

// ── 障害物 SDF ────────────────────────────────────────────────────────────

void PyroEngine::setColliderSDF(const std::vector<float>& mortonSDF) {
  if(mortonSDF.size() != totalCells())
    throw std::runtime_error("PyroEngine::setColliderSDF: mortonSDF size (" + std::to_string(mortonSDF.size()) + ") must equal totalCells() (" + std::to_string(totalCells()) + ")");
  if(colliderSDFIdx_ == 0) {
    colliderSDFIdx_ = attrBuf_.addAttribute("colliderSDF", sizeof(float), totalCells());
  }
  attrBuf_.upload("colliderSDF", mortonSDF.data(), mortonSDF.size() * sizeof(float), cmdPool_, queue_);
}

void PyroEngine::clearCollider() {
  colliderSDFIdx_ = 0;
  pc_.rotSdfIdx   = 0;
}

void PyroEngine::setMovingCollider(const std::vector<float>& localSDF, glm::uvec3 res, float localCell, glm::vec3 localMin, const RigidBodyState& initial, bool autoAdvance) {
  if(localSDF.size() != size_t(res.x) * res.y * res.z) throw std::runtime_error("PyroEngine::setMovingCollider: localSDF size must equal res.x*res.y*res.z");
  if(res.x == 0 || res.y == 0 || res.z == 0 || res.x > 1023 || res.y > 1023 || res.z > 1023) throw std::runtime_error("PyroEngine::setMovingCollider: local SDF resolution per axis must be 1..1023");
  if(colliderSDFIdx_ == 0) colliderSDFIdx_ = attrBuf_.addAttribute("colliderSDF", sizeof(float), totalCells());
  pc_.rotSdfIdx = attrBuf_.addAttribute("rotSDF", sizeof(float), localSDF.size());
  attrBuf_.upload("rotSDF", localSDF.data(), localSDF.size() * sizeof(float), cmdPool_, queue_);
  pc_.rotResPacked = res.x | (res.y << 10u) | (res.z << 20u);
  pc_.rotCell      = localCell;
  pc_.rotMin[0]    = localMin.x;
  pc_.rotMin[1]    = localMin.y;
  pc_.rotMin[2]    = localMin.z;
  body_            = initial;
  bodyAutoAdvance_ = autoAdvance;
}

// ── Emitter ───────────────────────────────────────────────────────────────

void PyroEngine::addEmitter(std::shared_ptr<Emitter> emitter) {
  emitters_.push_back(std::move(emitter));
  emitterStepsDone_.push_back(0);
}

void PyroEngine::clearEmitters() {
  emitters_.clear();
  emitterStepsDone_.clear();
}

void PyroEngine::updateEmitters(float dt) {
  emittersActiveCount_ = 0;
  if(emitters_.empty()) return;

  std::vector<EmitterGPU> active;
  active.reserve(emitters_.size());

  for(size_t si = 0; si < emitters_.size(); si++) {
    Emitter& emitter = *emitters_[si];
    int& done        = emitterStepsDone_[si];

    bool shouldEmit;
    if(emitter.step_count == -1)
      shouldEmit = (done == 0);
    else if(emitter.step_count == 0)
      shouldEmit = true;
    else
      shouldEmit = (done < emitter.step_count);

    if(!shouldEmit) continue;
    if(active.size() < maxEmitters) active.push_back(emitter.pack());
    done++;
    emitter.center += emitter.center_vel * dt;
  }

  if(active.empty()) return;
  attrBuf_.upload("pyroEmitters", active.data(), active.size() * sizeof(EmitterGPU), cmdPool_, queue_);
  emittersActiveCount_ = uint32_t(active.size());
}

// ── GPUパス単位プロファイリング ──────────────────────────────────────────
#ifdef NEBULA_GPU_PROFILING

void PyroEngine::enableGpuProfiling(VkPhysicalDevice physicalDevice) {
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

void PyroEngine::printGpuProfile() {
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
  std::fprintf(stderr, "=== [PyroEngine GPU profile] total=%.4f ms ===\n", total);
  for(const auto& [label, ms] : sorted) {
    std::fprintf(stderr, "  %-26s %9.4f ms  (%5.1f%%, x%d)\n", label.c_str(), ms, ms / total * 100.0, counts[label]);
  }
}

#endif // NEBULA_GPU_PROFILING

// ── Push Constants 構築 ──────────────────────────────────────────────────

PyroSimPC PyroEngine::buildPC(float dt) const {
  PyroSimPC pc       = pc_; // 物理パラメータ(buoyancyAlpha等)はpc_に設定済みの値をそのまま使う
  pc.velIdxA         = velIdx_[cur_];
  pc.velIdxB         = velIdx_[1 - cur_];
  pc.densityIdxA     = densityIdx_[cur_];
  pc.densityIdxB     = densityIdx_[1 - cur_];
  pc.temperatureIdxA = temperatureIdx_[cur_];
  pc.temperatureIdxB = temperatureIdx_[1 - cur_];
  pc.fuelIdxA        = fuelIdx_[cur_];
  pc.fuelIdxB        = fuelIdx_[1 - cur_];
  pc.flameIdx        = flameIdx_;
  pc.pressureIdxA    = pressureIdx_;
  pc.gsColor         = 0;
  pc.divergenceIdx   = divergenceIdx_;
  pc.curlIdx         = curlIdx_;
  pc.channelMask     = channels;
  const glm::quat q  = glm::normalize(body_.rotation);
  for(int k = 0; k < 3; k++) {
    pc.bodyPos[k]    = body_.position[k];
    pc.bodyLinVel[k] = body_.linearVelocity[k];
    pc.bodyAngVel[k] = body_.angularVelocity[k];
  }
  pc.bodyQuat[0] = q.x;
  pc.bodyQuat[1] = q.y;
  pc.bodyQuat[2] = q.z;
  pc.bodyQuat[3] = q.w;
  pc.colliderSDFIdx  = colliderSDFIdx_;
  // emittersIdx/emitterCount は updateEmitters() 後に step() 側で設定する (0=無効)
  pc.emittersIdx  = 0;
  pc.emitterCount = 0;
  pc.hashCells    = totalCells();
  pc.gridRes      = gridRes();

  pc.dt       = dt;
  pc.cellSize = cellSize;
  pc.worldMin = glm::vec3(0.0f);
  pc.worldMax = domainSize;

  pc.forceBufIdx = forcesIdx_;
  pc.forceCount  = (uint32_t)forces_.size();
  return pc;
}

// ── ステップ ──────────────────────────────────────────────────────────────

void PyroEngine::step(VkCommandBuffer cmd, float dt) {
  ZoneScoped;
  FrameMark;
  uploadForces(dt);

  const float subDt = dt / float(std::max(1, numSubsteps));

#ifdef NEBULA_GPU_PROFILING
  if(profEnabled_) {
    vkCmdResetQueryPool(cmd, profPool_, 0, kProfMaxQueries);
    profLabels_.clear();
  }
  profQueryIndex_ = 0;
#endif

  for(int s = 0; s < numSubsteps; s++) {
    updateEmitters(subDt);

    PyroSimPC pc = buildPC(subDt);
    if(emittersActiveCount_ > 0) {
      pc.emittersIdx  = emittersIdx_;
      pc.emitterCount = emittersActiveCount_;
    }

    // ⓪ 回転コライダー: colliderSDF を現在の角度で焼き直す
    if(pc.rotSdfIdx != 0) dispatchAndBarrier(cmd, kRotateSDF_, pc, "rotate_sdf");

    // ① Emitter 注入 (A バッファへインプレース)
    dispatchAndBarrier(cmd, kEmit_, pc, "emit");

    // ①b 燃焼反応 (A バッファへインプレース)
    if((channels & (kChFuel | kChTemperature | kChFlame | kChDensity)) == (kChFuel | kChTemperature | kChFlame | kChDensity)) dispatchAndBarrier(cmd, kCombustion_, pc, "combustion");

    // ② 浮力 (A バッファへインプレース)
    dispatchAndBarrier(cmd, kForces_, pc, "forces");

    // ②b 渦度閉じ込め (2パス: curl → 閉じ込め力適用、A バッファへインプレース)
    if(pc.vorticityEps > 0.0f && (channels & kChCurl)) {
      dispatchAndBarrier(cmd, kCurl_, pc, "curl");
      dispatchAndBarrier(cmd, kVorticityForce_, pc, "vorticity_force");
    }

    // ③ 障害物 BC (投影前, A バッファへインプレース)
    dispatchAndBarrier(cmd, kObstacleBC_, pc, "obstacle_bc_pre");

    // ③b 圧力投影 (非圧縮化): divergence → Red-Black Gauss-Seidel 反復 → project
    dispatchAndBarrier(cmd, kDivergence_, pc, "divergence");

    for(int sweep = 0; sweep < numPressureIters; sweep++) {
      for(uint32_t color = 0; color < 2; color++) {
        PyroSimPC ppc = pc;
        ppc.gsColor   = color;
        dispatchAndBarrier(cmd, kPressureGS_, ppc, "pressure_gs");
      }
    }
    dispatchAndBarrier(cmd, kProject_, pc, "project");

    // ③c 障害物 BC (投影後, A バッファへインプレース)
    dispatchAndBarrier(cmd, kObstacleBC_, pc, "obstacle_bc_post_proj");

    // ④ 移流 (A→B)
    dispatchAndBarrier(cmd, kAdvect_, pc, "advect");

    // ⑤ 障害物 BC (移流後, B バッファへインプレース)
    PyroSimPC pcPost = pc;
    pcPost.velIdxA   = pc.velIdxB;
    dispatchAndBarrier(cmd, kObstacleBC_, pcPost, "obstacle_bc_post_advect");

    // ダブルバッファ入れ替え: 次フレームは B が現在値になる
    cur_ = 1 - cur_;
    if(bodyAutoAdvance_ && pc_.rotSdfIdx != 0) { // 移動コライダーを一定の線速度・角速度で進める (次サブステップの buildPC に反映)
      body_.position += body_.linearVelocity * subDt;
      const float w = glm::length(body_.angularVelocity);
      if(w > 0.0f) body_.rotation = glm::normalize(glm::angleAxis(w * subDt, body_.angularVelocity / w) * body_.rotation);
    }
  }
}

// ── バッファ取得 ──────────────────────────────────────────────────────────

VkBuffer PyroEngine::getDensityBuffer() const { return attrBuf_.getBuffer(cur_ == 0 ? "densA" : "densB"); }
VkBuffer PyroEngine::getTemperatureBuffer() const { return attrBuf_.getBuffer(cur_ == 0 ? "tempA" : "tempB"); }
VkBuffer PyroEngine::getFuelBuffer() const { return attrBuf_.getBuffer(cur_ == 0 ? "fuelA" : "fuelB"); }
VkBuffer PyroEngine::getFlameBuffer() const { return attrBuf_.getBuffer("flame"); }
VkBuffer PyroEngine::getVelocityBuffer() const { return attrBuf_.getBuffer(cur_ == 0 ? "velA" : "velB"); }

// ── ボクセルダンプ ────────────────────────────────────────────────────────

void PyroEngine::readBufferToCPU(VkBuffer src, void* dst, size_t bytes) const {
  VkBufferCreateInfo bci{};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size  = VkDeviceSize(bytes);
  bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
  aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VkBuffer stageBuf;
  VmaAllocation stageAlloc;
  VmaAllocationInfo info;
  vmaCreateBuffer(allocator_, &bci, &aci, &stageBuf, &stageAlloc, &info);

  VkCommandBufferAllocateInfo ai{};
  ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool        = cmdPool_;
  ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(device_, &ai, &cmd);

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);

  VkBufferCopy region{0, 0, VkDeviceSize(bytes)};
  vkCmdCopyBuffer(cmd, src, stageBuf, 1, &region);

  vkEndCommandBuffer(cmd);
  VkSubmitInfo si{};
  si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers    = &cmd;
  vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue_);
  vkFreeCommandBuffers(device_, cmdPool_, 1, &cmd);

  std::memcpy(dst, info.pMappedData, bytes);
  vmaDestroyBuffer(allocator_, stageBuf, stageAlloc);
}

namespace {

// GPU 側 pyroMortonCompact/pyroMortonDecodeI (shaders/pyro_common.glsl) と同一ロジック
uint32_t mortonCompact(uint32_t v) {
  v &= 0x09249249u;
  v = (v | (v >> 2u)) & 0x030C30C3u;
  v = (v | (v >> 4u)) & 0x0300F00Fu;
  v = (v | (v >> 8u)) & 0x030000FFu;
  v = (v | (v >> 16u)) & 0x000003FFu;
  return v;
}
glm::ivec3 mortonDecode(uint32_t code) { return {int(mortonCompact(code)), int(mortonCompact(code >> 1u)), int(mortonCompact(code >> 2u))}; }

// Morton 順(cubeCells要素、パディング含む)の配列から実セル範囲(realRes)内のみを取り出し、
// 線形 (x + y*nx + z*nx*ny) 順に並べ替えて channelsOut へ追記する(パディングセルは除外)。
// components=1: float スカラー。components=3: vec4 の xyz のみ (w は捨てる)。
void appendChannelLinear(const std::vector<uint8_t>& mortonRaw, const glm::uvec3& realRes, uint32_t cubeCells, uint32_t components, std::vector<float>& out) {
  out.resize(size_t(realRes.x) * realRes.y * realRes.z * components);
  const float* src = reinterpret_cast<const float*>(mortonRaw.data());
  for(uint32_t code = 0; code < cubeCells; ++code) {
    glm::ivec3 c = mortonDecode(code);
    if(uint32_t(c.x) >= realRes.x || uint32_t(c.y) >= realRes.y || uint32_t(c.z) >= realRes.z) continue; // パディングセルをスキップ
    size_t lin = size_t(c.x) + size_t(c.y) * realRes.x + size_t(c.z) * realRes.x * realRes.y;
    for(uint32_t k = 0; k < components; ++k) out[lin * components + k] = src[size_t(code) * (components == 1 ? 1u : 4u) + k];
  }
}

} // namespace

void PyroEngine::dumpFrame(const std::string& path, float simTime) const {
  const glm::uvec3 realRes   = gridRes();
  const uint32_t cubeCells   = totalCells();
  const uint32_t realCellCnt = realRes.x * realRes.y * realRes.z;

  // sdf は障害物未設定 (hasCollider()==false) だと GPU バッファが存在しないため、
  // その場合のみ GPU 読み戻しをスキップし全セル背景値 (1e6 = 障害物なし) を書く。
  struct Channel {
    const char* name;
    uint32_t components;
    VkBuffer buf;
    bool present;
  };
  const Channel chans[] = {
    {"density", 1u, getDensityBuffer(), (channels & kChDensity) != 0},   {"temperature", 1u, getTemperatureBuffer(), (channels & kChTemperature) != 0}, {"fuel", 1u, getFuelBuffer(), (channels & kChFuel) != 0},
    {"flame", 1u, getFlameBuffer(), (channels & kChFlame) != 0},         {"velocity", 3u, getVelocityBuffer(), true},                                    {"sdf", 1u, hasCollider() ? attrBuf_.getBuffer("colliderSDF") : VK_NULL_HANDLE, hasCollider()},
  };

  std::vector<std::vector<float>> linearData(std::size(chans));
  for(size_t i = 0; i < std::size(chans); ++i) {
    const auto& ch = chans[i];
    if(!ch.present) {
      linearData[i].assign(size_t(realCellCnt), 1e6f);
      continue;
    }
    const VkDeviceSize elemBytes = ch.components == 1 ? sizeof(float) : sizeof(glm::vec4);
    std::vector<uint8_t> raw(size_t(cubeCells) * elemBytes);
    readBufferToCPU(ch.buf, raw.data(), raw.size());
    appendChannelLinear(raw, realRes, cubeCells, ch.components, linearData[i]);
  }

  std::ofstream f(path, std::ios::binary);
  if(!f) throw std::runtime_error("dumpFrame: failed to open " + path);

  f.write("PVX1", 4);
  f.write(reinterpret_cast<const char*>(&realRes.x), sizeof(uint32_t));
  f.write(reinterpret_cast<const char*>(&realRes.y), sizeof(uint32_t));
  f.write(reinterpret_cast<const char*>(&realRes.z), sizeof(uint32_t));
  float cs = cellSize; // 旧 worldSize(単一float)。直方体対応につき意味を cellSize に変更 (バイト数不変)
  f.write(reinterpret_cast<const char*>(&cs), sizeof(float));
  f.write(reinterpret_cast<const char*>(&simTime), sizeof(float));
  uint32_t numChannels = uint32_t(std::size(chans));
  f.write(reinterpret_cast<const char*>(&numChannels), sizeof(uint32_t));

  for(const auto& ch : chans) {
    char name[16] = {};
    std::strncpy(name, ch.name, sizeof(name) - 1);
    f.write(name, sizeof(name));
    f.write(reinterpret_cast<const char*>(&ch.components), sizeof(uint32_t));
  }
  for(const auto& data : linearData) f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size() * sizeof(float)));
}
