#include "PyroEngine.h"

#include <algorithm>
#include <cstring>
#include <fstream>
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
  vkCmdDispatch(cmd, cfg_.nGroups(), 1, 1);
}

// ── 初期化 ────────────────────────────────────────────────────────────────

void PyroEngine::init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descriptorPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir, const PyroConfig& cfg) {
  if(cfg.grid_res == 0 || (cfg.grid_res & (cfg.grid_res - 1)) != 0) throw std::runtime_error("PyroConfig.grid_res must be a power of two (Morton encoding requirement): " + std::to_string(cfg.grid_res));

  cfg_       = cfg;
  device_    = device;
  allocator_ = allocator;
  cmdPool_   = cmdPool;
  queue_     = queue;

  attrBuf_.init(device, allocator, descriptorPool);

  const uint32_t NC = cfg_.totalCells();

  velIdx_[0]         = attrBuf_.addAttribute("velA",  sizeof(glm::vec4), NC);
  velIdx_[1]         = attrBuf_.addAttribute("velB",  sizeof(glm::vec4), NC);
  densityIdx_[0]     = attrBuf_.addAttribute("densA", sizeof(float),     NC);
  densityIdx_[1]     = attrBuf_.addAttribute("densB", sizeof(float),     NC);
  temperatureIdx_[0] = attrBuf_.addAttribute("tempA", sizeof(float),     NC);
  temperatureIdx_[1] = attrBuf_.addAttribute("tempB", sizeof(float),     NC);
  fuelIdx_[0]        = attrBuf_.addAttribute("fuelA", sizeof(float),     NC);
  fuelIdx_[1]        = attrBuf_.addAttribute("fuelB", sizeof(float),     NC);
  flameIdx_          = attrBuf_.addAttribute("flame", sizeof(float),     NC);
  pressureIdx_       = attrBuf_.addAttribute("pres",  sizeof(float),     NC);
  divergenceIdx_     = attrBuf_.addAttribute("div",   sizeof(float),     NC);
  curlIdx_           = attrBuf_.addAttribute("curl",  sizeof(glm::vec4), NC);
  emittersIdx_       = attrBuf_.addAttribute("pyroEmitters", sizeof(EmitterGPU), cfg_.maxEmitters);
  cur_ = 0;

  // GPU バッファは未初期化のため、全フィールドを明示的にゼロクリアする
  {
    std::vector<glm::vec4> zeroVec4(NC, glm::vec4(0.0f));
    std::vector<float> zeroFloat(NC, 0.0f);
    auto up = [&](const std::string& name, const void* data, size_t bytes) { attrBuf_.upload(name, data, bytes, cmdPool_, queue_); };
    up("velA", zeroVec4.data(), NC * sizeof(glm::vec4));
    up("velB", zeroVec4.data(), NC * sizeof(glm::vec4));
    up("densA", zeroFloat.data(), NC * sizeof(float));
    up("densB", zeroFloat.data(), NC * sizeof(float));
    up("tempA", zeroFloat.data(), NC * sizeof(float));
    up("tempB", zeroFloat.data(), NC * sizeof(float));
    up("fuelA", zeroFloat.data(), NC * sizeof(float));
    up("fuelB", zeroFloat.data(), NC * sizeof(float));
    up("flame", zeroFloat.data(), NC * sizeof(float));
    up("pres",  zeroFloat.data(), NC * sizeof(float));
    up("div",   zeroFloat.data(), NC * sizeof(float));
    up("curl",  zeroVec4.data(),  NC * sizeof(glm::vec4));
  }

  auto load = [&](ComputePipeline& k, const char* name) {
    k.init(device, attrBuf_.descriptorSetLayout, shaderDir + "/" + name + ".spv");
  };
  load(kEmit_,           "pyro_emit.comp");
  load(kCombustion_,     "pyro_combustion.comp");
  load(kForces_,         "pyro_forces.comp");
  load(kObstacleBC_,     "pyro_obstacle_bc.comp");
  load(kAdvect_,         "pyro_advect.comp");
  load(kAdvectMC_,       "pyro_advect_mc.comp");
  load(kCurl_,           "pyro_curl.comp");
  load(kVorticityForce_, "pyro_vorticity_force.comp");
  load(kDivergence_,     "pyro_divergence.comp");
  load(kPressureGS_,     "pyro_pressure_gs.comp");
  load(kProject_,        "pyro_project.comp");

  descriptorSetLayout = attrBuf_.descriptorSetLayout;
  descriptorSet       = attrBuf_.descriptorSet;
}

void PyroEngine::cleanup() {
  for (auto* k : {&kEmit_, &kCombustion_, &kForces_, &kObstacleBC_, &kAdvect_, &kAdvectMC_,
                  &kCurl_, &kVorticityForce_, &kDivergence_, &kPressureGS_, &kProject_})
    k->cleanup();
  attrBuf_.cleanup();
}

// ── 障害物 SDF ────────────────────────────────────────────────────────────

void PyroEngine::setColliderSDF(const std::vector<float>& mortonSDF) {
  if(colliderSDFIdx_ == 0) {
    colliderSDFIdx_ = attrBuf_.addAttribute("colliderSDF", sizeof(float), cfg_.totalCells());
  }
  attrBuf_.upload("colliderSDF", mortonSDF.data(), mortonSDF.size() * sizeof(float), cmdPool_, queue_);
}

void PyroEngine::clearCollider() { colliderSDFIdx_ = 0; }

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
    if(active.size() < cfg_.maxEmitters) active.push_back(emitter.pack());
    done++;
    emitter.center += emitter.center_vel * dt;
  }

  if(active.empty()) return;
  attrBuf_.upload("pyroEmitters", active.data(), active.size() * sizeof(EmitterGPU), cmdPool_, queue_);
  emittersActiveCount_ = uint32_t(active.size());
}

// ── Push Constants 構築 ──────────────────────────────────────────────────

PyroSimPC PyroEngine::buildPC(float dt) const {
  PyroSimPC pc{};
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
  pc.colliderSDFIdx  = colliderSDFIdx_;
  // emittersIdx/emitterCount は updateEmitters() 後に step() 側で設定する (0=無効)
  pc.emittersIdx  = 0;
  pc.emitterCount = 0;
  pc.gridRes      = cfg_.grid_res;

  pc.dt       = dt;
  pc.cellSize = cfg_.cellSize();
  pc.worldMin = 0.0f;
  pc.worldMax = cfg_.world_size;

  pc.buoyancyAlpha = buoyancyAlpha;
  pc.buoyancyBeta  = buoyancyBeta;
  pc.ambientTemp   = ambientTemp;
  pc.vorticityEps  = vorticityEps;

  pc.densityDissipation = densityDissipation;
  pc.tempDissipation    = tempDissipation;
  pc.velocityDissipation = velocityDissipation;
  pc.maxVelocity         = maxVelocity;
  pc.ignitionTemp       = ignitionTemp;
  pc.burnRate           = burnRate;

  pc.heatRelease       = heatRelease;
  pc.smokeYieldPerFuel = smokeYieldPerFuel;
  pc.flameBrightness   = flameBrightness;
  return pc;
}

// ── ステップ ──────────────────────────────────────────────────────────────

void PyroEngine::step(VkCommandBuffer cmd, float dt) {
  const float subDt = dt / float(std::max(1, numSubsteps));

  for(int s = 0; s < numSubsteps; s++) {
    updateEmitters(subDt);

    PyroSimPC pc = buildPC(subDt);
    if(emittersActiveCount_ > 0) {
      pc.emittersIdx  = emittersIdx_;
      pc.emitterCount = emittersActiveCount_;
    }

    // ① Emitter 注入 (A バッファへインプレース)
    dispatchPyro(cmd, kEmit_, pc);
    computeBarrier(cmd);

    // ①b 燃焼反応 (A バッファへインプレース)
    dispatchPyro(cmd, kCombustion_, pc);
    computeBarrier(cmd);

    // ② 浮力 (A バッファへインプレース)
    dispatchPyro(cmd, kForces_, pc);
    computeBarrier(cmd);

    // ②b 渦度閉じ込め (2パス: curl → 閉じ込め力適用、A バッファへインプレース)
    if(vorticityEps > 0.0f) {
      dispatchPyro(cmd, kCurl_, pc);
      computeBarrier(cmd);
      dispatchPyro(cmd, kVorticityForce_, pc);
      computeBarrier(cmd);
    }

    // ③ 障害物 BC (投影前, A バッファへインプレース)
    dispatchPyro(cmd, kObstacleBC_, pc);
    computeBarrier(cmd);

    // ③b 圧力投影 (非圧縮化): divergence → Red-Black Gauss-Seidel 反復 → project
    // 単一バッファ (pressureIdx_) を in-place 更新するチェッカーボード分割。
    // 同一 sweep 内で red→black の順に適用することで、black パスは red パスで
    // 更新済みの隣接値を使えるため、同じ反復回数でも Jacobi よりも速く収束する。
    dispatchPyro(cmd, kDivergence_, pc);
    computeBarrier(cmd);

    for (int it = 0; it < numPressureIters; it++) {
      PyroSimPC pcRed = pc;
      pcRed.gsColor   = 0;
      dispatchPyro(cmd, kPressureGS_, pcRed);
      computeBarrier(cmd);

      PyroSimPC pcBlack = pc;
      pcBlack.gsColor   = 1;
      dispatchPyro(cmd, kPressureGS_, pcBlack);
      computeBarrier(cmd);
    }
    dispatchPyro(cmd, kProject_, pc);
    computeBarrier(cmd);

    // ③c 障害物 BC (投影後, A バッファへインプレース)
    dispatchPyro(cmd, kObstacleBC_, pc);
    computeBarrier(cmd);

    // ④ 移流 (MacCormack 2パス: 前進推定→スクラッチ、後退補正+クランプ→B)
    dispatchPyro(cmd, kAdvect_, pc);
    computeBarrier(cmd);
    dispatchPyro(cmd, kAdvectMC_, pc);
    computeBarrier(cmd);

    // ⑤ 障害物 BC (移流後, B バッファへインプレース)
    PyroSimPC pcPost = pc;
    pcPost.velIdxA   = pc.velIdxB;
    dispatchPyro(cmd, kObstacleBC_, pcPost);
    computeBarrier(cmd);

    // ダブルバッファ入れ替え: 次フレームは B が現在値になる
    cur_ = 1 - cur_;
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

// Morton 順の配列を線形 (x + y*nx + z*nx*ny) 順に並べ替えて channelsOut へ追記する。
// components=1: float スカラー。components=3: vec4 の xyz のみ (w は捨てる)。
void appendChannelLinear(const std::vector<uint8_t>& mortonRaw, uint32_t gridRes, uint32_t components, std::vector<float>& out) {
  const uint32_t NC = gridRes * gridRes * gridRes;
  out.resize(size_t(NC) * components);
  const float* src = reinterpret_cast<const float*>(mortonRaw.data());
  for(uint32_t code = 0; code < NC; ++code) {
    glm::ivec3 c = mortonDecode(code);
    size_t lin   = size_t(c.x) + size_t(c.y) * gridRes + size_t(c.z) * gridRes * gridRes;
    for(uint32_t k = 0; k < components; ++k) out[lin * components + k] = src[size_t(code) * (components == 1 ? 1u : 4u) + k];
  }
}

} // namespace

void PyroEngine::dumpFrame(const std::string& path, float simTime) const {
  const uint32_t gridRes = cfg_.grid_res;
  const uint32_t NC      = cfg_.totalCells();

  // sdf は障害物未設定 (hasCollider()==false) だと GPU バッファが存在しないため、
  // その場合のみ GPU 読み戻しをスキップし全セル背景値 (1e6 = 障害物なし) を書く。
  struct Channel {
    const char* name;
    uint32_t components;
    VkBuffer buf;
    bool present;
  };
  const Channel channels[] = {
    {"density", 1u, getDensityBuffer(), true}, {"temperature", 1u, getTemperatureBuffer(), true}, {"fuel", 1u, getFuelBuffer(), true},
    {"flame", 1u, getFlameBuffer(), true},     {"velocity", 3u, getVelocityBuffer(), true},       {"sdf", 1u, hasCollider() ? attrBuf_.getBuffer("colliderSDF") : VK_NULL_HANDLE, hasCollider()},
  };

  std::vector<std::vector<float>> linearData(std::size(channels));
  for(size_t i = 0; i < std::size(channels); ++i) {
    const auto& ch = channels[i];
    if(!ch.present) {
      linearData[i].assign(size_t(NC), 1e6f);
      continue;
    }
    const VkDeviceSize elemBytes = ch.components == 1 ? sizeof(float) : sizeof(glm::vec4);
    std::vector<uint8_t> raw(size_t(NC) * elemBytes);
    readBufferToCPU(ch.buf, raw.data(), raw.size());
    appendChannelLinear(raw, gridRes, ch.components, linearData[i]);
  }

  std::ofstream f(path, std::ios::binary);
  if(!f) throw std::runtime_error("dumpFrame: failed to open " + path);

  f.write("PVX1", 4);
  f.write(reinterpret_cast<const char*>(&gridRes), sizeof(uint32_t));
  f.write(reinterpret_cast<const char*>(&gridRes), sizeof(uint32_t));
  f.write(reinterpret_cast<const char*>(&gridRes), sizeof(uint32_t));
  float worldSize = cfg_.world_size;
  f.write(reinterpret_cast<const char*>(&worldSize), sizeof(float));
  f.write(reinterpret_cast<const char*>(&simTime), sizeof(float));
  uint32_t numChannels = uint32_t(std::size(channels));
  f.write(reinterpret_cast<const char*>(&numChannels), sizeof(uint32_t));

  for(const auto& ch : channels) {
    char name[16] = {};
    std::strncpy(name, ch.name, sizeof(name) - 1);
    f.write(name, sizeof(name));
    f.write(reinterpret_cast<const char*>(&ch.components), sizeof(uint32_t));
  }
  for(const auto& data : linearData) f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size() * sizeof(float)));
}
