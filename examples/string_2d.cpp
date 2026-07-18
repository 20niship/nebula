// 2D 糸シミュレーション — XPBD 距離拘束 + 空間ハッシュ自己衝突検証
// 既存の compute シェーダー (predict / solve_stretch / solve_density 等) を流用し、
// string2d.vert/.frag で LINE_STRIP + POINT_LIST の 2D 正射影描画を行う。

#include "AttributeBuffer.h"
#include "ComputePipeline.h"
#include "Force.h"
#include "ForceShaderCompiler.h"
#include "SimPC.h"
#include "VulkanContext.h"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── シミュレーション定数 ─────────────────────────────────────────────────────
static constexpr uint32_t STR_N      = 64;
static constexpr float STR_SPACE     = 0.1f; // 粒子間隔 (m)
static constexpr uint32_t STR_EDGES  = STR_N - 1;
static constexpr float STR_WORLD     = 10.0f;
static constexpr uint32_t STR_GRID   = 64;
static constexpr float STR_CELL      = STR_WORLD / STR_GRID; // ≈ 0.156m
static constexpr uint32_t STR_CELLS  = STR_GRID * STR_GRID * STR_GRID;
static constexpr uint32_t STR_GROUPS = (STR_CELLS + 255u) / 256u;
// typeFlag=2 (Cloth) → solve_density で colDiam = r*0.5
// r=0.14m → colDiam=0.07m < STR_SPACE 0.1m なので隣接粒子は常時衝突しない
static constexpr float STR_RADIUS    = 0.14f;
static constexpr uint32_t MAX_FRAMES = 2;

// ── ユーティリティ ─────────────────────────────────────────────────────────
static uint32_t floatBits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

static VkShaderModule loadSPV(VkDevice dev, const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if(!file.is_open()) throw std::runtime_error("Cannot open shader: " + path);
  size_t sz = file.tellg();
  std::vector<char> code(sz);
  file.seekg(0);
  file.read(code.data(), sz);
  VkShaderModuleCreateInfo info{};
  info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  info.codeSize = sz;
  info.pCode    = reinterpret_cast<const uint32_t*>(code.data());
  VkShaderModule mod;
  if(vkCreateShaderModule(dev, &info, nullptr, &mod) != VK_SUCCESS) throw std::runtime_error("Failed to create shader module: " + path);
  return mod;
}

// ─────────────────────────────────────────────────────────────────────────────
// LineRenderer: LINE_STRIP (糸) + POINT_LIST (粒子) の 2 パイプライン
// ─────────────────────────────────────────────────────────────────────────────
class LineRenderer {
public:
  VkPipeline linePipeline         = VK_NULL_HANDLE;
  VkPipeline pointPipeline        = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

  void init(VkDevice device, VkRenderPass rp, VkDescriptorSetLayout dsLayout, const std::string& shaderDir);
  void draw(VkCommandBuffer cmd, VkDescriptorSet ds, const SimPC& pc, uint32_t n);
  void cleanup();

private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkPipeline buildPipeline(VkRenderPass rp, VkShaderModule vert, VkShaderModule frag, VkPrimitiveTopology topo);
};

void LineRenderer::init(VkDevice device, VkRenderPass rp, VkDescriptorSetLayout dsLayout, const std::string& shaderDir) {
  device_ = device;

  VkPushConstantRange pcRange{};
  pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  pcRange.size       = sizeof(SimPC);

  VkPipelineLayoutCreateInfo li{};
  li.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  li.setLayoutCount         = 1;
  li.pSetLayouts            = &dsLayout;
  li.pushConstantRangeCount = 1;
  li.pPushConstantRanges    = &pcRange;
  if(vkCreatePipelineLayout(device_, &li, nullptr, &pipelineLayout) != VK_SUCCESS) throw std::runtime_error("LineRenderer: failed to create pipeline layout");

  VkShaderModule vert = loadSPV(device_, shaderDir + "/string2d.vert.spv");
  VkShaderModule frag = loadSPV(device_, shaderDir + "/string2d.frag.spv");

  linePipeline  = buildPipeline(rp, vert, frag, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP);
  pointPipeline = buildPipeline(rp, vert, frag, VK_PRIMITIVE_TOPOLOGY_POINT_LIST);

  vkDestroyShaderModule(device_, vert, nullptr);
  vkDestroyShaderModule(device_, frag, nullptr);
}

VkPipeline LineRenderer::buildPipeline(VkRenderPass rp, VkShaderModule vert, VkShaderModule frag, VkPrimitiveTopology topo) {
  std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
  stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName  = "main";
  stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName  = "main";

  VkPipelineVertexInputStateCreateInfo vi{};
  vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo ia{};
  ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  ia.topology = topo;

  VkPipelineViewportStateCreateInfo vps{};
  vps.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  vps.viewportCount = 1;
  vps.scissorCount  = 1;

  VkPipelineRasterizationStateCreateInfo raster{};
  raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode    = VK_CULL_MODE_NONE;
  raster.lineWidth   = 1.0f;

  VkPipelineMultisampleStateCreateInfo ms{};
  ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState blendA{};
  blendA.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

  VkPipelineColorBlendStateCreateInfo blend{};
  blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  blend.attachmentCount = 1;
  blend.pAttachments    = &blendA;

  std::array<VkDynamicState, 2> dynS = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dyn{};
  dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dyn.dynamicStateCount = 2;
  dyn.pDynamicStates    = dynS.data();

  VkGraphicsPipelineCreateInfo pi{};
  pi.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pi.stageCount          = 2;
  pi.pStages             = stages.data();
  pi.pVertexInputState   = &vi;
  pi.pInputAssemblyState = &ia;
  pi.pViewportState      = &vps;
  pi.pRasterizationState = &raster;
  pi.pMultisampleState   = &ms;
  pi.pColorBlendState    = &blend;
  pi.pDynamicState       = &dyn;
  pi.layout              = pipelineLayout;
  pi.renderPass          = rp;

  VkPipeline pipeline;
  if(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline) != VK_SUCCESS) throw std::runtime_error("LineRenderer: failed to create graphics pipeline");
  return pipeline;
}

void LineRenderer::draw(VkCommandBuffer cmd, VkDescriptorSet ds, const SimPC& pc, uint32_t n) {
  // 糸を折れ線で描画
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &ds, 0, nullptr);
  vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SimPC), &pc);
  vkCmdDraw(cmd, n, 1, 0, 0);

  // 粒子位置を点で描画
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pointPipeline);
  vkCmdDraw(cmd, n, 1, 0, 0);
}

void LineRenderer::cleanup() {
  vkDestroyPipeline(device_, linePipeline, nullptr);
  vkDestroyPipeline(device_, pointPipeline, nullptr);
  vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// String2DSim: GPU XPBD 糸シミュレーションエンジン
// ─────────────────────────────────────────────────────────────────────────────
class String2DSim {
public:
  // ImGui で変更するパラメータ
  float gravity            = -9.8f;
  float particleRadius     = STR_RADIUS;
  float restitution        = 0.3f;
  float friction           = 0.2f;
  float stretchCompliance  = 0.0f;
  float windX              = 0.0f;
  int solverIterations     = 5;
  int numSubsteps          = 8;
  bool enableSelfCollision = true;

  // 描画用に公開
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet             = VK_NULL_HANDLE;
  uint32_t posIdx                           = 0;
  uint32_t velIdx                           = 0;

  void init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir);
  void step(VkCommandBuffer cmd, float dt);
  void cleanup();

  VkBuffer getPositionBuffer() const { return attrBuf_.getBuffer("P"); }
  glm::vec4 readParticlePos(uint32_t i) const;

private:
  VkDevice device_        = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VkCommandPool cmdPool_  = VK_NULL_HANDLE;
  VkQueue queue_          = VK_NULL_HANDLE;

  AttributeBuffer attrBuf_;

  uint32_t predPIdx_        = 0;
  uint32_t invMassIdx_      = 0;
  uint32_t typeFlagIdx_     = 0;
  uint32_t cellCountIdx_    = 0;
  uint32_t cellOffsetIdx_   = 0;
  uint32_t sortedIdxIdx_    = 0;
  uint32_t stretchEdgesIdx_ = 0;
  uint32_t lambdasIdx_      = 0;

  std::vector<uint32_t> colorBatch_;
  int nColors_ = 0;

  // Force (issue #30): gravity/windX 互換の既定Forceを常時登録 (型追加はしないため
  // rebuildForceShaderはinit()で1回のみ呼ぶ)
  std::vector<std::shared_ptr<Force>> forces_;
  std::shared_ptr<GravityForce> legacyGravity_;
  std::shared_ptr<ConstantWindForce> legacyWind_;
  uint32_t forcesIdx_ = 0;
  void uploadForces();

  ComputePipeline kPredict_, kSdfCollision_;
  ComputePipeline kHashCount_, kHashScanLocal_, kHashScanGlobal_, kHashSort_;
  ComputePipeline kSolveDensity_, kSolveStretch_, kUpdateVelocity_;

  void computeBarrier(VkCommandBuffer cmd);
};

void String2DSim::init(VkDevice device, VmaAllocator allocator, VkDescriptorPool descPool, VkCommandPool cmdPool, VkQueue queue, const std::string& shaderDir) {
  device_    = device;
  allocator_ = allocator;
  cmdPool_   = cmdPool;
  queue_     = queue;

  attrBuf_.init(device, allocator, descPool);

  // ── 属性バッファ登録 ───────────────────────────────────────────────
  posIdx      = attrBuf_.addAttribute("P", sizeof(glm::vec4), STR_N);
  velIdx      = attrBuf_.addAttribute("v", sizeof(glm::vec4), STR_N);
  predPIdx_   = attrBuf_.addAttribute("predP", sizeof(glm::vec4), STR_N);
  invMassIdx_ = attrBuf_.addAttribute("invMass", sizeof(glm::vec4), STR_N);
  // typeFlag: uint32 per 粒子 (readUint(typeFlagIdx, i) = data[i] が正しく動く)
  typeFlagIdx_     = attrBuf_.addAttribute("typeFlag", sizeof(uint32_t), STR_N);
  cellCountIdx_    = attrBuf_.addAttribute("cellCount", sizeof(uint32_t), STR_CELLS);
  cellOffsetIdx_   = attrBuf_.addAttribute("cellOffset", sizeof(uint32_t), STR_CELLS + STR_GROUPS);
  sortedIdxIdx_    = attrBuf_.addAttribute("sortedIdx", sizeof(uint32_t), STR_N);
  stretchEdgesIdx_ = attrBuf_.addAttribute("stretchEdges", sizeof(uint32_t), STR_EDGES * 3);
  lambdasIdx_      = attrBuf_.addAttribute("lambdas", sizeof(float), STR_EDGES);

  // ── 粒子初期化 ────────────────────────────────────────────────────
  // 粒子 0 をピン留め、残りを X 軸方向に水平に並べてスタート
  // 全長 = (STR_N-1)*STR_SPACE = 6.3m を x=[1.85, 8.15] に配置
  std::vector<glm::vec4> pos(STR_N), vel(STR_N, glm::vec4(0.0f)), invM(STR_N);
  std::vector<uint32_t> flags(STR_N, 2u); // 2 = Cloth: wind 有効 + tight colDiam

  const float startX = 5.0f - (STR_N - 1) * STR_SPACE * 0.5f; // ≈ 1.85
  const float startZ = 8.0f;
  for(uint32_t i = 0; i < STR_N; ++i) {
    pos[i]  = glm::vec4(startX + i * STR_SPACE, 5.0f, startZ, 1.0f);
    invM[i] = glm::vec4(i == 0 ? 0.0f : 1.0f, 0.0f, 0.0f, 0.0f);
  }

  attrBuf_.upload("P", pos.data(), sizeof(glm::vec4) * STR_N, cmdPool, queue);
  attrBuf_.upload("v", vel.data(), sizeof(glm::vec4) * STR_N, cmdPool, queue);
  attrBuf_.upload("predP", pos.data(), sizeof(glm::vec4) * STR_N, cmdPool, queue);
  attrBuf_.upload("invMass", invM.data(), sizeof(glm::vec4) * STR_N, cmdPool, queue);
  attrBuf_.upload("typeFlag", flags.data(), sizeof(uint32_t) * STR_N, cmdPool, queue);

  // ── エッジデータ (2 色グラフ彩色) ─────────────────────────────────
  // 色 0: 偶数エッジ (0-1, 2-3, 4-5, ...) — 頂点を共有しないので競合なし
  // 色 1: 奇数エッジ (1-2, 3-4, 5-6, ...)
  std::vector<uint32_t> edgeData;
  edgeData.reserve(STR_EDGES * 3);

  for(uint32_t i = 0; i + 1 < STR_N; i += 2) {
    edgeData.push_back(i);
    edgeData.push_back(i + 1);
    edgeData.push_back(floatBits(STR_SPACE));
  }
  uint32_t color0End = static_cast<uint32_t>(edgeData.size() / 3);

  for(uint32_t i = 1; i + 1 < STR_N; i += 2) {
    edgeData.push_back(i);
    edgeData.push_back(i + 1);
    edgeData.push_back(floatBits(STR_SPACE));
  }
  uint32_t totalEdges = static_cast<uint32_t>(edgeData.size() / 3);

  colorBatch_ = {0, color0End, totalEdges};
  nColors_    = 2;

  attrBuf_.upload("stretchEdges", edgeData.data(), sizeof(uint32_t) * edgeData.size(), cmdPool, queue);

  std::vector<float> zeros(STR_EDGES, 0.0f);
  attrBuf_.upload("lambdas", zeros.data(), sizeof(float) * STR_EDGES, cmdPool, queue);

  // ── 外部公開 ──────────────────────────────────────────────────────
  descriptorSetLayout = attrBuf_.descriptorSetLayout;
  descriptorSet       = attrBuf_.descriptorSet;

  // Force (issue #30): gravity/windX 互換の既定Forceを常時登録する
  forcesIdx_     = attrBuf_.addAttribute("forces", sizeof(ForceGPU), 2);
  legacyGravity_ = GravityForce::FromDirection({0.0f, 0.0f, 1.0f}, gravity); // Z-up; strengthに符号を持たせる
  legacyWind_    = ConstantWindForce::FromDirection({windX, 0.0f, 0.0f}, 1.0f);
  legacyWind_->affectMask = ForceAffectTypeFlag(2u); // 布頂点 (typeFlag==2) のみ
  forces_                 = {legacyGravity_, legacyWind_};
  {
    std::vector<uint32_t> spirv = ForceShaderCompiler::compile(forces_, "predict.comp");
    kPredict_.initFromSpirv(device, attrBuf_.descriptorSetLayout, spirv);
  }

  // ── Compute パイプライン ───────────────────────────────────────────
  auto load = [&](ComputePipeline& k, const std::string& name) { k.init(device, attrBuf_.descriptorSetLayout, shaderDir + "/" + name + ".spv"); };
  load(kSdfCollision_, "sdf_collision.comp");
  load(kHashCount_, "hash_count.comp");
  load(kHashScanLocal_, "hash_scan_local.comp");
  load(kHashScanGlobal_, "hash_scan_global.comp");
  load(kHashSort_, "hash_sort.comp");
  load(kSolveDensity_, "solve_density.comp");
  load(kSolveStretch_, "solve_stretch.comp");
  load(kUpdateVelocity_, "update_velocity.comp");

  printf("[String2D] %u particles, %u edges, %d colors\n", STR_N, totalEdges, nColors_);
}

void String2DSim::computeBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier b{};
  b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void String2DSim::uploadForces() {
  std::vector<ForceGPU> packed;
  packed.reserve(forces_.size());
  for(const auto& f : forces_) packed.push_back(f->pack());
  if(!packed.empty()) attrBuf_.upload("forces", packed.data(), sizeof(ForceGPU) * packed.size(), cmdPool_, queue_);
}

void String2DSim::step(VkCommandBuffer cmd, float dt) {
  auto ds = attrBuf_.descriptorSet;

  // Force (issue #30): gravity/windX 互換値を毎フレーム反映してアップロード
  legacyGravity_->strength = gravity;
  legacyWind_->direction   = glm::vec3(windX, 0.0f, 0.0f);
  uploadForces();

  float subDt = dt / std::max(1, numSubsteps);

  for(int sub = 0; sub < numSubsteps; ++sub) {
    SimPC pc{};
    pc.posIdx            = posIdx;
    pc.velIdx            = velIdx;
    pc.predPIdx          = predPIdx_;
    pc.invMassIdx        = invMassIdx_;
    pc.typeFlagIdx       = typeFlagIdx_;
    pc.cellCountIdx      = cellCountIdx_;
    pc.cellOffsetIdx     = cellOffsetIdx_;
    pc.sortedIdxIdx      = sortedIdxIdx_;
    pc.particleCount     = STR_N;
    pc.gridRes           = STR_GRID;
    pc.stretchEdgesIdx   = stretchEdgesIdx_;
    pc.lambdasIdx        = lambdasIdx_;
    pc.dt                = subDt;
    pc.cellSize          = STR_CELL;
    pc.worldMin          = 0.0f;
    pc.worldMax          = STR_WORLD;
    pc.restitution       = restitution;
    pc.friction          = friction;
    pc.particleRadius    = particleRadius;
    pc.forceBufIdx       = forcesIdx_;
    pc.clothVertexCount  = STR_N;
    pc.edgeCount         = STR_EDGES;
    pc.stretchCompliance = stretchCompliance;
    pc.forceCount        = (uint32_t)forces_.size();

    // ① Predict (重力 + 風)
    kPredict_.dispatch(cmd, ds, pc, STR_N);
    computeBarrier(cmd);

    // ② SDF 境界衝突
    kSdfCollision_.dispatch(cmd, ds, pc, STR_N);
    computeBarrier(cmd);

    // ③ λ リセット
    vkCmdFillBuffer(cmd, attrBuf_.getBuffer("lambdas"), 0, VK_WHOLE_SIZE, 0);
    {
      VkBufferMemoryBarrier b{};
      b.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      b.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
      b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.buffer              = attrBuf_.getBuffer("lambdas");
      b.offset              = 0;
      b.size                = VK_WHOLE_SIZE;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &b, 0, nullptr);
    }

    // ④ XPBD 距離拘束 (2 色 × solverIterations 反復)
    for(int iter = 0; iter < solverIterations; ++iter) {
      for(int c = 0; c < nColors_; ++c) {
        uint32_t start = colorBatch_[c];
        uint32_t end   = colorBatch_[c + 1];
        if(end <= start) continue;
        pc.batchEdgeStart = start;
        pc.batchEdgeEnd   = end;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kSolveStretch_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kSolveStretch_.pipelineLayout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, kSolveStretch_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
        vkCmdDispatch(cmd, ((end - start) + 255u) / 256u, 1, 1);
        computeBarrier(cmd);
      }
    }

    // ⑤ SDF 再適用
    kSdfCollision_.dispatch(cmd, ds, pc, STR_N);
    computeBarrier(cmd);

    // ⑥ 自己衝突 (空間ハッシュ)
    if(enableSelfCollision) {
      vkCmdFillBuffer(cmd, attrBuf_.getBuffer("cellCount"), 0, VK_WHOLE_SIZE, 0);
      {
        VkBufferMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer              = attrBuf_.getBuffer("cellCount");
        b.offset              = 0;
        b.size                = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &b, 0, nullptr);
      }

      kHashCount_.dispatch(cmd, ds, pc, STR_N);
      computeBarrier(cmd);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanLocal_.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanLocal_.pipelineLayout, 0, 1, &ds, 0, nullptr);
      vkCmdPushConstants(cmd, kHashScanLocal_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
      vkCmdDispatch(cmd, (STR_CELLS + 255u) / 256u, 1, 1);
      computeBarrier(cmd);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanGlobal_.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kHashScanGlobal_.pipelineLayout, 0, 1, &ds, 0, nullptr);
      vkCmdPushConstants(cmd, kHashScanGlobal_.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimPC), &pc);
      vkCmdDispatch(cmd, 1, 1, 1);
      computeBarrier(cmd);

      kHashSort_.dispatch(cmd, ds, pc, STR_N);
      computeBarrier(cmd);

      // Jacobi solve x2
      kSolveDensity_.dispatch(cmd, ds, pc, STR_N);
      computeBarrier(cmd);
      kSolveDensity_.dispatch(cmd, ds, pc, STR_N);
      computeBarrier(cmd);

      kSdfCollision_.dispatch(cmd, ds, pc, STR_N);
      computeBarrier(cmd);
    }

    // ⑦ 速度更新
    kUpdateVelocity_.dispatch(cmd, ds, pc, STR_N);
    computeBarrier(cmd);
  }
}

void String2DSim::cleanup() {
  kPredict_.cleanup();
  kSdfCollision_.cleanup();
  kHashCount_.cleanup();
  kHashScanLocal_.cleanup();
  kHashScanGlobal_.cleanup();
  kHashSort_.cleanup();
  kSolveDensity_.cleanup();
  kSolveStretch_.cleanup();
  kUpdateVelocity_.cleanup();
  attrBuf_.cleanup();
}

// 粒子 i の位置を GPU から読み出す (デバッグ用; vkDeviceWaitIdle を内部で呼ぶ)
glm::vec4 String2DSim::readParticlePos(uint32_t i) const {
  vkDeviceWaitIdle(device_);

  VkBufferCreateInfo bufInfo{};
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufInfo.size  = sizeof(glm::vec4);
  bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  VmaAllocationCreateInfo allocCI{};
  allocCI.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;

  VkBuffer stageBuf;
  VmaAllocation stageAlloc;
  vmaCreateBuffer(allocator_, &bufInfo, &allocCI, &stageBuf, &stageAlloc, nullptr);

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

  VkBufferCopy region{i * sizeof(glm::vec4), 0, sizeof(glm::vec4)};
  vkCmdCopyBuffer(cmd, attrBuf_.getBuffer("P"), stageBuf, 1, &region);
  vkEndCommandBuffer(cmd);

  VkSubmitInfo si{};
  si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers    = &cmd;
  vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue_);
  vkFreeCommandBuffers(device_, cmdPool_, 1, &cmd);

  void* mapped;
  vmaMapMemory(allocator_, stageAlloc, &mapped);
  glm::vec4 pos = *reinterpret_cast<glm::vec4*>(mapped);
  vmaUnmapMemory(allocator_, stageAlloc);
  vmaDestroyBuffer(allocator_, stageBuf, stageAlloc);
  return pos;
}

// ─────────────────────────────────────────────────────────────────────────────
// String2DApp: Vulkan ウィンドウ + メインループ
// ─────────────────────────────────────────────────────────────────────────────
class String2DApp {
public:
  void run();

private:
  GLFWwindow* window_ = nullptr;
  VulkanContext ctx_;
  String2DSim sim_;
  LineRenderer renderer_;

  struct FrameData {
    VkSemaphore imageAvailable    = VK_NULL_HANDLE;
    VkSemaphore renderFinished    = VK_NULL_HANDLE;
    VkSemaphore timelineSemaphore = VK_NULL_HANDLE;
    VkFence inFlightFence         = VK_NULL_HANDLE;
    VkCommandBuffer graphicsCmd   = VK_NULL_HANDLE;
    VkCommandBuffer computeCmd    = VK_NULL_HANDLE;
    uint64_t timelineValue        = 0;
  };
  std::array<FrameData, MAX_FRAMES> frames_;
  uint32_t currentFrame_ = 0;

  VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
  bool framebufferResized_         = false;
  float simTime_                   = 0.0f;
  using Clock                      = std::chrono::steady_clock;
  Clock::time_point lastTime_;

  void initWindow();
  void initVulkan();
  void initImGui();
  void mainLoop();
  void drawFrame(float dt);
  void cleanup();
  void createDescriptorPool();
  void createFrameData();
  void recordComputeCmd(VkCommandBuffer cmd, float dt);
  void recordGraphicsCmd(VkCommandBuffer cmd, uint32_t imageIdx);

  static void resizeCallback(GLFWwindow* w, int, int) {
    auto* a                = static_cast<String2DApp*>(glfwGetWindowUserPointer(w));
    a->framebufferResized_ = true;
  }
};

void String2DApp::run() {
  initWindow();
  initVulkan();
  mainLoop();
  cleanup();
}

void String2DApp::initWindow() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window_ = glfwCreateWindow(1280, 720, "String 2D – Self-Collision Test", nullptr, nullptr);
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, resizeCallback);
}

void String2DApp::createDescriptorPool() {
  VkDescriptorPoolSize ps{};
  ps.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = MAX_BINDLESS_BUFFERS;

  VkDescriptorPoolCreateInfo info{};
  info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  info.maxSets       = 4;
  info.poolSizeCount = 1;
  info.pPoolSizes    = &ps;
  info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
  if(vkCreateDescriptorPool(ctx_.device, &info, nullptr, &descriptorPool_) != VK_SUCCESS) throw std::runtime_error("Failed to create descriptor pool");
}

void String2DApp::createFrameData() {
  VkSemaphoreTypeCreateInfo typeInfo{};
  typeInfo.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  typeInfo.initialValue  = 0;

  VkSemaphoreCreateInfo bsem{};
  bsem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkSemaphoreCreateInfo tsem{};
  tsem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  tsem.pNext = &typeInfo;

  VkFenceCreateInfo fence{};
  fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for(uint32_t i = 0; i < MAX_FRAMES; ++i) {
    auto& f         = frames_[i];
    f.timelineValue = 0;
    vkCreateSemaphore(ctx_.device, &bsem, nullptr, &f.imageAvailable);
    vkCreateSemaphore(ctx_.device, &bsem, nullptr, &f.renderFinished);
    vkCreateSemaphore(ctx_.device, &tsem, nullptr, &f.timelineSemaphore);
    vkCreateFence(ctx_.device, &fence, nullptr, &f.inFlightFence);

    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    ai.commandPool = ctx_.graphicsCommandPool;
    vkAllocateCommandBuffers(ctx_.device, &ai, &f.graphicsCmd);
    ai.commandPool = ctx_.computeCommandPool;
    vkAllocateCommandBuffers(ctx_.device, &ai, &f.computeCmd);
  }
}

void String2DApp::initImGui() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForVulkan(window_, true);

  ImGui_ImplVulkan_InitInfo ii{};
  ii.ApiVersion                   = VK_API_VERSION_1_2;
  ii.Instance                     = ctx_.instance;
  ii.PhysicalDevice               = ctx_.physicalDevice;
  ii.Device                       = ctx_.device;
  ii.QueueFamily                  = ctx_.graphicsFamily;
  ii.Queue                        = ctx_.graphicsQueue;
  ii.DescriptorPool               = VK_NULL_HANDLE;
  ii.DescriptorPoolSize           = 1000;
  ii.MinImageCount                = 2;
  ii.ImageCount                   = static_cast<uint32_t>(ctx_.swapchainImages.size());
  ii.PipelineInfoMain.RenderPass  = ctx_.renderPass;
  ii.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  ImGui_ImplVulkan_Init(&ii);
}

void String2DApp::initVulkan() {
  ctx_.vsync = false; // VSync 無効: MAILBOX/IMMEDIATE を優先
  ctx_.init(window_);
  createDescriptorPool();

  sim_.init(ctx_.device, ctx_.allocator, descriptorPool_, ctx_.graphicsCommandPool, ctx_.graphicsQueue, SHADER_DIR_STR);

  renderer_.init(ctx_.device, ctx_.renderPass, sim_.descriptorSetLayout, SHADER_DIR_STR);

  createFrameData();
  initImGui();
}

void String2DApp::recordComputeCmd(VkCommandBuffer cmd, float dt) {
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);

  sim_.step(cmd, dt);

  VkBufferMemoryBarrier barrier{};
  barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
  barrier.srcQueueFamilyIndex = ctx_.computeFamily;
  barrier.dstQueueFamilyIndex = ctx_.graphicsFamily;
  barrier.buffer              = sim_.getPositionBuffer();
  barrier.offset              = 0;
  barrier.size                = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);

  vkEndCommandBuffer(cmd);
}

void String2DApp::recordGraphicsCmd(VkCommandBuffer cmd, uint32_t imageIdx) {
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);

  VkClearValue clear{};
  clear.color = {{0.05f, 0.05f, 0.1f, 1.0f}};

  VkRenderPassBeginInfo rpInfo{};
  rpInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpInfo.renderPass        = ctx_.renderPass;
  rpInfo.framebuffer       = ctx_.framebuffers[imageIdx];
  rpInfo.renderArea.extent = ctx_.swapchainExtent;
  rpInfo.clearValueCount   = 1;
  rpInfo.pClearValues      = &clear;

  vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

  VkViewport vp{};
  vp.width    = static_cast<float>(ctx_.swapchainExtent.width);
  vp.height   = static_cast<float>(ctx_.swapchainExtent.height);
  vp.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &vp);

  VkRect2D sc{};
  sc.extent = ctx_.swapchainExtent;
  vkCmdSetScissor(cmd, 0, 1, &sc);

  SimPC pc{};
  pc.posIdx   = sim_.posIdx;
  pc.velIdx   = sim_.velIdx;
  pc.worldMin = 0.0f;
  pc.worldMax = STR_WORLD;

  renderer_.draw(cmd, sim_.descriptorSet, pc, STR_N);

  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
  vkCmdEndRenderPass(cmd);
  vkEndCommandBuffer(cmd);
}

void String2DApp::mainLoop() {
  const auto wallStart = Clock::now();
  lastTime_            = wallStart;
  auto lastPrint       = wallStart;

  printf("\n%-8s  %-8s  %-8s   (先端 = 粒子 %u)\n", "wall(s)", "tip_x", "tip_z", STR_N - 1);
  printf("--------  --------  --------\n");
  fflush(stdout);

  while(!glfwWindowShouldClose(window_)) {
    glfwPollEvents();

    const auto now          = Clock::now();
    const float wallElapsed = std::chrono::duration<float>(now - wallStart).count();

    // 3 秒経過で自動終了
    if(wallElapsed >= 3.0f) {
      printf("\n[App] %.1f 秒経過 → 自動終了\n", wallElapsed);
      fflush(stdout);
      break;
    }

    // 先端座標を 0.5 秒ごとにプリント
    if(std::chrono::duration<float>(now - lastPrint).count() >= 0.5f) {
      glm::vec4 pos = sim_.readParticlePos(STR_N - 1);
      printf("%7.2f   %7.3f   %7.3f\n", wallElapsed, pos.x, pos.z);
      fflush(stdout);
      lastPrint = now;
    }

    float dt  = std::chrono::duration<float>(now - lastTime_).count();
    lastTime_ = now;
    dt        = std::min(dt, 1.0f / 30.0f);
    drawFrame(dt);
  }
  vkDeviceWaitIdle(ctx_.device);
}

void String2DApp::drawFrame(float dt) {
  auto& f = frames_[currentFrame_];
  vkWaitForFences(ctx_.device, 1, &f.inFlightFence, VK_TRUE, UINT64_MAX);

  uint32_t imageIdx;
  VkResult result = vkAcquireNextImageKHR(ctx_.device, ctx_.swapchain, UINT64_MAX, f.imageAvailable, VK_NULL_HANDLE, &imageIdx);
  if(result == VK_ERROR_OUT_OF_DATE_KHR) {
    ctx_.recreateSwapchain();
    return;
  }

  vkResetFences(ctx_.device, 1, &f.inFlightFence);

  // ImGui フレーム
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
  ImGui::SetNextWindowSize(ImVec2(280, 260), ImGuiCond_Once);
  ImGui::Begin("String 2D Control");
  ImGui::Text("FPS: %.1f  dt: %.2fms", ImGui::GetIO().Framerate, dt * 1000.0f);
  ImGui::Text("Particles: %u  Edges: %u", STR_N, STR_EDGES);
  ImGui::Text("SimTime: %.2f s", simTime_);
  ImGui::Separator();
  ImGui::SliderFloat("Gravity", &sim_.gravity, -20.0f, 0.0f);
  ImGui::SliderFloat("WindX", &sim_.windX, -15.0f, 15.0f);
  ImGui::SliderFloat("Stretch Comp.", &sim_.stretchCompliance, 0.0f, 1e-2f);
  ImGui::SliderInt("Solver Iter.", &sim_.solverIterations, 1, 10);
  ImGui::SliderInt("Substeps", &sim_.numSubsteps, 1, 20);
  ImGui::Checkbox("Self-Collision", &sim_.enableSelfCollision);
  if(ImGui::Button("WindX = 0")) sim_.windX = 0.0f;
  ImGui::SameLine();
  if(ImGui::Button("Wind Burst+")) sim_.windX = 8.0f;
  ImGui::SameLine();
  if(ImGui::Button("Wind Burst-")) sim_.windX = -8.0f;
  ImGui::End();

  ImGui::Render();

  simTime_ += dt;

  // Compute submit
  f.timelineValue++;
  vkResetCommandBuffer(f.computeCmd, 0);
  recordComputeCmd(f.computeCmd, dt);

  VkTimelineSemaphoreSubmitInfo tsSig{};
  tsSig.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  tsSig.signalSemaphoreValueCount = 1;
  tsSig.pSignalSemaphoreValues    = &f.timelineValue;

  VkSubmitInfo computeSubmit{};
  computeSubmit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  computeSubmit.pNext                = &tsSig;
  computeSubmit.commandBufferCount   = 1;
  computeSubmit.pCommandBuffers      = &f.computeCmd;
  computeSubmit.signalSemaphoreCount = 1;
  computeSubmit.pSignalSemaphores    = &f.timelineSemaphore;
  vkQueueSubmit(ctx_.computeQueue, 1, &computeSubmit, VK_NULL_HANDLE);

  // Graphics submit
  vkResetCommandBuffer(f.graphicsCmd, 0);
  recordGraphicsCmd(f.graphicsCmd, imageIdx);

  std::array<uint64_t, 2> waitValues = {0, f.timelineValue};

  VkTimelineSemaphoreSubmitInfo tsWait{};
  tsWait.sType                   = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  tsWait.waitSemaphoreValueCount = 2;
  tsWait.pWaitSemaphoreValues    = waitValues.data();

  std::array<VkSemaphore, 2> waitSems            = {f.imageAvailable, f.timelineSemaphore};
  std::array<VkPipelineStageFlags, 2> waitStages = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT};

  VkSubmitInfo graphicsSubmit{};
  graphicsSubmit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  graphicsSubmit.pNext                = &tsWait;
  graphicsSubmit.waitSemaphoreCount   = 2;
  graphicsSubmit.pWaitSemaphores      = waitSems.data();
  graphicsSubmit.pWaitDstStageMask    = waitStages.data();
  graphicsSubmit.commandBufferCount   = 1;
  graphicsSubmit.pCommandBuffers      = &f.graphicsCmd;
  graphicsSubmit.signalSemaphoreCount = 1;
  graphicsSubmit.pSignalSemaphores    = &f.renderFinished;
  vkQueueSubmit(ctx_.graphicsQueue, 1, &graphicsSubmit, f.inFlightFence);

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores    = &f.renderFinished;
  presentInfo.swapchainCount     = 1;
  presentInfo.pSwapchains        = &ctx_.swapchain;
  presentInfo.pImageIndices      = &imageIdx;

  result = vkQueuePresentKHR(ctx_.graphicsQueue, &presentInfo);
  if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized_) {
    framebufferResized_ = false;
    ctx_.recreateSwapchain();
  }

  currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
}

void String2DApp::cleanup() {
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  for(auto& f : frames_) {
    vkDestroySemaphore(ctx_.device, f.imageAvailable, nullptr);
    vkDestroySemaphore(ctx_.device, f.renderFinished, nullptr);
    vkDestroySemaphore(ctx_.device, f.timelineSemaphore, nullptr);
    vkDestroyFence(ctx_.device, f.inFlightFence, nullptr);
  }

  renderer_.cleanup();
  sim_.cleanup();
  vkDestroyDescriptorPool(ctx_.device, descriptorPool_, nullptr);
  ctx_.cleanup();

  glfwDestroyWindow(window_);
  glfwTerminate();
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
  String2DApp app;
  try {
    app.run();
  } catch(const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
