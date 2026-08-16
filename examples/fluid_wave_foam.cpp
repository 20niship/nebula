// 波打ちパドル + Bunny 障害物 + 泡 (spray/foam/bubble) デモシーン (issue #47 検証用)。
// ドメイン左側で単振動(SHM)する壁パドルが波を起こし、右側に並べた複数の Bunny
// メッシュへ波が衝突して泡を生成する。海面のリアルなレンダリング検証を想定。
#include "App.h"
#include "core/DefineShaderCompiler.h"
#include "core/Emitter.h"
#include "core/Force.h"
#include "engine/BoundaryParticles.h"
#include "engine/FluidEngine.h"
#include "graphics/GraphicsPipeline.h"
#include "utils.hpp"

#include <argparse/argparse.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <glm/glm.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>

static const std::string SHADER_DIR_STR = SHADER_DIR;
static const std::string ASSET_DIR_STR  = ASSET_DIR;

constexpr float HALF_Y                 = 3.0f; // 流体薄層の奥行き半幅 [m]
constexpr float Y_MARGIN               = 1.0f; // パドル/水面がドメインY境界にちょうど触れないようにする余白 [m]
constexpr float BOUNDARY_SPACING_RATIO = 0.75f;

// ── CLI ───────────────────────────────────────────────────────────────────────

struct WaveFoamArgs : public argparse::Args {
  float& world_size           = kwarg("world-size", "simulation domain size [m] (X/Z)").set_default(24.0f);
  float& particle_radius      = kwarg("particle-radius", "流体粒子半径 [m]").set_default(24.0f / (134.0f * 2.0f));
  float& water_depth          = kwarg("water-depth", "初期水深 [m]").set_default(11.0f * (24.0f / 134.0f));
  float& dt                   = kwarg("dt", "timestep (sec)").set_default(1.0f / 120.0f);
  float& paddle_amp           = kwarg("paddle-amp", "波発生パドル 振幅 [m]").set_default(3.0f);
  float& paddle_omega         = kwarg("paddle-omega", "波発生パドル 角振動数 [rad/s]").set_default(1.2f);
  int& max_diffuse            = kwarg("max-diffuse", "泡の最大パーティクル数").set_default(20000);
  bool& large                 = flag("large", "高解像度プリセット: particle-radius÷2, max-diffuse×4");
  bool& half                  = flag("half", "half floatにしてメモリ帯域を削減");
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

// ── 手続きパドル形状生成（薄い壁の前後2面をシェルサンプリング）──────────────────

static std::vector<glm::vec4> generatePaddleShell(glm::vec3 center, glm::vec3 halfExtents, float spacing) {
  std::vector<glm::vec4> pts;
  int ny = std::max(1, int(halfExtents.y * 2.0f / spacing));
  int nz = std::max(1, int(halfExtents.z * 2.0f / spacing));
  for(int iz = 0; iz <= nz; ++iz) {
    float z = center.z - halfExtents.z + float(iz) / float(nz) * halfExtents.z * 2.0f;
    for(int iy = 0; iy <= ny; ++iy) {
      float y = center.y - halfExtents.y + float(iy) / float(ny) * halfExtents.y * 2.0f;
      pts.push_back(glm::vec4(center.x - halfExtents.x, y, z, 1.0f)); // 背面
      pts.push_back(glm::vec4(center.x + halfExtents.x, y, z, 1.0f)); // 前面
    }
  }
  return pts;
}

// ── Bunny 配置ヘルパー ───────────────────────────────────────────────────────
// assets/bunny.obj のバウンディングボックス (Y-up, 実測値)。
static const glm::vec3 kBunnyMinYup(-0.09469f, 0.032987f, -0.061874f);
static const glm::vec3 kBunnyMaxYup(0.061009f, 0.187321f, 0.058800f);

// floorCenter (X,Y は中心、Z は接地面) にバニーの底面が接地するようオフセットを計算する。
// yup_to_zup 変換 (x,y,z)->(x,z,y) 後の座標系で計算する。
static glm::vec3 bunnyOffsetForFloorCenter(glm::vec3 floorCenter, float scale) {
  glm::vec3 minZup(kBunnyMinYup.x, kBunnyMinYup.z, kBunnyMinYup.y);
  glm::vec3 maxZup(kBunnyMaxYup.x, kBunnyMaxYup.z, kBunnyMaxYup.y);
  glm::vec3 centerZup = (minZup + maxZup) * 0.5f;
  glm::vec3 offset;
  offset.x = floorCenter.x - centerZup.x * scale;
  offset.y = floorCenter.y - centerZup.y * scale;
  offset.z = floorCenter.z - minZup.z * scale; // 底面を floorCenter.z に接地
  return offset;
}

// ── KinematicPaddle ──────────────────────────────────────────────────────────
// 前後2面シェル形状を X 方向へ単振動 (SHM) させる波発生パドル。
struct KinematicPaddle {
  std::vector<glm::vec4> restPositions; // 変位0 (t=0 相当) でのワールド座標
  float amplitude    = 1.5f;            // [m]
  float omega        = 3.49f;           // 角振動数 [rad/s] (= 2π/period)
  uint32_t gpuOffset = 0;               // GPU バッファ先頭インデックス (常に0)
  uint32_t count     = 0;

  static constexpr float rampTime = 2.0f; // [s]
  static float ramp(float t) {
    float x = std::clamp(t / rampTime, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x); // smoothstep
  }

  float displacementX(float t) const { return amplitude * ramp(t) * std::sin(omega * t); }
  float velocityX(float t) const { return amplitude * ramp(t) * omega * std::cos(omega * t); }
};

// ── App ───────────────────────────────────────────────────────────────────────

class WaveFoamApp {
public:
  void run(const WaveFoamArgs& args) {
    dt_                 = args.dt;
    base_.screenshotDir = args.screenshot_dir;

    // --large: 粒子半径を半分(解像度2倍)、泡数を4倍にする高解像度プリセット
    // waterDepth は radius に反比例しないため --large でも変わらない
    const float particleRadius = args.large ? args.particle_radius * 0.5f : args.particle_radius;
    const int max_diff         = args.large ? args.max_diffuse * 4 : args.max_diffuse;

    const float domainSizeY = HALF_Y * 2.0f + Y_MARGIN * 2.0f;
    const float domainSizeZ = args.world_size / 6.0f;

    FluidConfig cfg;
    cfg.particleRadius      = particleRadius;
    cfg.cellSize            = particleRadius * 4.0f; // h=2d 推奨 (d=spacing=2r)
    cfg.domainSize          = glm::vec3(args.world_size, domainSizeY, domainSizeZ);
    cfg.halfVec4Vel         = args.half;
    cfg.max_boundary        = 20000;
    cfg.maxDiffuseParticles = uint32_t(max_diff);

    paddle_.amplitude = args.paddle_amp;
    paddle_.omega     = args.paddle_omega;

    base_.initWindow(args.large ? "Vulkan Sim - Wave Paddle + Foam (Large)" : "Vulkan Sim - Wave Paddle + Bunny + Foam (issue #47)");
    initVulkan(cfg, args.water_depth);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  FluidEngine engine_;
  GraphicsPipeline graphicsPipe_;
  GraphicsPipeline foamGraphicsPipe_;
  std::shared_ptr<GravityForce> gravity_;
  FluidEngine::FoamParams foamParams_;

  KinematicPaddle paddle_;
  std::vector<glm::vec4> paddleNewPos_;
  std::vector<glm::vec4> paddleNewVel_;

  float dt_      = 1.0f / 60.0f;
  float simTime_ = 0.0f;

  void initVulkan(const FluidConfig& cfg, float waterDepth) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR, cfg);

    engine_.rho0             = cfg.computeRestDensity();
    engine_.viscosityC       = 0.002f;
    engine_.linearDamping    = 0.003f;
    engine_.vorticityEnabled = false;
    engine_.vorticityEpsilon = 0.15f;
    engine_.pbfIterations    = 3;
    engine_.numSubsteps      = 2;

    gravity_ = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 9.8f); // Z-up

    engine_.addForce(gravity_);

    const float w           = cfg.domainSize.x;
    const float d           = cfg.particleSpacing();
    const float domainSizeY = cfg.domainSize.y;
    const float centerY     = domainSizeY * 0.5f;

    // ── 波発生パドル (ドメイン左側、X方向に単振動) ────────────────────────────
    // Z: 下端はドメイン下端よりわずかに潜らせて隙間からの流体漏れを防ぎ、
    //    上端はドメイン上端を超えないようにする (超えるとハッシュセル範囲外の
    //    境界パーティクルが生じ、近傍探索が破綻し得る)。
    // Y: ドメインYのAABBを全てカバーする (centerY ± domainSizeY/2)。
    const float boundarySpacing = d * BOUNDARY_SPACING_RATIO; // 流体より密な境界パーティクル間隔
    const float paddleRestX     = paddle_.amplitude + 1.2f;   // 左壁からの最小マージンを確保
    const float paddleZMargin   = 2.0f * boundarySpacing;     // ドメイン下端よりわずかに下へ潜らせる
    const float paddleZMin      = -paddleZMargin;
    // 旧実装は paddleZMax=domainSizeZ*0.85 で上端15%を覆っておらず、波の飛沫が
    // パドル上端を飛び越えて反対側(左壁側)に着水し蓄積する貫通漏れを実機確認した
    // (frame0180/0360スクリーンショットでパドル背後に青い流体粒子が溜まる様子を確認)。
    // 下端と同様に、ドメイン上端をわずかに残してほぼ全高を覆うことで飛び越えを防ぐ。
    const float paddleZMax = cfg.domainSize.z - paddleZMargin;
    // パドルは前後2枚のシェル(x=center.x±paddleHalfThicknessX)で近似しているため、
    // 2枚の間隔(=paddleHalfThicknessX*2)が近傍探索の半径 h=cfg.cellSize より大きいと、
    // 2枚のどちらのカーネル範囲にも入らない「死角」がシェル内部にでき、そこを
    // 流体粒子が貫通できてしまう。h より確実に小さくして死角を無くす。
    const float paddleHalfThicknessX = cfg.cellSize * 0.4f;
    const glm::vec3 paddleCenter(paddleRestX, centerY, (paddleZMin + paddleZMax) * 0.5f);
    const glm::vec3 paddleHalfExtents(paddleHalfThicknessX, domainSizeY * 0.5f, (paddleZMax - paddleZMin) * 0.5f);
    paddle_.restPositions = generatePaddleShell(paddleCenter, paddleHalfExtents, boundarySpacing);
    paddle_.count         = uint32_t(std::min(paddle_.restPositions.size(), size_t(cfg.max_boundary)));
    paddle_.gpuOffset     = 0;
    paddleNewPos_.resize(paddle_.count);
    paddleNewVel_.resize(paddle_.count);

    // ── パドルを境界パーティクルとして登録 ────────────────────────────────
    engine_.loadBoundaryParticles(paddle_.restPositions);
    engine_.initKinematicBoundaryStaging(paddle_.count);

    // ── 流体: パドルのすぐ右に浅い「海」を配置 ──────────────────────────────
    const float oceanMargin = 2.0f * d; // パドル静止面とのごく小さいクリアランス
    const float oceanXStart = paddleRestX + paddleHalfExtents.x + oceanMargin;
    const float oceanXEnd   = w - 1.0f; // ドメイン右壁の手前まで満たす
    glm::vec3 oceanSize(oceanXEnd - oceanXStart, HALF_Y * 2.0f, waterDepth);
    auto src                = std::make_shared<AABBEmitter>();
    src->center             = glm::vec3((oceanXStart + oceanXEnd) * 0.5f, centerY, waterDepth * 0.5f);
    src->size               = oceanSize;
    src->vel                = glm::vec3(0.0f);
    src->particles_per_step = std::max(1, int((oceanSize.x * oceanSize.y * oceanSize.z) / (d * d * d)));
    src->step_count         = -1; // 初回のみ一括生成
    src->particleType       = 1u;
    engine_.addEmitter(src);

    // ── 泡 (spray/foam/bubble) ────────────────────────────────────────────
    foamParams_.kTa                 = 1500.0f; // 既定4000→生成量を抑制
    foamParams_.kWc                 = 1500.0f;
    foamParams_.taLo                = 8.0f; // 既定5→表面の乱れが大きい箇所のみ生成
    foamParams_.taHi                = 25.0f;
    foamParams_.wcLo                = 2.0f; // 既定1
    foamParams_.wcHi                = 6.0f;
    foamParams_.keLo                = 8.0f;  // 既定5→高速な粒子のみ生成対象
    foamParams_.surfaceDensityRatio = 0.85f; // 既定0.95→表面ゲートを厳しくして対象粒子数を削減
    foamParams_.lifetimeMin         = 0.6f;  // 既定1.0→同時生存数(=advectの実効負荷)を削減
    foamParams_.lifetimeMax         = 1.8f;  // 既定3.0
    engine_.foamEnabled             = cfg.maxDiffuseParticles > 0;
    engine_.setFoamParams(foamParams_);

    // fluid_particle_wave.vert / foam_particle_wave.vert は本シーン専用のカメラ
    if(cfg.halfVec4 || cfg.halfVec4Vel) {
      // vがpackHalf2x16詰めのため、静的SPV(FP32想定)のままだとストライド不一致でゴミ値を読む。FluidEngine::init()のextraForceDefines_と同じ判定。
      const std::vector<std::pair<std::string, std::string>> halfDefines =
          cfg.halfVec4 ? std::vector<std::pair<std::string, std::string>>{{"HALF_VEC4", "1"}, {"HALF_VEC4_V", "1"}}
                       : std::vector<std::pair<std::string, std::string>>{{"HALF_VEC4_V", "1"}};
      auto fluidVert = DefineShaderCompiler::compile("fluid_particle_wave.vert", halfDefines, /*isVertexShader=*/true);
      auto foamVert  = DefineShaderCompiler::compile("foam_particle_wave.vert", halfDefines, /*isVertexShader=*/true);
      graphicsPipe_.initVertFromSpirv(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, fluidVert, SHADER_DIR_STR + "/fluid.frag.spv");
      foamGraphicsPipe_.initVertFromSpirv(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, foamVert, SHADER_DIR_STR + "/foam.frag.spv", VK_PRIMITIVE_TOPOLOGY_POINT_LIST, /*enableBlend=*/true);
    } else {
      graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/fluid_particle_wave.vert.spv", SHADER_DIR_STR + "/fluid.frag.spv");
      foamGraphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/foam_particle_wave.vert.spv", SHADER_DIR_STR + "/foam.frag.spv", VK_PRIMITIVE_TOPOLOGY_POINT_LIST, /*enableBlend=*/true);
    }

    base_.createFrameData();
    base_.initImGui();
  }

  void recordComputeCmd(VkCommandBuffer cmd) {
    // 容量拡張によるバッファ再確保は、recordKinematicBoundaryUpdate が
    // その時点の VkBuffer ハンドルを焼き込む前に解決しておく (screw_fluid.cpp と同じ理由)。
    engine_.emitFromEmitters(dt_);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    engine_.recordKinematicBoundaryUpdate(cmd, base_.currentFrame, paddle_.gpuOffset, paddle_.count, paddleNewPos_.data(), paddleNewVel_.data());
    engine_.step(cmd, dt_);

    VkBufferMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = base_.ctx.computeFamily;
    barrier.dstQueueFamilyIndex = base_.ctx.graphicsFamily;
    barrier.buffer              = engine_.getPositionBuffer();
    barrier.offset              = 0;
    barrier.size                = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);

    vkEndCommandBuffer(cmd);
  }

  void recordGraphicsCmd(VkCommandBuffer cmd, uint32_t imageIdx) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    VkClearValue clear{};
    clear.color = {{0.02f, 0.05f, 0.09f, 1.0f}};
    VkRenderPassBeginInfo rp{};
    rp.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass        = base_.ctx.renderPass;
    rp.framebuffer       = base_.ctx.framebuffers[imageIdx];
    rp.renderArea.extent = base_.ctx.swapchainExtent;
    rp.clearValueCount   = 1;
    rp.pClearValues      = &clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{};
    vp.width    = float(base_.ctx.swapchainExtent.width);
    vp.height   = float(base_.ctx.swapchainExtent.height);
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{};
    sc.extent = base_.ctx.swapchainExtent;
    vkCmdSetScissor(cmd, 0, 1, &sc);

    SimPC pc{};
    pc.posIdx   = engine_.posIdx();
    pc.velIdx   = engine_.velIdx();
    pc.worldMin = glm::vec3(0.0f);
    pc.worldMax = engine_.config().domainSize;

    // 境界 (パドル + Bunny): buffer index 0 から
    pc.boundaryStart = 0;
    pc.particleCount = engine_.nBoundary;
    graphicsPipe_.draw(cmd, engine_.descriptorSet, pc, engine_.nBoundary);

    // 流体
    pc.boundaryStart = engine_.config().max_boundary;
    pc.particleCount = engine_.nFluid();
    graphicsPipe_.draw(cmd, engine_.descriptorSet, pc, engine_.nFluid());

    // 泡 (spray/foam/bubble)
    if(engine_.config().maxDiffuseParticles > 0) {
      SimPC foamPc{};
      foamPc.posIdx        = engine_.foamPosIdx();
      foamPc.velIdx        = engine_.foamVelIdx();
      foamPc.typeFlagIdx   = engine_.foamKindIdx();
      foamPc.particleCount = engine_.config().maxDiffuseParticles;
      foamPc.worldMin      = glm::vec3(0.0f);
      foamPc.worldMax      = engine_.config().domainSize;
      foamPc.boundaryStart = 0;
      foamGraphicsPipe_.draw(cmd, engine_.descriptorSet, foamPc, engine_.config().maxDiffuseParticles);
    }

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
  }

  void drawFrame(int nShots) {
    auto& f = base_.frames[base_.currentFrame];
    vkWaitForFences(base_.ctx.device, 1, &f.inFlightFence, VK_TRUE, UINT64_MAX);

    for(uint32_t i = 0; i < paddle_.count; ++i) {
      glm::vec4 p = paddle_.restPositions[i];
      p.x += paddle_.displacementX(simTime_);
      paddleNewPos_[i] = p;
      paddleNewVel_[i] = glm::vec4(paddle_.velocityX(simTime_), 0.0f, 0.0f, 0.0f);
    }

    uint32_t imageIdx;
    VkResult result = vkAcquireNextImageKHR(base_.ctx.device, base_.ctx.swapchain, UINT64_MAX, f.imageAvailable, VK_NULL_HANDLE, &imageIdx);
    if(result == VK_ERROR_OUT_OF_DATE_KHR) {
      base_.ctx.recreateSwapchain();
      return;
    }

    vkResetFences(base_.ctx.device, 1, &f.inFlightFence);

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // 録画時にシーンが見えるよう、パネルは左上に小さく畳んでおく (issue #47 検証用)。
    ImGui::SetNextWindowSize(ImVec2(300.0f, 160.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
    ImGui::Begin("Wave+Bunny+Foam");
    ImGui::Text("FPS %.1f | fluid %u/%u | t=%.2fs", ImGui::GetIO().Framerate, engine_.nFluid(), engine_.config().fluidCount(), simTime_);
    sim_ui::fluid_reset_button(engine_, simTime_);
    ImGui::SliderFloat("paddle amp", &paddle_.amplitude, 0.0f, 3.0f);
    ImGui::SliderFloat("paddle omega", &paddle_.omega, 0.5f, 8.0f);
    if(sim_ui::foam_params(engine_, foamParams_)) engine_.setFoamParams(foamParams_);
    ImGui::End();

    ImGui::Render();
    simTime_ += dt_;

    f.timelineValue++;
    vkResetCommandBuffer(f.computeCmd, 0);
    recordComputeCmd(f.computeCmd);

    VkTimelineSemaphoreSubmitInfo tsSig{};
    tsSig.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tsSig.signalSemaphoreValueCount = 1;
    tsSig.pSignalSemaphoreValues    = &f.timelineValue;
    VkSubmitInfo compSub{};
    compSub.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    compSub.pNext                = &tsSig;
    compSub.commandBufferCount   = 1;
    compSub.pCommandBuffers      = &f.computeCmd;
    compSub.signalSemaphoreCount = 1;
    compSub.pSignalSemaphores    = &f.timelineSemaphore;
    vkQueueSubmit(base_.ctx.computeQueue, 1, &compSub, VK_NULL_HANDLE);

    vkResetCommandBuffer(f.graphicsCmd, 0);
    recordGraphicsCmd(f.graphicsCmd, imageIdx);

    std::array<uint64_t, 2> waitVals = {0, f.timelineValue};
    VkTimelineSemaphoreSubmitInfo tsWait{};
    tsWait.sType                                   = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tsWait.waitSemaphoreValueCount                 = 2;
    tsWait.pWaitSemaphoreValues                    = waitVals.data();
    std::array<VkSemaphore, 2> waitSems            = {f.imageAvailable, f.timelineSemaphore};
    std::array<VkPipelineStageFlags, 2> waitStages = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT};
    VkSubmitInfo grSub{};
    grSub.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    grSub.pNext                = &tsWait;
    grSub.waitSemaphoreCount   = 2;
    grSub.pWaitSemaphores      = waitSems.data();
    grSub.pWaitDstStageMask    = waitStages.data();
    grSub.commandBufferCount   = 1;
    grSub.pCommandBuffers      = &f.graphicsCmd;
    grSub.signalSemaphoreCount = 1;
    grSub.pSignalSemaphores    = &f.renderFinished;
    vkQueueSubmit(base_.ctx.graphicsQueue, 1, &grSub, f.inFlightFence);

    VkPresentInfoKHR present{};
    present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores    = &f.renderFinished;
    present.swapchainCount     = 1;
    present.pSwapchains        = &base_.ctx.swapchain;
    present.pImageIndices      = &imageIdx;

    if(nShots > 0) base_.saveScreenshot(imageIdx, nShots);

    result = vkQueuePresentKHR(base_.ctx.graphicsQueue, &present);
    if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || base_.framebufferResized) {
      base_.framebufferResized = false;
      base_.ctx.recreateSwapchain();
    }

    base_.currentFrame = (base_.currentFrame + 1) % BaseApp::MAX_FRAMES;
  }

  void mainLoop(int nShots) {
    while(!glfwWindowShouldClose(base_.window) && !base_.shouldExit) {
      glfwPollEvents();
      drawFrame(nShots);
    }
    vkDeviceWaitIdle(base_.ctx.device);
  }

  void cleanup() {
    graphicsPipe_.cleanup();
    foamGraphicsPipe_.cleanup();
    engine_.cleanup();
    base_.cleanupBase();
  }
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  auto args = argparse::parse<WaveFoamArgs>(argc, argv);
  WaveFoamApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
