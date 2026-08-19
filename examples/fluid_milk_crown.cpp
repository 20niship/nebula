#include "App.h"
#include "core/DefineShaderCompiler.h"
#include "core/Emitter.h"
#include "core/Force.h"
#include "engine/FluidEngine.h"
#include "graphics/GraphicsPipeline.h"
#include "utils.hpp"

#include <argparse/argparse.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── シナリオ定数: 通常モード (水たまりエミッタ + 一発水滴エミッタ) ────────────────
static constexpr float kWorldSize    = 20.0f;
static constexpr float kPoolRadius   = 4.0f;   // 水たまりの半径 [m]
static constexpr float kDropRadius   = 1.0f;   // 水滴の半径 [m]
static constexpr float kDropHeight   = 14.0f;  // 水滴の初期高さ [m] (床からの落下距離で水たまりが沈降する時間を確保)
static constexpr float kSigmaDefault = 0.001f; // h=cellSize≈1.25m でチューニングした値

// ── シナリオ定数: --large モード (cohesion係数はh^-3でスケールしh=0.02mは通常h≈1.25m比で約2.44e5倍強くなるためsigmaは逆数程度が出発点、実測調整前提) ──
static constexpr float kLargeWorldSize     = 2.0f;
static constexpr float kLargeSpacing       = 0.01f; // 目標粒子間隔 [m] (≈1cm)
static constexpr float kLargePoolThickness = 0.10f; // 水たまりの厚み [m] (数cm〜数十cmの範囲で10cmを既定に)
static constexpr float kLargeRainHalfSize  = 0.85f; // 降雨エリアの半辺長 [m] (水たまりよりひと回り内側)
static constexpr float kLargeRainHeight    = 1.7f;  // 降雨エミッタの高さ [m]
static constexpr float kLargeRainRate      = 60.0f; // 降雨生成レート [粒子/s]
static constexpr float kSigmaDefaultLarge  = 4e-9f; // h^-3スケーリングから逆算した出発点(実測調整前提)

// --largeの水たまりに必要な粒子数(FluidConfig::fluidCount()の理論値=800万粒子をそのまま初期容量にするとMoltenVKのbindless resident化コストが跳ね上がるため実量から算出、超過分はgrowFluidCapacityに任せる)
static uint32_t largePoolParticleCount() {
  const float margin       = kLargeSpacing;
  const float poolHalfSize = kLargeWorldSize * 0.5f - margin;
  const float d            = kLargeSpacing;
  const float volume       = (2.0f * poolHalfSize) * (2.0f * poolHalfSize) * kLargePoolThickness;
  return (uint32_t)(volume / (d * d * d));
}

// ── CLI ───────────────────────────────────────────────────────────────────────

struct MilkCrownArgs : public argparse::Args {
  float& dt = kwarg("dt", "timestep [s]").set_default(1.0f / 60.0f);
  // -1 = 未指定(--largeの有無に応じてkSigmaDefault/kSigmaDefaultLargeを自動選択)
  float& surface_tension      = kwarg("surface-tension", "surface tension cohesion sigma (Akinci 2013); 未指定なら--largeに応じて自動選択").set_default(-1.0f);
  bool& large                 = flag("large", "2m四方ドメイン・1cm前後解像度・水たまり+連続降雨シナリオ");
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
  bool& no_half               = flag("no-half", "計測用: --large時のhalfVec4を強制的に無効化する");
};

// ── App ───────────────────────────────────────────────────────────────────────

class MilkCrownApp {
public:
  void run(const MilkCrownArgs& args) {
    dt_                 = args.dt;
    base_.screenshotDir = args.screenshot_dir;
    large_              = args.large;

    FluidConfig cfg;
    if(large_) {
      cfg.particleRadius = kLargeSpacing * 0.5f;
      cfg.domainSize     = glm::vec3(kLargeWorldSize, kLargeWorldSize, kLargeWorldSize);
      cfg.initialCapacityHint = largePoolParticleCount() * 2u; // 水たまり+連続降雨の蓄積分として2倍の余裕(超過分はgrowFluidCapacityが追い足す)
      cfg.halfVec4            = !args.no_half; // --largeはcommon.glslの通常モード発散知見と異なるスケールのためここでのみ有効化(--no-halfで計測用に強制無効化)
    } else {
      cfg.particleRadius = (kWorldSize / 32.0f) / 2.0f; // spacing=20/32=0.625m
      cfg.domainSize     = glm::vec3(kWorldSize, kWorldSize, kWorldSize);
    }
    cfg.cellSize     = 2.0f * cfg.particleSpacing(); // h/d比≈2 (fluid_absorb.cppと同方針)
    cfg.max_boundary = 0;                            // 床はドメインのSDF境界がそのまま機能する (専用境界メッシュ不要)

    const float sigma = args.surface_tension >= 0.0f ? args.surface_tension : (large_ ? kSigmaDefaultLarge : kSigmaDefault);

    base_.initWindow(large_ ? "Vulkan Sim – Milk Crown (Large / Surface Tension)" : "Vulkan Sim – Milk Crown (Surface Tension)");
    initVulkan(cfg, sigma);
    setupScenario(cfg);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  FluidEngine engine_;
  GraphicsPipeline graphicsPipe_;
  std::shared_ptr<GravityForce> gravity_;

  float dt_      = 1.0f / 60.0f;
  float simTime_ = 0.0f;
  bool large_    = false;

  void initVulkan(const FluidConfig& cfg, float surfaceTension) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR, cfg);

    gravity_ = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 9.8f); // Z-up
    engine_.addForce(gravity_);
    engine_.viscosityC    = 0.01f;
    engine_.pbfIterations = 2;
    engine_.numSubsteps   = 2;
    engine_.rho0            = large_ ? cfg.computeRestDensity() : 30.0f; // h/dの絶対スケールが--largeで大きく変わるためハードコード値でなくcomputeRestDensity()を使う(通常モードは実測チューニング済みの30.0fを踏襲)
    engine_.linearDamping  = 0.02f;
    engine_.surfaceTension = surfaceTension;
    if(large_) {
      // cfmEpsilon/scorrKの既定値はh≈1.25m規模のチューニング値で、--large(h=0.02m)ではCFM緩和項がgrad項優位になり密度拘束が過剰に硬くなり暴走するため大幅に緩める(TC13で実測確認)
      engine_.cfmEpsilon = 1e6f;
      engine_.scorrK     = 0.0f;
    }

    if(large_) {
      // cfg.halfVec4==trueだとバッファがpackHalf2x16詰め(8 bytes/vec4)になり静的.spv(FP32想定)のまま読むとストライド不一致で範囲外読み出しになるためHALF_VEC4注入の実行時コンパイルに切り替える
      std::vector<uint32_t> vertSpirv = DefineShaderCompiler::compile("fluid_particle.vert", {{"HALF_VEC4", "1"}}, /*isVertexShader=*/true);
      graphicsPipe_.initVertFromSpirv(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, vertSpirv, SHADER_DIR_STR + "/fluid.frag.spv");
    } else {
      graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/fluid_particle.vert.spv", SHADER_DIR_STR + "/fluid.frag.spv");
    }

    base_.createFrameData();
    base_.initImGui();
  }

  void setupScenario(const FluidConfig& cfg) {
    if(large_) {
      setupLargeScenario(cfg);
      return;
    }

    const float particleR = cfg.cellSize * 0.5f;               // SDF衝突距離
    const float floorZ    = particleR + cfg.particleSpacing(); // 床ちょうど上
    // EllipseEmitter(単層)だと全粒子が「表面」扱いで表面張力により薄膜ごと丸まる(Plateau-Rayleigh不安定の誇張)ためAABBEmitterで厚みを持たせ内部粒子を作る
    const float poolThickness = 3.0f * cfg.particleSpacing();

    // 水たまりエミッタ: 床付近に浅い直方体の水たまりを一括投入
    auto pool                = std::make_shared<AABBEmitter>();
    pool->center             = glm::vec3(kWorldSize * 0.5f, kWorldSize * 0.5f, floorZ + poolThickness * 0.5f);
    pool->size               = glm::vec3(2.0f * kPoolRadius, 2.0f * kPoolRadius, poolThickness);
    pool->vel                = glm::vec3(0.0f);
    pool->particles_per_step = 4500;
    pool->step_count         = -1; // 初回1回のみ
    engine_.addEmitter(pool);

    // 水滴エミッタ: step_count=-1で初回即発生するが落下に数十フレームかかるため水たまりが自然に沈降・安定する時間が確保される(タイマー等不要)
    auto drop                = std::make_shared<SphereEmitter>();
    drop->center             = glm::vec3(kWorldSize * 0.5f, kWorldSize * 0.5f, kDropHeight);
    drop->radius             = kDropRadius;
    drop->vel                = glm::vec3(0.0f, 0.0f, -2.0f); // 初速: 軽く下向き
    drop->particles_per_step = 900;
    drop->step_count         = -1;
    engine_.addEmitter(drop);
  }

  // --large: ドメイン下部全体を覆う厚みのある水たまり + 常時降り続く雨エミッタ
  void setupLargeScenario(const FluidConfig& cfg) {
    const float particleR = cfg.cellSize * 0.5f;               // SDF衝突距離
    const float floorZ    = particleR + cfg.particleSpacing(); // 床ちょうど上
    const float d         = cfg.particleSpacing();

    const float margin        = d; // 壁際に1粒子分の余白
    const float poolHalfSize  = kLargeWorldSize * 0.5f - margin;
    const float poolThickness = kLargePoolThickness;

    auto pool                = std::make_shared<AABBEmitter>();
    pool->center             = glm::vec3(kLargeWorldSize * 0.5f, kLargeWorldSize * 0.5f, floorZ + poolThickness * 0.5f);
    pool->size               = glm::vec3(2.0f * poolHalfSize, 2.0f * poolHalfSize, poolThickness);
    pool->vel                = glm::vec3(0.0f);
    pool->particles_per_step = largePoolParticleCount();
    pool->step_count         = -1; // 初回1回のみ
    engine_.addEmitter(pool);

    auto rain                = std::make_shared<AABBEmitter>();
    rain->center             = glm::vec3(kLargeWorldSize * 0.5f, kLargeWorldSize * 0.5f, kLargeRainHeight);
    rain->size               = glm::vec3(2.0f * kLargeRainHalfSize, 2.0f * kLargeRainHalfSize, d);
    rain->vel                = glm::vec3(0.0f, 0.0f, -3.0f); // 初速: 下向き
    rain->particles_per_step = std::max(1u, (uint32_t)(kLargeRainRate * dt_));
    rain->step_count         = 0; // 無限
    engine_.addEmitter(rain);
  }

  void recordComputeCmd(VkCommandBuffer cmd) {
    // 容量拡張によるバッファ再確保はコマンドバッファ記録前に解決しておく
    engine_.emitFromEmitters(dt_);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

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
    clear.color = {{0.02f, 0.04f, 0.08f, 1.0f}};

    VkRenderPassBeginInfo rp{};
    rp.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass        = base_.ctx.renderPass;
    rp.framebuffer       = base_.ctx.framebuffers[imageIdx];
    rp.renderArea.extent = base_.ctx.swapchainExtent;
    rp.clearValueCount   = 1;
    rp.pClearValues      = &clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{};
    vp.width    = (float)base_.ctx.swapchainExtent.width;
    vp.height   = (float)base_.ctx.swapchainExtent.height;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{};
    sc.extent = base_.ctx.swapchainExtent;
    vkCmdSetScissor(cmd, 0, 1, &sc);

    SimPC pc{};
    pc.posIdx        = engine_.posIdx();
    pc.velIdx        = engine_.velIdx();
    pc.particleCount = engine_.nFluid();
    pc.worldMin      = glm::vec3(0.0f);
    pc.worldMax      = engine_.config().domainSize;
    pc.boundaryStart = engine_.config().max_boundary;

    graphicsPipe_.draw(cmd, engine_.descriptorSet, pc, engine_.nFluid());

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
  }

  void drawFrame(int nShots) {
    auto& f = base_.frames[base_.currentFrame];
    vkWaitForFences(base_.ctx.device, 1, &f.inFlightFence, VK_TRUE, UINT64_MAX);

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

    ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({320, 0}, ImGuiCond_Once);
    ImGui::Begin("Milk Crown");
    ImGui::Text("FPS: %.1f  |  流体: %u / %u  経過: %.2f s", ImGui::GetIO().Framerate, engine_.nFluid(), engine_.config().fluidCount(), simTime_);
    ImGui::Separator();
    sim_ui::fluid_reset_button(engine_, simTime_);
    ImGui::Separator();
    sim_ui::fluid_params(engine_, *gravity_);
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
    tsWait.sType                   = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tsWait.waitSemaphoreValueCount = 2;
    tsWait.pWaitSemaphoreValues    = waitVals.data();

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
    engine_.cleanup();
    base_.cleanupBase();
  }
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  auto args = argparse::parse<MilkCrownArgs>(argc, argv);
  MilkCrownApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
