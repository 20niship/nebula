#include "App.h"
#include "core/Emitter.h"
#include "engine/FluidEngine.h"
#include "graphics/GraphicsPipeline.h"
#include "utils.hpp"

#include <argparse/argparse.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <glm/glm.hpp>
#include <stdexcept>
#include <string>
#include <vk_mem_alloc.h>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── CLI ───────────────────────────────────────────────────────────────────────

struct SmokeArgs : public argparse::Args {
  float& domain_size_x        = kwarg("domain-size-x", "domain physical size X [m]").set_default(20.0f);
  float& domain_size_y        = kwarg("domain-size-y", "domain physical size Y [m]").set_default(20.0f);
  float& domain_size_z        = kwarg("domain-size-z", "domain physical size Z [m]").set_default(20.0f);
  int& max_particles          = kwarg("max-particles", "max particle count").set_default(8192);
  float& dt                   = kwarg("dt", "timestep (sec)").set_default(1.0f / 60.0f);
  float& rise_accel           = kwarg("rise-accel", "smoke buoyancy acceleration").set_default(8.0f);
  float& smoke_damping        = kwarg("smoke-damping", "smoke velocity damping [1/s]").set_default(0.4f);
  float& emit_radius          = kwarg("emit-radius", "emitter sphere radius").set_default(0.8f);
  int& particles_per_step     = kwarg("pps", "particles emitted per step").set_default(96);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

// ── App ───────────────────────────────────────────────────────────────────────

class SmokeApp {
public:
  void run(const SmokeArgs& args) {
    dt_                 = args.dt;
    base_.screenshotDir = args.screenshot_dir;

    // FluidConfig: 煙専用の小さな設定。cellSize は cellSize >= 2*spacing を満たすよう調整。
    FluidConfig cfg;
    cfg.fluid_nx     = uint32_t(std::cbrt(double(args.max_particles)));
    cfg.fluid_ny     = cfg.fluid_nx;
    cfg.fluid_nz     = cfg.fluid_nx;
    cfg.domainSize   = glm::vec3(args.domain_size_x, args.domain_size_y, args.domain_size_z);
    cfg.cellSize     = args.domain_size_x / float(std::max(16u, cfg.fluid_nx / 2u));
    cfg.max_boundary = 0;

    base_.initWindow("Vulkan Sim – Smoke");
    initVulkan(cfg, args);
    setupSmoke(cfg, args);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  FluidEngine engine_;
  GraphicsPipeline graphicsPipe_;

  float dt_      = 1.0f / 60.0f;
  float simTime_ = 0.0f;

  static constexpr float DIAG_INTERVAL = 2.0f;
  float nextDiagTime_                  = DIAG_INTERVAL;

  // issue #30 デモ: addForce() で風を1回だけ登録し (forces_ の型・個数は以後
  // 不変 = シェーダー再生成は起きない)、毎フレーム wind_->direction/strength
  // だけを書き換えて FluidEngine::step() 内の uploadForces() で SSBO を
  // リアルタイム更新する。風向きは時間とともに水平面(XY)を回転する。
  std::shared_ptr<ConstantWindForce> wind_;
  // issue #30 レビュー対応: gravity は FluidEngine の public メンバとしては廃止
  // されたため、addForce() で登録した GravityForce への参照を自前で保持する。
  std::shared_ptr<GravityForce> gravity_;
  bool windEnabled_        = true;
  float windStrength_      = 12.0f; // 元の6.0から2倍に変更
  float windRotationSpeed_ = 7.5f;  // [rad/s] 元の0.5から5倍速→さらに3倍(計15倍速)

  void updateWind() {
    float angle      = simTime_ * windRotationSpeed_;
    wind_->direction  = glm::vec3(std::cos(angle), std::sin(angle), 0.0f);
    wind_->strength   = windEnabled_ ? windStrength_ : 0.0f;
  }

  void setupSmoke(const FluidConfig& cfg, const SmokeArgs& args) {
    const glm::vec3 w = cfg.domainSize;
    const float cx = w.x * 0.5f, cy = w.y * 0.5f;

    // 底面中央から連続放出するソース（typeFlag=4 = 煙）
    auto src                = std::make_shared<SphereEmitter>();
    src->center             = glm::vec3(cx, cy, 1.0f);
    src->radius             = args.emit_radius;
    src->vel                = glm::vec3(0.0f, 0.0f, 3.0f); // 初期上向き速度
    src->particles_per_step = args.particles_per_step;
    src->step_count         = 0;  // 無限放出
    src->particleType       = 4u; // 煙
    engine_.addEmitter(src);
  }

  void initVulkan(const FluidConfig& cfg, const SmokeArgs& args) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR, cfg);

    // 煙パラメータ
    gravity_ = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 2.0f); // 弱い重力（浮力が上回る）; Z-up
    engine_.addForce(gravity_);
    engine_.smokeRiseAccel   = args.rise_accel;
    engine_.smokeDamping     = args.smoke_damping;
    engine_.linearDamping    = 0.02f;
    engine_.pbfIterations    = 0; // 密度拘束なし（煙は圧縮可能）
    engine_.numSubsteps      = 2;
    engine_.vorticityEnabled = true; // 渦度閉じ込めで煙らしい揺らぎ
    engine_.vorticityEpsilon = 0.5f;
    engine_.rho0             = cfg.computeRestDensity();

    // issue #30 デモ: 風Forceをここで1回だけ登録する (以後 forces_ の型・個数は
    // 不変なので addForce() によるシェーダー再生成はこの1回のみ発生する)
    wind_             = ConstantWindForce::FromDirection({1.0f, 0.0f, 0.0f}, windStrength_);
    wind_->affectMask = ForceAffectTypeFlag(4u); // 煙粒子 (typeFlag==4) のみ
    engine_.addForce(wind_);

    graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/fluid_particle.vert.spv", SHADER_DIR_STR + "/fluid.frag.spv");

    base_.createFrameData();
    base_.initImGui();
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
    clear.color = {{0.02f, 0.02f, 0.03f, 1.0f}};

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
    pc.posIdx        = engine_.posIdx;
    pc.velIdx        = engine_.velIdx;
    pc.particleCount = engine_.nFluid();
    pc.worldMin      = glm::vec3(0.0f);
    pc.worldMax      = engine_.config().domainSize;
    pc.boundaryStart = engine_.config().max_boundary; // 流体パーティクル領域の開始オフセット

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

    ImGui::Begin("Smoke Control");
    ImGui::Text("FPS: %.1f  |  煙粒子: %u / %u  経過: %.2f s", ImGui::GetIO().Framerate, engine_.nFluid(), engine_.config().fluidCount(), simTime_);
    ImGui::Separator();
    sim_ui::fluid_reset_button(engine_, simTime_);
    ImGui::Separator();
    ImGui::Text("煙パラメータ");
    ImGui::SliderFloat("浮力加速度", &engine_.smokeRiseAccel, 0.0f, 20.0f);
    ImGui::SliderFloat("煙の減衰", &engine_.smokeDamping, 0.0f, 2.0f, "%.3f");
    ImGui::SliderFloat("重力", &gravity_->strength, 0.0f, 10.0f);
    ImGui::Separator();
    ImGui::Checkbox("渦度閉じ込め", &engine_.vorticityEnabled);
    if(engine_.vorticityEnabled) {
      ImGui::SliderFloat("渦度 epsilon", &engine_.vorticityEpsilon, 0.0f, 5.0f, "%.3f");
    }
    ImGui::SliderFloat("線形ダンピング", &engine_.linearDamping, 0.0f, 2.0f, "%.3f");
    ImGui::Separator();
    ImGui::Text("issue #30: 動的な風デモ");
    ImGui::TextWrapped("Force一覧(型・個数)は起動時に1回登録したまま不変。"
                        "direction/strengthのみ毎フレームSSBOへ再アップロードされる"
                        "(シェーダー再生成は発生しない)。");
    ImGui::Checkbox("風を有効化", &windEnabled_);
    ImGui::SliderFloat("風の強さ", &windStrength_, 0.0f, 20.0f);
    ImGui::SliderFloat("風の回転速度 [rad/s]", &windRotationSpeed_, 0.0f, 10.0f, "%.2f");
    ImGui::Text("現在の風向き: (%.2f, %.2f)", wind_->direction.x, wind_->direction.y);
    ImGui::End();

    ImGui::Render();
    simTime_ += dt_;
    updateWind();

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

    if(simTime_ >= nextDiagTime_) {
      std::printf("[%.1fs] 煙粒子数: %u\n", simTime_, engine_.nFluid());
      std::fflush(stdout);
      nextDiagTime_ += DIAG_INTERVAL;
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
  auto args = argparse::parse<SmokeArgs>(argc, argv);
  SmokeApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
