#include "App.h"
#include "engine/MPMEngine.h"
#include "MaterialParams.h"
#include "Collider.h"
#include "graphics/GraphicsPipeline.h"
#include "../core/source.h"

#include <argparse/argparse.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <stdexcept>
#include <string>
#include <memory>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── CLI ───────────────────────────────────────────────────────────────────

struct MpmFountainArgs : public argparse::Args {
  float&       world_size     = kwarg("world-size",     "world size [m]").set_default(10.0f);
  int&         grid_res       = kwarg("grid-res",       "MPM grid resolution").set_default(64);
  int&         max_n          = kwarg("max-n",          "max particle count").set_default(32768);
  int&         emit_n         = kwarg("emit-n",         "particles per emit step").set_default(512);
  float&       dt             = kwarg("dt",             "frame timestep [s]").set_default(1.0f / 60.0f);
  int&         substeps       = kwarg("substeps",       "substeps per frame").set_default(25);
  int&         n_shots        = kwarg("n-shots",        "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
  float&       flip_ratio_arg = kwarg("flip-ratio",     "transfer mode: 0=PIC -1=APIC 0~1=FLIP").set_default(-1.0f);
};

// ── App ───────────────────────────────────────────────────────────────────

class MpmFountainApp {
public:
  void run(const MpmFountainArgs& args) {
    dt_ = args.dt;
    worldSize_ = args.world_size;
    base_.screenshotDir = args.screenshot_dir;

    MPMConfig cfg;
    cfg.nx         = 0;  // 初期パーティクルなし (ソースで生成)
    cfg.ny         = 0;
    cfg.nz         = 0;
    cfg.maxParticles = uint32_t(args.max_n);
    cfg.world_size = args.world_size;
    cfg.grid_res   = uint32_t(args.grid_res);
    cfg.E    = 1e4f;
    cfg.nu   = 0.3f;
    cfg.rho0 = 1600.0f;

    base_.initWindow("MPM Fountain – コライダー + ソースエミッタ");
    initVulkan(cfg, args.substeps, args.emit_n, args.flip_ratio_arg);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp          base_;
  MPMEngine        engine_;
  GraphicsPipeline graphicsPipe_;
  float dt_        = 1.0f / 60.0f;
  float worldSize_ = 10.0f;
  float simTime_   = 0.0f;

  // UI 用コライダーパラメータ
  float sphere_cx_ = 5.0f, sphere_cy_ = 2.0f, sphere_cz_ = 5.0f;
  float sphere_r_  = 1.5f;
  bool  colliderEnabled_ = true;

  void rebuildColliders() {
    ColliderSet cols;
    // 床平面
    cols.addPlane({0, 0.5f, 0}, {0, 1, 0}, 0.2f, 0.4f);
    if (colliderEnabled_) {
      // 球コライダー (粒子が迂回するように配置)
      cols.addSphere({sphere_cx_, sphere_cy_, sphere_cz_}, sphere_r_, 0.3f, 0.1f);
    }
    engine_.setColliders(cols);
  }

  void initVulkan(const MPMConfig& cfg, int substeps, int emitN, float flipRatio) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool,
                 base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue,
                 SHADER_DIR_STR, cfg);
    engine_.numSubsteps = substeps;
    engine_.gravity     = -9.8f;
    engine_.flip_ratio  = flipRatio;

    // 砂マテリアル (slot 0) + 泥マテリアル (slot 1)
    engine_.setMaterials({
        presetSand(5e4f, 0.3f, 1600.0f),
        presetMud( 2e4f, 0.35f, 1800.0f)
    });

    // ソース: AABB から砂を連続放出
    float cx = cfg.world_size * 0.5f;
    float cy = cfg.world_size * 0.75f;
    float cz = cfg.world_size * 0.5f;
    auto src0 = std::make_shared<AABBSource>(
        AABBSource::FromAABB(
            {cx - 0.5f, cy, cz - 0.5f},
            {cx + 0.5f, cy + 0.3f, cz + 0.5f},
            {0, -2.0f, 0},
            emitN));
    src0->step_count   = 0;  // 無限放出
    src0->particleType = 0u; // 砂 (material id 0)
    engine_.addSource(src0);

    rebuildColliders();

    graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass,
                       engine_.descriptorSetLayout,
                       SHADER_DIR_STR + "/particle.vert.spv",
                       SHADER_DIR_STR + "/particle.frag.spv");
    base_.createFrameData();
    base_.initImGui();
  }

  void recordComputeCmd(VkCommandBuffer cmd) {
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
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
        0, 0, nullptr, 1, &barrier, 0, nullptr);

    vkEndCommandBuffer(cmd);
  }

  void recordGraphicsCmd(VkCommandBuffer cmd, uint32_t imageIdx) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    VkClearValue clear{};
    clear.color = {{0.02f, 0.02f, 0.04f, 1.0f}};

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
    VkRect2D sc{}; sc.extent = base_.ctx.swapchainExtent;
    vkCmdSetScissor(cmd, 0, 1, &sc);

    SimPC renderPc{};
    renderPc.posIdx        = engine_.posIdx;
    renderPc.velIdx        = engine_.velIdx;
    renderPc.particleCount = engine_.liveParticleCount();
    renderPc.worldMin      = 0.0f;
    renderPc.worldMax      = engine_.config().world_size;

    graphicsPipe_.draw(cmd, engine_.descriptorSet, renderPc, engine_.liveParticleCount());

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
  }

  void drawFrame(int nShots) {
    auto& f = base_.frames[base_.currentFrame];
    vkWaitForFences(base_.ctx.device, 1, &f.inFlightFence, VK_TRUE, UINT64_MAX);

    uint32_t imageIdx;
    VkResult result = vkAcquireNextImageKHR(base_.ctx.device, base_.ctx.swapchain,
                                             UINT64_MAX, f.imageAvailable,
                                             VK_NULL_HANDLE, &imageIdx);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) { base_.ctx.recreateSwapchain(); return; }

    vkResetFences(base_.ctx.device, 1, &f.inFlightFence);

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({330, 0}, ImGuiCond_Once);
    ImGui::Begin("MPM Fountain");
    const char* mode = (engine_.flip_ratio < -0.5f) ? "APIC"
                     : (engine_.flip_ratio > 0.01f) ? "FLIP" : "PIC";
    ImGui::Text("FPS: %.1f | N=%u | t=%.2f s | %s (r=%.2f)",
                ImGui::GetIO().Framerate, engine_.liveParticleCount(), simTime_,
                mode, engine_.flip_ratio);
    ImGui::Separator();
    ImGui::SliderFloat("重力", &engine_.gravity, -20.0f, 0.0f);
    ImGui::SliderInt("サブステップ", &engine_.numSubsteps, 1, 50);
    ImGui::Separator();
    ImGui::Text("解析コライダー (球)");
    bool changed = false;
    changed |= ImGui::SliderFloat("X",  &sphere_cx_, 1.0f, worldSize_ - 1.0f);
    changed |= ImGui::SliderFloat("Y",  &sphere_cy_, 0.5f, worldSize_ * 0.6f);
    changed |= ImGui::SliderFloat("Z",  &sphere_cz_, 1.0f, worldSize_ - 1.0f);
    changed |= ImGui::SliderFloat("半径", &sphere_r_, 0.3f, 3.0f);
    changed |= ImGui::Checkbox("球コライダー有効", &colliderEnabled_);
    if (changed) rebuildColliders();
    ImGui::Separator();
    if (ImGui::Button("リセット")) {
      engine_.resetParticles();
      engine_.clearSources();
      auto src0 = std::make_shared<AABBSource>(
          AABBSource::FromAABB(
              {worldSize_*0.5f-0.5f, worldSize_*0.75f, worldSize_*0.5f-0.5f},
              {worldSize_*0.5f+0.5f, worldSize_*0.75f+0.3f, worldSize_*0.5f+0.5f},
              {0, -2.0f, 0}, 512));
      src0->step_count   = 0;
      src0->particleType = 0u;
      engine_.addSource(src0);
      simTime_ = 0.0f;
    }
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

    std::array<VkSemaphore, 2>          waitSems   = {f.imageAvailable, f.timelineSemaphore};
    std::array<VkPipelineStageFlags, 2> waitStages = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT};

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
    if (nShots > 0) base_.saveScreenshot(imageIdx, nShots);

    result = vkQueuePresentKHR(base_.ctx.graphicsQueue, &present);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR
        || base_.framebufferResized) {
      base_.framebufferResized = false;
      base_.ctx.recreateSwapchain();
    }
    base_.currentFrame = (base_.currentFrame + 1) % BaseApp::MAX_FRAMES;
  }

  void mainLoop(int nShots) {
    while (!glfwWindowShouldClose(base_.window) && !base_.shouldExit) {
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

// ── main ──────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  auto args = argparse::parse<MpmFountainArgs>(argc, argv);
  MpmFountainApp app;
  try {
    app.run(args);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
