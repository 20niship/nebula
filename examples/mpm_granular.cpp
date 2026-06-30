#include "App.h"
#include "engine/MPMEngine.h"
#include "MaterialParams.h"
#include "graphics/GraphicsPipeline.h"

#include <argparse/argparse.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <stdexcept>
#include <string>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── CLI ───────────────────────────────────────────────────────────────────

struct MpmGranularArgs : public argparse::Args {
  int&   nx         = kwarg("nx",  "column width X").set_default(8);
  int&   ny         = kwarg("ny",  "column height Y").set_default(32);
  int&   nz         = kwarg("nz",  "column depth Z").set_default(8);
  float& world_size = kwarg("world-size", "world size [m]").set_default(10.0f);
  int&   grid_res   = kwarg("grid-res", "MPM grid resolution").set_default(64);
  float& E          = kwarg("E",   "Young modulus [Pa] for sand").set_default(5e4f);
  float& nu         = kwarg("nu",  "Poisson ratio").set_default(0.3f);
  float& rho0       = kwarg("rho0","density [kg/m^3]").set_default(1600.0f);
  float& dt         = kwarg("dt",  "frame timestep [s]").set_default(1.0f / 60.0f);
  int&   substeps   = kwarg("substeps", "substeps per frame").set_default(25);
};

// ── App ───────────────────────────────────────────────────────────────────

class MpmGranularApp {
public:
  void run(const MpmGranularArgs& args) {
    dt_ = args.dt;

    MPMConfig cfg;
    cfg.nx         = uint32_t(args.nx);
    cfg.ny         = uint32_t(args.ny);
    cfg.nz         = uint32_t(args.nz);
    cfg.world_size = args.world_size;
    cfg.grid_res   = uint32_t(args.grid_res);
    cfg.E          = args.E;
    cfg.nu         = args.nu;
    cfg.rho0       = args.rho0;

    base_.initWindow("MPM Granular – 砂柱崩壊");
    initVulkan(cfg, args.substeps);
    mainLoop();
    cleanup();
  }

private:
  BaseApp          base_;
  MPMEngine        engine_;
  GraphicsPipeline graphicsPipe_;
  float dt_      = 1.0f / 60.0f;
  float simTime_ = 0.0f;

  void initVulkan(const MPMConfig& cfg, int substeps) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool,
                 base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue,
                 SHADER_DIR_STR, cfg);
    engine_.numSubsteps = substeps;
    engine_.gravity     = -9.8f;

    // Drucker-Prager 砂マテリアルをスロット 0 に設定
    engine_.setMaterials({presetSand(cfg.E, cfg.nu, cfg.rho0)});

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

    graphicsPipe_.draw(cmd, engine_.descriptorSet, renderPc,
                       engine_.liveParticleCount());

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
  }

  void drawFrame() {
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
    ImGui::SetNextWindowSize({300, 0}, ImGuiCond_Once);
    ImGui::Begin("MPM Granular – 砂柱崩壊");
    const auto& cfg = engine_.config();
    ImGui::Text("FPS: %.1f | N=%u | gridRes=%u",
                ImGui::GetIO().Framerate,
                engine_.liveParticleCount(), cfg.grid_res);
    ImGui::Text("E=%.0f Pa, nu=%.2f, rho0=%.0f", cfg.E, cfg.nu, cfg.rho0);
    ImGui::Text("Column: %u x %u x %u | t=%.2f s",
                cfg.nx, cfg.ny, cfg.nz, simTime_);
    ImGui::Text("材料: Drucker-Prager (砂)");
    ImGui::Separator();
    ImGui::SliderFloat("重力", &engine_.gravity, -20.0f, 0.0f);
    ImGui::SliderInt("サブステップ", &engine_.numSubsteps, 1, 50);
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
    result = vkQueuePresentKHR(base_.ctx.graphicsQueue, &present);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR
        || base_.framebufferResized) {
      base_.framebufferResized = false;
      base_.ctx.recreateSwapchain();
    }
    base_.currentFrame = (base_.currentFrame + 1) % BaseApp::MAX_FRAMES;
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(base_.window) && !base_.shouldExit) {
      glfwPollEvents();
      drawFrame();
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
  auto args = argparse::parse<MpmGranularArgs>(argc, argv);
  MpmGranularApp app;
  try {
    app.run(args);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
