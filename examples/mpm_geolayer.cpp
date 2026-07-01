#include "App.h"
#include "engine/MPMEngine.h"
#include "MaterialParams.h"
#include "Collider.h"
#include "graphics/GraphicsPipeline.h"

#include <argparse/argparse.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <stdexcept>
#include <string>
#include <vector>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── CLI ───────────────────────────────────────────────────────────────────

struct MpmGeoLayerArgs : public argparse::Args {
  float& world_size    = kwarg("world-size",    "world size [m]").set_default(10.0f);
  int&   grid_res      = kwarg("grid-res",      "MPM grid resolution").set_default(64);
  float& dt            = kwarg("dt",            "frame timestep [s]").set_default(1.0f / 60.0f);
  int&   substeps      = kwarg("substeps",      "substeps per frame").set_default(25);
  int&   n_shots       = kwarg("n-shots",       "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

// ── App ───────────────────────────────────────────────────────────────────

class MpmGeoLayerApp {
public:
  void run(const MpmGeoLayerArgs& args) {
    dt_ = args.dt;
    base_.screenshotDir = args.screenshot_dir;

    MPMConfig cfg;
    cfg.nx         = 8;
    cfg.ny         = 30;
    cfg.nz         = 8;
    cfg.world_size = args.world_size;
    cfg.grid_res   = uint32_t(args.grid_res);

    base_.initWindow("MPM Geo-Layer – 地層崩壊シミュレーション");
    initVulkan(cfg, args.substeps);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp          base_;
  MPMEngine        engine_;
  GraphicsPipeline graphicsPipe_;
  float dt_      = 1.0f / 60.0f;
  float simTime_ = 0.0f;

  float sphere_cx_     = 4.0f;
  float sphere_cy_     = 7.0f;
  float sphere_cz_     = 5.0f;
  float sphere_r_      = 0.8f;
  bool  sphereEnabled_ = false;

  void rebuildColliders() {
    ColliderSet cols;
    cols.addPlane({0.0f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.2f, 0.4f);
    if (sphereEnabled_)
      cols.addSphere({sphere_cx_, sphere_cy_, sphere_cz_}, sphere_r_, 0.3f, 0.1f);
    engine_.setColliders(cols);
  }

  void initVulkan(const MPMConfig& cfg, int substeps) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool,
                 base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue,
                 SHADER_DIR_STR, cfg);
    engine_.numSubsteps = substeps;
    engine_.gravity     = -9.8f;
    engine_.flip_ratio  = -1.0f; // APIC

    // Slot 0: 硬岩 (ELASTIC)
    MaterialParams mat0 = presetJelly(4e5f, 0.2f, 2500.0f);

    // Slot 1: 弱粘土 (VON_MISES, 低降伏応力)
    MaterialParams mat1{};
    mat1.mu     = calcMu(1e4f, 0.4f);
    mat1.lambda = calcLambda(1e4f, 0.4f);
    mat1.rho0   = 1500.0f;
    mat1.model  = uint32_t(MaterialModel::VON_MISES);
    mat1.q_max  = 800.0f;

    // Slot 2: 緩い土 (DRUCKER_PRAGER, 低摩擦角)
    MaterialParams mat2 = presetSand(3e4f, 0.3f, 1200.0f);
    mat2.M_friction = 0.35f;

    engine_.setMaterials({mat0, mat1, mat2});

    // Y インデックスで3層に分割
    const uint32_t N  = cfg.particleCount(); // 8*30*8 = 1920
    const uint32_t nx = cfg.nx;
    const uint32_t ny = cfg.ny;
    std::vector<uint32_t> matIds(N);
    for (uint32_t i = 0; i < N; i++) {
      uint32_t iy = (i / nx) % ny;
      if      (iy < ny / 3)       matIds[i] = 0u; // 下1/3: 硬岩
      else if (iy < 2 * ny / 3)   matIds[i] = 1u; // 中1/3: 弱粘土
      else                         matIds[i] = 2u; // 上1/3: 緩い土
    }
    engine_.setParticleMaterialIds(matIds);

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

    graphicsPipe_.draw(cmd, engine_.descriptorSet, renderPc,
                       engine_.liveParticleCount());

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
    ImGui::SetNextWindowSize({360, 0}, ImGuiCond_Once);
    ImGui::Begin("MPM Geo-Layer Collapse");

    const auto& cfg = engine_.config();
    ImGui::Text("FPS: %.1f | N=%u | t=%.2f s",
                ImGui::GetIO().Framerate, engine_.liveParticleCount(), simTime_);
    ImGui::Text("Grid: %u x %u x %u | gridRes=%u", cfg.nx, cfg.ny, cfg.nz, cfg.grid_res);
    ImGui::Separator();
    ImGui::Text("Slot 0 (y < ny/3)   : ELASTIC     硬岩   E=400kPa rho=2500");
    ImGui::Text("Slot 1 (ny/3..2ny/3): VON_MISES   弱粘土 E=10kPa  q=800Pa");
    ImGui::Text("Slot 2 (y >= 2ny/3) : DRUCKER_PR  緩土   E=30kPa  M=0.35");
    ImGui::Separator();
    ImGui::SliderFloat("重力",       &engine_.gravity,     -20.0f, 0.0f);
    ImGui::SliderInt("サブステップ", &engine_.numSubsteps,  1,     50);
    ImGui::Separator();
    ImGui::Text("球コライダー (横から押し当て):");
    bool changed = false;
    changed |= ImGui::SliderFloat("X",  &sphere_cx_, 0.5f, cfg.world_size - 0.5f);
    changed |= ImGui::SliderFloat("Y",  &sphere_cy_, 0.5f, cfg.world_size * 0.95f);
    changed |= ImGui::SliderFloat("Z",  &sphere_cz_, 0.5f, cfg.world_size - 0.5f);
    changed |= ImGui::SliderFloat("半径", &sphere_r_, 0.2f, 3.0f);
    changed |= ImGui::Checkbox("球コライダー有効", &sphereEnabled_);
    if (changed) rebuildColliders();

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
  auto args = argparse::parse<MpmGeoLayerArgs>(argc, argv);
  MpmGeoLayerApp app;
  try {
    app.run(args);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
