#include "App.h"
#include "Collider.h"
#include "MaterialParams.h"
#include "engine/MPMEngine.h"
#include "graphics/GraphicsPipeline.h"

#include <argparse/argparse.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <stdexcept>
#include <string>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── CLI ───────────────────────────────────────────────────────────────────

struct MpmSnowImpactArgs : public argparse::Args {
  float& world_size           = kwarg("world-size", "world size [m]").set_default(10.0f);
  int& grid_res               = kwarg("grid-res", "MPM grid resolution").set_default(64);
  float& dt                   = kwarg("dt", "frame timestep [s]").set_default(1.0f / 60.0f);
  int& substeps               = kwarg("substeps", "substeps per frame").set_default(25);
  int& pn                     = kwarg("pn", "particle grid per side (N^3 total)").set_default(44);
  float& box_speed            = kwarg("box-speed", "obstacle box speed [m/s]").set_default(6.0f);
  float& box_scale            = kwarg("box-scale", "obstacle box half-extent scale (1=original)").set_default(0.5f);
  int& launch_frame           = kwarg("launch-frame", "box starts moving automatically at this frame (-1=manual button only)").set_default(60);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

// ── App ───────────────────────────────────────────────────────────────────

class MpmSnowImpactApp {
public:
  void run(const MpmSnowImpactArgs& args) {
    dt_                 = args.dt;
    base_.screenshotDir = args.screenshot_dir;
    boxSpeed_           = args.box_speed;
    boxScale_           = args.box_scale;
    launchFrame_        = args.launch_frame;

    MPMConfig cfg;
    cfg.nx         = uint32_t(args.pn);
    cfg.ny         = uint32_t(args.pn);
    cfg.nz         = uint32_t(args.pn);
    cfg.world_size = args.world_size;
    cfg.grid_res   = uint32_t(args.grid_res);

    base_.initWindow("MPM Snow Impact – 移動箱コライダー衝突");
    initVulkan(cfg, args.substeps);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  MPMEngine engine_;
  GraphicsPipeline graphicsPipe_;
  float dt_        = 1.0f / 60.0f;
  float simTime_   = 0.0f;
  int frameCount_  = 0;
  int launchFrame_ = 60;

  float boxPosX_  = 0.0f;
  float boxSpeed_ = 6.0f;
  float boxScale_ = 0.5f;
  bool boxMoving_ = false;

  void rebuildColliders() {
    const float ws = engine_.config().world_size;
    ColliderSet cols;
    cols.addPlane({0.0f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.1f, 0.5f);
    glm::vec3 vel = boxMoving_ ? glm::vec3(-boxSpeed_, 0.0f, 0.0f) : glm::vec3(0.0f);
    cols.addBox({boxPosX_, 1.5f * boxScale_, ws * 0.5f}, {0.5f * boxScale_, 1.5f * boxScale_, ws * 0.3f * boxScale_}, 0.1f, 0.6f, vel);
    engine_.setColliders(cols);
  }

  void initVulkan(const MPMConfig& cfg, int substeps) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR, cfg);
    engine_.numSubsteps = substeps;
    engine_.gravity     = -9.8f;
    engine_.flip_ratio  = -1.0f; // APIC

    // 雪: Von Mises 塑性 (低密度・低降伏応力)
    MaterialParams snow{};
    snow.mu     = calcMu(5e4f, 0.3f);
    snow.lambda = calcLambda(5e4f, 0.3f);
    snow.rho0   = 300.0f;
    snow.model  = uint32_t(MaterialModel::VON_MISES);
    snow.q_max  = 3000.0f;
    engine_.setMaterials({snow});

    boxPosX_ = cfg.world_size * 0.85f;
    rebuildColliders();

    graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/particle.vert.spv", SHADER_DIR_STR + "/particle.frag.spv");
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
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);

    vkEndCommandBuffer(cmd);
  }

  void recordGraphicsCmd(VkCommandBuffer cmd, uint32_t imageIdx) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    VkClearValue clear{};
    clear.color = {{0.02f, 0.03f, 0.06f, 1.0f}};

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
    ImGui::SetNextWindowSize({310, 0}, ImGuiCond_Once);
    ImGui::Begin("MPM Snow Impact");

    ImGui::Text("FPS: %.1f | N=%u | t=%.2f s", ImGui::GetIO().Framerate, engine_.liveParticleCount(), simTime_);
    ImGui::Text("Snow: VON_MISES  E=50kPa  q=3kPa  rho=300 kg/m3");
    ImGui::Text("Box X: %.2f  %s", boxPosX_, boxMoving_ ? "[移動中]" : "[停止]");
    ImGui::Separator();

    if(!boxMoving_) {
      if(launchFrame_ >= 0 && frameCount_ < launchFrame_) {
        ImGui::TextDisabled("自動発進まで: %d フレーム", launchFrame_ - frameCount_);
      }
      if(ImGui::Button("Launch Box →衝突開始")) {
        boxPosX_   = engine_.config().world_size * 0.85f;
        boxMoving_ = true;
        rebuildColliders();
      }
    } else {
      ImGui::TextDisabled("箱が移動中...");
    }
    ImGui::SliderFloat("速度 [m/s]", &boxSpeed_, 0.1f, 10.0f);
    ImGui::Separator();
    ImGui::SliderFloat("重力", &engine_.gravity, -20.0f, 0.0f);
    ImGui::SliderInt("サブステップ", &engine_.numSubsteps, 1, 50);

    ImGui::End();
    ImGui::Render();

    // 固定フレームに到達したらボタン操作なしで自動的に箱を発進させる
    if(!boxMoving_ && launchFrame_ >= 0 && frameCount_ >= launchFrame_) {
      boxPosX_   = engine_.config().world_size * 0.85f;
      boxMoving_ = true;
    }

    // 箱の位置を更新してからコライダーを再アップロード (compute より前)
    if(boxMoving_) {
      boxPosX_ -= boxSpeed_ * dt_;
      if(boxPosX_ - 0.5f * boxScale_ < 0.5f) boxMoving_ = false;
      rebuildColliders();
    }
    simTime_ += dt_;
    ++frameCount_;

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

// ── main ──────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  auto args = argparse::parse<MpmSnowImpactArgs>(argc, argv);
  MpmSnowImpactApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
