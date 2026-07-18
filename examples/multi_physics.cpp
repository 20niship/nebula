#include "App.h"
#include "engine/MultiPhysicsEngine.h"
#include "graphics/ClothRenderer.h"
#include "graphics/GraphicsPipeline.h"
#include "utils.hpp"

#include <argparse/argparse.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <array>
#include <stdexcept>
#include <string>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── CLI ───────────────────────────────────────────────────────────────────────

struct MultiPhysicsArgs : public argparse::Args {
  int& cloth_n                = kwarg("cloth-n", "cloth grid size NxN").set_default(64);
  int& fluid_nx               = kwarg("nx", "fluid grid X").set_default(32);
  int& fluid_ny               = kwarg("ny", "fluid grid Y").set_default(16);
  int& fluid_nz               = kwarg("nz", "fluid grid Z").set_default(32);
  float& domain_size_x        = kwarg("domain-size-x", "domain physical size X [m]").set_default(10.0f);
  float& domain_size_y        = kwarg("domain-size-y", "domain physical size Y [m]").set_default(10.0f);
  float& domain_size_z        = kwarg("domain-size-z", "domain physical size Z [m]").set_default(10.0f);
  float& cell_size            = kwarg("cell-size", "hash grid cell size [m]").set_default(10.0f / 16.0f);
  float& dt                   = kwarg("dt", "timestep (sec)").set_default(1.0f / 60.0f);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

// ── App ───────────────────────────────────────────────────────────────────────

class MultiPhysicsApp {
public:
  void run(const MultiPhysicsArgs& args) {
    dt_                 = args.dt;
    base_.screenshotDir = args.screenshot_dir;

    MultiPhysicsConfig cfg;
    cfg.cloth_grid_n = (uint32_t)args.cloth_n;
    cfg.fluid_nx     = (uint32_t)args.fluid_nx;
    cfg.fluid_ny     = (uint32_t)args.fluid_ny;
    cfg.fluid_nz     = (uint32_t)args.fluid_nz;
    cfg.domainSize   = glm::vec3(args.domain_size_x, args.domain_size_y, args.domain_size_z);
    cfg.cellSize     = args.cell_size;

    base_.initWindow("Vulkan Sim – Multi-Physics");
    initVulkan(cfg);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  MultiPhysicsEngine engine_;
  GraphicsPipeline fluidPipe_;
  ClothRenderer clothRenderer_;

  // issue #30 デモ: addForce() で布・流体の混在バッファに Turbulence を追加できる
  // ことを示す (affectMask 未設定=全typeFlag対象のため布・流体双方に作用する)
  bool turbulenceEnabled_ = false;
  std::shared_ptr<TurbulenceForce> turbulence_;

  // issue #30 レビュー対応: gravity/windX/windZ は MultiPhysicsEngine の public
  // メンバとしては廃止されたため、addForce() で登録した Force への参照を保持する。
  std::shared_ptr<GravityForce> gravity_;
  std::shared_ptr<ConstantWindForce> wind_;

  float dt_      = 1.0f / 60.0f;
  float simTime_ = 0.0f;

  void initVulkan(const MultiPhysicsConfig& cfg) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR, cfg);

    gravity_ = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 9.8f); // Z-up
    wind_    = ConstantWindForce::FromDirection({0.0f, 0.0f, 0.0f}, 1.0f);
    engine_.addForce(gravity_);
    engine_.addForce(wind_);

    fluidPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/fluid_particle.vert.spv", SHADER_DIR_STR + "/fluid.frag.spv");

    clothRenderer_.init(base_.ctx.device, base_.ctx.allocator, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR);
    clothRenderer_.uploadIndices(engine_.getClothMesh().triIndices, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue);

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
    clear.color = {{0.03f, 0.04f, 0.08f, 1.0f}};

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

    {
      SimPC pc{};
      pc.posIdx           = engine_.posIdx;
      pc.velIdx           = engine_.velIdx;
      pc.particleCount    = engine_.config().clothCount();
      pc.worldMin         = glm::vec3(0.0f);
      pc.worldMax         = engine_.config().domainSize;
      pc.couplingForceIdx = 0;
      pc.clothVertexCount = engine_.config().clothCount();
      clothRenderer_.draw(cmd, engine_.descriptorSet, pc, engine_.config().clothCount());
    }

    {
      SimPC pc{};
      pc.posIdx        = engine_.posIdx;
      pc.velIdx        = engine_.velIdx;
      pc.particleCount = engine_.config().fluidCount();
      pc.worldMin      = glm::vec3(0.0f);
      pc.worldMax      = engine_.config().domainSize;
      pc.boundaryStart = engine_.config().fluidStart();
      fluidPipe_.draw(cmd, engine_.descriptorSet, pc, engine_.config().fluidCount());
    }

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

    ImGui::Begin("Multi-Physics Control");
    ImGui::Text("FPS: %.1f  |  布: %u  流体: %u  経過: %.2f s", ImGui::GetIO().Framerate, engine_.config().clothCount(), engine_.config().fluidCount(), simTime_);
    ImGui::Separator();
    sim_ui::multi_physics_params(engine_, *gravity_, *wind_);
    ImGui::Separator();
    ImGui::Text("issue #30: Force API デモ");
    if(ImGui::Checkbox("Turbulence (布+流体双方に作用) を追加", &turbulenceEnabled_)) {
      if(turbulenceEnabled_) {
        turbulence_            = std::make_shared<TurbulenceForce>();
        turbulence_->strength  = 3.0f;
        turbulence_->frequency = 0.25f;
        engine_.addForce(turbulence_);
      } else {
        engine_.removeForce(turbulence_);
        turbulence_.reset();
      }
    }
    if(turbulenceEnabled_) ImGui::SliderFloat("Turbulence 強さ", &turbulence_->strength, 0.0f, 15.0f);
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
    clothRenderer_.cleanup();
    fluidPipe_.cleanup();
    engine_.cleanup();
    base_.cleanupBase();
  }
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  auto args = argparse::parse<MultiPhysicsArgs>(argc, argv);
  MultiPhysicsApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
