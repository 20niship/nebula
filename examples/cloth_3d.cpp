#include "App.h"
#include "engine/ClothSceneEngine.h"
#include "graphics/ClothRenderer.h"
#include "graphics/GraphicsPipeline.h"

#include <argparse/argparse.hpp>

#include <array>
#include <chrono>
#include <stdexcept>
#include <string>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── CLI ───────────────────────────────────────────────────────────────────────

struct ClothArgs : public argparse::Args {
  int& cloth_n                = kwarg("n,cloth-n", "cloth grid size NxN").set_default(128);
  float& domain_size_x        = kwarg("domain-size-x", "domain physical size X [m]").set_default(10.0f);
  float& domain_size_y        = kwarg("domain-size-y", "domain physical size Y [m]").set_default(10.0f);
  float& domain_size_z        = kwarg("domain-size-z", "domain physical size Z [m]").set_default(10.0f);
  float& cell_size            = kwarg("cell-size", "hash grid cell size [m]").set_default(10.0f / 64.0f);
  float& dt                   = kwarg("dt", "timestep (sec)").set_default(1.0f / 60.0f);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

// ── App ───────────────────────────────────────────────────────────────────────

class ClothApp {
public:
  void run(const ClothArgs& args) {
    dt_                 = args.dt;
    base_.screenshotDir = args.screenshot_dir;

    gridN_      = (uint32_t)args.cloth_n;
    domainSize_ = glm::vec3(args.domain_size_x, args.domain_size_y, args.domain_size_z);
    cellSize_   = args.cell_size;

    base_.initWindow("Vulkan Sim – Cloth 3D");
    initVulkan();
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  ClothSceneEngine sim_;
  GraphicsPipeline graphicsPipe_;
  ClothRenderer clothRenderer_;

  uint32_t gridN_ = 128;
  glm::vec3 domainSize_{10.0f, 10.0f, 10.0f};
  float cellSize_      = 10.0f / 64.0f;
  float dt_      = 1.0f / 60.0f;
  float simTime_ = 0.0f;

  // issue #30 レビュー対応: gravity/windX/windZ の public メンバは廃止されたため
  // ここで Force を作って addForce() する (Engineには自動登録しない)。
  std::shared_ptr<GravityForce> gravity_;
  std::shared_ptr<ConstantWindForce> wind_;

  void initVulkan() {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    // 単一クロスをaddCloth()し、上端行をPin制約で固定する(issue #98: XPBDEngineはClothSceneEngineの機能サブセットのため統合)。
    ClothMesh mesh;
    mesh.build((int)gridN_, 0.065f, domainSize_.x * 0.5f, domainSize_.y * 0.5f, domainSize_.z * 0.85f); // Z-up、箱上部付近
    uint32_t offset = sim_.addCloth(mesh);
    for(uint32_t j = 0; j < gridN_; ++j) {
      uint32_t vi = offset + (uint32_t)mesh.idx(0, j);
      sim_.addConstraint({ClothConstraint::Type::Pin, vi, glm::vec3(mesh.positions[mesh.idx(0, j)])});
    }

    sim_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR, domainSize_, cellSize_);

    gravity_ = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 9.8f); // Z-up
    wind_    = ConstantWindForce::FromDirection({0.0f, 0.0f, 0.0f}, 1.0f);
    wind_->affectMask = ForceAffectTypeFlag(2u); // 布頂点 (typeFlag==2) のみ
    sim_.addForce(gravity_);
    sim_.addForce(wind_);

    graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, sim_.descriptorSetLayout, SHADER_DIR_STR + "/particle.vert.spv", SHADER_DIR_STR + "/particle.frag.spv");

    clothRenderer_.init(base_.ctx.device, base_.ctx.allocator, base_.ctx.renderPass, sim_.descriptorSetLayout, SHADER_DIR_STR);
    clothRenderer_.uploadIndices(sim_.getMesh(0).triIndices, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue);

    base_.createFrameData();
  }

  void recordComputeCmd(VkCommandBuffer cmd) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    sim_.step(cmd, dt_);

    VkBufferMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = base_.ctx.computeFamily;
    barrier.dstQueueFamilyIndex = base_.ctx.graphicsFamily;
    barrier.buffer              = sim_.getPositionBuffer();
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
    clear.color = {{0.05f, 0.05f, 0.08f, 1.0f}};

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

    const uint32_t clothN = sim_.totalParticleCount();
    SimPC pc{};
    pc.posIdx           = sim_.posIdx;
    pc.velIdx           = sim_.velIdx;
    pc.particleCount    = clothN;
    pc.worldMin         = glm::vec3(0.0f);
    pc.worldMax         = domainSize_;
    pc.couplingForceIdx = 0;
    pc.clothVertexCount = clothN;

    clothRenderer_.draw(cmd, sim_.descriptorSet, pc, clothN);
    graphicsPipe_.draw(cmd, sim_.descriptorSet, pc, clothN);

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
    graphicsPipe_.cleanup();
    sim_.cleanup();
    base_.cleanupBase();
  }
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  auto args = argparse::parse<ClothArgs>(argc, argv);
  ClothApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
