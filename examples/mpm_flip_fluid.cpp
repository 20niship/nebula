// MPM FLIP 液体ダムブレイク (fluid_wave_foam.cpp の泡なし版をPBFではなくMPMEngineのFLIP転写+FLUID構成則で実装)
#include "App.h"
#include "Collider.h"
#include "MaterialParams.h"
#include "core/SimPC.h"
#include "engine/MPMEngine.h"
#include "graphics/GraphicsPipeline.h"

#include <argparse/argparse.hpp>
#include <glm/glm.hpp>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── CLI ───────────────────────────────────────────────────────────────────

struct FlipFluidArgs : public argparse::Args {
  float& domain_size_x        = kwarg("domain-size-x", "domain physical size X [m]").set_default(16.0f);
  float& domain_size_y        = kwarg("domain-size-y", "domain physical size Y [m]").set_default(16.0f);
  float& domain_size_z        = kwarg("domain-size-z", "domain physical size Z [m]").set_default(16.0f);
  float& cell_size            = kwarg("cell-size", "MPM grid cell size [m]").set_default(16.0f / 128.0f);
  float& dt                   = kwarg("dt", "frame timestep [s]").set_default(1.0f / 60.0f);
  int& substeps               = kwarg("substeps", "substeps per frame").set_default(10);
  float& flip_ratio           = kwarg("flip-ratio", "0=PIC -1=APIC 0~1=FLIP").set_default(0.95f);
  int& target_particles       = kwarg("target-particles", "ダム柱に詰める目標粒子数").set_default(300000);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

// ── App ───────────────────────────────────────────────────────────────────

class FlipFluidApp {
public:
  void run(const FlipFluidArgs& args) {
    dt_                 = args.dt;
    base_.screenshotDir = args.screenshot_dir;

    engine_.domainSize = glm::vec3(args.domain_size_x, args.domain_size_y, args.domain_size_z);
    engine_.cellSize   = args.cell_size;

    base_.initWindow("MPM FLIP Fluid — Dam Break");
    initVulkan(args.substeps, args.flip_ratio, uint32_t(args.target_particles));
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  MPMEngine engine_;
  GraphicsPipeline graphicsPipe_;
  std::shared_ptr<GravityForce> gravity_;
  float dt_      = 1.0f / 60.0f;
  float simTime_ = 0.0f;

  // ドメイン隅(壁2枚+床)に接する水柱を古典的ダムブレイク配置。体積と目標粒子数から等方spacingを逆算する。
  void placeWaterColumn(uint32_t targetCount) {
    const glm::vec3 D = engine_.domainSize;
    const glm::vec3 colMin(0.5f, 0.0f, 0.5f);
    const glm::vec3 colExt(D.x * 0.32f, D.y * 0.55f, D.z * 0.32f);

    const float volume  = colExt.x * colExt.y * colExt.z;
    const float spacing = std::cbrt(volume / float(targetCount));
    const int nx        = std::max(1, int(colExt.x / spacing));
    const int ny        = std::max(1, int(colExt.y / spacing));
    const int nz        = std::max(1, int(colExt.z / spacing));
    const float Vp      = spacing * spacing * spacing;

    const int maxN = int(engine_.maxParticles);
    std::vector<glm::vec4> pos, vel;
    pos.reserve(std::min(size_t(nx) * ny * nz, size_t(maxN)));
    vel.reserve(pos.capacity());

    for(int iy = 0; iy < ny && int(pos.size()) < maxN; ++iy)
      for(int iz = 0; iz < nz && int(pos.size()) < maxN; ++iz)
        for(int ix = 0; ix < nx && int(pos.size()) < maxN; ++ix) {
          glm::vec3 p = colMin + glm::vec3((ix + 0.5f) * spacing, (iy + 0.5f) * spacing, (iz + 0.5f) * spacing);
          pos.push_back(glm::vec4(p, Vp));
          vel.push_back(glm::vec4(0.0f));
        }
    std::printf("  水柱配置: %zu 個 (目標 %u, spacing=%.4fm)\n", pos.size(), targetCount, spacing);
    engine_.appendParticles(pos, vel);
  }

  void initVulkan(int substeps, float flipRatio, uint32_t targetParticles) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    // appendParticles で後から詰めるため、init() 時の自動シードは行わない (maxParticles>0 指定)。
    engine_.maxParticles = uint32_t(targetParticles * 1.2f);
    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR);
    engine_.numSubsteps = substeps;
    gravity_            = GravityForce::FromDirection({0.0f, -1.0f, 0.0f}, 9.8f); // Y-up; dir=加速度方向そのもの (実測確認: +Yだと天井に張り付く)
    engine_.addForce(gravity_);
    engine_.flip_ratio = flipRatio;

    // 弱圧縮流体 (Tait EOS)。壁際の跳ね返りを抑えるため摩擦は低め。
    MaterialParams water = presetWater(1000.0f, 5e3f);
    engine_.setMaterials({water});

    // 床 + 側壁4枚 (天井なし)。水は摩擦の少ない壁を想定。
    {
      ColliderSet cols;
      const glm::vec3 D = engine_.domainSize;
      cols.addPlane({D.x * 0.5f, 0.0f, D.z * 0.5f}, {0, 1, 0}, 0.0f, 0.05f);
      cols.addPlane({0.0f, D.y * 0.5f, D.z * 0.5f}, {1, 0, 0}, 0.0f, 0.02f);
      cols.addPlane({D.x, D.y * 0.5f, D.z * 0.5f}, {-1, 0, 0}, 0.0f, 0.02f);
      cols.addPlane({D.x * 0.5f, D.y * 0.5f, 0.0f}, {0, 0, 1}, 0.0f, 0.02f);
      cols.addPlane({D.x * 0.5f, D.y * 0.5f, D.z}, {0, 0, -1}, 0.0f, 0.02f);
      engine_.setColliders(cols);
    }

    placeWaterColumn(targetParticles);

    graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/particle.vert.spv", SHADER_DIR_STR + "/particle.frag.spv");
    base_.createFrameData();
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
    clear.color = {{0.02f, 0.04f, 0.09f, 1.0f}};

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
    renderPc.worldMin      = glm::vec3(0.0f);
    renderPc.worldMax      = engine_.domainSize;
    graphicsPipe_.draw(cmd, engine_.descriptorSet, renderPc, engine_.liveParticleCount());

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
    graphicsPipe_.cleanup();
    engine_.cleanup();
    base_.cleanupBase();
  }
};

// ── main ──────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  auto args = argparse::parse<FlipFluidArgs>(argc, argv);
  FlipFluidApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
