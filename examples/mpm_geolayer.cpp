#include "App.h"
#include "Collider.h"
#include "MaterialParams.h"
#include "engine/MPMEngine.h"
#include "graphics/GraphicsPipeline.h"

#include <argparse/argparse.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── CLI ───────────────────────────────────────────────────────────────────

struct MpmGeoLayerArgs : public argparse::Args {
  float& domain_size_x        = kwarg("domain-size-x", "domain physical size X [m]").set_default(10.0f);
  float& domain_size_y        = kwarg("domain-size-y", "domain physical size Y [m]").set_default(10.0f);
  float& domain_size_z        = kwarg("domain-size-z", "domain physical size Z [m]").set_default(10.0f);
  float& cell_size            = kwarg("cell-size", "MPM grid cell size [m]").set_default(10.0f / 64.0f);
  int& particles              = kwarg("particles", "total seed particle count (split evenly across the 3 layers)").set_default(1920);
  float& dt                   = kwarg("dt", "frame timestep [s]").set_default(1.0f / 60.0f);
  int& substeps               = kwarg("substeps", "substeps per frame").set_default(25);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

// ── App ───────────────────────────────────────────────────────────────────

class MpmGeoLayerApp {
public:
  void run(const MpmGeoLayerArgs& args) {
    dt_                 = args.dt;
    base_.screenshotDir = args.screenshot_dir;

    engine_.domainSize   = glm::vec3(args.domain_size_x, args.domain_size_y, args.domain_size_z);
    engine_.cellSize     = args.cell_size;
    engine_.maxParticles = uint32_t(args.particles);

    base_.initWindow("MPM Geo-Layer – 地層崩壊シミュレーション");
    initVulkan(args.substeps, uint32_t(args.particles));
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

  float sphere_cx_    = 4.0f;
  float sphere_cy_    = 7.0f;
  float sphere_cz_    = 5.0f;
  float sphere_r_     = 0.8f;
  bool sphereEnabled_ = false;

  void rebuildColliders() {
    ColliderSet cols;
    cols.addPlane({0.0f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.2f, 0.4f);
    if(sphereEnabled_) cols.addSphere({sphere_cx_, sphere_cy_, sphere_cz_}, sphere_r_, 0.3f, 0.1f);
    engine_.setColliders(cols);
  }

  // AABB内に一括放出する初期ブロックEmitterを追加する(粒子数をgridResと切り離すため)。
  void addSeedEmitter(const glm::vec3& center, const glm::vec3& size, uint32_t count, uint32_t materialId) {
    auto seed                = std::make_shared<AABBEmitter>();
    seed->center             = center;
    seed->size               = size;
    seed->particleType       = materialId;
    seed->particles_per_step = int(count);
    seed->step_count         = -1;
    engine_.addEmitter(seed);
  }

  void initVulkan(int substeps, uint32_t particleCount) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR);
    engine_.numSubsteps = substeps;
    gravity_            = GravityForce::FromDirection({0.0f, 1.0f, 0.0f}, 9.8f); // Y-up
    engine_.addForce(gravity_);
    engine_.flip_ratio = -1.0f; // APIC

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
    mat2.M_friction     = 0.35f;

    engine_.setMaterials({mat0, mat1, mat2});

    // 縦に細長い柱 (旧cfg.nx/ny/nz=8,30,8相当) を3層に分けたEmitterで再現する。
    const glm::vec3& d    = engine_.domainSize;
    const float minDomain = std::min({d.x, d.y, d.z});
    const float sp        = minDomain * 0.40f / 30.0f;
    const glm::vec3 fullSize(sp * 7.0f, sp * 29.0f, sp * 7.0f);
    const glm::vec3 cx(d.x * 0.5f, d.y * 0.70f, d.z * 0.5f);
    const glm::vec3 layerSize(fullSize.x, fullSize.y / 3.0f, fullSize.z);
    const uint32_t layerCount = particleCount / 3u;
    addSeedEmitter(cx - glm::vec3(0.0f, fullSize.y / 3.0f, 0.0f), layerSize, layerCount, 0u); // 下1/3: 硬岩
    addSeedEmitter(cx, layerSize, layerCount, 1u);                                            // 中1/3: 弱粘土
    addSeedEmitter(cx + glm::vec3(0.0f, fullSize.y / 3.0f, 0.0f), layerSize, layerCount, 2u);  // 上1/3: 緩い土

    rebuildColliders();

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
  auto args = argparse::parse<MpmGeoLayerArgs>(argc, argv);
  MpmGeoLayerApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
