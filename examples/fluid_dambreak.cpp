// ダム崩壊 + 障害物 (box) のデモ。既定は PBF (FluidEngine)、--flip で非圧縮 FLIP (FLIPEngine) に切り替わる。
// 泡 (spray/foam/bubble) は --max-diffuse N (N>0) で有効化 (どちらのソルバーも同じ描画を使う)。Z-up。
#include "App.h"
#include "Collider.h"
#include "core/SimPC.h"
#include "engine/FLIPEngine.h"
#include "engine/FluidEngine.h"
#include "graphics/GraphicsPipeline.h"

#include <argparse/argparse.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static const std::string SHADER_DIR_STR = SHADER_DIR;

struct DamArgs : public argparse::Args {
  bool& flip                  = flag("flip", "FLIPEngine (incompressible FLIP) instead of PBF FluidEngine");
  float& cell_size            = kwarg("cell-size", "grid cell size [m] (8 particles/cell for both solvers)").set_default(1.0f / 16);
  int& max_n                  = kwarg("max-n", "max particle count").set_default(800000);
  int& max_diffuse            = kwarg("max-diffuse", "max spray/foam/bubble particle count (0=no foam)").set_default(0);
  float& dt                   = kwarg("dt", "frame timestep [s]").set_default(1.0f / 60.0f);
  int& substeps               = kwarg("substeps", "substeps per frame (0=solver default)").set_default(0);
  float& flip_ratio           = kwarg("flip-ratio", "FLIP share of PIC/FLIP blend (--flip only)").set_default(0.95f);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

class DamApp {
public:
  void run(const DamArgs& args) {
    dt_                 = args.dt;
    flip_               = args.flip;
    cell_               = args.cell_size;
    maxN_               = uint32_t(args.max_n);
    maxDiffuse_         = uint32_t(args.max_diffuse);
    base_.screenshotDir = args.screenshot_dir;

    base_.initWindow(flip_ ? "Dam break - FLIP" : "Dam break - PBF");
    initVulkan(args);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  FluidEngine pbf_;
  FLIPEngine flipEngine_;
  GraphicsPipeline graphicsPipe_;
  GraphicsPipeline foamGraphicsPipe_;
  bool flip_ = false;
  float cell_ = 1.0f / 16;
  uint32_t maxN_ = 0, maxDiffuse_ = 0;
  glm::vec3 domain_{8.0f, 4.0f, 4.0f};
  float dt_ = 1.0f / 60.0f;

  // 障害物 (右寄りの box、床置き)
  static constexpr glm::vec3 kBoxCenter{5.0f, 2.0f, 0.8f};
  static constexpr glm::vec3 kBoxHalf{0.4f, 0.9f, 0.8f};

  VkDescriptorSetLayout descriptorSetLayout() const { return flip_ ? flipEngine_.descriptorSetLayout : pbf_.descriptorSetLayout; }
  VkDescriptorSet descriptorSet() const { return flip_ ? flipEngine_.descriptorSet : pbf_.descriptorSet; }
  VkBuffer positionBuffer() const { return flip_ ? flipEngine_.getPositionBuffer() : pbf_.getPositionBuffer(); }
  bool foamOn() const { return maxDiffuse_ > 0; }

  // PBF は SDF コライダーを持たないため、box の表面 2 層を境界粒子として置く
  std::vector<glm::vec4> boxBoundaryParticles(float spacing) const {
    std::vector<glm::vec4> pts;
    const glm::vec3 lo = kBoxCenter - kBoxHalf, hi = kBoxCenter + kBoxHalf;
    for(float z = lo.z; z <= hi.z + 1e-4f; z += spacing)
      for(float y = lo.y; y <= hi.y + 1e-4f; y += spacing)
        for(float x = lo.x; x <= hi.x + 1e-4f; x += spacing) {
          const glm::vec3 p(x, y, z);
          const float depth = std::min({p.x - lo.x, hi.x - p.x, p.y - lo.y, hi.y - p.y, hi.z - p.z});
          if(depth < 2.0f * spacing - 1e-4f) pts.emplace_back(p, 0.0f);
        }
    return pts;
  }

  void initVulkan(const DamArgs& args) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();
    VkDevice dev = base_.ctx.device;

    // 両ソルバーとも 8 粒子/セル: 粒子間隔 d = cell/2
    const float d = cell_ * 0.5f;
    auto dam      = AABBEmitter::FromAABB({0.3f, 0.3f, 0.3f}, {2.5f, 3.7f, 2.7f}, {0.0f, 0.0f, 0.0f}, 0);
    dam->particles_per_step = std::min<int>(int(dam->size.x * dam->size.y * dam->size.z / (d * d * d)), int(maxN_));
    dam->step_count         = -1;
    auto gravity            = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 9.8f);

    if(flip_) {
      flipEngine_.maxParticles        = maxN_;
      flipEngine_.maxDiffuseParticles = maxDiffuse_;
      flipEngine_.domainSize          = domain_;
      flipEngine_.cellSize            = cell_;
      if(args.substeps > 0) flipEngine_.numSubsteps = args.substeps;
      flipEngine_.flip_ratio = args.flip_ratio;
      flipEngine_.init(dev, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR);
      flipEngine_.addForce(gravity);
      ColliderSet cols;
      cols.addBox(kBoxCenter, kBoxHalf, 0.0f, 0.0f);
      flipEngine_.setColliders(cols);
      flipEngine_.addEmitter(dam);
      flipEngine_.foamEnabled = foamOn();
    } else {
      FluidConfig cfg;
      cfg.particleRadius       = d * 0.5f;
      cfg.domainSize           = domain_;
      cfg.cellSize             = cell_; // h = 2d
      cfg.max_boundary         = 50000;
      cfg.maxDiffuseParticles  = maxDiffuse_;
      cfg.initialCapacityHint  = maxN_;
      pbf_.init(dev, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR, cfg);
      if(args.substeps > 0) pbf_.numSubsteps = args.substeps;
      pbf_.addForce(gravity);
      pbf_.loadBoundaryParticles(boxBoundaryParticles(d));
      pbf_.addEmitter(dam);
      pbf_.foamEnabled = foamOn();
      if(foamOn()) pbf_.setFoamParams(FluidEngine::FoamParams{});
    }

    // FLIP は粒子バッファが MPM 形式 (particle.vert)、PBF は fluid_particle.vert。泡の描画は共通。
    const char* vert = flip_ ? "/particle.vert.spv" : "/fluid_particle.vert.spv";
    const char* frag = flip_ ? "/particle.frag.spv" : "/fluid.frag.spv";
    graphicsPipe_.init(dev, base_.ctx.renderPass, descriptorSetLayout(), SHADER_DIR_STR + vert, SHADER_DIR_STR + frag);
    foamGraphicsPipe_.init(dev, base_.ctx.renderPass, descriptorSetLayout(), SHADER_DIR_STR + "/foam_particle.vert.spv", SHADER_DIR_STR + "/foam.frag.spv", VK_PRIMITIVE_TOPOLOGY_POINT_LIST, /*enableBlend=*/true);
    base_.createFrameData();
  }

  void recordComputeCmd(VkCommandBuffer cmd) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    if(flip_) {
      flipEngine_.step(cmd, dt_);
    } else {
      pbf_.emitFromEmitters(dt_); // 容量拡張による再確保はコマンド記録前に解決する
      pbf_.step(cmd, dt_);
    }

    VkBufferMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = base_.ctx.computeFamily;
    barrier.dstQueueFamilyIndex = base_.ctx.graphicsFamily;
    barrier.buffer              = positionBuffer();
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
    clear.color = {{0.03f, 0.05f, 0.10f, 1.0f}};

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
    renderPc.worldMin = glm::vec3(0.0f);
    renderPc.worldMax = domain_;
    uint32_t n;
    if(flip_) {
      renderPc.posIdx = flipEngine_.posIdx;
      renderPc.velIdx = flipEngine_.velIdx;
      n               = flipEngine_.liveParticleCount();
    } else {
      renderPc.posIdx        = pbf_.posIdx();
      renderPc.velIdx        = pbf_.velIdx();
      renderPc.boundaryStart = pbf_.config().max_boundary; // 流体パーティクル領域の開始オフセット
      n                      = pbf_.nFluid();
    }
    renderPc.particleCount = n;
    graphicsPipe_.draw(cmd, descriptorSet(), renderPc, n);

    if(foamOn()) {
      SimPC foamPc{};
      foamPc.posIdx        = flip_ ? flipEngine_.foamPosIdx() : pbf_.foamPosIdx();
      foamPc.velIdx        = flip_ ? flipEngine_.foamVelIdx() : pbf_.foamVelIdx();
      foamPc.typeFlagIdx   = flip_ ? flipEngine_.foamKindIdx() : pbf_.foamKindIdx(); // foam_particle.vert は kind==0 をクリップする
      foamPc.particleCount = maxDiffuse_;
      foamPc.worldMin      = glm::vec3(0.0f);
      foamPc.worldMax      = domain_;
      foamPc.boundaryStart = 0;
      foamGraphicsPipe_.draw(cmd, descriptorSet(), foamPc, maxDiffuse_);
    }

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
    foamGraphicsPipe_.cleanup();
    graphicsPipe_.cleanup();
    if(flip_) flipEngine_.cleanup();
    else pbf_.cleanup();
    base_.cleanupBase();
  }
};

int main(int argc, char* argv[]) {
  auto args = argparse::parse<DamArgs>(argc, argv);
  DamApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
