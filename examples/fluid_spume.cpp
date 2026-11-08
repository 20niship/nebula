// spume (https://github.com/codepawl/spume) の README にある FLIP 液体 example を
// nebula で再現する。既定は MPMEngine (弱圧縮 FLUID + FLIP/PIC ブレンド)、--flip で FLIPEngine (圧力投影による非圧縮 FLIP)。
//   --scene geyser : 床から垂直に噴き上がるジェット
//   --scene outfall: 側面パイプから流れ出す急流
//   --scene culvert: 閉じた暗渠 (天井あり) を走るサージ
//   --scene slab   : 床ジェットがコンクリート板 (box 障害物) に当たる
//   --scene dual   : 床ジェット + 側面ジェットの 2 inflow
// 未対応: outflow-box / whitewater / marching-cubes メッシュ出力 (粒子描画のみ)。
#include "App.h"
#include "Collider.h"
#include "MaterialParams.h"
#include "core/SimPC.h"
#include "engine/FLIPEngine.h"
#include "engine/MPMEngine.h"
#include "graphics/GraphicsPipeline.h"

#include <argparse/argparse.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <memory>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static const std::string SHADER_DIR_STR = SHADER_DIR;

struct SpumeArgs : public argparse::Args {
  std::string& scene          = kwarg("scene", "geyser|outfall|culvert|slab|dual").set_default(std::string("geyser"));
  float& cell_size            = kwarg("cell-size", "grid cell size [m]").set_default(0.0f);
  int& max_n                  = kwarg("max-n", "max particle count").set_default(400000);
  float& dt                   = kwarg("dt", "frame timestep [s]").set_default(1.0f / 60.0f);
  bool& flip                  = flag("flip", "FLIPEngine (incompressible FLIP) instead of MPMEngine weakly-compressible fluid");
  int& substeps               = kwarg("substeps", "substeps per frame (0=solver default: MPM 20, FLIP 4)").set_default(0);
  float& flip_ratio           = kwarg("flip-ratio", "FLIP share of PIC/FLIP blend").set_default(0.95f);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

// 連続流入口。size は開口の全幅 [m]、dir は噴出方向 (正規化済み)。
struct Inflow {
  glm::vec3 center, size, dir;
  float speed;
};

struct Scene {
  glm::vec3 domain;
  float cell;
  bool closed = false; // 天井も固体にする
  std::vector<Inflow> inflows;
  std::vector<std::pair<glm::vec3, glm::vec3>> boxes; // 障害物 (center, halfExt)
};

static Scene makeScene(const std::string& name) {
  Scene s;
  const glm::vec3 up{0, 1, 0}, px{1, 0, 0};
  if(name == "geyser") {
    s.domain  = {4, 8, 4};
    s.cell    = 1.0f / 16;
    s.inflows = {{{2, 0.15f, 2}, {0.5f, 0.3f, 0.5f}, up, 12.0f}};
  } else if(name == "outfall") {
    s.domain  = {10, 4, 4};
    s.cell    = 1.0f / 12;
    s.inflows = {{{0.2f, 1.6f, 2}, {0.4f, 1.2f, 1.2f}, px, 7.0f}};
  } else if(name == "culvert") {
    s.domain  = {10, 3, 3};
    s.cell    = 1.0f / 16;
    s.closed  = true;
    s.inflows = {{{0.2f, 0.6f, 1.5f}, {0.4f, 1.0f, 2.2f}, px, 6.0f}};
  } else if(name == "slab") {
    s.domain  = {6, 6, 6};
    s.cell    = 1.0f / 12;
    s.inflows = {{{2.25f, 0.15f, 3}, {0.5f, 0.3f, 0.5f}, up, 14.0f}};
    s.boxes   = {{{3, 3.3f, 3}, {1.0f, 0.25f, 1.0f}}}; // 噴流直上の板
  } else if(name == "dual") {
    s.domain  = {6, 8, 6};
    s.cell    = 1.0f / 12;
    s.inflows = {{{2, 0.15f, 3}, {0.7f, 0.3f, 0.7f}, up, 13.0f}, {{0.2f, 1.2f, 3.8f}, {0.4f, 1.0f, 1.0f}, px, 7.0f}};
  } else {
    throw std::runtime_error("unknown --scene: " + name);
  }
  // 上記は (x, 高さ, 奥行) で記述。描画カメラ/重力は Z-up なので Y<->Z を入れ替える。
  auto sw = [](glm::vec3 v) { return glm::vec3(v.x, v.z, v.y); };
  s.domain = sw(s.domain);
  for(auto& in : s.inflows) in = {sw(in.center), sw(in.size), sw(in.dir), in.speed};
  for(auto& [c, h] : s.boxes) { c = sw(c); h = sw(h); }
  return s;
}

class SpumeApp {
public:
  void run(const SpumeArgs& args) {
    dt_                 = args.dt;
    scene_              = makeScene(args.scene);
    if(args.cell_size > 0.0f) scene_.cell = args.cell_size;
    base_.screenshotDir = args.screenshot_dir;

    if(args.flip) engine_ = std::make_unique<FLIPEngine>();
    else engine_ = std::make_unique<MPMEngine>();
    engine_->maxParticles = uint32_t(args.max_n);
    engine_->domainSize   = scene_.domain;
    engine_->cellSize     = scene_.cell;

    base_.initWindow((std::string(args.flip ? "FLIP" : "MPM FLIP") + " (spume) - " + args.scene).c_str());
    initVulkan(args.substeps, args.flip_ratio);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  std::unique_ptr<ParticleGridEngine> engine_; // MPMEngine か FLIPEngine
  GraphicsPipeline graphicsPipe_;
  std::shared_ptr<GravityForce> gravity_;
  Scene scene_;
  float dt_ = 1.0f / 60.0f;

  void initVulkan(int substeps, float flipRatio) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_->init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR);
    if(substeps > 0) engine_->numSubsteps = substeps;
    engine_->flip_ratio = flipRatio;
    gravity_            = GravityForce::FromDirection({0.0f, 0.0f, 1.0f}, 9.8f);
    engine_->addForce(gravity_);
    engine_->setMaterials({presetWater()});

    const glm::vec3 D = scene_.domain;
    ColliderSet cols;
    cols.addPlane({D.x * 0.5f, D.y * 0.5f, 0}, {0, 0, 1}, 0.0f, 0.2f);
    cols.addPlane({0, D.y * 0.5f, D.z * 0.5f}, {1, 0, 0}, 0.0f, 0.0f);
    cols.addPlane({D.x, D.y * 0.5f, D.z * 0.5f}, {-1, 0, 0}, 0.0f, 0.0f);
    cols.addPlane({D.x * 0.5f, 0, D.z * 0.5f}, {0, 1, 0}, 0.0f, 0.0f);
    cols.addPlane({D.x * 0.5f, D.y, D.z * 0.5f}, {0, -1, 0}, 0.0f, 0.0f);
    if(scene_.closed) cols.addPlane({D.x * 0.5f, D.y * 0.5f, D.z}, {0, 0, -1}, 0.0f, 0.0f);
    for(auto& [c, h] : scene_.boxes) cols.addBox(c, h, 0.0f, 0.0f);
    engine_->setColliders(cols);

    // 1 粒子 = (cell/2)^3 (8 particles/cell)。1 フレームで開口を通過する体積から放出数を決める。
    const float Vp = std::pow(scene_.cell * 0.5f, 3.0f);
    for(const Inflow& in : scene_.inflows) {
      // 噴出方向に垂直な面積 × 速度 × dt = 1 フレームの流入体積
      const glm::vec3 a = glm::abs(in.dir);
      const float area  = a.x > 0.5f ? in.size.y * in.size.z : a.y > 0.5f ? in.size.x * in.size.z : in.size.x * in.size.y;
      auto e            = std::make_shared<AABBEmitter>();
      e->center         = in.center;
      e->size           = in.size;
      e->vel            = in.dir * in.speed;
      e->particleType   = 0u;
      e->particleVolume = Vp;
      e->particles_per_step = std::max(1, int(area * in.speed * dt_ / Vp));
      e->step_count     = 0; // 無限に放出
      engine_->addEmitter(e);
    }

    graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_->descriptorSetLayout, SHADER_DIR_STR + "/particle.vert.spv", SHADER_DIR_STR + "/particle.frag.spv");
    base_.createFrameData();
  }

  void recordComputeCmd(VkCommandBuffer cmd) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    engine_->step(cmd, dt_);

    VkBufferMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = base_.ctx.computeFamily;
    barrier.dstQueueFamilyIndex = base_.ctx.graphicsFamily;
    barrier.buffer              = engine_->getPositionBuffer();
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
    renderPc.posIdx        = engine_->posIdx;
    renderPc.velIdx        = engine_->velIdx;
    renderPc.particleCount = engine_->liveParticleCount();
    renderPc.worldMin      = glm::vec3(0.0f);
    renderPc.worldMax      = engine_->domainSize;
    graphicsPipe_.draw(cmd, engine_->descriptorSet, renderPc, engine_->liveParticleCount());

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
    graphicsPipe_.cleanup();
    engine_->cleanup();
    base_.cleanupBase();
  }
};

int main(int argc, char* argv[]) {
  auto args = argparse::parse<SpumeArgs>(argc, argv);
  SpumeApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
