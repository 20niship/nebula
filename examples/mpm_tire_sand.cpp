#include "App.h"
#include "Collider.h"
#include "MaterialParams.h"
#include "core/Profiling.h"
#include "engine/MPMEngine.h"
#include "graphics/GraphicsPipeline.h"

#include <argparse/argparse.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── CLI ───────────────────────────────────────────────────────────────────

struct TireSandArgs : public argparse::Args {
  float& domain_size_x        = kwarg("domain-size-x", "domain physical size X [m]").set_default(3.0f);
  float& domain_size_y        = kwarg("domain-size-y", "domain physical size Y [m]").set_default(3.0f);
  float& domain_size_z        = kwarg("domain-size-z", "domain physical size Z [m]").set_default(3.0f);
  float& cell_size            = kwarg("cell-size", "MPM grid cell size [m]").set_default(3.0f / 128.0f);
  float& dt                   = kwarg("dt", "frame timestep [s]").set_default(1.0f / 60.0f);
  int& substeps               = kwarg("substeps", "substeps per frame").set_default(40);
  int& sand_nx                = kwarg("sand-nx", "砂粒子配置 X方向グリッド数").set_default(300);
  int& sand_nz                = kwarg("sand-nz", "砂粒子配置 Z方向グリッド数").set_default(300);
  int& sand_layers            = kwarg("sand-layers", "砂粒子配置 厚み方向レイヤ数").set_default(4);
  float& sand_thickness       = kwarg("sand-thickness", "砂堆積層の厚み [m]").set_default(0.05f);
  float& tire_radius          = kwarg("tire-radius", "タイヤ半径 [m]").set_default(0.35f);
  float& tire_width           = kwarg("tire-width", "タイヤ幅(車軸方向) [m]").set_default(0.25f);
  float& tire_speed           = kwarg("tire-speed", "タイヤ移動速度 [m/s]").set_default(1.5f);
  float& tire_embed           = kwarg("tire-embed", "タイヤが砂層に沈み込む深さ [m]").set_default(0.12f);
  float& slip_ratio           = kwarg("slip-ratio", "すべりなし転がり角速度に対する倍率(1=スリップなし、>1でスピンして土を巻き上げる)").set_default(10.0f);
  int& launch_frame           = kwarg("launch-frame", "タイヤが自動発進するフレーム").set_default(30);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

// ── App ───────────────────────────────────────────────────────────────────

class TireSandApp {
public:
  void run(const TireSandArgs& args) {
    dt_                 = args.dt;
    base_.screenshotDir = args.screenshot_dir;
    tireSpeed_          = args.tire_speed;
    tireRadius_         = args.tire_radius;
    tireEmbed_          = args.tire_embed;
    slipRatio_          = args.slip_ratio;
    launchFrame_        = args.launch_frame;
    sandThickness_      = args.sand_thickness;

    engine_.domainSize   = glm::vec3(args.domain_size_x, args.domain_size_y, args.domain_size_z);
    engine_.cellSize     = args.cell_size;
    engine_.maxParticles = uint32_t(args.sand_nx) * uint32_t(args.sand_nz) * uint32_t(args.sand_layers);

    base_.initWindow("MPM Tire Sand — 転がるタイヤと砂の巻き上げ");
    initVulkan(args.substeps, args.sand_nx, args.sand_nz, args.sand_layers, args.tire_width);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  MPMEngine engine_;
  GraphicsPipeline graphicsPipe_;
  std::shared_ptr<GravityForce> gravity_;
  float dt_        = 1.0f / 60.0f;
  float simTime_   = 0.0f;
  int frameCount_  = 0;
  int launchFrame_ = 30;

  float tirePosX_      = 0.0f;
  float tireSpeed_     = 1.5f;
  float tireRadius_    = 0.35f;
  float tireEmbed_     = 0.12f;
  float slipRatio_     = 10.0f;
  float tireCenterY_   = 0.0f;
  float sandThickness_ = 0.05f;
  bool tireMoving_     = false;

  float tireHalfHeight_ = 0.0f;
  glm::quat baseRot_    = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // ローカルY(車軸)をワールドZへ向ける固定回転
  glm::quat spinQuat_   = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // 転がりによる蓄積スピン(ワールドZ軸まわり)

  // 砂粒子配置: ドメイン全体(余白5%)をsandThickness厚みで敷き詰める
  void placeSand(int nx_p, int nz_p, int nlay) {
    const float x0 = engine_.domainSize.x * 0.05f, x1 = engine_.domainSize.x * 0.95f;
    const float z0 = engine_.domainSize.z * 0.05f, z1 = engine_.domainSize.z * 0.95f;
    const float dx = (x1 - x0) / float(nx_p);
    const float dz = (z1 - z0) / float(nz_p);
    const float dy = sandThickness_ / float(nlay);
    const float Vp = dx * dz * dy;

    const int maxN = int(engine_.maxParticles);
    std::vector<glm::vec4> pos, vel;
    pos.reserve(std::min(nx_p * nz_p * nlay, maxN));
    vel.reserve(pos.capacity());
    for(int iz = 0; iz < nz_p && int(pos.size()) < maxN; ++iz)
      for(int ix = 0; ix < nx_p && int(pos.size()) < maxN; ++ix) {
        float px = x0 + (ix + 0.5f) * dx;
        float pz = z0 + (iz + 0.5f) * dz;
        for(int iy = 0; iy < nlay && int(pos.size()) < maxN; ++iy) {
          float py = (iy + 0.5f) * dy;
          pos.push_back({px, py, pz, Vp});
          vel.push_back({0.0f, 0.0f, 0.0f, 0.0f});
        }
      }
    std::printf("  砂粒子配置: %zu 個 (最大 %d)\n", pos.size(), maxN);
    engine_.appendParticles(pos, vel);
  }

  // 床(平面) + ドメイン4壁 + 転がるタイヤ(MESH_SDF)をまとめて登録
  void rebuildColliders() {
    const glm::vec3 ws = engine_.domainSize;
    ColliderSet cols;
    cols.addPlane({ws.x * 0.5f, 0.0f, ws.z * 0.5f}, {0, 1, 0}, 0.0f, 0.5f);
    cols.addPlane({0.0f, ws.y * 0.5f, ws.z * 0.5f}, {1, 0, 0}, 0.0f, 0.1f);
    cols.addPlane({ws.x, ws.y * 0.5f, ws.z * 0.5f}, {-1, 0, 0}, 0.0f, 0.1f);
    cols.addPlane({ws.x * 0.5f, ws.y * 0.5f, 0.0f}, {0, 0, 1}, 0.0f, 0.1f);
    cols.addPlane({ws.x * 0.5f, ws.y * 0.5f, ws.z}, {0, 0, -1}, 0.0f, 0.1f);

    glm::vec3 vel = tireMoving_ ? glm::vec3(-tireSpeed_, 0.0f, 0.0f) : glm::vec3(0.0f);
    glm::vec3 pos(tirePosX_, tireCenterY_, ws.z * 0.5f);
    // すべりなし転がり角速度(接地点速度0)に slipRatio_ を掛けてスピンさせ、スリップで土を巻き上げる
    glm::vec3 angVel = tireMoving_ ? glm::vec3(0.0f, 0.0f, slipRatio_ * tireSpeed_ / tireRadius_) : glm::vec3(0.0f);
    cols.addCylinder(pos, tireRadius_, tireHalfHeight_, 0.1f, 0.6f, vel, meshRot(), angVel);
    engine_.setColliders(cols);
  }

  glm::quat meshRot() const { return spinQuat_ * baseRot_; }

  void initVulkan(int substeps, int sandNx, int sandNz, int sandLayers, float tireWidth) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR);
    engine_.numSubsteps = substeps;
    gravity_            = GravityForce::FromDirection({0.0f, -1.0f, 0.0f}, 9.8f); // Y-up
    engine_.addForce(gravity_);
    engine_.flip_ratio = 0.0f; // PIC (DP材料での発散を避ける、mpm_avalancheと同じ知見)

    engine_.setMaterials({presetSand(5e4f, 0.3f, 1600.0f)});

    baseRot_        = glm::angleAxis(glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)); // ローカルY(高さ)→ワールドZ(車軸)
    tireHalfHeight_ = tireWidth * 0.5f;
    tireCenterY_    = tireRadius_ + sandThickness_ - tireEmbed_;
    tirePosX_       = engine_.domainSize.x * 0.85f;
    rebuildColliders();

    placeSand(sandNx, sandNz, sandLayers);

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
    clear.color = {{0.05f, 0.04f, 0.03f, 1.0f}};

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
    ZoneScoped;
    auto& f = base_.frames[base_.currentFrame];
    {
      ZoneScopedN("WaitForFence");
      vkWaitForFences(base_.ctx.device, 1, &f.inFlightFence, VK_TRUE, UINT64_MAX);
    }

    uint32_t imageIdx;
    VkResult result = vkAcquireNextImageKHR(base_.ctx.device, base_.ctx.swapchain, UINT64_MAX, f.imageAvailable, VK_NULL_HANDLE, &imageIdx);
    if(result == VK_ERROR_OUT_OF_DATE_KHR) {
      base_.ctx.recreateSwapchain();
      return;
    }

    vkResetFences(base_.ctx.device, 1, &f.inFlightFence);

    // 固定フレームで自動発進、端に到達したら停止 → 次に自動再発進(ループ)
    if(!tireMoving_ && launchFrame_ >= 0 && frameCount_ >= launchFrame_) {
      tirePosX_   = engine_.domainSize.x * 0.85f;
      spinQuat_   = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
      tireMoving_ = true;
    }
    if(tireMoving_) {
      tirePosX_ -= tireSpeed_ * dt_;
      float angVelZ = slipRatio_ * tireSpeed_ / tireRadius_;
      spinQuat_     = glm::angleAxis(angVelZ * dt_, glm::vec3(0.0f, 0.0f, 1.0f)) * spinQuat_;
      if(tirePosX_ - tireRadius_ < engine_.domainSize.x * 0.1f) tireMoving_ = false;
      ZoneScopedN("RebuildColliders");
      rebuildColliders();
    }
    simTime_ += dt_;
    ++frameCount_;

    f.timelineValue++;
    vkResetCommandBuffer(f.computeCmd, 0);
    {
      ZoneScopedN("RecordComputeCmd");
      recordComputeCmd(f.computeCmd);
    }

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
    {
      ZoneScopedN("QueueSubmitCompute");
      vkQueueSubmit(base_.ctx.computeQueue, 1, &compSub, VK_NULL_HANDLE);
    }

    vkResetCommandBuffer(f.graphicsCmd, 0);
    {
      ZoneScopedN("RecordGraphicsCmd");
      recordGraphicsCmd(f.graphicsCmd, imageIdx);
    }

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
    {
      ZoneScopedN("QueueSubmitGraphics");
      vkQueueSubmit(base_.ctx.graphicsQueue, 1, &grSub, f.inFlightFence);
    }

    VkPresentInfoKHR present{};
    present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores    = &f.renderFinished;
    present.swapchainCount     = 1;
    present.pSwapchains        = &base_.ctx.swapchain;
    present.pImageIndices      = &imageIdx;
    if(nShots > 0) {
      ZoneScopedN("SaveScreenshot");
      base_.saveScreenshot(imageIdx, nShots);
    }
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
  auto args = argparse::parse<TireSandArgs>(argc, argv);
  TireSandApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
