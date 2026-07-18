#include "App.h"
#include "core/Emitter.h"
#include "core/Force.h"
#include "engine/FluidEngine.h"
#include "graphics/GraphicsPipeline.h"
#include "utils.hpp"

#include <argparse/argparse.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <glm/glm.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── シナリオ定数 ─────────────────────────────────────────────────────────────

static constexpr float kPi        = 3.14159265f;
static constexpr float kWorldSize = 20.0f;
static constexpr float kMoveStart = 2.0f;
static constexpr float kMoveEnd   = 18.0f;
static constexpr float kMoveDur   = 5.0f; // 拭き取り速度2倍 (旧10.0fの半分の時間で走破)
static constexpr float kMoveSpeed = (kMoveEnd - kMoveStart) / kMoveDur;
static constexpr float kCylRadius = 1.5f;  // 移動円柱の外径 [m]
static constexpr float kCylY      = 10.0f; // 移動円柱の Y 座標 [m]
static constexpr float kAbsRadius = 1.2f;  // 吸収 SDF 半径 [m]
static constexpr float kAbsRate   = 0.8f;  // デフォルト吸収率

// ── 境界粒子ヘルパー ─────────────────────────────────────────────────────────

// 円柱側面の境界粒子を out に追記し、追記した粒子数を返す
static uint32_t appendCylinder(float cx, float cy, float r, float zBot, float zTop, float spacing, std::vector<glm::vec4>& out) {
  float circumf   = 2.0f * kPi * r;
  int nRings      = std::max(1, (int)std::round((zTop - zBot) / spacing));
  int nPerRing    = std::max(4, (int)std::round(circumf / spacing));
  uint32_t before = (uint32_t)out.size();
  for(int ri = 0; ri <= nRings; ++ri) {
    float z = zBot + ri * (zTop - zBot) / float(nRings);
    for(int k = 0; k < nPerRing; ++k) {
      float a = k * 2.0f * kPi / float(nPerRing);
      out.push_back({cx + r * std::cos(a), cy + r * std::sin(a), z, 1.0f});
    }
  }
  return (uint32_t)out.size() - before;
}

// 矩形枠の境界粒子を out に追記する
static void appendRectBorder(float x1, float y1, float x2, float y2, float zBot, float zTop, float spacing, std::vector<glm::vec4>& out) {
  int nz = std::max(1, (int)std::round((zTop - zBot) / spacing));
  for(int ri = 0; ri <= nz; ++ri) {
    float z = zBot + ri * (zTop - zBot) / float(nz);
    for(float x = x1; x <= x2 + 1e-4f; x += spacing) {
      out.push_back({x, y1, z, 1.0f});
      out.push_back({x, y2, z, 1.0f});
    }
    for(float y = y1 + spacing; y < y2 - 1e-4f; y += spacing) {
      out.push_back({x1, y, z, 1.0f});
      out.push_back({x2, y, z, 1.0f});
    }
  }
}

// 移動円柱の位置・速度ベクトルをキネマティック用に生成する
static void makeCylinderKinematic(float cx, float cy, float r, float zBot, float zTop, float spacing, float vx, std::vector<glm::vec4>& pos, std::vector<glm::vec4>& vel) {
  float circumf = 2.0f * kPi * r;
  int nRings    = std::max(1, (int)std::round((zTop - zBot) / spacing));
  int nPerRing  = std::max(4, (int)std::round(circumf / spacing));
  pos.clear();
  vel.clear();
  for(int ri = 0; ri <= nRings; ++ri) {
    float z = zBot + ri * (zTop - zBot) / float(nRings);
    for(int k = 0; k < nPerRing; ++k) {
      float a = k * 2.0f * kPi / float(nPerRing);
      pos.push_back({cx + r * std::cos(a), cy + r * std::sin(a), z, 1.0f});
      vel.push_back({vx, 0.0f, 0.0f, 0.0f});
    }
  }
}

// ── CLI ───────────────────────────────────────────────────────────────────────

struct AbsorbArgs : public argparse::Args {
  float& dt                   = kwarg("dt", "timestep [s]").set_default(1.0f / 60.0f);
  float& rate                 = kwarg("rate", "absorption probability per substep [0,1]").set_default(kAbsRate);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

// ── App ───────────────────────────────────────────────────────────────────────

class FluidAbsorbApp {
public:
  void run(const AbsorbArgs& args) {
    dt_                 = args.dt;
    absorbRate_         = args.rate;
    base_.screenshotDir = args.screenshot_dir;

    FluidConfig cfg;
    cfg.fluid_nx     = 26;
    cfg.fluid_ny     = 26;
    cfg.fluid_nz     = 2;
    cfg.world_size   = kWorldSize;
    cfg.grid_res     = 13; // cellSize≈1.54m, h/d≈2.0 ✓ (spacing=20/26≈0.77m)
    cfg.max_boundary = 5000;

    base_.initWindow("Vulkan Sim – Fluid Absorb");
    initVulkan(cfg);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  FluidEngine engine_;
  GraphicsPipeline graphicsPipe_;
  std::shared_ptr<GravityForce> gravity_;

  float dt_         = 1.0f / 60.0f;
  float simTime_    = 0.0f;
  float absorbRate_ = kAbsRate;

  uint32_t movingBoundaryStart_ = 0;
  uint32_t nMoving_             = 0;
  std::vector<glm::vec4> movingPos_, movingVel_;

  void initVulkan(const FluidConfig& cfg) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR, cfg);

    gravity_ = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 9.8f); // Z-up
    engine_.addForce(gravity_);
    engine_.viscosityC    = 0.01f;
    engine_.pbfIterations = 2;
    engine_.numSubsteps   = 2;
    engine_.rho0          = 30.0f;
    engine_.linearDamping = 0.02f;

    // 境界粒子の Z 範囲
    const float spacing = cfg.particleSpacing() * 0.9f;
    const float zBot    = cfg.particleSpacing(); // ≈ 0.25m (floor 直上)
    const float zTop    = 2.0f;

    std::vector<glm::vec4> allBoundary;

    // 固定シリンダー×2 (水たまりの端付近)
    appendCylinder(5.0f, 15.0f, 0.7f, zBot, 1.5f, spacing, allBoundary);
    appendCylinder(15.0f, 15.0f, 0.7f, zBot, 1.5f, spacing, allBoundary);

    // 固定矩形障害物
    appendRectBorder(3.0f, 3.0f, 7.0f, 6.0f, zBot, 1.5f, spacing, allBoundary);

    uint32_t nStaticBoundary = (uint32_t)allBoundary.size();

    // 移動円柱の初期位置 (x = kMoveStart)
    nMoving_ = appendCylinder(kMoveStart, kCylY, kCylRadius, zBot, zTop, spacing, allBoundary);

    engine_.loadBoundaryParticles(allBoundary);
    engine_.initKinematicBoundaryStaging(nMoving_);

    movingBoundaryStart_ = nStaticBoundary; // 境界パーティクルは常に buffer index 0 から配置される
    movingPos_.resize(nMoving_);
    movingVel_.resize(nMoving_);

    // 楕円水たまりを一発投入 (XY 中央, 床面の1粒子間隔上)
    const float particleR   = cfg.cellSize() * 0.5f;             // SDF 衝突距離
    const float floorZ      = particleR + cfg.particleSpacing(); // 床ちょうど上から投下
    auto src                = std::make_shared<EllipseEmitter>();
    src->center             = glm::vec3(kWorldSize * 0.5f, kWorldSize * 0.5f, floorZ);
    src->semiA              = 5.0f;
    src->semiB              = 3.0f;
    src->vel                = glm::vec3(0.0f);
    src->particles_per_step = cfg.fluidCount();
    src->step_count         = -1; // 初回 1 回のみ
    engine_.addEmitter(src);

    graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/fluid_particle.vert.spv", SHADER_DIR_STR + "/fluid.frag.spv");

    base_.createFrameData();
    base_.initImGui();
  }

  void recordComputeCmd(VkCommandBuffer cmd) {
    // 容量拡張によるバッファ再確保は、下の recordKinematicBoundaryUpdate が
    // その時点の VkBuffer ハンドルをコマンドバッファへ焼き込む前に解決しておく
    engine_.emitFromEmitters(dt_);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    // 移動円柱の現在 X 座標
    float cylX          = std::min(kMoveStart + kMoveSpeed * simTime_, kMoveEnd);
    float vx            = (cylX < kMoveEnd) ? kMoveSpeed : 0.0f;
    const float spacing = engine_.config().particleSpacing() * 0.9f;
    const float zBot    = engine_.config().particleSpacing();
    makeCylinderKinematic(cylX, kCylY, kCylRadius, zBot, 2.0f, spacing, vx, movingPos_, movingVel_);
    engine_.recordKinematicBoundaryUpdate(cmd, base_.currentFrame, movingBoundaryStart_, nMoving_, movingPos_.data(), movingVel_.data());

    engine_.step(cmd, dt_);

    // compute → vertex シェーダー所有権移転バリア
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
    clear.color = {{0.02f, 0.04f, 0.08f, 1.0f}};

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

    // typeFlagIdx を渡して吸収済み粒子 (typeFlag==0) をクリップ
    SimPC pc{};
    pc.posIdx        = engine_.posIdx;
    pc.velIdx        = engine_.velIdx;
    pc.typeFlagIdx   = engine_.typeFlagIdx;
    pc.particleCount = engine_.nFluid();
    pc.worldMin      = 0.0f;
    pc.worldMax      = engine_.config().world_size;
    pc.boundaryStart = engine_.config().max_boundary; // 流体パーティクル領域の開始オフセット

    graphicsPipe_.draw(cmd, engine_.descriptorSet, pc, engine_.nFluid());

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
  }

  void drawFrame(int nShots) {
    auto& f = base_.frames[base_.currentFrame];
    vkWaitForFences(base_.ctx.device, 1, &f.inFlightFence, VK_TRUE, UINT64_MAX);

    // 吸収ポートを移動円柱に追従させて更新（GPU アップロードを含む）
    float cylX = std::min(kMoveStart + kMoveSpeed * simTime_, kMoveEnd);
    engine_.setAbsorbers({FluidEngine::AbsorberDesc::CylinderZ(cylX, kCylY, 0.0f, kAbsRadius, 2.0f, absorbRate_)});

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

    ImGui::Begin("Fluid Absorb");
    ImGui::Text("FPS: %.1f  |  流体: %u / %u  経過: %.2f s", ImGui::GetIO().Framerate, engine_.nFluid(), engine_.config().fluidCount(), simTime_);
    ImGui::Text("移動円柱 X: %.2f m  (%.0f%%)", cylX, (cylX - kMoveStart) / (kMoveEnd - kMoveStart) * 100.0f);
    ImGui::Separator();
    ImGui::SliderFloat("吸収率", &absorbRate_, 0.0f, 1.0f);
    ImGui::Separator();
    sim_ui::fluid_reset_button(engine_, simTime_);
    ImGui::Separator();
    sim_ui::fluid_params(engine_, *gravity_);
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
    graphicsPipe_.cleanup();
    engine_.cleanup();
    base_.cleanupBase();
  }
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  auto args = argparse::parse<AbsorbArgs>(argc, argv);
  FluidAbsorbApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
