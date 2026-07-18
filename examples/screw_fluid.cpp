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

// ── CLI ───────────────────────────────────────────────────────────────────────

struct ScrewArgs : public argparse::Args {
  int& nx                     = kwarg("nx", "fluid grid X").set_default(192);
  int& ny                     = kwarg("ny", "fluid grid Y").set_default(192);
  int& nz                     = kwarg("nz", "fluid grid Z").set_default(6);
  float& domain_size_x        = kwarg("domain-size-x", "domain physical size X [m]").set_default(20.0f);
  float& domain_size_y        = kwarg("domain-size-y", "domain physical size Y [m]").set_default(20.0f);
  float& domain_size_z        = kwarg("domain-size-z", "domain physical size Z [m]").set_default(20.0f);
  float& cell_size            = kwarg("cell-size", "spatial hash cell size [m]").set_default(20.0f / 64.0f);
  float& dt                   = kwarg("dt", "timestep (sec)").set_default(1.0f / 60.0f);
  float& ang_vel              = kwarg("ang-vel", "screw angular velocity [rad/s]").set_default(2.0f);
  float& viscosity            = kwarg("viscosity", "XSPH viscosity coefficient").set_default(0.05f);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

// ── 手続きスクリュー形状生成 ───────────────────────────────────────────────────

static void sampleTri(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, float spacing, std::vector<glm::vec4>& out) {
  glm::vec3 ab = b - a;
  glm::vec3 ac = c - a;
  float lenAB  = glm::length(ab);
  float lenAC  = glm::length(ac);
  if(lenAB < 1e-6f || lenAC < 1e-6f) return;

  int stepsU = std::max(1, int(lenAB / spacing));
  int stepsV = std::max(1, int(lenAC / spacing));
  for(int iu = 0; iu <= stepsU; ++iu) {
    float u = float(iu) / float(stepsU);
    for(int iv = 0; iv <= stepsV - iu; ++iv) {
      float v = float(iv) / float(stepsV);
      if(u + v > 1.0f + 1e-5f) continue;
      out.push_back(glm::vec4(a + u * ab + v * ac, 1.0f));
    }
  }
}

// 4枚羽根プロペラ型境界粒子を生成する（rest フレーム, angle=0）
// pivotXY: XY 面の回転中心
// zBottom/zTop: ブレード Z 範囲
// outerRadius: 外径, shaftRadius: シャフト半径
// numBlades: 羽根枚数, spacing: 粒子間隔
static std::vector<glm::vec4> generate4BladePropeller(glm::vec2 pivotXY, float zBottom, float zTop, float outerRadius, float shaftRadius, int numBlades, float spacing) {
  std::vector<glm::vec4> pts;
  const float pi2 = 2.0f * 3.14159265f;

  // ① シャフト円柱側面
  const int shaftSegs = 32;
  for(int k = 0; k < shaftSegs; ++k) {
    float th0 = float(k) / float(shaftSegs) * pi2;
    float th1 = float(k + 1) / float(shaftSegs) * pi2;
    glm::vec3 b0(pivotXY.x + shaftRadius * std::cos(th0), pivotXY.y + shaftRadius * std::sin(th0), zBottom);
    glm::vec3 b1(pivotXY.x + shaftRadius * std::cos(th1), pivotXY.y + shaftRadius * std::sin(th1), zBottom);
    glm::vec3 t0(b0.x, b0.y, zTop);
    glm::vec3 t1(b1.x, b1.y, zTop);
    sampleTri(b0, b1, t0, spacing, pts);
    sampleTri(b1, t1, t0, spacing, pts);
  }

  // ② numBlades 枚の平板ブレード（等間隔, 半径方向に延伸）
  // 各ブレードは角度 θ 方向のラジアル面（R-Z 面）に配置する。
  // 回転することで接線方向の運動量を XSPH 粘性を通じて流体に伝達する。
  for(int b = 0; b < numBlades; ++b) {
    float theta = float(b) / float(numBlades) * pi2;
    float cosT  = std::cos(theta);
    float sinT  = std::sin(theta);

    glm::vec3 innerBottom(pivotXY.x + shaftRadius * cosT, pivotXY.y + shaftRadius * sinT, zBottom);
    glm::vec3 outerBottom(pivotXY.x + outerRadius * cosT, pivotXY.y + outerRadius * sinT, zBottom);
    glm::vec3 innerTop(innerBottom.x, innerBottom.y, zTop);
    glm::vec3 outerTop(outerBottom.x, outerBottom.y, zTop);

    sampleTri(innerBottom, outerBottom, innerTop, spacing, pts);
    sampleTri(outerBottom, outerTop, innerTop, spacing, pts);
  }
  return pts;
}

// ── KinematicScrew ────────────────────────────────────────────────────────────

struct KinematicScrew {
  std::vector<glm::vec4> restPositions; // angle=0 のワールド座標
  glm::vec2 pivotXY;
  float angVelZ      = 2.0f; // [rad/s]
  uint32_t gpuOffset = 0;    // GPU バッファ先頭インデックス (= fluidCount)
  uint32_t count     = 0;
  float currentAngle = 0.0f;

  glm::vec4 rotateParticle(const glm::vec4& p, float angle) const {
    float c = std::cos(angle), s = std::sin(angle);
    float rx = p.x - pivotXY.x, ry = p.y - pivotXY.y;
    return glm::vec4(c * rx - s * ry + pivotXY.x, s * rx + c * ry + pivotXY.y, p.z, 1.0f);
  }

  // v = ω × (pos - pivot);  ω = (0, 0, angVelZ) → v = (-ω*dy, ω*dx, 0)
  glm::vec4 kineVelocity(const glm::vec4& worldPos) const {
    float dx = worldPos.x - pivotXY.x;
    float dy = worldPos.y - pivotXY.y;
    return glm::vec4(-angVelZ * dy, angVelZ * dx, 0.0f, 0.0f);
  }
};

// ── App ───────────────────────────────────────────────────────────────────────

class ScrewFluidApp {
public:
  void run(const ScrewArgs& args) {
    dt_                 = args.dt;
    base_.screenshotDir = args.screenshot_dir;

    FluidConfig cfg;
    cfg.fluid_nx     = uint32_t(args.nx);
    cfg.fluid_ny     = uint32_t(args.ny);
    cfg.fluid_nz     = uint32_t(args.nz);
    cfg.domainSize   = glm::vec3(args.domain_size_x, args.domain_size_y, args.domain_size_z);
    cfg.cellSize     = args.cell_size;
    cfg.max_boundary = 20000;

    base_.initWindow("Vulkan Sim – Screw Fluid (TC8)");
    initVulkan(cfg, args);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  FluidEngine engine_;
  GraphicsPipeline graphicsPipe_;
  std::shared_ptr<GravityForce> gravity_;

  KinematicScrew screw_;
  std::vector<glm::vec4> screwNewPos_;
  std::vector<glm::vec4> screwNewVel_;

  float dt_      = 1.0f / 60.0f;
  float simTime_ = 0.0f;

  void initVulkan(const FluidConfig& cfg, const ScrewArgs& args) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR, cfg);

    gravity_ = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 9.8f); // Z-up
    engine_.addForce(gravity_);
    engine_.viscosityC    = args.viscosity;
    engine_.pbfIterations = 2;
    engine_.numSubsteps   = 2;
    engine_.rho0          = 30;

    // ── スクリュー境界粒子を手続き生成 → GPU 登録 ──────────────────────────
    const glm::vec3 w = cfg.domainSize;
    screw_.pivotXY     = glm::vec2(w.x * 0.5f, w.y * 0.5f);
    screw_.angVelZ   = args.ang_vel;
    screw_.gpuOffset = 0; // 境界パーティクルは常に buffer index 0 から配置される

    // 4枚羽根プロペラ：内径 0.3m, 外径 5m, Z=1.5〜7.5m, 粒子間隔 0.15m
    screw_.restPositions = generate4BladePropeller(screw_.pivotXY, 0.01f, 1.0f, 7.0f, 0.3f, 4, 0.15f);
    screw_.count         = uint32_t(std::min(screw_.restPositions.size(), size_t(cfg.max_boundary)));

    // invMass=0, typeFlag=3 を含む境界初期化
    engine_.loadBoundaryParticles(std::vector<glm::vec4>(screw_.restPositions.begin(), screw_.restPositions.begin() + screw_.count));

    screwNewPos_.resize(screw_.count);
    screwNewVel_.resize(screw_.count);

    // kinematic staging バッファを確保
    engine_.initKinematicBoundaryStaging(screw_.count);

    // ── 流体ソース: スクリュー周囲を充填 ──────────────────────────────────
    // スクリュー外径 5m に合わせた 12m 角タンクに集中配置して粒子密度を上げる
    auto src                = std::make_shared<AABBEmitter>();
    src->center             = glm::vec3(w.x * 0.2f, w.y * 0.3f, 4.0f);
    src->size               = glm::vec3(4.0f, 3.0f, 4.0f);
    src->vel                = glm::vec3(0.0f);
    src->particles_per_step = 1024;
    src->step_count         = 210;
    src->particleType       = 1u;
    engine_.addEmitter(src);

    graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/fluid_particle.vert.spv", SHADER_DIR_STR + "/fluid.frag.spv");

    base_.createFrameData();
    base_.initImGui();
  }

  void recordComputeCmd(VkCommandBuffer cmd) {
    // 容量拡張によるバッファ再確保は、下の recordKinematicBoundaryUpdate が
    // その時点の VkBuffer ハンドルをコマンドバッファへ焼き込む前に解決しておく
    // (先に解決しないと、再確保で破棄された古いバッファへのコピーが
    // コマンドバッファに残ってしまい GPU 実行時にクラッシュする)
    engine_.emitFromEmitters(dt_);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    // 境界粒子を現フレームの位置・速度で更新してから simulation step
    engine_.recordKinematicBoundaryUpdate(cmd, base_.currentFrame, screw_.gpuOffset, screw_.count, screwNewPos_.data(), screwNewVel_.data());

    engine_.step(cmd, dt_);

    // compute → vertex shader の所有権移転バリア
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

    // 流体粒子 + 境界（スクリュー）を描画（バッファ上は非連続のため2回に分けて描画）
    SimPC pc{};
    pc.posIdx   = engine_.posIdx;
    pc.velIdx   = engine_.velIdx;
    pc.worldMin = glm::vec3(0.0f);
    pc.worldMax = engine_.config().domainSize;

    // 境界（スクリュー）: buffer index 0 から
    pc.boundaryStart = 0;
    pc.particleCount = engine_.nBoundary;
    graphicsPipe_.draw(cmd, engine_.descriptorSet, pc, engine_.nBoundary);

    // 流体
    pc.boundaryStart = engine_.config().max_boundary;
    pc.particleCount = engine_.nFluid();
    graphicsPipe_.draw(cmd, engine_.descriptorSet, pc, engine_.nFluid());

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
  }

  void drawFrame(int nShots) {
    auto& f = base_.frames[base_.currentFrame];
    vkWaitForFences(base_.ctx.device, 1, &f.inFlightFence, VK_TRUE, UINT64_MAX);

    // vkWaitForFences 後 = このスロットの staging buffer は安全に書き込める
    screw_.currentAngle += screw_.angVelZ * dt_;
    for(uint32_t i = 0; i < screw_.count; ++i) {
      screwNewPos_[i] = screw_.rotateParticle(screw_.restPositions[i], screw_.currentAngle);
      screwNewVel_[i] = screw_.kineVelocity(screwNewPos_[i]);
    }

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

    ImGui::Begin("Screw Fluid (TC8)");
    ImGui::Text("FPS: %.1f  |  流体: %u / %u  t=%.2f s", ImGui::GetIO().Framerate, engine_.nFluid(), engine_.config().fluidCount(), simTime_);
    ImGui::Text("スクリュー: %.1f deg/s  |  境界粒子: %u", screw_.angVelZ * (180.0f / 3.14159265f), screw_.count);
    ImGui::Separator();
    sim_ui::fluid_reset_button(engine_, simTime_);
    ImGui::Separator();
    ImGui::SliderFloat("重力", &gravity_->strength, 0.0f, 15.0f);
    ImGui::SliderFloat("粘性係数", &engine_.viscosityC, 0.0f, 0.1f, "%.4f");
    ImGui::SliderFloat("角速度 [rad/s]", &screw_.angVelZ, -10.0f, 10.0f, "%.2f");
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
    tsWait.sType                                   = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tsWait.waitSemaphoreValueCount                 = 2;
    tsWait.pWaitSemaphoreValues                    = waitVals.data();
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
  auto args = argparse::parse<ScrewArgs>(argc, argv);
  ScrewFluidApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
