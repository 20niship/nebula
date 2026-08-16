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

#include <algorithm>
#include <array>
#include <cstdio>
#include <glm/glm.hpp>
#include <stdexcept>
#include <string>
#include <vk_mem_alloc.h>

static const std::string SHADER_DIR_STR = SHADER_DIR;
static const std::string ASSET_DIR_STR  = ASSET_DIR;

// ── CLI ───────────────────────────────────────────────────────────────────────

struct FluidArgs : public argparse::Args {
  float& domain_size_x        = kwarg("domain-size-x", "domain physical size X [m]").set_default(20.0f);
  float& domain_size_y        = kwarg("domain-size-y", "domain physical size Y [m]").set_default(20.0f);
  float& domain_size_z        = kwarg("domain-size-z", "domain physical size Z [m]").set_default(20.0f);
  float& cell_size            = kwarg("cell-size", "hash grid cell size [m]").set_default(20.0f / 32.0f);
  int& max_boundary           = kwarg("max-boundary", "max boundary particle count").set_default(200000);
  float& dt                   = kwarg("dt", "timestep (sec)").set_default(1.0f / 60.0f);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
  std::string& boundary_obj   = kwarg("boundary-obj", "auto-load boundary OBJ file").set_default(std::string(""));
  float& boundary_spacing     = kwarg("boundary-spacing", "boundary particle spacing").set_default(0.156f);
  float& rho0                 = kwarg("rho0", "rest density (0=auto from h/d ratio)").set_default(0);
  float& viscosity            = kwarg("viscosity", "XSPH viscosity c").set_default(0);
  float& cfm_eps              = kwarg("cfm-eps", "CFM relaxation epsilon").set_default(3000.0f);
  float& scorr_k              = kwarg("scorr-k", "artificial pressure k").set_default(0.0f);
  float& surface_tension      = kwarg("surface-tension", "surface tension cohesion sigma").set_default(0.0f);
  float& damping              = kwarg("damping", "linear velocity damping 1/s").set_default(0.6f);
  std::string& scenario       = kwarg("scenario", "dam-break | source-flow").set_default(std::string("dam-break"));
  int& max_diffuse            = kwarg("max-diffuse", "max spray/foam/bubble diffuse particle count (0=disabled, issue #47)").set_default(0);
};

// ── App ───────────────────────────────────────────────────────────────────────

class FluidApp {
public:
  void run(const FluidArgs& args) {
    dt_                 = args.dt;
    base_.screenshotDir = args.screenshot_dir;
    bSpacing_           = args.boundary_spacing;

    FluidConfig cfg;
    cfg.particleRadius      = (args.domain_size_x / 192.0f) / 2.0f; // 旧デフォルト nx=192 相当の粒子解像度
    cfg.domainSize          = glm::vec3(args.domain_size_x, args.domain_size_y, args.domain_size_z);
    cfg.cellSize            = args.cell_size;
    cfg.max_boundary        = (uint32_t)args.max_boundary;
    cfg.maxDiffuseParticles = (uint32_t)args.max_diffuse;

    base_.initWindow("Vulkan Sim – PBF Fluid");
    initVulkan(cfg, args.boundary_obj, args.rho0);
    engine_.viscosityC    = args.viscosity;
    engine_.cfmEpsilon    = args.cfm_eps;
    engine_.scorrK        = args.scorr_k;
    engine_.surfaceTension = args.surface_tension;
    engine_.linearDamping = args.damping;
    setupScenario(args.scenario, cfg);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  FluidEngine engine_;
  GraphicsPipeline graphicsPipe_;
  GraphicsPipeline foamGraphicsPipe_; // 泡 (spray/foam/bubble) 描画用（半透明合成; issue #47）
  std::shared_ptr<GravityForce> gravity_;
  FluidEngine::FoamParams foamParams_;

  float dt_       = 1.0f / 60.0f;
  float simTime_  = 0.0f;
  float bSpacing_ = 0.156f;

  char objPath_[256] = {};
  std::string loadStatus_;

  float nextDiagTime_                  = 0.0f;
  static constexpr float DIAG_INTERVAL = 1.0f;

  void setupScenario(const std::string& scenario, const FluidConfig& cfg) {
    const glm::vec3 w = cfg.domainSize;
    const float m     = cfg.cellSize * 0.5f;      // margin
    const float d     = cfg.particleSpacing();

    if(scenario == "source-flow") {
      // TC2: 左端から右方向へ移動するボックスソース
      // X が広いドメイン (domain-size-x=40 を推奨) で左から右へ流体が噴出
      auto src                = std::make_shared<AABBEmitter>();
      src->center             = glm::vec3(w.x * 0.05f, w.y * 0.5f, w.z * 0.5f);
      src->size               = glm::vec3(w.x * 0.07f, w.y * 0.35f, w.z * 0.35f);
      src->center_vel         = glm::vec3(w.x * 0.10f, 0.0f, 0.0f); // 10% domain/s で右移動
      src->vel                = glm::vec3(w.x * 0.08f, 0.0f, 0.0f); // 放出粒子に右向き初速
      const uint32_t boxCount = (uint32_t)(src->size.x * src->size.y * src->size.z / (d * d * d));
      src->particles_per_step = std::max(1u, boxCount / 400u);
      src->step_count         = 0; // 無限
      engine_.addEmitter(src);
    } else {
      // dam-break (デフォルト): 左半分上部 (X: 左半分, Z: 上半分) を一気に充填
      auto src                = std::make_shared<AABBEmitter>();
      src->center             = glm::vec3(w.x * 0.25f, w.y * 0.5f, w.z * 0.75f);
      src->size               = glm::vec3(w.x * 0.5f - 2.0f * m, w.y - 2.0f * m, w.z * 0.5f - 2.0f * m);
      src->vel                = glm::vec3(0.0f);
      src->particles_per_step = (uint32_t)(src->size.x * src->size.y * src->size.z / (d * d * d)); // 箱を一気に充填
      src->step_count         = -1;               // 1回のみ
      engine_.addEmitter(src);
    }
  }

  void initVulkan(const FluidConfig& cfg, const std::string& boundaryObj, float rho0Arg) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR, cfg);
    if(rho0Arg > 0.0f) engine_.rho0 = rho0Arg;

    gravity_ = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 9.8f); // Z-up
    engine_.addForce(gravity_);

    std::snprintf(objPath_, sizeof(objPath_), "%s", (ASSET_DIR_STR + "/sphere.obj").c_str());
    if(!boundaryObj.empty()) {
      try {
        engine_.loadBoundary(boundaryObj, bSpacing_);
        loadStatus_ = "OK: " + std::to_string(engine_.nBoundary) + " boundary particles";
      } catch(const std::exception& e) {
        loadStatus_ = std::string("Error: ") + e.what();
      }
    }

    graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/fluid_particle.vert.spv", SHADER_DIR_STR + "/fluid.frag.spv");

    // 泡 (spray/foam/bubble) 描画パイプライン (issue #47)。maxDiffuseParticles==0 でも
    // パイプライン自体は安価に作れるため無条件で初期化し、draw() 呼び出し側で
    // config().maxDiffuseParticles>0 のときのみ描画する。
    foamGraphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/foam_particle.vert.spv", SHADER_DIR_STR + "/foam.frag.spv", VK_PRIMITIVE_TOPOLOGY_POINT_LIST, /*enableBlend=*/true);
    if(cfg.maxDiffuseParticles > 0) {
      engine_.foamEnabled = true;
      engine_.setFoamParams(foamParams_);
    }

    base_.createFrameData();
    base_.initImGui();
  }

  void recordComputeCmd(VkCommandBuffer cmd) {
    // 容量拡張によるバッファ再確保はコマンドバッファ記録前に解決しておく
    engine_.emitFromEmitters(dt_);

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

    SimPC pc{};
    pc.posIdx        = engine_.posIdx();
    pc.velIdx        = engine_.velIdx();
    pc.particleCount = engine_.nFluid();
    pc.worldMin      = glm::vec3(0.0f);
    pc.worldMax      = engine_.config().domainSize;
    pc.boundaryStart = engine_.config().max_boundary; // 流体パーティクル領域の開始オフセット

    graphicsPipe_.draw(cmd, engine_.descriptorSet, pc, engine_.nFluid());

    // 泡 (spray/foam/bubble) 描画（issue #47; maxDiffuseParticles==0 のとき完全スキップ）
    if(engine_.config().maxDiffuseParticles > 0) {
      SimPC foamPc{};
      foamPc.posIdx        = engine_.foamPosIdx();
      foamPc.velIdx        = engine_.foamVelIdx();
      foamPc.typeFlagIdx   = engine_.foamKindIdx(); // foam_particle.vert は kind==0 をクリップする
      foamPc.particleCount = engine_.config().maxDiffuseParticles;
      foamPc.worldMin      = glm::vec3(0.0f);
      foamPc.worldMax      = engine_.config().domainSize;
      foamPc.boundaryStart = 0;
      foamGraphicsPipe_.draw(cmd, engine_.descriptorSet, foamPc, engine_.config().maxDiffuseParticles);
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

    ImGui::Begin("PBF Fluid Control");
    ImGui::Text("FPS: %.1f  |  流体: %u / %u  境界: %u  経過: %.2f s", ImGui::GetIO().Framerate, engine_.nFluid(), engine_.config().fluidCount(), engine_.nBoundary, simTime_);
    ImGui::Separator();
    sim_ui::fluid_reset_button(engine_, simTime_);
    ImGui::Separator();
    sim_ui::fluid_params(engine_, *gravity_);
    ImGui::Separator();
    if(sim_ui::foam_params(engine_, foamParams_)) engine_.setFoamParams(foamParams_);
    ImGui::Separator();
    ImGui::Text("境界粒子 (OBJ)");
    ImGui::InputText("OBJ パス", objPath_, sizeof(objPath_));
    ImGui::SliderFloat("粒子間隔", &bSpacing_, 0.05f, 0.5f);
    if(ImGui::IsItemHovered()) ImGui::SetTooltip("境界粒子の配置間隔 [m]。小さいほど密になるがメモリが増える。");
    if(ImGui::Button("ロード")) {
      try {
        engine_.loadBoundary(objPath_, bSpacing_);
        loadStatus_ = "OK: " + std::to_string(engine_.nBoundary) + " 境界粒子";
      } catch(const std::exception& e) {
        loadStatus_ = std::string("エラー: ") + e.what();
      }
    }
    ImGui::SameLine();
    if(ImGui::Button("クリア")) {
      engine_.clearBoundary();
      loadStatus_ = "クリア済み";
    }
    if(!loadStatus_.empty()) ImGui::TextWrapped("%s", loadStatus_.c_str());
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
    foamGraphicsPipe_.cleanup();
    engine_.cleanup();
    base_.cleanupBase();
  }
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  auto args = argparse::parse<FluidArgs>(argc, argv);
  FluidApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
