#include "App.h"
#include "core/Emitter.h"
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
  int& fluid_nx               = kwarg("nx", "fluid grid X").set_default(192);
  int& fluid_ny               = kwarg("ny", "fluid grid Y").set_default(3);
  int& fluid_nz               = kwarg("nz", "fluid grid Z").set_default(192);
  float& world_size           = kwarg("world-size", "simulation world size").set_default(20.0f);
  int& grid_res               = kwarg("grid-res", "hash grid resolution").set_default(64);
  int& max_boundary           = kwarg("max-boundary", "max boundary particle count").set_default(50000);
  float& dt                   = kwarg("dt", "timestep (sec)").set_default(1.0f / 60.0f);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
  std::string& boundary_obj   = kwarg("boundary-obj", "auto-load boundary OBJ file").set_default(std::string(""));
  float& boundary_spacing     = kwarg("boundary-spacing", "boundary particle spacing").set_default(0.156f);
  float& rho0                 = kwarg("rho0", "rest density (0=auto from h/d ratio)").set_default(0);
  float& viscosity            = kwarg("viscosity", "XSPH viscosity c").set_default(0);
  float& cfm_eps              = kwarg("cfm-eps", "CFM relaxation epsilon").set_default(3000.0f);
  float& scorr_k              = kwarg("scorr-k", "artificial pressure k").set_default(0.0f);
  float& damping              = kwarg("damping", "linear velocity damping 1/s").set_default(0.6f);
  std::string& scenario       = kwarg("scenario", "dam-break | source-flow").set_default(std::string("dam-break"));
};

// ── App ───────────────────────────────────────────────────────────────────────

class FluidApp {
public:
  void run(const FluidArgs& args) {
    dt_                 = args.dt;
    base_.screenshotDir = args.screenshot_dir;
    bSpacing_           = args.boundary_spacing;

    FluidConfig cfg;
    cfg.fluid_nx     = (uint32_t)args.fluid_nx;
    cfg.fluid_ny     = (uint32_t)args.fluid_ny;
    cfg.fluid_nz     = (uint32_t)args.fluid_nz;
    cfg.world_size   = args.world_size;
    cfg.grid_res     = (uint32_t)args.grid_res;
    cfg.max_boundary = (uint32_t)args.max_boundary;

    base_.initWindow("Vulkan Sim – PBF Fluid");
    initVulkan(cfg, args.boundary_obj, args.rho0);
    engine_.viscosityC    = args.viscosity;
    engine_.cfmEpsilon    = args.cfm_eps;
    engine_.scorrK        = args.scorr_k;
    engine_.linearDamping = args.damping;
    setupScenario(args.scenario, cfg);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  FluidEngine engine_;
  GraphicsPipeline graphicsPipe_;

  float dt_       = 1.0f / 60.0f;
  float simTime_  = 0.0f;
  float bSpacing_ = 0.156f;

  char objPath_[256] = {};
  std::string loadStatus_;

  float nextDiagTime_                  = 0.0f;
  static constexpr float DIAG_INTERVAL = 1.0f;

  void setupScenario(const std::string& scenario, const FluidConfig& cfg) {
    const float w = cfg.world_size;
    const float m = cfg.cellSize() * 0.5f; // margin

    if(scenario == "source-flow") {
      // TC2: 左端から右方向へ移動するボックスソース
      // X が広いワールド (world_size=40 を推奨) で左から右へ流体が噴出
      auto src                = std::make_shared<AABBEmitter>();
      src->center             = glm::vec3(w * 0.05f, w * 0.5f, w * 0.5f);
      src->size               = glm::vec3(w * 0.07f, w * 0.35f, w * 0.35f);
      src->center_vel         = glm::vec3(w * 0.10f, 0.0f, 0.0f); // 10% world/s で右移動
      src->vel                = glm::vec3(w * 0.08f, 0.0f, 0.0f); // 放出粒子に右向き初速
      src->particles_per_step = std::max(1u, cfg.fluidCount() / 400u);
      src->step_count         = 0; // 無限
      engine_.addEmitter(src);
    } else {
      // dam-break (デフォルト): 左半分上部 (X: 左半分, Z: 上半分) を一気に充填
      auto src                = std::make_shared<AABBEmitter>();
      src->center             = glm::vec3(w * 0.25f, w * 0.5f, w * 0.75f);
      src->size               = glm::vec3(w * 0.5f - 2.0f * m, w - 2.0f * m, w * 0.5f - 2.0f * m);
      src->vel                = glm::vec3(0.0f);
      src->particles_per_step = cfg.fluidCount(); // 全粒子を一気に
      src->step_count         = -1;               // 1回のみ
      engine_.addEmitter(src);
    }
  }

  void initVulkan(const FluidConfig& cfg, const std::string& boundaryObj, float rho0Arg) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR, cfg);
    if(rho0Arg > 0.0f) engine_.rho0 = rho0Arg;

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
    pc.posIdx        = engine_.posIdx;
    pc.velIdx        = engine_.velIdx;
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
    sim_ui::fluid_params(engine_);
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

    if(simTime_ >= nextDiagTime_) {
      printDiag();
      nextDiagTime_ += DIAG_INTERVAL;
    }

    base_.currentFrame = (base_.currentFrame + 1) % BaseApp::MAX_FRAMES;
  }

  void printDiag() {
    vkDeviceWaitIdle(base_.ctx.device);
    const uint32_t N      = engine_.nFluid();
    VkDeviceSize byteSize = (VkDeviceSize)N * sizeof(glm::vec4);

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = byteSize;
    bci.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stagingBuf        = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    VmaAllocationInfo allocInfo{};
    if(vmaCreateBuffer(base_.ctx.allocator, &bci, &aci, &stagingBuf, &stagingAlloc, &allocInfo) != VK_SUCCESS) return;

    VkCommandBufferAllocateInfo cai{};
    cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool        = base_.ctx.graphicsCommandPool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd    = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(base_.ctx.device, &cai, &cmd);

    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbi);

    VkBufferCopy region{};
    region.size = byteSize;
    vkCmdCopyBuffer(cmd, engine_.getPositionBuffer(), stagingBuf, 1, &region);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo sub{};
    sub.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    sub.commandBufferCount = 1;
    sub.pCommandBuffers    = &cmd;
    vkQueueSubmit(base_.ctx.graphicsQueue, 1, &sub, VK_NULL_HANDLE);
    vkQueueWaitIdle(base_.ctx.graphicsQueue);
    vkFreeCommandBuffers(base_.ctx.device, base_.ctx.graphicsCommandPool, 1, &cmd);

    vmaInvalidateAllocation(base_.ctx.allocator, stagingAlloc, 0, VK_WHOLE_SIZE);
    const float* d = static_cast<const float*>(allocInfo.pMappedData);

    glm::vec3 centroid(0.0f), bmin(1e9f), bmax(-1e9f);
    for(uint32_t i = 0; i < N; ++i) {
      float x = d[i * 4 + 0], y = d[i * 4 + 1], z = d[i * 4 + 2];
      centroid.x += x;
      centroid.y += y;
      centroid.z += z;
      bmin.x = std::min(bmin.x, x);
      bmax.x = std::max(bmax.x, x);
      bmin.y = std::min(bmin.y, y);
      bmax.y = std::max(bmax.y, y);
      bmin.z = std::min(bmin.z, z);
      bmax.z = std::max(bmax.z, z);
    }
    centroid /= float(N);

    std::printf("[DIAG t=%.2fs] centroid=(%.3f,%.3f,%.3f) AABB_X=[%.3f,%.3f] AABB_Y=[%.3f,%.3f] AABB_Z=[%.3f,%.3f]\n", simTime_, centroid.x, centroid.y, centroid.z, bmin.x, bmax.x, bmin.y, bmax.y, bmin.z, bmax.z);
    std::fflush(stdout);
    vmaDestroyBuffer(base_.ctx.allocator, stagingBuf, stagingAlloc);
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
