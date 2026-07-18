#include "App.h"
#include "engine/SoftBodyEngine.h"
#include "graphics/GraphicsPipeline.h"

#include <argparse/argparse.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <fstream>
#include <stdexcept>
#include <string>

static const std::string SHADER_DIR_STR = SHADER_DIR;
static const std::string ASSET_DIR_STR  = ASSET_DIR;

// ── CLI ───────────────────────────────────────────────────────────────────────

struct SoftBodyArgs : public argparse::Args {
  float& domain_size_x  = kwarg("domain-size-x", "domain physical size X [m]").set_default(10.0f);
  float& domain_size_y  = kwarg("domain-size-y", "domain physical size Y [m]").set_default(10.0f);
  float& domain_size_z  = kwarg("domain-size-z", "domain physical size Z [m]").set_default(10.0f);
  float& cell_size      = kwarg("cell-size", "spatial grid cell size [m]").set_default(10.0f / 64.0f);
  float& dt             = kwarg("dt", "frame timestep [s]").set_default(1.0f / 60.0f);
  int& substeps         = kwarg("substeps", "substeps per frame").set_default(15);
  int& n_shots          = kwarg("n-shots", "screenshot count (0=off)").set_default(0);
  std::string& shot_dir = kwarg("screenshot-dir", "screenshot output dir").set_default(std::string(""));
};

// ── App ───────────────────────────────────────────────────────────────────────

class SoftBodyApp {
public:
  void run(const SoftBodyArgs& args) {
    dt_     = args.dt;
    nShots_ = args.n_shots;
    if(!args.shot_dir.empty()) base_.screenshotDir = args.shot_dir;

    base_.initWindow("XPBD Soft Body – Vulkan GPU");
    initVulkan(args);
    mainLoop();
    cleanup();
  }

private:
  BaseApp base_;
  SoftBodyEngine engine_;
  GraphicsPipeline graphicsPipe_;
  std::shared_ptr<GravityForce> gravity_;
  float dt_      = 1.0f / 60.0f;
  float simTime_ = 0.0f;
  int nShots_    = 0;

  void initVulkan(const SoftBodyArgs& args) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    float ws   = args.domain_size_x; // 配置ロジックは等方(cube)前提のスカラーwsを使用
    float mid  = ws * 0.5f;
    float high = ws * 0.75f;

    // Stanford Bunny (中央から少し右上)
    std::string bunnyPath = ASSET_DIR_STR + "/bunny_sb.sb";
    std::string cubePath  = ASSET_DIR_STR + "/cube_sb.sb";

    // Bunny が存在すれば追加 (中央上部)
    {
      std::ifstream check(bunnyPath);
      if(check.good()) {
        engine_.addInstance({bunnyPath, {mid, mid + 4.0, high + 1.0f}, 1.0f});
      } else {
        std::cout << "[SoftBody] bunny_sb.sb not found, skipping bunny\n";
      }
    }

    // ジェリーキューブ 9 個 — 3×3 グリッドで高さをずらして落下・衝突させる
    {
      std::ifstream check(cubePath);
      if(check.good()) {
        const float sp = 2.0f;     // 格子間隔 [m]
        const float x0 = mid - sp; // 3.5 (世界中心から -1 格子)
        const float y0 = mid - sp;
        int k          = 0;
        for(int iy = 0; iy < 3; iy++) {
          for(int ix = 0; ix < 3; ix++) {
            float x = x0 + ix * sp;
            float y = y0 + iy * sp;
            float z = high + k * 0.25f; // 0.25m ずつ高さをずらす
            engine_.addInstance({cubePath, {x, y, z}, 1.0f});
            k++;
          }
        }
      } else {
        std::cout << "[SoftBody] cube_sb.sb not found. "
                  << "Run: python tools/gen_softbody.py --cube -o assets/cube_sb.sb\n";
      }
    }

    if(engine_.totalParticleCount() == 0) {
      throw std::runtime_error("No .sb files found. Generate them first:\n"
                               "  python tools/gen_softbody.py --cube -o assets/cube_sb.sb\n"
                               "  python tools/gen_softbody.py --bunny --input assets/bunny.obj "
                               "-o assets/bunny_sb.sb");
    }

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR, glm::vec3(args.domain_size_x, args.domain_size_y, args.domain_size_z), args.cell_size);
    gravity_ = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 9.8f); // Z-up
    engine_.addForce(gravity_);
    engine_.numSubsteps = args.substeps;

    graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/sb_wire.vert.spv", SHADER_DIR_STR + "/sb_wire.frag.spv", VK_PRIMITIVE_TOPOLOGY_LINE_LIST);

    base_.createFrameData();
    base_.initImGui();
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
    clear.color = {{0.02f, 0.02f, 0.05f, 1.0f}};

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

    // sb_wire.vert: posIdx + stretchEdgesIdx でエッジワイヤーフレームを描画
    SimPC renderPc{};
    renderPc.posIdx          = engine_.posIdx;
    renderPc.stretchEdgesIdx = engine_.edgeDataIdx;
    renderPc.worldMin        = glm::vec3(0.0f);
    renderPc.worldMax        = glm::vec3(10.0f);

    // LINE_LIST: 1辺 = 頂点2個
    graphicsPipe_.draw(cmd, engine_.descriptorSet, renderPc, engine_.totalEdgeCount() * 2);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
  }

  void drawFrame() {
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

    ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({310, 0}, ImGuiCond_Once);
    ImGui::Begin("XPBD Soft Body");
    ImGui::Text("FPS: %.1f  |  N=%u  |  t=%.2f s", ImGui::GetIO().Framerate, engine_.totalParticleCount(), simTime_);
    ImGui::Text("dt_sub=%.4f s", dt_ / float(engine_.numSubsteps));
    ImGui::Separator();
    ImGui::SliderFloat("重力", &gravity_->strength, 0.0f, 20.0f);
    ImGui::SliderFloat("反発係数", &engine_.restitution, 0.0f, 1.0f);
    ImGui::SliderFloat("摩擦係数", &engine_.friction, 0.0f, 1.0f);
    ImGui::SliderFloat("エッジ剛性", &engine_.stretchCompliance, 1e-7f, 1e-3f, "%.2e");
    ImGui::SliderFloat("体積剛性", &engine_.volCompliance, 1e-6f, 1e-1f, "%.2e");
    ImGui::SliderFloat("線形減衰", &engine_.linearDamping, 0.0f, 0.2f);
    ImGui::SliderFloat("衝突半径", &engine_.particleCollisionRadius, 0.0f, 1.0f);
    ImGui::SliderInt("ソルバー反復", &engine_.solverIterations, 1, 20);
    ImGui::SliderInt("サブステップ", &engine_.numSubsteps, 1, 50);
    ImGui::End();

    ImGui::Render();
    simTime_ += dt_;

    // スクリーンショット
    if(nShots_ > 0) base_.saveScreenshot(imageIdx, nShots_);

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
    result                     = vkQueuePresentKHR(base_.ctx.graphicsQueue, &present);
    if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || base_.framebufferResized) {
      base_.framebufferResized = false;
      base_.ctx.recreateSwapchain();
    }
    base_.currentFrame = (base_.currentFrame + 1) % BaseApp::MAX_FRAMES;
  }

  void mainLoop() {
    while(!glfwWindowShouldClose(base_.window) && !base_.shouldExit) {
      glfwPollEvents();
      drawFrame();
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
  auto args = argparse::parse<SoftBodyArgs>(argc, argv);
  SoftBodyApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
