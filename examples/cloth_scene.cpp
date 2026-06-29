#include "App.h"
#include "engine/ClothConstraint.h"
#include "engine/ClothMesh.h"
#include "engine/ClothSceneEngine.h"
#include "graphics/ClothRenderer.h"
#include "graphics/GraphicsPipeline.h"

#include <argparse/argparse.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── CLI ───────────────────────────────────────────────────────────────────────

struct SceneArgs : public argparse::Args {
  int&   scene     = kwarg("scene",      "5=two cloths, 7=4-corner twist").set_default(5);
  int&   cloth_n   = kwarg("n,cloth-n",  "cloth grid size NxN").set_default(48);
  float& world_size= kwarg("world-size", "simulation world size").set_default(10.0f);
  int&   grid_res  = kwarg("grid-res",   "hash grid resolution").set_default(64);
  float& dt        = kwarg("dt",         "timestep (sec)").set_default(1.0f / 60.0f);
  int&   n_shots   = kwarg("n-shots",    "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

// ── ヘルパー ──────────────────────────────────────────────────────────────────

// ClothMesh の全頂点を pivot 周りに Z 軸で angle 回転 (XY 平面内で回転)
static void rotateClothAroundZ(ClothMesh& mesh, const glm::vec3& pivot, float angle) {
  float c = glm::cos(angle);
  float s = glm::sin(angle);
  for(auto& pos : mesh.positions) {
    float dx = pos.x - pivot.x;
    float dy = pos.y - pivot.y;
    pos.x = pivot.x + dx * c - dy * s;
    pos.y = pivot.y + dx * s + dy * c;
  }
}

// point を pivot 周りに Y 軸で angle 回転 (XZ 平面内で回転)
// 左端 (dx<0) は +Z 方向、右端 (dx>0) は -Z 方向へ — 布の絞り動作
static glm::vec3 rotateAroundY(const glm::vec3& point, const glm::vec3& pivot, float angle) {
  float dx = point.x - pivot.x;
  float dz = point.z - pivot.z;
  float c  = glm::cos(angle);
  float s  = glm::sin(angle);
  return glm::vec3(pivot.x + dx * c + dz * s,
                   point.y,
                   pivot.z - dx * s + dz * c);
}

// ── App ───────────────────────────────────────────────────────────────────────

class ClothSceneApp {
public:
  void run(const SceneArgs& args) {
    scene_    = args.scene;
    dt_       = args.dt;
    clothN_   = (uint32_t)args.cloth_n;
    worldSize_= args.world_size;
    gridRes_  = (uint32_t)args.grid_res;
    base_.screenshotDir = args.screenshot_dir;

    base_.initWindow("Vulkan Sim – Cloth Scene");
    initVulkan();
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp            base_;
  ClothSceneEngine   engine_;
  GraphicsPipeline   particlePipe_;
  std::vector<ClothRenderer> clothRenderers_;

  int      scene_     = 5;
  float    dt_        = 1.0f / 60.0f;
  float    simTime_   = 0.0f;
  uint32_t clothN_    = 48;
  float    worldSize_ = 10.0f;
  uint32_t gridRes_   = 64;

  // TC7 用: 4隅の初期位置
  glm::vec3 corners_[4]{};

  // buildScene* で収集した triIndices (engine_.init() 後にレンダラーへ渡す)
  std::vector<std::vector<uint32_t>> meshTriIndices_;

  void buildScene5() {
    // sp × 0.60: 45°回転後の対角線幅 (side×√2) がワールド内に収まるよう抑制
    float sp = worldSize_ / float(clothN_ + 1) * 0.60f;
    float cx = worldSize_ * 0.5f;
    float cy = worldSize_ * 0.5f;

    ClothMesh m1, m2;
    // Z 位置をワールド中央寄りに配置 (旧: 0.80/0.55 → 新: 0.65/0.40)
    m1.build((int)clothN_, sp, cx, cy, worldSize_ * 0.65f);
    m2.build((int)clothN_, sp, cx, cy, worldSize_ * 0.40f);

    // Z 軸周りに 45° 回転して斜め配置
    const glm::vec3 pivot(cx, cy, 0.0f);
    const float     angle45 = glm::quarter_pi<float>();
    rotateClothAroundZ(m1, pivot, angle45);
    rotateClothAroundZ(m2, pivot, angle45);

    uint32_t off1 = engine_.addCloth(m1);
    uint32_t off2 = engine_.addCloth(m2);

    for(int j = 0; j < (int)clothN_; ++j) {
      engine_.addConstraint({ClothConstraint::Type::Pin,
          off1 + (uint32_t)m1.idx(0, j),
          glm::vec3(m1.positions[m1.idx(0, j)])});
      engine_.addConstraint({ClothConstraint::Type::Pin,
          off2 + (uint32_t)m2.idx(0, j),
          glm::vec3(m2.positions[m2.idx(0, j)])});
    }

    meshTriIndices_.push_back(m1.triIndices);
    meshTriIndices_.push_back(m2.triIndices);
  }

  void buildScene7() {
    float sp = worldSize_ / float(clothN_ + 1) * 0.85f;
    float cx = worldSize_ * 0.5f;
    float cy = worldSize_ * 0.5f;
    float cz = worldSize_ * 0.5f;  // 中央高さ: 回転時に上下の境界を超えないよう

    ClothMesh m;
    m.build((int)clothN_, sp, cx, cy, cz);

    uint32_t off = engine_.addCloth(m);
    int N = (int)clothN_;

    corners_[0] = glm::vec3(m.positions[m.idx(0,   0  )]);
    corners_[1] = glm::vec3(m.positions[m.idx(0,   N-1)]);
    corners_[2] = glm::vec3(m.positions[m.idx(N-1, 0  )]);
    corners_[3] = glm::vec3(m.positions[m.idx(N-1, N-1)]);

    engine_.addConstraint({ClothConstraint::Type::PinAnimated, off + (uint32_t)m.idx(0,   0  ), corners_[0]});
    engine_.addConstraint({ClothConstraint::Type::PinAnimated, off + (uint32_t)m.idx(0,   N-1), corners_[1]});
    engine_.addConstraint({ClothConstraint::Type::PinAnimated, off + (uint32_t)m.idx(N-1, 0  ), corners_[2]});
    engine_.addConstraint({ClothConstraint::Type::PinAnimated, off + (uint32_t)m.idx(N-1, N-1), corners_[3]});

    meshTriIndices_.push_back(m.triIndices);
  }

  void initVulkan() {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    // ① メッシュ構築・制約登録 (engine_.init() 前: descriptorSetLayout 未確定)
    if(scene_ == 5)      buildScene5();
    else if(scene_ == 7) buildScene7();
    else throw std::runtime_error("Unknown scene: " + std::to_string(scene_));

    // ② エンジン初期化 → descriptorSetLayout が確定する
    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool,
                 base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue,
                 SHADER_DIR_STR, worldSize_, gridRes_);

    // TC7 は高速回転 (7回転/5秒) のため substep を増やして安定化
    if(scene_ == 7) {
      engine_.numSubsteps     = 15;
      engine_.solverIterations = 5;
    }

    // ③ レンダラー初期化 (engine_.init() 後に descriptorSetLayout を使う)
    clothRenderers_.resize(meshTriIndices_.size());
    for(size_t i = 0; i < meshTriIndices_.size(); ++i) {
      clothRenderers_[i].init(base_.ctx.device, base_.ctx.allocator, base_.ctx.renderPass,
                              engine_.descriptorSetLayout, SHADER_DIR_STR);
      clothRenderers_[i].uploadIndices(meshTriIndices_[i],
                                       base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue);
    }

    particlePipe_.init(base_.ctx.device, base_.ctx.renderPass,
                       engine_.descriptorSetLayout,
                       SHADER_DIR_STR + "/particle.vert.spv",
                       SHADER_DIR_STR + "/particle.frag.spv");

    base_.createFrameData();
    base_.initImGui();
  }

  void updateAnimatedPins() {
    if(scene_ != 7) return;
    // Y軸周りに絞り回転: 左端(dx<0)と右端(dx>0)が同一軸上で逆Z方向へ
    // 300フレーム×(1/60s) = 5秒で 7回転
    const float angVel = glm::two_pi<float>() * 7.0f / 5.0f;
    float angle = simTime_ * angVel;
    float cx = worldSize_ * 0.5f;
    float cz = worldSize_ * 0.5f;
    for(int k = 0; k < 4; ++k) {
      glm::vec3 pivot(cx, corners_[k].y, cz);
      engine_.updateConstraint(k, rotateAroundY(corners_[k], pivot, angle));
    }
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
    vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
      0, 0, nullptr, 1, &barrier, 0, nullptr);

    vkEndCommandBuffer(cmd);
  }

  void recordGraphicsCmd(VkCommandBuffer cmd, uint32_t imageIdx) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    VkClearValue clear{};
    clear.color = {{0.05f, 0.05f, 0.08f, 1.0f}};

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
    VkRect2D sc{}; sc.extent = base_.ctx.swapchainExtent;
    vkCmdSetScissor(cmd, 0, 1, &sc);

    SimPC pc{};
    pc.posIdx           = engine_.posIdx;
    pc.velIdx           = engine_.velIdx;
    pc.particleCount    = engine_.totalParticleCount();
    pc.worldMin         = 0.0f;
    pc.worldMax         = worldSize_;
    pc.clothVertexCount = engine_.totalParticleCount();
    pc.couplingForceIdx = 0;

    for(uint32_t i = 0; i < engine_.clothCount(); ++i) {
      clothRenderers_[i].draw(cmd, engine_.descriptorSet, pc,
                              engine_.getMesh(i).vertexCount(),
                              (int32_t)engine_.meshOffset(i));
    }
    particlePipe_.draw(cmd, engine_.descriptorSet, pc, engine_.totalParticleCount());

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
  }

  void drawFrame(int nShots) {
    auto& f = base_.frames[base_.currentFrame];
    vkWaitForFences(base_.ctx.device, 1, &f.inFlightFence, VK_TRUE, UINT64_MAX);

    uint32_t imageIdx;
    VkResult result = vkAcquireNextImageKHR(base_.ctx.device, base_.ctx.swapchain,
                                             UINT64_MAX, f.imageAvailable,
                                             VK_NULL_HANDLE, &imageIdx);
    if(result == VK_ERROR_OUT_OF_DATE_KHR) { base_.ctx.recreateSwapchain(); return; }

    vkResetFences(base_.ctx.device, 1, &f.inFlightFence);

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Cloth Scene Control");
    ImGui::Text("Scene %d | FPS: %.1f | 頂点: %u | t=%.2fs",
                scene_, ImGui::GetIO().Framerate, engine_.totalParticleCount(), simTime_);
    ImGui::Separator();
    ImGui::SliderFloat("重力",        &engine_.gravity,          -20.0f,  0.0f);
    ImGui::SliderFloat("反発係数",    &engine_.restitution,        0.0f,  1.0f);
    ImGui::SliderFloat("摩擦係数",    &engine_.friction,           0.0f,  1.0f);
    ImGui::SliderFloat("伸び剛性",    &engine_.stretchCompliance,  0.0f,  1e-2f, "%.6f");
    ImGui::SliderFloat("曲げ剛性",    &engine_.bendCompliance,     0.0f,  1e-1f, "%.6f");
    ImGui::SliderFloat("風 X",        &engine_.windX,            -10.0f, 10.0f);
    ImGui::SliderFloat("風 Z",        &engine_.windZ,            -10.0f, 10.0f);
    ImGui::SliderInt("反復",          &engine_.solverIterations,    1,    10);
    ImGui::SliderInt("サブステップ",  &engine_.numSubsteps,         1,    20);
    ImGui::Checkbox("自己衝突",       &engine_.enableSelfCollision);
    ImGui::End();

    ImGui::Render();

    updateAnimatedPins();
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

    std::array<VkSemaphore, 2>          waitSems   = {f.imageAvailable, f.timelineSemaphore};
    std::array<VkPipelineStageFlags, 2> waitStages = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
    };

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
    for(auto& r : clothRenderers_) r.cleanup();
    particlePipe_.cleanup();
    engine_.cleanup();
    base_.cleanupBase();
  }
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  auto args = argparse::parse<SceneArgs>(argc, argv);
  ClothSceneApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
