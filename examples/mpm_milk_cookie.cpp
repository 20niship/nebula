#include "App.h"
#include "Collider.h"
#include "MaterialParams.h"
#include "engine/MPMEngine.h"
#include "graphics/GraphicsPipeline.h"

#include <argparse/argparse.hpp>

#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <stdexcept>
#include <string>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── CLI ───────────────────────────────────────────────────────────────────

struct MpmMultiArgs : public argparse::Args {
  float& cell_size            = kwarg("cell-size", "MPM grid cell size [m] (domain is 1m cube)").set_default(1.0f / 48.0f);
  float& dt                   = kwarg("dt", "frame timestep [s]").set_default(1.0f / 60.0f);
  int& substeps               = kwarg("substeps", "substeps per frame (0=auto: CFLを保つためセルサイズに反比例、1/48で30)").set_default(0);
  float& blade_speed          = kwarg("blade-speed", "blender blade angular speed [rad/s]").set_default(14.0f);
  int& blade_start            = kwarg("blade-start", "frame at which the blade starts to spin (ramps up over 60 frames)").set_default(90);
  float& cookie_q_max         = kwarg("cookie-q-max", "cookie Von Mises yield stress (smaller = crumbles easier)").set_default(3e3f);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

// ── App ───────────────────────────────────────────────────────────────────

class MpmMultiApp {
public:
  void run(const MpmMultiArgs& args) {
    dt_                 = args.dt;
    base_.screenshotDir = args.screenshot_dir;

    engine_.domainSize = glm::vec3(1.0f);
    engine_.cellSize   = args.cell_size;
    // 1粒子 = (cell/2)^3 (8粒子/セル)。ミルク+クッキーの総体積から粒子数の上限を決める(--cell-sizeを細かくすれば1000万粒子級にも拡大できる)
    const float Vp       = std::pow(args.cell_size * 0.5f, 3.0f);
    const float milkSide = kWallMax - kWallMin - 0.02f;
    engine_.maxParticles = uint32_t((milkSide * milkSide * 0.32f + 6.0f * 0.09f * 0.09f * 0.09f) / Vp * 1.05f) + 1024u;
    std::printf("cell=%.5f m, 粒子上限 %u\n", args.cell_size, engine_.maxParticles);
    bladeSpeed_          = args.blade_speed;
    bladeStart_          = args.blade_start;
    cookieQMax_          = args.cookie_q_max;

    base_.initWindow("MPM Milk + Cookie + Blender");
    initVulkan(args.substeps > 0 ? args.substeps : int(std::ceil(30.0f * (1.0f / 48.0f) / args.cell_size)));
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  MPMEngine engine_;
  GraphicsPipeline graphicsPipe_;
  std::shared_ptr<GravityForce> gravity_;
  float dt_       = 1.0f / 60.0f;
  float simTime_  = 0.0f;
  int frameCount_ = 0;

  // 容器(角柱グラス)と刃の寸法 [m]
  static constexpr float kMassScale = 1000.0f;
  static constexpr float kWallMin = 0.2f, kWallMax = 0.8f, kFloorZ = 0.1f;
  float bladeSpeed_ = 14.0f;
  int bladeStart_   = 90;
  float cookieQMax_ = 3e3f;
  float bladeAngle_ = 0.0f;

  // AABB内に一括放出する(1回のみ)。particleVolumeを明示して粒子数と密度を整合させる。
  void addBlock(const glm::vec3& center, const glm::vec3& size, float Vp, uint32_t materialId, const glm::vec3& vel = glm::vec3(0.0f)) {
    auto e            = std::make_shared<AABBEmitter>();
    e->center         = center;
    e->size           = size;
    e->particleType   = materialId;
    e->vel            = vel;
    e->particleVolume = Vp;
    e->particles_per_step = std::max(1, int(size.x * size.y * size.z / Vp));
    e->step_count     = -1;
    engine_.addEmitter(e);
  }

  // 容器(床+4壁)と回転する刃(十字2枚の薄いBOX + 軸の円柱)。毎フレーム再登録して姿勢を更新する。
  void rebuildColliders() {
    const glm::vec3 c(0.5f, 0.5f, 0.0f);
    ColliderSet cols;
    cols.addPlane({0.5f, 0.5f, kFloorZ}, {0, 0, 1}, 0.0f, 0.3f);
    cols.addPlane({kWallMin, 0.5f, 0.5f}, {1, 0, 0}, 0.0f, 0.1f);
    cols.addPlane({kWallMax, 0.5f, 0.5f}, {-1, 0, 0}, 0.0f, 0.1f);
    cols.addPlane({0.5f, kWallMin, 0.5f}, {0, 1, 0}, 0.0f, 0.1f);
    cols.addPlane({0.5f, kWallMax, 0.5f}, {0, -1, 0}, 0.0f, 0.1f);

    // 立ち上がり時は角速度を線形ランプ(急加速で粒子が貫通しないように)
    const float ramp = std::clamp(float(frameCount_ - bladeStart_) / 60.0f, 0.0f, 1.0f);
    const float w    = bladeSpeed_ * ramp;
    const glm::vec3 angVel(0.0f, 0.0f, w);
    const float bladeZ = kFloorZ + 0.07f;
    for(int k = 0; k < 2; ++k) {
      const glm::quat q = glm::angleAxis(bladeAngle_ + float(k) * glm::half_pi<float>(), glm::vec3(0, 0, 1));
      cols.addBox({c.x, c.y, bladeZ}, {0.22f, 0.035f, 0.012f}, 0.0f, 0.4f, {0, 0, 0}, q, angVel);
    }
    cols.addCylinder({c.x, c.y, 0.4f}, 0.03f, 0.35f, 0.0f, 0.4f, {0, 0, 0}, glm::angleAxis(glm::half_pi<float>(), glm::vec3(1, 0, 0)), angVel); // ローカルY(軸)→ワールドZ
    engine_.setColliders(cols);
  }

  void initVulkan(int substeps) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR);
    engine_.numSubsteps = substeps;
    gravity_            = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 9.8f); // Z-up (particle.vertのカメラ規約)
    engine_.addForce(gravity_);
    engine_.flip_ratio = -1.0f; // APIC: 角運動量を保存し撹拌の渦が残る。FLIPは弱圧縮流体で体積が潰れるため不採用(実測)

    // 粒子が軽すぎるとP2Gの固定小数点(1/65536)で運動量が丸められ重力が積算されない(PIC/APICで実測)。
    // 密度・剛性を一律kMassScale倍して粒子質量を稼ぐ(c=sqrt(K/rho)・加速度は不変、応力の単位だけが変わる)。
    constexpr float S = kMassScale;
    // slot 0: ミルク(弱圧縮流体)  slot 1: クッキー(Von Mises 脆性固体。降伏後は塑性流動で崩れる)
    MaterialParams milk   = presetWater(1030.0f * S, 2e5f * S);
    MaterialParams cookie = presetJelly(2e5f * S, 0.25f, 1300.0f * S);
    cookie.model          = uint32_t(MaterialModel::VON_MISES);
    cookie.q_max          = cookieQMax_ * S;
    engine_.setMaterials({milk, cookie});

    const float Vp = std::pow(engine_.cellSize * 0.5f, 3.0f);
    const float W  = kWallMax - kWallMin - 0.02f;
    addBlock({0.5f, 0.5f, kFloorZ + 0.16f}, {W, W, 0.32f}, Vp, 0u); // ミルク(軸/刃も含めて満たし、コライダー側が押しのける)

    // クッキー: 軸を避けて円周上に6個、時間差をつけず高さを散らして投入
    for(int i = 0; i < 6; ++i) {
      const float a = glm::two_pi<float>() * float(i) / 6.0f;
      addBlock({0.5f + 0.17f * std::cos(a), 0.5f + 0.17f * std::sin(a), 0.55f + 0.07f * float(i)}, glm::vec3(0.09f), Vp, 1u, glm::vec3(0.0f, 0.0f, -2.0f)); // 初速で液面に突入させる
    }
    rebuildColliders();

    graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/particle_mat.vert.spv", SHADER_DIR_STR + "/particle_mat.frag.spv");

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

    if(frameCount_ >= bladeStart_) { // 刃が回り始めるまでは静的コライダーのまま
      const float ramp = std::clamp(float(frameCount_ - bladeStart_) / 60.0f, 0.0f, 1.0f);
      bladeAngle_ += bladeSpeed_ * ramp * dt_;
      rebuildColliders();
    }
    simTime_ += dt_;
    ++frameCount_;

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
  auto args = argparse::parse<MpmMultiArgs>(argc, argv);
  MpmMultiApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
