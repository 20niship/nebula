// 波打ちパドル + Bunny 障害物 + 泡 (spray/foam/bubble) デモシーン (issue #47 検証用)。
// ドメイン左側で単振動(SHM)する壁パドルが波を起こし、右側に並べた複数の Bunny
// メッシュへ波が衝突して泡を生成する。海面のリアルなレンダリング検証を想定。
#include "App.h"
#include "core/Emitter.h"
#include "core/Force.h"
#include "engine/BoundaryParticles.h"
#include "engine/FluidEngine.h"
#include "graphics/GraphicsPipeline.h"
#include "utils.hpp"

#include <argparse/argparse.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <glm/glm.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>

static const std::string SHADER_DIR_STR = SHADER_DIR;
static const std::string ASSET_DIR_STR  = ASSET_DIR;

// ── CLI ───────────────────────────────────────────────────────────────────────

struct WaveFoamArgs : public argparse::Args {
  float& world_size           = kwarg("world-size", "simulation domain size [m]").set_default(24.0f);
  int& grid_res                = kwarg("grid-res", "spatial hash grid resolution").set_default(88);
  int& fluid_nx                 = kwarg("nx", "fluid particle grid X (密度基準)").set_default(200);
  int& fluid_nz                 = kwarg("nz", "fluid particle grid Z / 水深基準").set_default(24);
  float& dt                     = kwarg("dt", "timestep (sec)").set_default(1.0f / 60.0f);
  float& paddle_amp             = kwarg("paddle-amp", "波発生パドル SHM 振幅 [m]").set_default(0.8f);
  float& paddle_period          = kwarg("paddle-period", "波発生パドル SHM 周期 [s]").set_default(2.5f);
  int& max_diffuse              = kwarg("max-diffuse", "泡(spray/foam/bubble)の最大パーティクル数").set_default(20000);
  int& n_shots                  = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir   = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
};

// ── 手続きパドル形状生成（薄い壁の前後2面をシェルサンプリング）──────────────────

static std::vector<glm::vec4> generatePaddleShell(glm::vec3 center, glm::vec3 halfExtents, float spacing) {
  std::vector<glm::vec4> pts;
  int ny = std::max(1, int(halfExtents.y * 2.0f / spacing));
  int nz = std::max(1, int(halfExtents.z * 2.0f / spacing));
  for(int iz = 0; iz <= nz; ++iz) {
    float z = center.z - halfExtents.z + float(iz) / float(nz) * halfExtents.z * 2.0f;
    for(int iy = 0; iy <= ny; ++iy) {
      float y = center.y - halfExtents.y + float(iy) / float(ny) * halfExtents.y * 2.0f;
      pts.push_back(glm::vec4(center.x - halfExtents.x, y, z, 1.0f)); // 背面
      pts.push_back(glm::vec4(center.x + halfExtents.x, y, z, 1.0f)); // 前面
    }
  }
  return pts;
}

// ── Bunny 配置ヘルパー ───────────────────────────────────────────────────────
// assets/bunny.obj のバウンディングボックス (Y-up, 実測値)。
static const glm::vec3 kBunnyMinYup(-0.09469f, 0.032987f, -0.061874f);
static const glm::vec3 kBunnyMaxYup(0.061009f, 0.187321f, 0.058800f);

// floorCenter (X,Y は中心、Z は接地面) にバニーの底面が接地するようオフセットを計算する。
// yup_to_zup 変換 (x,y,z)->(x,z,y) 後の座標系で計算する。
static glm::vec3 bunnyOffsetForFloorCenter(glm::vec3 floorCenter, float scale) {
  glm::vec3 minZup(kBunnyMinYup.x, kBunnyMinYup.z, kBunnyMinYup.y);
  glm::vec3 maxZup(kBunnyMaxYup.x, kBunnyMaxYup.z, kBunnyMaxYup.y);
  glm::vec3 centerZup = (minZup + maxZup) * 0.5f;
  glm::vec3 offset;
  offset.x = floorCenter.x - centerZup.x * scale;
  offset.y = floorCenter.y - centerZup.y * scale;
  offset.z = floorCenter.z - minZup.z * scale; // 底面を floorCenter.z に接地
  return offset;
}

// ── KinematicPaddle ──────────────────────────────────────────────────────────
// 前後2面シェル形状を X 方向へ単振動 (SHM) させる波発生パドル。
struct KinematicPaddle {
  std::vector<glm::vec4> restPositions; // 変位0 (t=0 相当) でのワールド座標
  float amplitude    = 1.5f; // [m]
  float omega         = 3.49f; // 角振動数 [rad/s] (= 2π/period)
  uint32_t gpuOffset  = 0;    // GPU バッファ先頭インデックス (常に0)
  uint32_t count      = 0;

  // 起動直後にいきなり最大速度 (amplitude*omega) で動き出すと、静止していた
  // 流体塊への衝撃的な初速入力となり PBF が発散し得る (issue #47 検証時に実測:
  // GPU タイムアウト後シミュレーションが完全停止する不具合を確認)。
  // rampTime 秒かけて滑らかに立ち上げることで衝撃を緩和する。
  static constexpr float rampTime = 2.0f; // [s]
  static float ramp(float t) {
    float x = std::clamp(t / rampTime, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x); // smoothstep
  }

  float displacementX(float t) const { return amplitude * ramp(t) * std::sin(omega * t); }
  float velocityX(float t) const { return amplitude * ramp(t) * omega * std::cos(omega * t); }
};

// ── App ───────────────────────────────────────────────────────────────────────

class WaveFoamApp {
public:
  void run(const WaveFoamArgs& args) {
    dt_                 = args.dt;
    base_.screenshotDir = args.screenshot_dir;

    FluidConfig cfg;
    cfg.fluid_nx            = uint32_t(args.fluid_nx);
    cfg.fluid_ny             = 16; // 流体薄層の奥行き分解能
    cfg.fluid_nz             = uint32_t(args.fluid_nz);
    cfg.domainSize           = glm::vec3(args.world_size);
    cfg.cellSize             = args.world_size / float(args.grid_res);
    cfg.max_boundary         = 20000;
    cfg.maxDiffuseParticles  = uint32_t(args.max_diffuse);

    paddle_.amplitude = args.paddle_amp;
    paddle_.omega     = 2.0f * 3.14159265f / std::max(0.1f, args.paddle_period);

    base_.initWindow("Vulkan Sim - Wave Paddle + Bunny + Foam (issue #47)");
    initVulkan(cfg);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  FluidEngine engine_;
  GraphicsPipeline graphicsPipe_;
  GraphicsPipeline foamGraphicsPipe_;
  std::shared_ptr<GravityForce> gravity_;
  FluidEngine::FoamParams foamParams_;

  KinematicPaddle paddle_;
  std::vector<glm::vec4> paddleNewPos_;
  std::vector<glm::vec4> paddleNewVel_;

  float dt_       = 1.0f / 60.0f;
  float simTime_  = 0.0f;

  void initVulkan(const FluidConfig& cfg) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR, cfg);

    engine_.rho0          = cfg.computeRestDensity();
    engine_.viscosityC     = 0.02f;
    engine_.pbfIterations  = 4;
    engine_.numSubsteps    = 3;
    engine_.linearDamping   = 0.1f;   // 既定 0.02 よりやや強めに減衰させ、パドル往復による
                                        // エネルギー蓄積 (GPU タイムアウトを誘発する発散) を抑える
    engine_.vorticityEnabled = false;
    engine_.vorticityEpsilon = 0.15f;

    gravity_ = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 9.8f); // Z-up
    engine_.addForce(gravity_);

    // カメラ (fluid_particle.vert 共通) は target=(mid,mid,mid), mid=(worldMin+worldMax)/2 を
    // 全軸共通で見る。カメラ距離は world_size に比例して決まるため、コンテンツの
    // 画面占有率は「絶対サイズの world_size に対する比率」で決まる（world_size を
    // 変えても比率が同じなら見た目のサイズは変わらない）。dam-break 等の既存シーンは
    // 水柱の高さ = world_size 相当で画面の4割程度を占めており、これに合わせて
    // パドル高さは world_size の 0.85 倍 (≒画面の4割強) を目安に取る。
    const float w          = cfg.domainSize.x; // 立方体ドメイン想定 (x=y=z)
    const float d          = cfg.particleSpacing(); // 流体粒子間隔（密度の基準）
    const float centerY    = w * 0.5f;
    const float halfY      = 1.5f; // 流体薄層の奥行き半幅
    const float waterDepth = cfg.fluid_nz * d;
    const float bSpacing   = 0.15f;

    // ── 波発生パドル (ドメイン左側、X方向に単振動) ────────────────────────────
    const float paddleRestX  = paddle_.amplitude + 1.2f; // 左壁からの最小マージンを確保
    const float paddleHeight = w * 0.85f;                 // フレーム中央〜上寄りに収まる高さ
    paddle_.restPositions = generatePaddleShell(glm::vec3(paddleRestX, centerY, paddleHeight * 0.5f), glm::vec3(0.25f, halfY, paddleHeight * 0.5f), bSpacing);
    paddle_.count          = uint32_t(std::min(paddle_.restPositions.size(), size_t(cfg.max_boundary)));
    paddle_.gpuOffset       = 0;
    paddleNewPos_.resize(paddle_.count);
    paddleNewVel_.resize(paddle_.count);

    // ── Bunny 障害物 (ドメイン右側に3体、床から突き出た岩礁のように配置) ────────
    std::vector<glm::vec4> boundaryPts = paddle_.restPositions;
    const float bunnyNativeHeight = kBunnyMaxYup.y - kBunnyMinYup.y;   // Y-up ローカル座標での高さ (実測)
    const float bunnyTargetHeight = w * 0.13f;                        // world_size に対する目標高さ（隣接バニー同士が重ならない大きさ）
    const float bunnyScale        = bunnyTargetHeight / bunnyNativeHeight;
    struct BunnyPlacement {
      float x, yOff;
    };
    const std::array<BunnyPlacement, 3> bunnies = {{
        {w * 0.60f, -0.2f},
        {w * 0.75f, 0.2f},
        {w * 0.90f, -0.1f},
    }};
    BoundaryParticles bp;
    for(const auto& b : bunnies) {
      glm::vec3 floorCenter(b.x, centerY + b.yOff, 0.0f);
      glm::vec3 offset = bunnyOffsetForFloorCenter(floorCenter, bunnyScale);
      BoundaryMesh mesh = bp.loadOBJ(ASSET_DIR_STR + "/bunny.obj", bSpacing, bunnyScale, offset, /*yup_to_zup=*/true);
      boundaryPts.insert(boundaryPts.end(), mesh.particles.begin(), mesh.particles.end());
    }
    engine_.loadBoundaryParticles(boundaryPts);
    engine_.initKinematicBoundaryStaging(paddle_.count);

    // ── 流体: パドルと Bunny の間を満たす浅い「海」 ─────────────────────────
    const float oceanXStart = paddleRestX + paddle_.amplitude + 0.8f;
    const float oceanXEnd   = w * 0.51f; // 最初の Bunny 手前まで（波が伝播する余地を残す）
    glm::vec3 oceanSize(oceanXEnd - oceanXStart, halfY * 2.0f, waterDepth);
    auto src     = std::make_shared<AABBEmitter>();
    src->center  = glm::vec3((oceanXStart + oceanXEnd) * 0.5f, centerY, waterDepth * 0.5f);
    src->size    = oceanSize;
    src->vel     = glm::vec3(0.0f);
    // 密度基準スペーシング d = particleSpacing() を実際の充填密度と一致させる。
    // cfg.fluidCount() をそのまま使うと (このシーンのように emitter box が
    // world_size に対して独立比率のとき) 過密充填となり PBF が発散して
    // NaN 位置 -> 空間ハッシュのセルインデックス破損 -> 近傍探索ループ暴走で
    // GPU タイムアウトに至ることを確認済み (issue #47 検証時に実測)。
    src->particles_per_step = std::max(1, int((oceanSize.x * oceanSize.y * oceanSize.z) / (d * d * d)));
    src->step_count          = -1; // 初回のみ一括生成
    src->particleType        = 1u;
    engine_.addEmitter(src);

    // particles_per_step が初期容量 cfg.fluidCount() を超える場合は growFluidCapacity()
    // による動的拡張が発生する。この拡張は emitFromEmitters() 内 (recordComputeCmd の
    // コマンド記録前) で解決されるため、ウィンドウ有りの compute/graphics 分離
    // タイムラインセマフォ経路でも安全 (screw_fluid.cpp のコメント参照)。

    // ── 泡 (spray/foam/bubble) ────────────────────────────────────────────
    // 既定 (FluidEngine.h) より生成しきい値を上げ・生成係数を下げて発生数を抑える。
    // kFoamAdvect_ は maxDiffuseParticles 全スロットを毎 substep 無条件で走査するため、
    // 生成数そのものを絞ることに加えて --max-diffuse (バッファ容量) 自体を下げるのが
    // 最も直接的な負荷削減になる (両方を控えめな既定値にしている)。
    foamParams_.kTa                = 1500.0f; // 既定4000→生成量を抑制
    foamParams_.kWc                = 1500.0f;
    foamParams_.taLo               = 8.0f;    // 既定5→表面の乱れが大きい箇所のみ生成
    foamParams_.taHi               = 25.0f;
    foamParams_.wcLo               = 2.0f;    // 既定1
    foamParams_.wcHi               = 6.0f;
    foamParams_.keLo               = 8.0f;    // 既定5→高速な粒子のみ生成対象
    foamParams_.surfaceDensityRatio = 0.85f;  // 既定0.95→表面ゲートを厳しくして対象粒子数を削減
    foamParams_.lifetimeMin        = 0.6f;    // 既定1.0→同時生存数(=advectの実効負荷)を削減
    foamParams_.lifetimeMax        = 1.8f;    // 既定3.0
    engine_.foamEnabled = true;
    engine_.setFoamParams(foamParams_);

    graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/fluid_particle.vert.spv", SHADER_DIR_STR + "/fluid.frag.spv");
    foamGraphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/foam_particle.vert.spv", SHADER_DIR_STR + "/foam.frag.spv", VK_PRIMITIVE_TOPOLOGY_POINT_LIST, /*enableBlend=*/true);

    base_.createFrameData();
    base_.initImGui();
  }

  void recordComputeCmd(VkCommandBuffer cmd) {
    // 容量拡張によるバッファ再確保は、recordKinematicBoundaryUpdate が
    // その時点の VkBuffer ハンドルを焼き込む前に解決しておく (screw_fluid.cpp と同じ理由)。
    engine_.emitFromEmitters(dt_);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    engine_.recordKinematicBoundaryUpdate(cmd, base_.currentFrame, paddle_.gpuOffset, paddle_.count, paddleNewPos_.data(), paddleNewVel_.data());
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
    clear.color = {{0.02f, 0.05f, 0.09f, 1.0f}};
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

    SimPC pc{};
    pc.posIdx   = engine_.posIdx;
    pc.velIdx   = engine_.velIdx;
    pc.worldMin = glm::vec3(0.0f);
    pc.worldMax = engine_.config().domainSize;

    // 境界 (パドル + Bunny): buffer index 0 から
    pc.boundaryStart = 0;
    pc.particleCount = engine_.nBoundary;
    graphicsPipe_.draw(cmd, engine_.descriptorSet, pc, engine_.nBoundary);

    // 流体
    pc.boundaryStart = engine_.config().max_boundary;
    pc.particleCount = engine_.nFluid();
    graphicsPipe_.draw(cmd, engine_.descriptorSet, pc, engine_.nFluid());

    // 泡 (spray/foam/bubble)
    if(engine_.config().maxDiffuseParticles > 0) {
      SimPC foamPc{};
      foamPc.posIdx        = engine_.foamPosIdx();
      foamPc.velIdx        = engine_.foamVelIdx();
      foamPc.typeFlagIdx   = engine_.foamKindIdx();
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

    for(uint32_t i = 0; i < paddle_.count; ++i) {
      glm::vec4 p = paddle_.restPositions[i];
      p.x += paddle_.displacementX(simTime_);
      paddleNewPos_[i] = p;
      paddleNewVel_[i] = glm::vec4(paddle_.velocityX(simTime_), 0.0f, 0.0f, 0.0f);
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

    // 録画時にシーンが見えるよう、パネルは左上に小さく畳んでおく (issue #47 検証用)。
    ImGui::SetNextWindowSize(ImVec2(300.0f, 160.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
    ImGui::Begin("Wave+Bunny+Foam");
    ImGui::Text("FPS %.1f | fluid %u/%u | t=%.2fs", ImGui::GetIO().Framerate, engine_.nFluid(), engine_.config().fluidCount(), simTime_);
    sim_ui::fluid_reset_button(engine_, simTime_);
    ImGui::SliderFloat("paddle amp", &paddle_.amplitude, 0.0f, 3.0f);
    ImGui::SliderFloat("paddle omega", &paddle_.omega, 0.5f, 8.0f);
    if(sim_ui::foam_params(engine_, foamParams_)) engine_.setFoamParams(foamParams_);
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
    foamGraphicsPipe_.cleanup();
    engine_.cleanup();
    base_.cleanupBase();
  }
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  auto args = argparse::parse<WaveFoamArgs>(argc, argv);
  WaveFoamApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
