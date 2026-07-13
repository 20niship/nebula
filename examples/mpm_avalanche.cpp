#include "App.h"
#include "Collider.h"
#include "MaterialParams.h"
#include "core/SimPC.h"
#include "engine/MPMEngine.h"
#include "graphics/GraphicsPipeline.h"

#include <argparse/argparse.hpp>
#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── 地形関数 ────────────────────────────────────────────────────────────────

// SDF 構築用 (平滑版): rough/texture 高周波項を除外
// 高周波項の勾配は最大 ≈1.0 rad/m → セル幅 cs 以上のSDF誤差を生じ粒子が固体内に入る原因
static float terrainHeightSmooth(float x, float z, float W) {
  float slope   = W * 0.55f * (1.0f - z / W);
  float ridgeL  = W * 0.10f * std::exp(-std::pow((x - W * 0.28f) / (W * 0.06f), 2.0f));
  float ridgeR  = W * 0.10f * std::exp(-std::pow((x - W * 0.72f) / (W * 0.06f), 2.0f));
  float couloir = -W * 0.12f * std::exp(-std::pow((x - W * 0.50f) / (W * 0.10f + z * 0.006f), 2.0f));
  return std::max(0.2f, slope + ridgeL + ridgeR + couloir);
}

// 粒子初期配置用 (詳細版): rough/texture を加えて自然な凹凸に配置
static float terrainHeight(float x, float z, float W) {
  float base    = terrainHeightSmooth(x, z, W);
  float rough   = W * 0.020f * std::sin(x * 4.0f) * std::cos(z * 3.0f);
  float texture = W * 0.010f * std::sin(x * 9.0f + z * 7.0f);
  return base + rough + texture;
}

// ── Morton 符号化 (MPMEngine.cpp と同一) ────────────────────────────────────

static uint32_t mortonExpand(uint32_t v) {
  v &= 0x000003ffu;
  v = (v | (v << 16u)) & 0x030000ffu;
  v = (v | (v << 8u)) & 0x0300f00fu;
  v = (v | (v << 4u)) & 0x030c30c3u;
  v = (v | (v << 2u)) & 0x09249249u;
  return v;
}
static uint32_t mortonEncode(uint32_t x, uint32_t y, uint32_t z) { return mortonExpand(x) | (mortonExpand(y) << 1u) | (mortonExpand(z) << 2u); }

// ── CLI ────────────────────────────────────────────────────────────────────

struct AvalancheArgs : public argparse::Args {
  float& world_size           = kwarg("world-size", "world size [m]").set_default(10.0f);
  int& grid_res               = kwarg("grid-res", "MPM grid resolution").set_default(128);
  int& max_n                  = kwarg("max-n", "max particle count").set_default(80000);
  float& dt                   = kwarg("dt", "frame timestep [s]").set_default(1.0f / 60.0f);
  int& substeps               = kwarg("substeps", "substeps per frame").set_default(40);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
  // PIC (0.0) がデフォルト: 速度拡散で安定; APIC (-1.0) は角運動量保存だが DP 材料で発散しやすい
  float& flip_ratio = kwarg("flip-ratio", "0=PIC -1=APIC 0~1=FLIP").set_default(0.0f);
  int& vel_check    = kwarg("vel-check", "速度チェック間隔 (フレーム数)").set_default(30);
};

// ── App ────────────────────────────────────────────────────────────────────

class AvalancheApp {
public:
  void run(const AvalancheArgs& args) {
    dt_                 = args.dt;
    worldSize_          = args.world_size;
    velCheckEvery_      = args.vel_check;
    base_.screenshotDir = args.screenshot_dir;

    MPMConfig cfg;
    cfg.nx           = 0;
    cfg.ny           = 0;
    cfg.nz           = 0;
    cfg.maxParticles = uint32_t(args.max_n);
    cfg.world_size   = args.world_size;
    cfg.grid_res     = uint32_t(args.grid_res);

    base_.initWindow("MPM Mountain Avalanche — Drucker-Prager Snow");
    initVulkan(cfg, args.substeps, args.flip_ratio);
    mainLoop(args.n_shots);
    cleanup();
  }

private:
  BaseApp base_;
  MPMEngine engine_;
  GraphicsPipeline graphicsPipe_;
  float dt_        = 1.0f / 60.0f;
  float worldSize_ = 10.0f;
  float simTime_   = 0.0f;

  // ── 速度モニタリング ────────────────────────────────────────────────
  VkBuffer velStaging_           = VK_NULL_HANDLE;
  VmaAllocation velStagingAlloc_ = VK_NULL_HANDLE;
  int frameCount_                = 0;
  int velCheckEvery_             = 30;
  float velHistory_[120]{};
  int velHistHead_  = 0;
  float maxVelCur_  = 0.0f;
  float maxVelPrev_ = 0.0f;

  void createVelStaging(uint32_t maxParticles) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = VkDeviceSize(maxParticles) * sizeof(glm::vec4);
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    vmaCreateBuffer(base_.ctx.allocator, &bci, &aci, &velStaging_, &velStagingAlloc_, nullptr);
  }

  // GPU 速度バッファをホストに読み戻して最大速度を返す
  // 呼び出し前に前フレームの compute が完了していること (vkWaitForFences 済み)
  float readbackMaxVel() {
    uint32_t N = engine_.liveParticleCount();
    if(N == 0 || velStaging_ == VK_NULL_HANDLE) return 0.0f;

    VkDeviceSize sz = VkDeviceSize(N) * sizeof(glm::vec4);

    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = base_.ctx.graphicsCommandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(base_.ctx.device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    VkBufferCopy region{};
    region.size = sz;
    vkCmdCopyBuffer(cmd, engine_.getVelocityBuffer(), velStaging_, 1, &region);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(base_.ctx.graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(base_.ctx.graphicsQueue);
    vkFreeCommandBuffers(base_.ctx.device, base_.ctx.graphicsCommandPool, 1, &cmd);

    void* data;
    vmaMapMemory(base_.ctx.allocator, velStagingAlloc_, &data);
    const glm::vec4* vels = static_cast<const glm::vec4*>(data);
    float maxV            = 0.0f;
    for(uint32_t i = 0; i < N; ++i) {
      float spd = std::sqrt(vels[i].x * vels[i].x + vels[i].y * vels[i].y + vels[i].z * vels[i].z);
      if(spd > maxV) maxV = spd;
    }
    vmaUnmapMemory(base_.ctx.allocator, velStagingAlloc_);
    return maxV;
  }

  // ── 地形 SDF ───────────────────────────────────────────────────────
  // smooth版を使用: 高周波 rough/texture を含まないのでSDF境界が安定
  // さらに cs*0.5 の安全マージンを加算 → 粒子が固体内に入りにくくなる
  std::vector<float> buildTerrainSDF(const MPMConfig& cfg) {
    const uint32_t G = cfg.grid_res;
    const float cs   = cfg.cellSize();
    const float W    = cfg.world_size;
    std::vector<float> sdf(cfg.totalCells(), 1e9f);
    for(uint32_t iz = 0; iz < G; ++iz)
      for(uint32_t iy = 0; iy < G; ++iy)
        for(uint32_t ix = 0; ix < G; ++ix) {
          float cx                      = (ix + 0.5f) * cs;
          float cy                      = (iy + 0.5f) * cs;
          float cz                      = (iz + 0.5f) * cs;
          float h                       = terrainHeightSmooth(cx, cz, W) + cs * 0.5f; // 安全マージン
          sdf[mortonEncode(ix, iy, iz)] = cy - h;
        }
    return sdf;
  }

  // ── 雪粒子配置 ──────────────────────────────────────────────────────
  // オフセット = 2*cs: 地形 SDF のセル中心誤差 (terrain 高周波成分) を吸収
  // rough/texture の最大勾配は ~1.0 rad/m → セル幅 cs で最大 cs 程度の誤差
  // → 2*cs のマージンで初期位置が SDF 内部に入るのを防ぐ
  void placeSnow(const MPMConfig& cfg) {
    const float W  = cfg.world_size;
    const float cs = cfg.cellSize();

    const float x0 = W * 0.05f, x1 = W * 0.95f;
    const float z0 = W * 0.05f, z1 = W * 0.45f;
    const int nx_p   = 160;
    const int nz_p   = 160;
    const int nlay   = 3;
    const float dx   = (x1 - x0) / float(nx_p);
    const float dz_p = (z1 - z0) / float(nz_p);
    const float dy   = cs;
    const float Vp   = dx * dz_p * dy;

    const int maxN = int(cfg.maxParticleCount());
    std::vector<glm::vec4> pos, vel;
    pos.reserve(std::min(nx_p * nz_p * nlay, maxN));
    vel.reserve(std::min(nx_p * nz_p * nlay, maxN));

    for(int iz = 0; iz < nz_p && int(pos.size()) < maxN; ++iz)
      for(int ix = 0; ix < nx_p && int(pos.size()) < maxN; ++ix) {
        float px = x0 + (ix + 0.5f) * dx;
        float pz = z0 + (iz + 0.5f) * dz_p;
        float h  = terrainHeight(px, pz, W);
        for(int iy = 0; iy < nlay && int(pos.size()) < maxN; ++iy) {
          // 2*cs マージンで確実に SDF の外側に配置
          float py = h + 2.0f * cs + (iy + 0.5f) * dy;
          if(py >= W - cs) continue;
          pos.push_back({px, py, pz, Vp});
          vel.push_back({0.0f, 0.0f, 0.0f, 0.0f});
        }
      }
    std::printf("  雪粒子配置: %zu 個 (最大 %d)\n", pos.size(), maxN);
    engine_.appendParticles(pos, vel);
  }

  void initVulkan(const MPMConfig& cfg, int substeps, float flipRatio) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR, cfg);
    engine_.numSubsteps = substeps;
    engine_.gravity     = -9.8f;
    engine_.flip_ratio  = flipRatio;

    // ── Drucker-Prager 雪マテリアル ─────────────────────────────
    // E: 1e3 Pa (非常に柔らかい設定。J-クランプと組み合わせて蓄積圧縮爆発を防ぐ)
    // q_cohesion: 500 Pa (DP コーン頂点付近の張力不安定を防ぐ最小粘着力)
    MaterialParams snow{};
    snow.mu         = calcMu(1e3f, 0.3f);
    snow.lambda     = calcLambda(1e3f, 0.3f);
    snow.rho0       = 400.0f;
    snow.model      = uint32_t(MaterialModel::DRUCKER_PRAGER);
    snow.M_friction = 0.40f;
    snow.q_cohesion = 500.0f;
    engine_.setMaterials({snow});

    // 地形 SDF コライダー
    engine_.setColliderSDF(buildTerrainSDF(cfg));

    // ドメイン境界 (床・4壁)
    {
      ColliderSet cols;
      float W = cfg.world_size;
      cols.addPlane({W * 0.5f, 0.0f, W * 0.5f}, {0, 1, 0}, 0.0f, 0.3f);
      cols.addPlane({0.0f, W * 0.5f, W * 0.5f}, {1, 0, 0}, 0.0f, 0.1f);
      cols.addPlane({W, W * 0.5f, W * 0.5f}, {-1, 0, 0}, 0.0f, 0.1f);
      cols.addPlane({W * 0.5f, W * 0.5f, 0.0f}, {0, 0, 1}, 0.0f, 0.1f);
      cols.addPlane({W * 0.5f, W * 0.5f, W}, {0, 0, -1}, 0.0f, 0.1f);
      engine_.setColliders(cols);
    }

    placeSnow(cfg);
    createVelStaging(cfg.maxParticleCount());

    graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/particle.vert.spv", SHADER_DIR_STR + "/particle.frag.spv");
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
    renderPc.posIdx        = engine_.posIdx;
    renderPc.velIdx        = engine_.velIdx;
    renderPc.particleCount = engine_.liveParticleCount();
    renderPc.worldMin      = 0.0f;
    renderPc.worldMax      = engine_.config().world_size;
    graphicsPipe_.draw(cmd, engine_.descriptorSet, renderPc, engine_.liveParticleCount());

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
  }

  void drawFrame(int nShots) {
    auto& f = base_.frames[base_.currentFrame];
    vkWaitForFences(base_.ctx.device, 1, &f.inFlightFence, VK_TRUE, UINT64_MAX);

    // 速度チェック: 前フレームの compute 完了後に読み戻す
    ++frameCount_;
    if(velCheckEvery_ > 0 && frameCount_ % velCheckEvery_ == 0) {
      maxVelPrev_                       = maxVelCur_;
      maxVelCur_                        = readbackMaxVel();
      velHistory_[velHistHead_++ % 120] = maxVelCur_;
      // 急激な速度増加を検出してコンソールに出力
      if(maxVelPrev_ > 0.1f && maxVelCur_ > maxVelPrev_ * 4.0f) {
        std::printf("[frame %4d  t=%6.2fs] 速度急増! prev=%.2f → cur=%.2f m/s\n", frameCount_, simTime_, maxVelPrev_, maxVelCur_);
        std::fflush(stdout);
      } else {
        std::printf("[frame %4d  t=%6.2fs] max_vel=%.3f m/s\n", frameCount_, simTime_, maxVelCur_);
        std::fflush(stdout);
      }
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

    ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({370, 0}, ImGuiCond_Once);
    ImGui::Begin("MPM Mountain Avalanche");
    const char* mode = (engine_.flip_ratio < -0.5f) ? "APIC" : (engine_.flip_ratio > 0.01f) ? "FLIP" : "PIC";
    ImGui::Text("FPS: %.1f | N=%u | t=%.2f s | %s", ImGui::GetIO().Framerate, engine_.liveParticleCount(), simTime_, mode);
    ImGui::Separator();
    ImGui::SliderFloat("重力", &engine_.gravity, -20.0f, 0.0f);
    ImGui::SliderInt("サブステップ", &engine_.numSubsteps, 1, 60);
    ImGui::Separator();
    // 速度モニタリング
    bool exploding = (maxVelPrev_ > 0.1f && maxVelCur_ > maxVelPrev_ * 4.0f);
    if(exploding) ImGui::PushStyleColor(ImGuiCol_Text, {1, 0.2f, 0.2f, 1});
    ImGui::Text("max |v| = %.3f m/s %s", maxVelCur_, exploding ? "<!爆発>" : "");
    if(exploding) ImGui::PopStyleColor();
    // 直近 120 サンプルの速度履歴グラフ
    float dispHist[120];
    for(int i = 0; i < 120; ++i) dispHist[i] = velHistory_[(velHistHead_ - 120 + i + 120 * 2) % 120];
    ImGui::PlotLines("##velplot", dispHist, 120, 0, nullptr, 0.0f, std::max(20.0f, maxVelCur_ * 1.5f), {350, 60});
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
    if(velStaging_ != VK_NULL_HANDLE) {
      vmaDestroyBuffer(base_.ctx.allocator, velStaging_, velStagingAlloc_);
      velStaging_      = VK_NULL_HANDLE;
      velStagingAlloc_ = VK_NULL_HANDLE;
    }
    graphicsPipe_.cleanup();
    engine_.cleanup();
    base_.cleanupBase();
  }
};

// ── main ──────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  auto args = argparse::parse<AvalancheArgs>(argc, argv);
  AvalancheApp app;
  try {
    app.run(args);
  } catch(const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
