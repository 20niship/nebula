#include "App.h"
#include "engine/MPMEngine.h"
#include "MaterialParams.h"
#include "Collider.h"
#include "graphics/GraphicsPipeline.h"
#include "core/SimPC.h"

#include <argparse/argparse.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

static const std::string SHADER_DIR_STR = SHADER_DIR;

// ── 地形関数 (Python ツールと同一) ─────────────────────────────────────────

static float terrainHeight(float x, float z, float W) {
    float slope   = W * 0.55f * (1.0f - z / W);
    float ridgeL  = W * 0.10f * std::exp(-std::pow((x - W*0.28f) / (W*0.06f), 2.0f));
    float ridgeR  = W * 0.10f * std::exp(-std::pow((x - W*0.72f) / (W*0.06f), 2.0f));
    float couloir = -W * 0.12f * std::exp(-std::pow((x - W*0.50f) / (W*0.10f + z*0.006f), 2.0f));
    float rough   = W * 0.020f * std::sin(x * 4.0f) * std::cos(z * 3.0f);
    float texture = W * 0.010f * std::sin(x * 9.0f + z * 7.0f);
    return std::max(0.2f, slope + ridgeL + ridgeR + couloir + rough + texture);
}

// ── Morton 符号化 (MPMEngine.cpp と同一) ────────────────────────────────────

static uint32_t mortonExpand(uint32_t v) {
    v &= 0x000003ffu;
    v = (v | (v << 16u)) & 0x030000ffu;
    v = (v | (v <<  8u)) & 0x0300f00fu;
    v = (v | (v <<  4u)) & 0x030c30c3u;
    v = (v | (v <<  2u)) & 0x09249249u;
    return v;
}
static uint32_t mortonEncode(uint32_t x, uint32_t y, uint32_t z) {
    return mortonExpand(x) | (mortonExpand(y) << 1u) | (mortonExpand(z) << 2u);
}

// ── CLI ────────────────────────────────────────────────────────────────────

struct AvalancheArgs : public argparse::Args {
    float&       world_size     = kwarg("world-size",     "world size [m]").set_default(10.0f);
    int&         grid_res       = kwarg("grid-res",       "MPM grid resolution").set_default(128);
    int&         max_n          = kwarg("max-n",          "max particle count").set_default(80000);
    float&       dt             = kwarg("dt",             "frame timestep [s]").set_default(1.0f/60.0f);
    int&         substeps       = kwarg("substeps",       "substeps per frame").set_default(30);
    int&         n_shots        = kwarg("n-shots",        "screenshot count (0=disabled)").set_default(0);
    std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
    float&       flip_ratio     = kwarg("flip-ratio",     "0=PIC -1=APIC 0~1=FLIP").set_default(-1.0f);
};

// ── App ────────────────────────────────────────────────────────────────────

class AvalancheApp {
public:
    void run(const AvalancheArgs& args) {
        dt_        = args.dt;
        worldSize_ = args.world_size;
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
    BaseApp          base_;
    MPMEngine        engine_;
    GraphicsPipeline graphicsPipe_;
    float            dt_        = 1.0f / 60.0f;
    float            worldSize_ = 10.0f;
    float            simTime_   = 0.0f;

    // 地形 SDF (高さ場の符号付き距離) を Morton 順バッファとして構築
    std::vector<float> buildTerrainSDF(const MPMConfig& cfg) {
        const uint32_t G  = cfg.grid_res;
        const float    cs = cfg.cellSize();
        const float    W  = cfg.world_size;
        std::vector<float> sdf(cfg.totalCells(), 1e9f);
        for (uint32_t iz = 0; iz < G; ++iz)
        for (uint32_t iy = 0; iy < G; ++iy)
        for (uint32_t ix = 0; ix < G; ++ix) {
            float cx = (ix + 0.5f) * cs;
            float cy = (iy + 0.5f) * cs;
            float cz = (iz + 0.5f) * cs;
            float h  = terrainHeight(cx, cz, W);
            sdf[mortonEncode(ix, iy, iz)] = cy - h;  // < 0: 地形内部 (固体)
        }
        return sdf;
    }

    // 雪粒子を山肌上部に配置して appendParticles() で追加
    void placeSnow(const MPMConfig& cfg) {
        const float W   = cfg.world_size;
        const float cs  = cfg.cellSize();

        // 雪の覆う領域: x 全幅, z は上部 40% (斜面上部)
        const float x0   = W * 0.05f,  x1 = W * 0.95f;
        const float z0   = W * 0.05f,  z1 = W * 0.45f;
        const int   nx_p = 160;
        const int   nz_p = 160;
        const int   nlay = 3;
        const float dx   = (x1 - x0) / float(nx_p);
        const float dz_p = (z1 - z0) / float(nz_p);
        const float dy   = cs;
        const float Vp   = dx * dz_p * dy;

        const int maxN = int(cfg.maxParticleCount());
        std::vector<glm::vec4> pos, vel;
        pos.reserve(std::min(nx_p * nz_p * nlay, maxN));
        vel.reserve(std::min(nx_p * nz_p * nlay, maxN));

        for (int iz = 0; iz < nz_p && int(pos.size()) < maxN; ++iz)
        for (int ix = 0; ix < nx_p && int(pos.size()) < maxN; ++ix) {
            float px = x0 + (ix + 0.5f) * dx;
            float pz = z0 + (iz + 0.5f) * dz_p;
            float h  = terrainHeight(px, pz, W);
            for (int iy = 0; iy < nlay && int(pos.size()) < maxN; ++iy) {
                float py = h + (iy + 0.5f) * dy + cs * 0.15f;
                if (py >= W) continue;
                pos.push_back({px, py, pz, Vp});
                vel.push_back({0.0f, 0.0f, 0.0f, 0.0f});  // material id 0
            }
        }
        engine_.appendParticles(pos, vel);
    }

    void initVulkan(const MPMConfig& cfg, int substeps, float flipRatio) {
        base_.ctx.init(base_.window);
        base_.createDescriptorPool();

        engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool,
                     base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue,
                     SHADER_DIR_STR, cfg);
        engine_.numSubsteps = substeps;
        engine_.gravity     = -9.8f;
        engine_.flip_ratio  = flipRatio;

        // Drucker-Prager 雪マテリアル (スロット 0)
        MaterialParams snow{};
        snow.mu         = calcMu(5e4f, 0.3f);
        snow.lambda     = calcLambda(5e4f, 0.3f);
        snow.rho0       = 400.0f;
        snow.model      = uint32_t(MaterialModel::DRUCKER_PRAGER);
        snow.M_friction = 0.40f;  // tan(~22°)
        snow.q_cohesion = 0.0f;
        engine_.setMaterials({snow});

        // 地形 SDF コライダー
        engine_.setColliderSDF(buildTerrainSDF(cfg));

        // ドメイン境界 (壁・天井)
        {
            ColliderSet cols;
            float W = cfg.world_size;
            cols.addPlane({W*0.5f, 0.0f,   W*0.5f}, { 0, 1, 0}, 0.0f, 0.3f);
            cols.addPlane({0.0f,   W*0.5f, W*0.5f}, { 1, 0, 0}, 0.0f, 0.1f);
            cols.addPlane({W,      W*0.5f, W*0.5f}, {-1, 0, 0}, 0.0f, 0.1f);
            cols.addPlane({W*0.5f, W*0.5f, 0.0f  }, { 0, 0, 1}, 0.0f, 0.1f);
            cols.addPlane({W*0.5f, W*0.5f, W     }, { 0, 0,-1}, 0.0f, 0.1f);
            engine_.setColliders(cols);
        }

        // 初期雪粒子配置
        placeSnow(cfg);

        graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass,
                           engine_.descriptorSetLayout,
                           SHADER_DIR_STR + "/particle.vert.spv",
                           SHADER_DIR_STR + "/particle.frag.spv");
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
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
            0, 0, nullptr, 1, &barrier, 0, nullptr);

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
        VkRect2D sc{}; sc.extent = base_.ctx.swapchainExtent;
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

        uint32_t imageIdx;
        VkResult result = vkAcquireNextImageKHR(base_.ctx.device, base_.ctx.swapchain,
                                                 UINT64_MAX, f.imageAvailable,
                                                 VK_NULL_HANDLE, &imageIdx);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) { base_.ctx.recreateSwapchain(); return; }

        vkResetFences(base_.ctx.device, 1, &f.inFlightFence);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
        ImGui::SetNextWindowSize({360, 0}, ImGuiCond_Once);
        ImGui::Begin("MPM Mountain Avalanche");
        const char* mode = (engine_.flip_ratio < -0.5f) ? "APIC"
                         : (engine_.flip_ratio > 0.01f) ? "FLIP" : "PIC";
        ImGui::Text("FPS: %.1f | N=%u | t=%.2f s | %s",
                    ImGui::GetIO().Framerate, engine_.liveParticleCount(), simTime_, mode);
        ImGui::Separator();
        ImGui::SliderFloat("重力",       &engine_.gravity,      -20.0f, 0.0f);
        ImGui::SliderInt("サブステップ",  &engine_.numSubsteps,  1, 60);
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

        std::array<VkSemaphore, 2>          waitSems   = {f.imageAvailable, f.timelineSemaphore};
        std::array<VkPipelineStageFlags, 2> waitStages = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT};

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
        if (nShots > 0) base_.saveScreenshot(imageIdx, nShots);

        result = vkQueuePresentKHR(base_.ctx.graphicsQueue, &present);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR
            || base_.framebufferResized) {
            base_.framebufferResized = false;
            base_.ctx.recreateSwapchain();
        }
        base_.currentFrame = (base_.currentFrame + 1) % BaseApp::MAX_FRAMES;
    }

    void mainLoop(int nShots) {
        while (!glfwWindowShouldClose(base_.window) && !base_.shouldExit) {
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
    auto args = argparse::parse<AvalancheArgs>(argc, argv);
    AvalancheApp app;
    try {
        app.run(args);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
