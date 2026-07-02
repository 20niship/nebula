/**
 * mpm_stl_drop — STL 障害物 SDF + 上部からパーティクル落下
 *
 * 手順:
 *  1. binary STL を読み込み (gen_sphere_stl.py で生成した球体など)
 *  2. 三角メッシュから Morton 順 SDF バッファを構築
 *     - 各グリッドセル中心 p に対し、最近傍三角形への符号付き距離を計算
 *     - 符号: p - closest_point ドット 面法線 ≥ 0 → 外側 (正)
 *  3. setColliderSDF() でアップロード → mpm_nanovdb_bc.comp が境界処理
 *  4. 解析平面コライダー (床・壁) を setColliders() で追加
 *  5. AABBSource で障害物の上部にパーティクルを継続放出
 */

#include "App.h"
#include "engine/MPMEngine.h"
#include "MaterialParams.h"
#include "Collider.h"
#include "graphics/GraphicsPipeline.h"
#include "core/SimPC.h"
#include "core/MeshSDF.h"
#include "../core/source.h"

#include <argparse/argparse.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static const std::string SHADER_DIR_STR = SHADER_DIR;
static const std::string ASSET_DIR_STR  = ASSET_DIR;

// STL ローダー・三角形→SDF 構築ロジックは src/core/MeshSDF.h (mpm_stl_drop 由来を
// PyroEngine 等でも再利用できるよう一般化したもの) を使う。

// ── CLI ────────────────────────────────────────────────────────────────────

struct StlDropArgs : public argparse::Args {
    std::string& stl_path      = kwarg("stl",          "障害物 STL ファイルパス")
                                       .set_default(ASSET_DIR_STR + "/sphere_obstacle.stl");
    float&       world_size    = kwarg("world-size",   "world size [m]").set_default(10.0f);
    int&         grid_res      = kwarg("grid-res",     "MPM grid resolution").set_default(64);
    int&         max_n         = kwarg("max-n",        "max particle count").set_default(32768);
    int&         emit_n        = kwarg("emit-n",       "particles per emit step").set_default(256);
    float&       dt            = kwarg("dt",           "frame timestep [s]").set_default(1.0f/60.0f);
    int&         substeps      = kwarg("substeps",     "substeps per frame").set_default(25);
    int&         n_shots       = kwarg("n-shots",      "screenshot count (0=disabled)").set_default(0);
    std::string& screenshot_dir= kwarg("screenshot-dir","screenshot output dir").set_default(std::string(""));
    float&       flip_ratio    = kwarg("flip-ratio",   "0=PIC -1=APIC 0~1=FLIP").set_default(-1.0f);
};

// ── App ────────────────────────────────────────────────────────────────────

class StlDropApp {
public:
    void run(const StlDropArgs& args) {
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

        stlPath_ = args.stl_path;
        emitN_   = args.emit_n;

        base_.initWindow("MPM STL Drop — 球体 STL 障害物");
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
    std::string      stlPath_;
    int              emitN_     = 256;

    void initVulkan(const MPMConfig& cfg, int substeps, float flipRatio) {
        base_.ctx.init(base_.window);
        base_.createDescriptorPool();

        engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool,
                     base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue,
                     SHADER_DIR_STR, cfg);
        engine_.numSubsteps = substeps;
        engine_.gravity     = -9.8f;
        engine_.flip_ratio  = flipRatio;

        // 砂マテリアル (Drucker-Prager)
        engine_.setMaterials({ presetSand(8e4f, 0.3f, 1500.0f) });

        // STL 読み込み → SDF 構築
        std::printf("STL 読み込み: %s\n", stlPath_.c_str());
        auto tris = loadBinarySTL(stlPath_);
        std::printf("  三角形数: %zu\n", tris.size());
        engine_.setColliderSDF(buildMeshSDF(tris, cfg.grid_res, cfg.world_size));

        // 床・壁の解析コライダー
        {
            ColliderSet cols;
            float W = cfg.world_size;
            cols.addPlane({W*0.5f, 0.1f,  W*0.5f}, { 0, 1, 0}, 0.0f, 0.5f);
            cols.addPlane({0.1f,  W*0.5f, W*0.5f}, { 1, 0, 0}, 0.0f, 0.2f);
            cols.addPlane({W-0.1f,W*0.5f, W*0.5f}, {-1, 0, 0}, 0.0f, 0.2f);
            cols.addPlane({W*0.5f,W*0.5f, 0.1f  }, { 0, 0, 1}, 0.0f, 0.2f);
            cols.addPlane({W*0.5f,W*0.5f, W-0.1f}, { 0, 0,-1}, 0.0f, 0.2f);
            engine_.setColliders(cols);
        }

        // ソース: 球体の真上から砂を継続放出
        float W = cfg.world_size;
        auto src = std::make_shared<AABBSource>(
            AABBSource::FromAABB(
                {W*0.3f, W*0.82f, W*0.3f},
                {W*0.7f, W*0.88f, W*0.7f},
                {0.0f, -2.0f, 0.0f},
                emitN_));
        src->step_count   = 0;   // 無限放出
        src->particleType = 0u;  // material id 0 (砂)
        engine_.addSource(src);

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
        clear.color = {{0.02f, 0.02f, 0.06f, 1.0f}};

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
        ImGui::SetNextWindowSize({380, 0}, ImGuiCond_Once);
        ImGui::Begin("MPM STL Drop");
        const char* mode = (engine_.flip_ratio < -0.5f) ? "APIC"
                         : (engine_.flip_ratio > 0.01f) ? "FLIP" : "PIC";
        ImGui::Text("FPS: %.1f | N=%u | t=%.2f s | %s",
                    ImGui::GetIO().Framerate, engine_.liveParticleCount(), simTime_, mode);
        ImGui::Text("STL: %s", stlPath_.c_str());
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
    auto args = argparse::parse<StlDropArgs>(argc, argv);
    StlDropApp app;
    try {
        app.run(args);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
