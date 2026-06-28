#pragma once
#include "HeadlessCtx.h"
#include "AttributeBuffer.h"
#include "ComputePipeline.h"
#include "SimPC.h"
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>

// GPU XPBD physics harness for headless tests.
// Handles rope/string (with stretch edges) and particle-box/fluid (without edges),
// with optional spatial-hash self-collision (solve_density).
class PhysicsHarness {
public:
    struct Config {
        uint32_t N          = 16;
        float    worldMin   = 0.0f;
        float    worldMax   = 10.0f;
        float    radius     = 0.14f;
        float    gravity    = -9.8f;
        float    windX      = 0.0f;
        float    windZ      = 0.0f;
        float    restitution= 0.3f;
        float    friction   = 0.2f;
        float    stretchCompliance = 0.0f;
        uint32_t gridRes    = 16;         // hash grid cells per axis
        int      numSubsteps      = 4;
        int      solverIterations = 4;
        uint32_t typeFlag         = 2;    // 2=cloth/string (colDiam=r*0.5), 0=sand (colDiam=r*2)
        bool     selfCollision    = false;

        // Packed edge triplets [pIdx, qIdx, floatBits(restLen), ...]
        // Color-0 edges must be stored first, then color-1 edges.
        std::vector<uint32_t> edgeData;
        uint32_t edgeColor0End = 0;   // boundary between the two color groups
    };

    void init(const HeadlessCtx& ctx, const Config& cfg,
              const std::string& shaderDir,
              const std::vector<glm::vec4>& initPos,
              const std::vector<glm::vec4>& initInvMass);

    // Simulate one frame (blocking GPU dispatch)
    void step(float dt);

    glm::vec4 readPos(uint32_t i) const;
    glm::vec4 readVel(uint32_t i) const;

    void cleanup();

    // Build 2-color graph-coloring edges for a particle chain:
    //   color-0: even edges (0-1, 2-3, 4-5, …)
    //   color-1: odd  edges (1-2, 3-4, 5-6, …)
    static void buildChainEdges(uint32_t N, float spacing,
                                std::vector<uint32_t>& edgeDataOut,
                                uint32_t& color0EndOut);

private:
    const HeadlessCtx* ctx_ = nullptr;
    Config cfg_;
    uint32_t nEdges_  = 0;
    uint32_t nCells_  = 0;
    uint32_t nGroups_ = 0;

    AttributeBuffer attrBuf_;
    uint32_t posIdx_    = 0, velIdx_    = 0, predPIdx_  = 0;
    uint32_t invMIdx_   = 0, typeFIdx_  = 0;
    uint32_t cellCntIdx_= 0, cellOffIdx_= 0, sortedIdx_ = 0;
    uint32_t edgesIdx_  = 0, lambdaIdx_ = 0;

    ComputePipeline kPredict_, kSdf_;
    ComputePipeline kHashCnt_, kScanLoc_, kScanGlob_, kSort_;
    ComputePipeline kSolveDen_, kSolveSt_, kUpdateVel_;

    void recordSubstep(VkCommandBuffer cmd, float subDt);
    void computeBarrier(VkCommandBuffer cmd);
    void fillBarrier(VkCommandBuffer cmd, VkBuffer buf);
};
