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

// run()/initVulkan() の両方から参照する定数 (旧: 同名のローカル変数が
// それぞれの関数内で重複定義されていた)。
constexpr float HALF_Y   = 3.0f; // 流体薄層の奥行き半幅 [m]
constexpr float Y_MARGIN = 1.0f; // パドル/水面がドメインY境界にちょうど触れないようにする余白 [m]
// パドル境界パーティクルの配置間隔は、流体粒子間隔 d=particleSpacing() に対する比率で決める
// (固定値0.15mだと d=0.12m(nx=200時) より粗く、境界の方が疎になってしまう)。
// PBFの境界拘束は密度制約(rho=Σ poly6(r))に基づく「柔らかい」拘束で、境界パーティクルが
// 疎だと同じ体積でも計算される密度が低く出て C=rho/rho0-1 が小さいまま→λが弱く
// →deltaPによる押し返しが弱く、パドルの往復のたびに背後へ回り込んだ流体がほぼ無抵抗で
// 貫通してしまう不具合を実測 (--n-shots で AABB/貫通粒子数をprintして確認) で確認した。
// 境界を流体より密に (d比0.75) 敷き詰めることで、壁際の計算密度を底上げし拘束を強化する。
constexpr float BOUNDARY_SPACING_RATIO = 0.75f;

// ── CLI ───────────────────────────────────────────────────────────────────────

struct WaveFoamArgs : public argparse::Args {
  float& world_size = kwarg("world-size", "simulation domain size [m] (X/Z)").set_default(24.0f);
  // h=cellSize は d=particleSpacing()=world_size/nx の2倍を推奨値とする
  // (FluidEngine.h の h>=2d 推奨に合わせる: nx=200 なら grid_res=nx/2=100)。
  // grid_res=64 (h/d≈3.1) だと1セルあたりの粒子数が (h/d)^3≈30 に膨らみ、
  // pbf_density/pbf_delta_p の27近傍セル探索が O(1セルあたりの粒子数) で
  // 効いてくるため、実測で ~2.6倍 遅くなることを確認した (issue #47 検証時)。
  // セル総数(hashCells)は grid_res を上げると増えるが、そちらのクリア/構築
  // コストより1セルあたりの粒子数超過の方が支配的だった。
  // 粒子数 ~10万 (nx=200,nz=24 で海の体積から逆算) を優先する設定。
  // grid_res は h=2d 推奨 (nx/2=100) を維持。~14fps程度になることを確認済み
  // (issue #47 検証, Apple M2 Pro)。フレームレートより粒子解像度を優先する。
  // 粒子間距離 d=world_size/nx (=粒子半径相当) を旧値の1.5倍にするため、nx を
  // 旧200から1/1.5倍の134に変更 (d: 0.12m→0.179m、約1.49倍)。grid_res は
  // h=2d 推奨を維持するため nx/2=67 に連動して下げている。
  int& grid_res               = kwarg("grid-res", "spatial hash grid resolution").set_default(67);
  int& fluid_nx               = kwarg("nx", "fluid particle grid X (密度基準)").set_default(134);
  // nz は水深方向の層数にほぼ等しい (waterDepth = fluid_nz * d より layers=waterDepth/d=fluid_nz)。
  // 層数を10程度にするため旧4から10へ変更 (副次的に粒子数も約2.5倍になる)。
  // 一時的に40まで上げたが、waterDepth(=nz*d)が domainSizeZ(world_size/6=4m) を
  // 超過し(7.16m)、境界付近で粒子が異常密集して性能が10倍以上悪化する不具合が
  // 判明したため10に戻す。粒子数の追加調整は kExtraParticleMultiplier
  // (下記 particles_per_step 算出箇所) で行い、水深とは切り離す。
  int& fluid_nz               = kwarg("nz", "fluid particle grid Z / 水深基準").set_default(10);
  float& dt                   = kwarg("dt", "timestep (sec)").set_default(1.0f / 60.0f);
  float& paddle_amp           = kwarg("paddle-amp", "波発生パドル SHM 振幅 [m]").set_default(3.0f);
  float& paddle_omega         = kwarg("paddle-omega", "波発生パドル SHM 角振動数 [rad/s]").set_default(1.2f);
  int& max_diffuse            = kwarg("max-diffuse", "泡(spray/foam/bubble)の最大パーティクル数").set_default(0);
  int& n_shots                = kwarg("n-shots", "screenshot count (0=disabled)").set_default(0);
  std::string& screenshot_dir = kwarg("screenshot-dir", "screenshot output directory").set_default(std::string(""));
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
  float amplitude    = 1.5f;            // [m]
  float omega        = 3.49f;           // 角振動数 [rad/s] (= 2π/period)
  uint32_t gpuOffset = 0;               // GPU バッファ先頭インデックス (常に0)
  uint32_t count     = 0;

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

    const float domainSizeY = HALF_Y * 2.0f + Y_MARGIN * 2.0f;
    const float domainSizeZ = args.world_size / 6.0f; // Z方向の高さは world_size の1/6 (旧1/3から半減)

    FluidConfig cfg;
    cfg.fluid_nx            = uint32_t(args.fluid_nx);
    cfg.fluid_ny            = 16; // 流体薄層の奥行き分解能
    cfg.fluid_nz            = uint32_t(args.fluid_nz);
    cfg.domainSize          = glm::vec3(args.world_size, domainSizeY, domainSizeZ);
    cfg.cellSize            = args.world_size / float(args.grid_res);
    cfg.max_boundary        = 20000;
    cfg.maxDiffuseParticles = uint32_t(args.max_diffuse);

    paddle_.amplitude = args.paddle_amp;
    paddle_.omega     = args.paddle_omega;

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

  float dt_      = 1.0f / 60.0f;
  float simTime_ = 0.0f;

  void initVulkan(const FluidConfig& cfg) {
    base_.ctx.init(base_.window);
    base_.createDescriptorPool();

    engine_.init(base_.ctx.device, base_.ctx.allocator, base_.descriptorPool, base_.ctx.graphicsCommandPool, base_.ctx.graphicsQueue, SHADER_DIR_STR, cfg);

    engine_.rho0             = cfg.computeRestDensity();
    engine_.viscosityC       = 0.002f;
    engine_.linearDamping    = 0.003f;
    engine_.vorticityEnabled = false;
    engine_.vorticityEpsilon = 0.15f;
    engine_.pbfIterations    = 2;
    engine_.numSubsteps      = 2;

    gravity_ = GravityForce::FromDirection({0.0f, 0.0f, -1.0f}, 9.8f); // Z-up

    engine_.addForce(gravity_);

    const float w           = cfg.domainSize.x;      // ドメイン X/Z サイズ (world_size)
    const float d           = cfg.particleSpacing(); // 流体粒子間隔（密度の基準）
    const float waterDepth  = cfg.fluid_nz * d;
    const float domainSizeY = cfg.domainSize.y; // run() で HALF_Y*2+Y_MARGIN*2 に設定済み
    const float centerY     = domainSizeY * 0.5f;

    // ── 波発生パドル (ドメイン左側、X方向に単振動) ────────────────────────────
    // Z: 下端はドメイン下端よりわずかに潜らせて隙間からの流体漏れを防ぎ、
    //    上端はドメイン上端を超えないようにする (超えるとハッシュセル範囲外の
    //    境界パーティクルが生じ、近傍探索が破綻し得る)。
    // Y: ドメインYのAABBを全てカバーする (centerY ± domainSizeY/2)。
    const float boundarySpacing = d * BOUNDARY_SPACING_RATIO; // 流体より密な境界パーティクル間隔
    const float paddleRestX     = paddle_.amplitude + 1.2f;   // 左壁からの最小マージンを確保
    const float paddleZMargin   = 2.0f * boundarySpacing;     // ドメイン下端よりわずかに下へ潜らせる
    const float paddleZMin      = -paddleZMargin;
    const float paddleZMax      = cfg.domainSize.z * 0.85f; // フレーム中央〜上寄りに収まる高さ (ドメインZ高さ基準)
    // パドルは前後2枚のシェル(x=center.x±paddleHalfThicknessX)で近似しているため、
    // 2枚の間隔(=paddleHalfThicknessX*2)が近傍探索の半径 h=cfg.cellSize より大きいと、
    // 2枚のどちらのカーネル範囲にも入らない「死角」がシェル内部にでき、そこを
    // 流体粒子が貫通できてしまう。h より確実に小さくして死角を無くす。
    const float paddleHalfThicknessX = cfg.cellSize * 0.4f;
    const glm::vec3 paddleCenter(paddleRestX, centerY, (paddleZMin + paddleZMax) * 0.5f);
    const glm::vec3 paddleHalfExtents(paddleHalfThicknessX, domainSizeY * 0.5f, (paddleZMax - paddleZMin) * 0.5f);
    paddle_.restPositions = generatePaddleShell(paddleCenter, paddleHalfExtents, boundarySpacing);
    paddle_.count         = uint32_t(std::min(paddle_.restPositions.size(), size_t(cfg.max_boundary)));
    paddle_.gpuOffset     = 0;
    paddleNewPos_.resize(paddle_.count);
    paddleNewVel_.resize(paddle_.count);

    // ── Bunny 障害物 (ドメイン右側に3体、床から突き出た岩礁のように配置) ────────
    // std::vector<glm::vec4> boundaryPts = paddle_.restPositions;
    // const float bunnyNativeHeight      = kBunnyMaxYup.y - kBunnyMinYup.y; // Y-up ローカル座標での高さ (実測)
    // const float bunnyTargetHeight      = w * 0.13f;                       // world_size に対する目標高さ（隣接バニー同士が重ならない大きさ）
    // const float bunnyScale             = bunnyTargetHeight / bunnyNativeHeight;
    // struct BunnyPlacement {
    //   float x, yOff;
    // };
    // const std::array<BunnyPlacement, 3> bunnies = {{
    //   {w * 0.60f, -0.2f},
    //   {w * 0.75f, 0.2f},
    //   {w * 0.90f, -0.1f},
    // }};
    // BoundaryParticles bp;
    // for(const auto& b : bunnies) {
    //   glm::vec3 floorCenter(b.x, centerY + b.yOff, 0.0f);
    //   glm::vec3 offset  = bunnyOffsetForFloorCenter(floorCenter, bunnyScale);
    //   BoundaryMesh mesh = bp.loadOBJ(ASSET_DIR_STR + "/bunny.obj", bSpacing, bunnyScale, offset, /*yup_to_zup=*/true);
    //   boundaryPts.insert(boundaryPts.end(), mesh.particles.begin(), mesh.particles.end());
    // }
    // engine_.loadBoundaryParticles(boundaryPts);
    // ── パドルを境界パーティクルとして登録 ────────────────────────────────
    // initKinematicBoundaryStaging() は毎フレーム位置/速度を更新するための
    // staging buffer を確保するだけで、P/predP/v/invMass/typeFlag の初期値は
    // 書き込まない。これらは loadBoundaryParticles() でしか設定されないため、
    // この呼び出しが無いと nBoundary が 0 のままになり、invMass/typeFlag が
    // 未初期化 (=境界扱いされない) のパドルは流体に一切干渉しない
    // (パーティクルがただ落下するだけで押し出されない不具合の直接原因)。
    engine_.loadBoundaryParticles(paddle_.restPositions);
    engine_.initKinematicBoundaryStaging(paddle_.count);

    // ── 流体: パドルのすぐ右に浅い「海」を配置 ──────────────────────────────
    // パドル静止位置の前面 (paddleRestX + halfExtents.x) のすぐ右に水面を置く。
    // 旧実装は oceanXStart = paddleRestX + amplitude + 0.8f としており、
    // パドル前面の最大到達位置 (paddleRestX + halfExtents.x + amplitude) より
    // さらに右 (デフォルト値で約0.55m先) から水を始めていたため、境界登録が
    // 直っていてもパドルが振動域内で水に一切接触せず、押し出す動きが起きない
    // 不具合があった。
    // Bunny を無効化した現状、water は w*0.51 までしか満たしておらず右半分が
    // 完全に空のままだった。ドメイン幅全体で波を起こすため、右壁際まで満たす
    // (右壁ぎりぎりに置くと SDF 境界反射で波が跳ね返ってくるので少し余白を残す)。
    const float oceanMargin = 2.0f * d; // パドル静止面とのごく小さいクリアランス
    const float oceanXStart = paddleRestX + paddleHalfExtents.x + oceanMargin;
    const float oceanXEnd   = w - 1.0f; // ドメイン右壁の手前まで満たす
    glm::vec3 oceanSize(oceanXEnd - oceanXStart, HALF_Y * 2.0f, waterDepth);
    auto src                = std::make_shared<AABBEmitter>();
    src->center             = glm::vec3((oceanXStart + oceanXEnd) * 0.5f, centerY, waterDepth * 0.5f);
    src->size               = oceanSize;
    src->vel                = glm::vec3(0.0f);
    src->particles_per_step = std::max(1, int((oceanSize.x * oceanSize.y * oceanSize.z) / (d * d * d)));
    src->step_count         = -1; // 初回のみ一括生成
    src->particleType       = 1u;
    engine_.addEmitter(src);

    // // ── 泡 (spray/foam/bubble) ────────────────────────────────────────────
    // foamParams_.kTa                 = 1500.0f; // 既定4000→生成量を抑制
    // foamParams_.kWc                 = 1500.0f;
    // foamParams_.taLo                = 8.0f; // 既定5→表面の乱れが大きい箇所のみ生成
    // foamParams_.taHi                = 25.0f;
    // foamParams_.wcLo                = 2.0f; // 既定1
    // foamParams_.wcHi                = 6.0f;
    // foamParams_.keLo                = 8.0f;  // 既定5→高速な粒子のみ生成対象
    // foamParams_.surfaceDensityRatio = 0.85f; // 既定0.95→表面ゲートを厳しくして対象粒子数を削減
    // foamParams_.lifetimeMin         = 0.6f;  // 既定1.0→同時生存数(=advectの実効負荷)を削減
    // foamParams_.lifetimeMax         = 1.8f;  // 既定3.0
    // engine_.foamEnabled             = false;
    // engine_.setFoamParams(foamParams_);

    // fluid_particle_wave.vert / foam_particle_wave.vert は本シーン専用のカメラ
    // (共有版より約2倍近く、Z軸まわりにさらに斜めから見下ろす) を使う複製シェーダー。
    // 共有シェーダーを直接変更すると screw_fluid 等 他の全シーンのカメラも
    // 変わってしまうため、複製して差し替えている。
    graphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/fluid_particle_wave.vert.spv", SHADER_DIR_STR + "/fluid.frag.spv");
    foamGraphicsPipe_.init(base_.ctx.device, base_.ctx.renderPass, engine_.descriptorSetLayout, SHADER_DIR_STR + "/foam_particle_wave.vert.spv", SHADER_DIR_STR + "/foam.frag.spv", VK_PRIMITIVE_TOPOLOGY_POINT_LIST, /*enableBlend=*/true);

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
