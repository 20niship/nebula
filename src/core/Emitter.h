#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <glm/glm.hpp>

// ── Emitter 形状種別 (EmitterGPU.shape / shaders/pyro_common.glsl の
//    EMITTER_SHAPE_* 定数と対応) ──────────────────────────────────────────
enum class EmitterShape : uint32_t {
  AABB    = 0u,
  SPHERE  = 1u,
  ELLIPSE = 2u,
};

// GPU 転送用 packed struct (64 bytes = 16 uint32_t)。旧 PyroSourceGPU と
// 完全に同一のメモリレイアウト (shaders/pyro_emit.comp の base=si*16u
// オフセット計算は変更不要)。
// [0]=shape [1-3]=center [4-6]=size [7]=densityRate
// [8-10]=inflowVelocity [11]=temperatureRate [12]=fuelRate [13-15]=pad
struct alignas(16) EmitterGPU {
  uint32_t shape;
  float    cx, cy, cz;
  float    sx, sy, sz;
  float    densityRate;
  float    ivx, ivy, ivz;
  float    temperatureRate;
  float    fuelRate;
  float    pad0, pad1, pad2;
};
static_assert(sizeof(EmitterGPU) == 64, "EmitterGPU must be 64 bytes");

// ── Emitter 基底クラス ───────────────────────────────────────────────────
// MPM / Fluid(PBF) / FLIP(MPMEngine::flip_ratio 経由) / Pyro が共有する発生源
// の抽象。各エンジンが実際には使わないフィールドも一律で保持する
// (issue #12: 一部エンジンしか使わない情報でも統一して持たせる方針)。
struct Emitter {
  virtual ~Emitter() = default;

  // ── MPM / Fluid(PBF): 粒子を直接生成する際に使うフィールド ──────────
  glm::vec3 center{0.0f};
  glm::vec3 vel{0.0f};                     // 放出粒子の初速
  glm::vec3 center_vel{0.0f};              // エミッタ自体の移動速度 [m/s]
  int       particles_per_step = 1024;
  uint32_t  particleType       = 1u;       // MPM: material id / Fluid: 1=流体,4=煙,5=粉体

  // ── Pyro: グリッドセルへ連続注入する際に使うフィールド ──────────────
  glm::vec3 inflowVelocity{0.0f};          // 放出時に加える初速 [m/s]
  float     densityRate     = 0.0f;        // 密度注入速度 [1/s]
  float     temperatureRate = 0.0f;        // 温度注入速度 [K/s]
  float     fuelRate        = 0.0f;        // 燃料注入速度 [1/s]

  // ── 共通: 放出継続制御 ───────────────────────────────────────────────
  // 0=無限, -1=最初の1回(MPM/Fluid)/1フレーム(Pyro)のみ, >0=指定回数
  int step_count = -1;

  // MPM/Fluid が使用: 形状内に一様分布する1点を CPU でサンプリングする。
  virtual glm::vec3 sample(std::mt19937& rng) const = 0;
  // Pyro が使用: GPU アップロード用の packed 表現へ変換する。
  virtual EmitterGPU pack() const = 0;

protected:
  // shape/center/size 以外の共通フィールドを EmitterGPU に詰める補助。
  // サブクラスの pack() から呼ぶ。
  void packCommon(EmitterGPU& g) const {
    g.densityRate     = densityRate;
    g.ivx = inflowVelocity.x; g.ivy = inflowVelocity.y; g.ivz = inflowVelocity.z;
    g.temperatureRate = temperatureRate;
    g.fuelRate        = fuelRate;
  }
};

// ── AABBEmitter ──────────────────────────────────────────────────────────
struct AABBEmitter : public Emitter {
  // sample(): 直方体の全辺長 (旧 AABBSource::size と同一単位)。
  // pack(): GPU 側は半辺長規約 (旧 PyroSourceShape::AABB) のため、pack() 内で
  // 0.5倍して書き出す (単位変換のみ、挙動は変えない)。
  glm::vec3 size{1.0f};

  glm::vec3 sample(std::mt19937& rng) const override {
    glm::vec3 half = size * 0.5f;
    std::uniform_real_distribution<float> dx(-half.x, half.x);
    std::uniform_real_distribution<float> dy(-half.y, half.y);
    std::uniform_real_distribution<float> dz(-half.z, half.z);
    return center + glm::vec3(dx(rng), dy(rng), dz(rng));
  }

  EmitterGPU pack() const override {
    EmitterGPU g{};
    g.shape = static_cast<uint32_t>(EmitterShape::AABB);
    g.cx = center.x; g.cy = center.y; g.cz = center.z;
    g.sx = size.x * 0.5f; g.sy = size.y * 0.5f; g.sz = size.z * 0.5f;
    packCommon(g);
    return g;
  }

  static std::shared_ptr<AABBEmitter> FromAABB(const glm::vec3& min, const glm::vec3& max,
                                                const glm::vec3& vel, int particles_per_step) {
    auto e = std::make_shared<AABBEmitter>();
    e->center             = (min + max) * 0.5f;
    e->size               = max - min;
    e->vel                = vel;
    e->particles_per_step = particles_per_step;
    return e;
  }
};

// ── SphereEmitter ────────────────────────────────────────────────────────
struct SphereEmitter : public Emitter {
  float radius = 1.0f;

  glm::vec3 sample(std::mt19937& rng) const override {
    std::uniform_real_distribution<float> dr(0.0f, 1.0f);
    std::uniform_real_distribution<float> dtheta(0.0f, 3.14159265f);
    std::uniform_real_distribution<float> dphi(0.0f, 6.28318530f);
    float r     = radius * std::cbrt(dr(rng));
    float theta = dtheta(rng);
    float phi   = dphi(rng);
    return center + glm::vec3(
        r * std::sin(theta) * std::cos(phi),
        r * std::sin(theta) * std::sin(phi),
        r * std::cos(theta));
  }

  EmitterGPU pack() const override {
    EmitterGPU g{};
    g.shape = static_cast<uint32_t>(EmitterShape::SPHERE);
    g.cx = center.x; g.cy = center.y; g.cz = center.z;
    g.sx = radius; g.sy = radius; g.sz = radius; // シェーダーは sz.x のみ参照 (球は対称)
    packCommon(g);
    return g;
  }

  static std::shared_ptr<SphereEmitter> FromSphere(const glm::vec3& center, float radius,
                                                     const glm::vec3& vel, int particles_per_step) {
    auto e = std::make_shared<SphereEmitter>();
    e->center             = center;
    e->radius              = radius;
    e->vel                 = vel;
    e->particles_per_step = particles_per_step;
    return e;
  }
};

// ── EllipseEmitter ───────────────────────────────────────────────────────
// 旧 Fluid::EllipseSource (XY 平面の扁平楕円、sample() のみで使用) と、旧
// Pyro::PyroSourceShape::ELLIPSE (XZ 平面の楕円柱+Y半高さ、pack() のみで使用)
// という、互いに異なる幾何規約を単一クラスへロスレスに統合したもの。
// どちらも同一インスタンスが両方の意味で消費されることはない (Fluid は
// sample() だけ、Pyro は pack() だけを呼ぶ) ため、semiB は「呼び出し経路に
// よって意味が変わる」フィールドとして許容している:
//   - sample() (Fluid 用): semiA=X半径, semiB=Y半径, Z は center.z に固定。
//     halfHeightY は無視する。
//   - pack()   (Pyro  用): semiA=X半径, semiB=Z半径, halfHeightY=Y半高さ。
// 将来どちらの意味も同時に必要になった場合はこの共有をやめ、専用フィールドに
// 分離すること。
struct EllipseEmitter : public Emitter {
  float semiA       = 1.0f; // X 方向の半径 [m]
  float semiB       = 1.0f; // sample(): Y半径 / pack(): Z半径 ← 文脈依存 (上記コメント参照)
  float halfHeightY = 0.0f; // pack() 専用: Y 方向の半高さ [m] (sample() では無視)

  glm::vec3 sample(std::mt19937& rng) const override {
    // rejection sampling: バウンディングボックス内の一様乱数を楕円内に制限
    std::uniform_real_distribution<float> dx(-semiA, semiA);
    std::uniform_real_distribution<float> dy(-semiB, semiB);
    float ex, ey;
    do {
      ex = dx(rng);
      ey = dy(rng);
    } while ((ex * ex) / (semiA * semiA) + (ey * ey) / (semiB * semiB) > 1.0f);
    return glm::vec3(center.x + ex, center.y + ey, center.z); // Z は center.z 固定
  }

  EmitterGPU pack() const override {
    EmitterGPU g{};
    g.shape = static_cast<uint32_t>(EmitterShape::ELLIPSE);
    g.cx = center.x; g.cy = center.y; g.cz = center.z;
    // 旧 PyroSourceShape::ELLIPSE 規約: x=半径a, y=半高さ, z=半径b
    g.sx = semiA; g.sy = halfHeightY; g.sz = semiB;
    packCommon(g);
    return g;
  }
};
