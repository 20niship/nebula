#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>

// ── Force 種別 (GPU側 shaders/force_common.glsl の FORCE_TYPE_* 定数と対応) ──
enum class ForceType : uint32_t {
  GRAVITY       = 0u,
  CONSTANT_WIND = 1u,
  TURBULENCE    = 2u,
  NOISE         = 3u,
};

// GPU 転送用 packed struct (64 bytes = 16 uint32_t)。EmitterGPU/ColliderPrimitive と
// 同一規約 (issue #30)。実データは毎フレーム bindless SSBO へアップロードし、
// push constant には バッファindex(forceBufIdx) + 個数(forceCount) のみ乗せる。
// [0]=type [1-3]=direction [4]=strength [5]=frequency [6]=octaves [7]=seed
// [8]=affectMask [9-15]=pad
struct alignas(16) ForceGPU {
  uint32_t type;
  float dx, dy, dz;    // 方向ベクトル。正規化必須ではない (strength との積が実加速度)
  float strength;      // 加速度スケール [m/s^2]
  float frequency;     // 空間周波数 [1/m] (Turbulence/Noise)
  uint32_t octaves;    // fBm オクターブ数 (Turbulence/Noise; 0は1として扱う)
  float seed;          // 空間/時間位相オフセット (時間発展は Force::advanceTime() が CPU 側で加算)
  uint32_t affectMask; // 適用対象 typeFlag のビットマスク (0=全対象、bit(n)=typeFlag==n)
  uint32_t pad0, pad1, pad2, pad3, pad4, pad5, pad6;
};
static_assert(sizeof(ForceGPU) == 64, "ForceGPU must be 64 bytes");

inline uint32_t ForceAffectTypeFlag(uint32_t typeFlag) { return 1u << typeFlag; }

// ── Force 基底クラス (Emitter.h と同型パターン: 継承で型を定義) ──────────────
struct Force {
  virtual ~Force() = default;

  virtual ForceType type() const = 0;
  // GPU アップロード用の packed 表現へ変換する。
  virtual ForceGPU pack() const = 0;
  // 時間発展を持つ Force (Turbulence/Noise) が seed 等を進める。既定は何もしない。
  virtual void advanceTime(float dt) {}

  uint32_t affectMask = 0u; // 0=全 typeFlag 対象

protected:
  void packCommon(ForceGPU& g) const { g.affectMask = affectMask; }
};

// ── GravityForce: 任意方向・強さの重力 (issue #30: 従来の下方向スカラーを一般化) ──
struct GravityForce : public Force {
  glm::vec3 direction{0.0f, 0.0f, -1.0f};
  float strength = 9.8f;

  ForceType type() const override { return ForceType::GRAVITY; }
  ForceGPU pack() const override {
    ForceGPU g{};
    g.type     = uint32_t(ForceType::GRAVITY);
    g.dx       = direction.x;
    g.dy       = direction.y;
    g.dz       = direction.z;
    g.strength = strength;
    packCommon(g);
    return g;
  }

  static std::shared_ptr<GravityForce> FromDirection(const glm::vec3& dir, float magnitude) {
    auto f       = std::make_shared<GravityForce>();
    f->direction = dir;
    f->strength  = magnitude;
    return f;
  }
};

// ── ConstantWindForce: 任意方向・強さの定常風 ────────────────────────────────
struct ConstantWindForce : public Force {
  glm::vec3 direction{1.0f, 0.0f, 0.0f};
  float strength = 1.0f;

  ForceType type() const override { return ForceType::CONSTANT_WIND; }
  ForceGPU pack() const override {
    ForceGPU g{};
    g.type     = uint32_t(ForceType::CONSTANT_WIND);
    g.dx       = direction.x;
    g.dy       = direction.y;
    g.dz       = direction.z;
    g.strength = strength;
    packCommon(g);
    return g;
  }

  static std::shared_ptr<ConstantWindForce> FromDirection(const glm::vec3& dir, float magnitude) {
    auto f       = std::make_shared<ConstantWindForce>();
    f->direction = dir;
    f->strength  = magnitude;
    return f;
  }
};

// ── TurbulenceForce: curl-noise による非圧縮的な乱流風 ──────────────────────
struct TurbulenceForce : public Force {
  float strength  = 1.0f; // 加速度スケール [m/s^2]
  float frequency = 0.5f; // 空間周波数 [1/m]
  uint32_t octaves = 3u;
  float seed  = 0.0f;
  float speed = 0.3f; // 時間発展速度 [1/s]

  ForceType type() const override { return ForceType::TURBULENCE; }
  ForceGPU pack() const override {
    ForceGPU g{};
    g.type      = uint32_t(ForceType::TURBULENCE);
    g.strength  = strength;
    g.frequency = frequency;
    g.octaves   = octaves;
    g.seed      = seed;
    packCommon(g);
    return g;
  }
  void advanceTime(float dt) override { seed += speed * dt; }
};

// ── NoiseForce: fBm スカラーノイズによる一方向の揺らぎ ───────────────────────
struct NoiseForce : public Force {
  glm::vec3 direction{0.0f, 1.0f, 0.0f}; // ノイズを乗せる方向 (正規化推奨)
  float strength  = 1.0f;
  float frequency = 1.0f;
  uint32_t octaves = 1u;
  float seed  = 0.0f;
  float speed = 0.0f;

  ForceType type() const override { return ForceType::NOISE; }
  ForceGPU pack() const override {
    ForceGPU g{};
    g.type      = uint32_t(ForceType::NOISE);
    g.dx        = direction.x;
    g.dy        = direction.y;
    g.dz        = direction.z;
    g.strength  = strength;
    g.frequency = frequency;
    g.octaves   = octaves;
    g.seed      = seed;
    packCommon(g);
    return g;
  }
  void advanceTime(float dt) override { seed += speed * dt; }
};
