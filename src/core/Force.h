#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

// ── Force 種別 (SSBO上のtypeフィールド用タグ。GLSL側はここから数値リテラルを
// 直接生成するため、GLSL側に対応するnamed定数を維持する必要はない) ───────────
enum class ForceType : uint32_t {
  GRAVITY       = 0u,
  CONSTANT_WIND = 1u,
  TURBULENCE    = 2u,
  NOISE         = 3u,
  Z_CLAMP       = 4u, // 位置制約の実証例 (粒子のz座標を固定する)
};

// GPU 転送用 packed struct (32 bytes = 8 words)。
// word0 の下位16bit=type、上位16bit=affectMask (GLSL側はビット演算で分離して読む)。
// [1-7]=data[0..6] (float単位の汎用スロット、既存Force群の最大使用量ちょうどで
// スラック無し。新しいForceを追加してdata[]が7要素を超える場合は表現を見直すこと)。
// data[] の意味は各 Force::variables() が宣言する ForceVar 群が定義し、GLSL側は
// shaders/force_common.glsl の FORCE_DATA_FLOAT/VEC3/UINT アクセサで読む。
struct ForceGPU {
  uint16_t type;
  uint16_t affectMask;
  float data[7];
};
static_assert(sizeof(ForceGPU) == 32, "ForceGPU must be 32 bytes");

inline uint32_t ForceAffectTypeFlag(uint32_t typeFlag) { return 1u << typeFlag; }

// Force固有データの変数記述 (GLSL関数の引数リスト自動生成に使う)
struct ForceVar {
  const char* name;
  enum Kind { FLOAT, VEC3, UINT } kind;
  uint32_t offset; // ForceGPU::data[] 内オフセット (float単位)
};

// ── Force 基底クラス ─────────────────────────────────────────────────────────
// 各派生クラスは「GLSL関数本体(テンプレ)を返す関数」と「変数一覧を返す関数」を
// 持ち、ForceShaderCompiler がそこから動的にGLSLソースを生成する (issue #30
// レビュー対応)。速度への力 (glslApplyVelocity) と位置制約 (glslApplyConstraint)
// は独立したフックであり、Force はどちらか一方、または両方を実装できる。
struct Force {
  virtual ~Force() = default;

  virtual ForceType type() const = 0;

  // 生成するGLSL関数名の一意化に使うsuffix (例: "gravity")。
  // 同一 type() の複数インスタンスは同じ値を返す前提 (ForceShaderCompiler が
  // type() 単位で関数定義を1回だけ生成するため)。
  virtual const char* glslFunctionSuffix() const = 0;

  // このForceが ForceGPU::data[] へ書き込む変数の一覧。offset は packData() と
  // 整合させること。既定は空(データを持たないForce用)。
  virtual std::vector<ForceVar> variables() const { return {}; }

  // variables() のoffsetに従い data[12] へ値を書き込む。既定は何もしない。
  virtual void packData(float* out) const {}

  // 速度に力を適用するGLSL関数本体 (グローバルスコープ、void main() より前に
  // 挿入される)。本体内では v(inout vec3), forcePos(vec3), forceTypeFlag(uint),
  // dt(float) および variables() で宣言した名前の引数を参照できる。
  // 既定は空文字列 = このForceは速度を変更しない。
  virtual std::string glslApplyVelocity() const { return ""; }

  // 位置制約を適用するGLSL関数本体 (predP計算後、FORCE_AUTOGEN_POST マーカーで
  // 呼ばれる)。本体内では predP(inout vec4), p(vec4), v(vec3), forcePos,
  // forceTypeFlag, dt および variables() の引数を参照できる。
  // 既定は空文字列 = このForceは位置を変更しない。
  // 注意: predP/p の概念を持たないグリッドドメインのシェーダー
  // (mpm_grid_update.comp/pyro_forces.comp) にはPOSTマーカー自体が存在しない。
  // この関数の定義自体はvoid main()前に常に挿入されるため単体では有効なGLSLだが
  // (predP/pは関数の仮引数でありグローバル変数への参照ではないため)、呼び出し側
  // (POSTマーカー内のcall)が存在しないため実際には一切呼ばれず静かに無効化される。
  // これらのEngineに glslApplyConstraint() を持つForceを addForce() しても
  // コンパイルエラーにはならないが、位置制約の効果も得られない点に注意。
  virtual std::string glslApplyConstraint() const { return ""; }

  // Turbulence/Noise等の時間発展を持つForceが seed 等を進める。
  // EngineBase::uploadForces(dt) が毎フレーム全Forceに対して呼ぶ。
  virtual void advanceTime(float dt) {}

  uint32_t affectMask = 0u; // 0=全typeFlag対象

  // ForceGPUへの変換 (共通実装、派生クラスはoverrideしない)。
  // affectMask は下位16bitのみが ForceGPU::affectMask (uint16_t) に収まる
  // (typeFlag は0-15の範囲で使うこと)。
  ForceGPU pack() const {
    ForceGPU g{};
    g.type       = uint16_t(type());
    g.affectMask = uint16_t(affectMask);
    packData(g.data);
    return g;
  }

protected:
  // ForceGPU::data[] は7要素ちょうど(スラック無し)なので範囲外書き込みを assert で検出する。
  static void packFloat(float* out, uint32_t off, float v) {
    assert(off < 7 && "ForceGPU::data[] overflow");
    out[off] = v;
  }
  static void packVec3(float* out, uint32_t off, const glm::vec3& v) {
    assert(off + 2 < 7 && "ForceGPU::data[] overflow");
    out[off]     = v.x;
    out[off + 1] = v.y;
    out[off + 2] = v.z;
  }
  // uint はビットパターンをそのまま書き込む (GLSL側 FORCE_DATA_UINT は
  // uintBitsToFloat変換をしないため、単純代入すると値が壊れる)。
  static void packUint(float* out, uint32_t off, uint32_t v) {
    assert(off < 7 && "ForceGPU::data[] overflow");
    std::memcpy(&out[off], &v, sizeof(uint32_t));
  }
};

// ── GravityForce: 任意方向・強さの重力 ──────────────────────────────────────
struct GravityForce : public Force {
  glm::vec3 direction{0.0f, 0.0f, -1.0f};
  float strength = 9.8f;

  ForceType type() const override { return ForceType::GRAVITY; }
  const char* glslFunctionSuffix() const override { return "gravity"; }

  std::vector<ForceVar> variables() const override {
    return {{"dir", ForceVar::VEC3, 0}, {"strength", ForceVar::FLOAT, 3}};
  }
  void packData(float* out) const override {
    packVec3(out, 0, direction);
    packFloat(out, 3, strength);
  }
  std::string glslApplyVelocity() const override {
    return R"(
void forceApplyVelocity_gravity(inout vec3 v, vec3 forcePos, uint forceTypeFlag, float dt, vec3 dir, float strength) {
    v += dir * strength * dt;
}
)";
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
  const char* glslFunctionSuffix() const override { return "wind"; }

  std::vector<ForceVar> variables() const override {
    return {{"dir", ForceVar::VEC3, 0}, {"strength", ForceVar::FLOAT, 3}};
  }
  void packData(float* out) const override {
    packVec3(out, 0, direction);
    packFloat(out, 3, strength);
  }
  std::string glslApplyVelocity() const override {
    return R"(
void forceApplyVelocity_wind(inout vec3 v, vec3 forcePos, uint forceTypeFlag, float dt, vec3 dir, float strength) {
    v += dir * strength * dt;
}
)";
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
  const char* glslFunctionSuffix() const override { return "turbulence"; }

  std::vector<ForceVar> variables() const override {
    return {
      {"strength", ForceVar::FLOAT, 0},
      {"frequency", ForceVar::FLOAT, 1},
      {"octaves", ForceVar::UINT, 2},
      {"seed", ForceVar::FLOAT, 3},
    };
  }
  void packData(float* out) const override {
    packFloat(out, 0, strength);
    packFloat(out, 1, frequency);
    packUint(out, 2, octaves);
    packFloat(out, 3, seed);
  }
  std::string glslApplyVelocity() const override {
    return R"(
void forceApplyVelocity_turbulence(inout vec3 v, vec3 forcePos, uint forceTypeFlag, float dt, float strength, float frequency, uint octaves, float seed) {
    vec3 curl = forceCurlNoise3D(forcePos * frequency + seed, octaves);
    v += curl * strength * dt;
}
)";
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
  const char* glslFunctionSuffix() const override { return "noise"; }

  std::vector<ForceVar> variables() const override {
    return {
      {"dir", ForceVar::VEC3, 0},
      {"strength", ForceVar::FLOAT, 3},
      {"frequency", ForceVar::FLOAT, 4},
      {"octaves", ForceVar::UINT, 5},
      {"seed", ForceVar::FLOAT, 6},
    };
  }
  void packData(float* out) const override {
    packVec3(out, 0, direction);
    packFloat(out, 3, strength);
    packFloat(out, 4, frequency);
    packUint(out, 5, octaves);
    packFloat(out, 6, seed);
  }
  std::string glslApplyVelocity() const override {
    return R"(
void forceApplyVelocity_noise(inout vec3 v, vec3 forcePos, uint forceTypeFlag, float dt, vec3 dir, float strength, float frequency, uint octaves, float seed) {
    float n = forceFbmNoise3D(forcePos * frequency + seed, octaves);
    v += dir * (n * strength * dt);
}
)";
  }
  void advanceTime(float dt) override { seed += speed * dt; }
};

// ── ZClampForce: 粒子のz座標を指定値に固定する位置制約 ──────────────────────
// glslApplyVelocity() はオーバーライドせず(何もしない)、glslApplyConstraint()
// のみで predP.z を上書きする。速度への力(加速度加算)ではなく位置制約が
// Force継承で実現できることの実証例。predP/p の概念を持つ predict.comp /
// predict_sdf.comp を使うEngineでのみ使用可能。
struct ZClampForce : public Force {
  float zValue = 0.0f;

  ForceType type() const override { return ForceType::Z_CLAMP; }
  const char* glslFunctionSuffix() const override { return "zClamp"; }

  std::vector<ForceVar> variables() const override { return {{"zValue", ForceVar::FLOAT, 0}}; }
  void packData(float* out) const override { packFloat(out, 0, zValue); }

  std::string glslApplyConstraint() const override {
    return R"(
void forceApplyConstraint_zClamp(inout vec4 predP, vec4 p, vec3 v, vec3 forcePos, uint forceTypeFlag, float dt, float zValue) {
    predP.z = zValue;
}
)";
  }

  static std::shared_ptr<ZClampForce> At(float z) {
    auto f     = std::make_shared<ZClampForce>();
    f->zValue  = z;
    return f;
  }
};
