#pragma once
#include <cstdint>
#include <glm/glm.hpp>

// Pyro (グリッドベース煙・火炎) 専用 Push Constants
// (issue #30: 末尾に forceBufIdx/forceCount を追加。
//  issue #46フォローアップ: 直方体ドメイン対応。worldMin/worldMax を vec3 化し、各軸独立の
//  gridRes(uvec3) を追加した。旧 gridRes(スカラー、Morton dispatch 用の立方体解像度)は
//  hashCells に改名した (MPMSimPC/mpm_p2g.comp/mpm_g2p.comp と同一の命名・用法。値は
//  domain::hashCells(gridRes()) = cubeRes^3)。
//  ComputePipeline の push constant range は sizeof(SimPC)=200 bytes で全エンジン共通
//  (src/core/ComputePipeline.h の static_assert(sizeof(PyroSimPC) <= sizeof(SimPC)) 参照)。
// MPMSimPC と同様、Bindless バッファインデックス + グリッド定数 + 物理パラメータを
// 固定サイズ構造体に詰める。velocity/density/temperature/fuel は front/back の
// ダブルバッファ (A/B) を持ち、CPU側 (PyroEngine::step) が毎フレーム役割を入れ替える。
//
// glm::vec3/uvec3 は GLSL push_constant ブロックの vec3 が要求する 16byte アライメントに
// 合わせるため、各 vec3/uvec3 直後にスカラーフィールドを1個「詰め物」として配置している
// (MPMSimPC.h の gridRes(64)+lambda_lame(76), worldMin(80)+particleVolume(92),
//  worldMax(96)+M_friction(108) と同一のテクニック)。
struct PyroSimPC {
  // ── フィールド バッファ (A/B ダブルバッファ) (16 bytes) ───────────────
  uint32_t velIdxA;     // 0   vec4×CELLS (xyz=速度)
  uint32_t velIdxB;     // 4
  uint32_t densityIdxA; // 8   float×CELLS
  uint32_t densityIdxB; // 12

  // (16 bytes)
  uint32_t temperatureIdxA; // 16  float×CELLS
  uint32_t temperatureIdxB; // 20
  uint32_t fuelIdxA;        // 24  float×CELLS
  uint32_t fuelIdxB;        // 28

  // ── 補助バッファ (16 bytes) ────────────────────────────────────────────
  uint32_t flameIdx;      // 32  float×CELLS (発光量、可視化用メタデータ)
  uint32_t pressureIdxA;  // 36  float×CELLS (Red-Black Gauss-Seidel、in-place更新)
  uint32_t gsColor;       // 40  Red-Black反復の色 (0=red/1=black、旧pressureIdxBを転用)
  uint32_t divergenceIdx; // 44  float×CELLS (スクラッチ)

  // ── コライダー / エミッタ / グリッド定数 (16 bytes) ────────────────────
  uint32_t colliderSDFIdx; // 48  float×CELLS (Morton SDF, 0=無効)
  uint32_t emittersIdx;    // 52  EmitterGPU×emitterCount (0=無効)
  uint32_t hashCells;      // 56  Morton dispatch用の稠密グリッド総セル数 (= cubeRes^3)
  uint32_t emitterCount;   // 60

  // ── グリッド解像度 / World (32 bytes) ────────────────────────────────
  glm::uvec3 gridRes; // 64  各軸の実セル数 (nx,ny,nz) = domain::gridRes(domainSize, cellSize)
  float dt;           // 76

  glm::vec3 worldMin; // 80  ドメイン下限座標 [m] (通常は原点 (0,0,0))
  float cellSize;     // 92  全軸共通のセルサイズ [m]

  glm::vec3 worldMax; // 96  ドメイン上限座標 [m] (= worldMin + domainSize)
  float buoyancyAlpha; // 108 温度浮力係数

  // ── 浮力 / 渦度 (16 bytes) ─────────────────────────────────────────────
  float buoyancyBeta;      // 112 密度による重さ (下向き) 係数
  float ambientTemp;       // 116 環境温度
  float vorticityEps;      // 120 渦度閉じ込め強度 (0=無効)
  float densityDissipation; // 124 密度減衰係数 [1/s]

  // ── 減衰 / 燃焼しきい値 (16 bytes) ─────────────────────────────────────
  float tempDissipation; // 128 温度減衰係数 [1/s] (環境温度への復帰)
  float ignitionTemp;    // 132 発火温度
  float burnRate;        // 136 燃料消費速度 [1/s]
  float heatRelease;     // 140 燃焼による温度上昇量 (burn量あたり)

  // ── 燃焼生成物 / Force (16 bytes) ──────────────────────────────────────
  float smokeYieldPerFuel; // 144 燃焼による密度生成量 (burn量あたり)
  float flameBrightness;   // 148 燃焼による発光量 (burn量あたり)
  uint32_t curlIdx;        // 152 vec4×CELLS 渦度 (curl) スクラッチ (渦度閉じ込め用)
  uint32_t forceBufIdx;    // 156 Force配列(ForceGPU×forceCount)のbindless index (0=無効)

  // ── Force (issue #30; 4 bytes) ─────────────────────────────────────────
  uint32_t forceCount; // 160 有効なForce数

  // 開境界ビットマスク (bit0..5 = -x,+x,-y,+y,-z,+z)。立っている面は領域外の圧力を 0 (大気圧) とみなし、風が出入りできる (0=従来どおり全面閉)
  uint32_t openBoundaryMask; // 164

  // 移動する剛体コライダー (任意の位置・姿勢・線速度・角速度)。rotSdfIdx != 0 のとき step() が colliderSDF を毎サブステップ剛体の姿勢に合わせて焼き直し
  // (pyro_rotate_sdf.comp)、障害物 BC / 発散は壁面速度 v = linVel + angVel × (p - pos) を課す。地面 (最下段の固体行) は静止。
  float bodyPos[3];     // 168 剛体原点の世界座標 [m]。角速度の回転中心でもある
  float bodyQuat[4];    // 180 剛体ローカル -> 世界 の回転 (x,y,z,w)
  float bodyLinVel[3];  // 196 線速度 (世界座標系) [m/s]
  float bodyAngVel[3];  // 208 角速度 (世界座標系, 原点まわり) [rad/s]
  uint32_t rotSdfIdx;   // 220 ローカル SDF (線形, x 最速)。節点値で、節点 (i,j,k) の位置は rotMin + (i,j,k)*rotCell。0=無効
  uint32_t rotResPacked; // 224 ローカル SDF の節点数 (x | y<<10 | z<<20, 各 1..1023)
  float rotCell;        // 228 節点間隔 [m]
  float rotMin[3];      // 232 ローカル SDF 原点 (剛体ローカル座標 [m])

  // 有効なチャンネル (PyroEngine::kChDensity 等のビット)。無効なチャンネルのバッファは 1 要素のダミーで、シェーダは読み書きしない
  uint32_t channelMask; // 244
};
static_assert(sizeof(PyroSimPC) == 248, "PyroSimPC must be 248 bytes (push constant 上限 256 に収める)");
static_assert(offsetof(PyroSimPC, gridRes) == 64, "vec3 alignment offset");
static_assert(offsetof(PyroSimPC, worldMin) == 80, "vec3 alignment offset");
static_assert(offsetof(PyroSimPC, worldMax) == 96, "vec3 alignment offset");
