#ifndef COMMON_GLSL
#define COMMON_GLSL

layout(set = 0, binding = 0) buffer StorageBuffers {
    uint data[];
} buffers[];

// SimPC.h と同一オフセット。cellCountIdx/cellOffsetIdx/hashCells は MPMSimPC と hash compat。
layout(push_constant) uniform PC {
    // Bindless indices
    uint  posIdx;
    uint  velIdx;
    uint  predPIdx;
    uint  invMassIdx;
    uint  typeFlagIdx;
    uint  cellCountIdx;   // ← hash compat
    uint  cellOffsetIdx;  // ← hash compat
    uint  sortedIdxIdx;
    // Particle / grid
    uint  particleCount;
    uint  hashCells;      // 空間ハッシュバッファの実要素数 (=cubeRes^3) ← hash compat
    uint  stretchEdgesIdx;
    uint  lambdasIdx;
    // World / time
    float dt;
    float cellSize;       // 全軸共通のセルサイズ [m]
    float restitution;
    float friction;
    // Grid resolution / world bounds
    uvec3 gridRes;         // 各軸の実セル数 (nx,ny,nz)
    float particleRadius;
    vec3  worldMin;         // ドメイン下限座標 [m]
    uint  forceBufIdx;      // Force配列 (ForceGPU×forceCount) の bindless index (0=無効)
    vec3  worldMax;         // ドメイン上限座標 [m]
    // Cloth / Coupling
    uint  couplingForceIdx;
    uint  clothVertexCount;
    uint  edgeCount;
    uint  batchEdgeStart;
    uint  batchEdgeEnd;
    uint  densityIdx;
    uint  lambdaPbfIdx;
    uint  boundaryStart;
    float stretchCompliance;
    float bendCompliance;
    float particleCollisionRadius; // 旧windX。SoftBodyEngine専用 (sb_particle_collision.comp)
    uint  forceCount;              // 有効な Force 数 (旧windZ の枠を充当)
    // ── PBF 流体専用 追加パラメータ（SimPC.h と同一レイアウト）──────
    float cfmEpsilon;       // CFM 緩和 ε
    float scorrK;           // 人工圧力 k
    float vorticityEpsilon; // 渦度閉じ込め ε
    float linearDamping;    // 速度減衰係数 [1/s]
    uint  omegaIdx;         // 渦度 ω バッファ index
    float smokeRiseAccel;   // 煙の浮力加速度 [m/s²] (typeFlag==4)
    float smokeDamping;     // 煙の速度減衰係数 [1/s] (typeFlag==4)
    uint  pinnedTargetIdx;  // アニメーションピン目標位置バッファ (ClothSceneEngine 専用)
    // 吸収ポート (fluid_absorb 専用; 他シェーダーは宣言のみで不使用)
    uint  absorberBufIdx;   // 吸収形状バッファの bindless index (8 floats × absorberCount)
    uint  absorberCount;    // 有効な吸収形状数 (0 = 吸収パスをスキップ)
    uint  fluidStart; // cfg_.max_boundary; 0=他エンジンで未使用
    // 泡 (foam/spray/bubble) 二次パーティクル (pbf_foam_generate/pbf_foam_advect 専用; issue #47)
    uint  foamPosIdx;
    uint  foamVelIdx;
    uint  foamKindIdx;
    uint  foamParamsIdx;
    uint  maxDiffuseParticles;
    // 表面張力 (pbf_delta_p 専用; Akinci 2013 cohesion。他シェーダーは宣言のみで不使用)
    float surfaceTension;
    // 近傍リストキャッシュ (pbf_density/pbf_delta_p 専用; issue #87 perf実験)
    uint  nbrListIdx;
    // 粒子バッファの物理ソート済みコピー (pbf_density/pbf_delta_p 専用; issue #87 perf実験)
    uint  predPSortedIdx;
    uint  densitySortedIdx;
    uint  lambdaPbfSortedIdx;
    uint  invSortedIdxIdx;
    // 最終pos/velのソート済みコピー (pbf_viscosity 専用; issue #87 perf実験 続き)
    uint  posSortedIdx;
    uint  velSortedIdx;
} pc;

#ifndef MAX_NBR
#define MAX_NBR 48u
#endif

// HALF_VEC4: FluidEngine --large 専用。小ドメインのみ安全 (大ドメインは位置精度不足で発散)。
#ifdef HALF_VEC4
#define readVec4(bufIdx, i) vec4( \
    unpackHalf2x16(buffers[(bufIdx)].data[(i) * 2u]), \
    unpackHalf2x16(buffers[(bufIdx)].data[(i) * 2u + 1u]))

#define writeVec4(bufIdx, i, v) { \
    uint _wb = (i) * 2u; \
    buffers[(bufIdx)].data[_wb     ] = packHalf2x16((v).xy); \
    buffers[(bufIdx)].data[_wb + 1u] = packHalf2x16((v).zw); }
#else
#define readVec4(bufIdx, i) vec4( \
    uintBitsToFloat(buffers[(bufIdx)].data[(i) * 4u     ]), \
    uintBitsToFloat(buffers[(bufIdx)].data[(i) * 4u + 1u]), \
    uintBitsToFloat(buffers[(bufIdx)].data[(i) * 4u + 2u]), \
    uintBitsToFloat(buffers[(bufIdx)].data[(i) * 4u + 3u]))

#define writeVec4(bufIdx, i, v) { \
    uint _wb = (i) * 4u; \
    buffers[(bufIdx)].data[_wb     ] = floatBitsToUint((v).x); \
    buffers[(bufIdx)].data[_wb + 1u] = floatBitsToUint((v).y); \
    buffers[(bufIdx)].data[_wb + 2u] = floatBitsToUint((v).z); \
    buffers[(bufIdx)].data[_wb + 3u] = floatBitsToUint((v).w); }
#endif

// HALF_VEC4_V: v/omega のみ half (大ドメイン安定用; HALF_VEC4 時は自動的に適用)
#if defined(HALF_VEC4) || defined(HALF_VEC4_V)
#define readVec4Vel(bufIdx, i) vec4( \
    unpackHalf2x16(buffers[(bufIdx)].data[(i) * 2u]), \
    unpackHalf2x16(buffers[(bufIdx)].data[(i) * 2u + 1u]))
#define writeVec4Vel(bufIdx, i, v) { \
    uint _wv = (i) * 2u; \
    buffers[(bufIdx)].data[_wv     ] = packHalf2x16((v).xy); \
    buffers[(bufIdx)].data[_wv + 1u] = packHalf2x16((v).zw); }
#else
#define readVec4Vel(bufIdx, i)      readVec4(bufIdx, i)
#define writeVec4Vel(bufIdx, i, v)  writeVec4(bufIdx, i, v)
#endif

#define readUint(bufIdx, i)      buffers[(bufIdx)].data[(i)]
#define writeUint(bufIdx, i, v)  buffers[(bufIdx)].data[(i)] = (v)

#define readFloat(bufIdx, i)     uintBitsToFloat(buffers[(bufIdx)].data[(i)])
#define writeFloat(bufIdx, i, v) buffers[(bufIdx)].data[(i)] = floatBitsToUint(v)

// アダプティブ Morton: 異方性ドメイン向け。定数は C++ 側から実行時コンパイルで注入。
// フォールバック 0 は「cellId() を呼ばないシェーダー用の保険」; 実使用シェーダーでは必ず注入すること。
#ifndef ADAPTIVE_MASK
#define ADAPTIVE_MASK 0u
#define ADAPTIVE_COMMON_BITS 0u
#define ADAPTIVE_SHIFT_X 0u
#define ADAPTIVE_SHIFT_Y 0u
#define ADAPTIVE_SHIFT_Z 0u
#endif

uint mortonExpand(uint v) {
    v = (v | (v << 16u)) & 0x030000FFu;
    v = (v | (v <<  8u)) & 0x0300F00Fu;
    v = (v | (v <<  4u)) & 0x030C30C3u;
    v = (v | (v <<  2u)) & 0x09249249u;
    return v;
}

uint cellId(vec3 p) {
    vec3 local = clamp((p - pc.worldMin) / pc.cellSize,
                       vec3(0.0), vec3(pc.gridRes) - vec3(1.0));
    uvec3 g = uvec3(local);
    uint cx = mortonExpand(g.x & ADAPTIVE_MASK) | ((g.x >> ADAPTIVE_COMMON_BITS) << ADAPTIVE_SHIFT_X);
    uint cy = (mortonExpand(g.y & ADAPTIVE_MASK) << 1u) | ((g.y >> ADAPTIVE_COMMON_BITS) << ADAPTIVE_SHIFT_Y);
    uint cz = (mortonExpand(g.z & ADAPTIVE_MASK) << 2u) | ((g.z >> ADAPTIVE_COMMON_BITS) << ADAPTIVE_SHIFT_Z);
    return cx | cy | cz;
}

// 27近傍: 軸ごとに3値を事前計算してループ内で |演算。26/27の重複計算を削減。
void mortonAxisTriples(ivec3 gi, out uint mx[3], out uint my[3], out uint mz[3]) {
    for(int k = 0; k < 3; ++k) {
        uint vx = uint(gi.x - 1 + k);
        uint vy = uint(gi.y - 1 + k);
        uint vz = uint(gi.z - 1 + k);
        mx[k] = mortonExpand(vx & ADAPTIVE_MASK) | ((vx >> ADAPTIVE_COMMON_BITS) << ADAPTIVE_SHIFT_X);
        my[k] = (mortonExpand(vy & ADAPTIVE_MASK) << 1u) | ((vy >> ADAPTIVE_COMMON_BITS) << ADAPTIVE_SHIFT_Y);
        mz[k] = (mortonExpand(vz & ADAPTIVE_MASK) << 2u) | ((vz >> ADAPTIVE_COMMON_BITS) << ADAPTIVE_SHIFT_Z);
    }
}

#endif // COMMON_GLSL
