#ifndef COMMON_GLSL
#define COMMON_GLSL

layout(set = 0, binding = 0) buffer StorageBuffers {
    uint data[];
} buffers[];

layout(push_constant) uniform PC {
    // Bindless indices
    uint  posIdx;
    uint  velIdx;
    uint  predPIdx;
    uint  invMassIdx;
    uint  typeFlagIdx;
    uint  cellCountIdx;
    uint  cellOffsetIdx;
    uint  sortedIdxIdx;
    // Particle / grid
    uint  particleCount;
    uint  gridRes;
    uint  stretchEdgesIdx;
    uint  lambdasIdx;
    // World / time
    float dt;
    float cellSize;
    float worldMin;
    float worldMax;
    // SDF
    float gravity;
    float restitution;
    float friction;
    float particleRadius;
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
    float windX;
    float windZ;
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
    // 流体パーティクル領域の開始オフセット (= FluidEngine の cfg_.max_boundary)。
    // 既定値0=オフセットなしのため、設定しない他エンジンのシェーダーは影響を受けない。
    uint  fluidStart;
} pc;

// ── vec4 読み書き（FP32, MoltenVK 関数化バグ回避のためマクロ）────
// FP16 は worldSize=20m で位置精度 ~2cm となり PBF 補正量と同オーダーになって
// シミュレーションが発散するため FP32 を維持する
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

#define readUint(bufIdx, i)      buffers[(bufIdx)].data[(i)]
#define writeUint(bufIdx, i, v)  buffers[(bufIdx)].data[(i)] = (v)

#define readFloat(bufIdx, i)     uintBitsToFloat(buffers[(bufIdx)].data[(i)])
#define writeFloat(bufIdx, i, v) buffers[(bufIdx)].data[(i)] = floatBitsToUint(v)

// ── Morton符号（Z-orderカーブ）でセルIDを計算 ────────────────────
// 標準10bit展開（gridRes=64の6bitでも正しく動作）
// メモリ局所性が向上し pbf_density / pbf_delta_p のキャッシュ効率が改善する
uint mortonExpand(uint v) {
    v = (v | (v << 16u)) & 0x030000FFu;
    v = (v | (v <<  8u)) & 0x0300F00Fu;
    v = (v | (v <<  4u)) & 0x030C30C3u;
    v = (v | (v <<  2u)) & 0x09249249u;
    return v;
}

uint cellId(vec3 p) {
    vec3 local = clamp((p - pc.worldMin) / pc.cellSize,
                       vec3(0.0), vec3(float(pc.gridRes) - 1.0));
    uvec3 g = uvec3(local);
    return mortonExpand(g.x) | (mortonExpand(g.y) << 1u) | (mortonExpand(g.z) << 2u);
}

// ── 27近傍セル走査用: 軸ごとのMorton成分を事前計算 ──────────────────
// dx,dy,dz∈{-1,0,1}の27通りを毎回 mortonExpand()×3 で計算すると、同じ軸値
// (gi.x-1, gi.x, gi.x+1 等) に対する重複計算が26/27発生する。
// 軸ごとに3値だけ計算しておき、ループ内では mx[dx+1]|(my[dy+1]<<1)|(mz[dz+1]<<2)
// で組み合わせることで mortonExpand の呼び出し回数を 81 回→9 回に削減する。
// 範囲外座標 (gi±1 が [0,gridRes) の外) の要素は呼び出し側の境界チェックで
// 使用されないため、値自体は計算しても捨てられるだけで安全。
void mortonAxisTriples(ivec3 gi, out uint mx[3], out uint my[3], out uint mz[3]) {
    mx[0] = mortonExpand(uint(gi.x - 1)); mx[1] = mortonExpand(uint(gi.x)); mx[2] = mortonExpand(uint(gi.x + 1));
    my[0] = mortonExpand(uint(gi.y - 1)); my[1] = mortonExpand(uint(gi.y)); my[2] = mortonExpand(uint(gi.y + 1));
    mz[0] = mortonExpand(uint(gi.z - 1)); mz[1] = mortonExpand(uint(gi.z)); mz[2] = mortonExpand(uint(gi.z + 1));
}

#endif // COMMON_GLSL
