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
    // PBF 密度制約 under-relaxation (IPBF 風; 他シェーダーは不使用)
    float relaxOmega;       // pbf_delta_p.comp の ΔP に乗じる緩和係数 (1.0=標準PBF互換)
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

#endif // COMMON_GLSL
