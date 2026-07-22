#ifndef COMMON_GLSL
#define COMMON_GLSL

layout(set = 0, binding = 0) buffer StorageBuffers {
    uint data[];
} buffers[];

// C++側 src/core/SimPC.h と同一オフセット順であること (offsetof の static_assert 参照)。
// hash compat: cellCountIdx(20)/cellOffsetIdx(24)/hashCells(36) は MPMSimPC (mpm_common.glsl)
// と完全一致。gridRes/worldMin/worldMax はこの3フィールドより後ろにあるため一致不要。
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
    // 流体パーティクル領域の開始オフセット (= FluidEngine の cfg_.max_boundary)。
    // 既定値0=オフセットなしのため、設定しない他エンジンのシェーダーは影響を受けない。
    uint  fluidStart;
    // 泡 (foam/spray/bubble) 二次パーティクル (pbf_foam_generate/pbf_foam_advect 専用; issue #47)
    uint  foamPosIdx;
    uint  foamVelIdx;
    uint  foamKindIdx;
    uint  foamParamsIdx;
    uint  maxDiffuseParticles;
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

// ── アダプティブ(直方体)Morton符号 ────────────────────────────────
// 通常のMorton(Z-order)符号化は3軸を常に固定3ビット周期でinterleaveするため、
// 異方性ドメイン(軸ごとの実セル数の差が大きい)では最大軸のビット幅で全軸を
// 揃えた立方体を確保する必要があり大きな無駄が生じる。アダプティブ版は
// 「全軸が共通して持つ下位ビット(commonBits=min(bx,by,bz))」だけ従来通り
// 3軸interleaveし、残りは軸ごとに連結するだけに留める。局所性(27近傍探索
// でのメモリ距離)は数値実験で通常Mortonと完全に同一と確認済みだが、セル数は
// bx+by+bz ビット分 (=2^bx*2^by*2^bz) まで縮小できる。
//
// ADAPTIVE_MASK/ADAPTIVE_COMMON_BITS/ADAPTIVE_SHIFT_X,Y,Z はシェーダー内で
// 計算せず、ドメイン形状(gridRes)から一度だけ C++側(src/core/Domain.h の
// AdaptiveMortonParams/computeAdaptiveMortonParams()) で算出した値を、各
// エンジンの実行時コンパイル(DefineShaderCompiler)で #define として注入する。
// このファイルはビルド時に静的コンパイルされるシェーダーにも #include される
// ため、未定義でもコンパイルが通るようフォールバック値を用意しておく
// (実際に cellId()/mortonAxisTriples() を呼ぶシェーダーは必ず実行時コンパイル
// 側で正しい値を注入すること。フォールバックの0のままだと mask=0 となり
// 近傍探索が壊れるため、フォールバックはあくまで「未使用シェーダーの保険」)。
#ifndef ADAPTIVE_MASK
#define ADAPTIVE_MASK 0u
#define ADAPTIVE_COMMON_BITS 0u
#define ADAPTIVE_SHIFT_X 0u
#define ADAPTIVE_SHIFT_Y 0u
#define ADAPTIVE_SHIFT_Z 0u
#endif

// 下位 commonBits ビットの3軸interleave部分に使う標準Morton展開
// (標準10bit展開、gridRes=64の6bitでも正しく動作)。
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

// ── 27近傍セル走査用: 軸ごとのアダプティブMorton成分を事前計算 ──────────
// dx,dy,dz∈{-1,0,1}の27通りを毎回 cellId() 相当で計算すると、同じ軸値
// (gi.x-1, gi.x, gi.x+1 等) に対する重複計算が26/27発生する。
// 軸ごとに3値だけ計算しておき、ループ内では mx[dx+1]|my[dy+1]|mz[dz+1] で
// 組み合わせる(interleave部の<<1u/<<2uと上位連結部のシフトは各軸の値自体に
// 焼き込み済みのため、呼び出し側で追加シフトは不要)。
// 範囲外座標 (gi±1 が [0,gridRes) の外) の要素は呼び出し側の境界チェックで
// 使用されないため、値自体は計算しても捨てられるだけで安全。
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
