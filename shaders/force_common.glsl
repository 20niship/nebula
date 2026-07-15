#ifndef FORCE_COMMON_GLSL
#define FORCE_COMMON_GLSL

// Force システム (issue #30) 共通定義。
// buffers[] / readUint / readFloat は common.glsl (または mpm_common.glsl /
// pyro_common.glsl) が既に宣言している前提で、ここでは再宣言しない。
// 各ドメインの predict.comp 等は本ファイルより先に該当の *_common.glsl を
// #include すること。

// Force 種別 (src/core/Force.h の ForceType と対応)
#define FORCE_TYPE_GRAVITY       0u
#define FORCE_TYPE_CONSTANT_WIND 1u
#define FORCE_TYPE_TURBULENCE    2u
#define FORCE_TYPE_NOISE         3u

// ForceGPU (64 bytes = 16 uint32_t, src/core/Force.h と同一レイアウト) の読み取り
// マクロ。base = fi * 16u (fi = 配列内インデックス)。
// [0]=type [1-3]=direction [4]=strength [5]=frequency [6]=octaves [7]=seed [8]=affectMask
#define FORCE_TYPE_AT(bufIdx, base)     buffers[(bufIdx)].data[(base)]
#define FORCE_DIR(bufIdx, base)         vec3(uintBitsToFloat(buffers[(bufIdx)].data[(base) + 1u]), \
                                              uintBitsToFloat(buffers[(bufIdx)].data[(base) + 2u]), \
                                              uintBitsToFloat(buffers[(bufIdx)].data[(base) + 3u]))
#define FORCE_STRENGTH(bufIdx, base)    uintBitsToFloat(buffers[(bufIdx)].data[(base) + 4u])
#define FORCE_FREQUENCY(bufIdx, base)   uintBitsToFloat(buffers[(bufIdx)].data[(base) + 5u])
#define FORCE_OCTAVES(bufIdx, base)     buffers[(bufIdx)].data[(base) + 6u]
#define FORCE_SEED(bufIdx, base)        uintBitsToFloat(buffers[(bufIdx)].data[(base) + 7u])
#define FORCE_AFFECT_MASK(bufIdx, base) buffers[(bufIdx)].data[(base) + 8u]

#endif // FORCE_COMMON_GLSL
