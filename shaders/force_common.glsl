#ifndef FORCE_COMMON_GLSL
#define FORCE_COMMON_GLSL

// Force システム (issue #30) 共通定義。
// buffers[] / readUint / readFloat は common.glsl (または mpm_common.glsl /
// pyro_common.glsl) が既に宣言している前提で、ここでは再宣言しない。
// 各ドメインの predict.comp 等は本ファイルより先に該当の *_common.glsl を
// #include すること。

// ForceGPU (64 bytes = 16 words, src/core/Force.h と同一レイアウト) の読み取り
// マクロ。base = fi * 16u (fi = 配列内インデックス)。
// [0]=type [1]=affectMask [2-3]=pad [4-15]=data[0..11]
// (型番号は C++ 側 Force::type() から ForceShaderCompiler が数値リテラルとして
//  直接コード生成するため、ここに FORCE_TYPE_* のような named 定数は持たない)
#define FORCE_TYPE_AT(bufIdx, base)     buffers[(bufIdx)].data[(base)]
#define FORCE_AFFECT_MASK(bufIdx, base) buffers[(bufIdx)].data[(base) + 1u]

// data[] 内オフセット o (float単位) の値を読む汎用アクセサ
#define FORCE_DATA_FLOAT(bufIdx, base, o) uintBitsToFloat(buffers[(bufIdx)].data[(base) + 4u + (o)])
#define FORCE_DATA_UINT(bufIdx, base, o)  buffers[(bufIdx)].data[(base) + 4u + (o)]
#define FORCE_DATA_VEC3(bufIdx, base, o)  vec3(FORCE_DATA_FLOAT(bufIdx, base, o), \
                                                FORCE_DATA_FLOAT(bufIdx, base, (o) + 1u), \
                                                FORCE_DATA_FLOAT(bufIdx, base, (o) + 2u))

#endif // FORCE_COMMON_GLSL
