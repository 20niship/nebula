#ifndef FLIP_FOAM_COMMON_GLSL
#define FLIP_FOAM_COMMON_GLSL

#include "mpm_common.glsl"

// FLIP 泡 (spray/foam/bubble) 共通定義。泡パスでは pc.flipGridIdx = foamParams バッファ、pc.pass = 乱数ソルト。
// foamParams レイアウト (32 要素): float と uint の混在 (readFloat/readUint で読む)
//   0:kTa 1:kWc 2:taLo 3:taHi 4:wcLo 5:wcHi 6:keLo 7:keHi 8:lifetimeMin 9:lifetimeMax
//   10:bubbleBuoyancy 11:dragCoeff 12-14:gravity(xyz) 15:densityLo 16:densityHi 17:surfaceDensityRatio
//   18:foamPosIdx(u) 19:foamVelIdx(u) 20:foamKindIdx(u) 21:maxDiffuse(u)
#define FP_KTA 0u
#define FP_KWC 1u
#define FP_TALO 2u
#define FP_TAHI 3u
#define FP_WCLO 4u
#define FP_WCHI 5u
#define FP_KELO 6u
#define FP_KEHI 7u
#define FP_LIFEMIN 8u
#define FP_LIFEMAX 9u
#define FP_BUOY 10u
#define FP_DRAG 11u
#define FP_GRAV 12u
#define FP_DENLO 15u
#define FP_DENHI 16u
#define FP_SURF 17u
#define FP_POSIDX 18u
#define FP_VELIDX 19u
#define FP_KINDIDX 20u
#define FP_MAXN 21u

float psi(float x, float lo, float hi) {
    return clamp((x - lo) / max(hi - lo, 1e-5), 0.0, 1.0);
}

float hashRand(uint idx, uint salt) {
    uint h = idx * 2654435761u ^ (salt * 7654321u) ^ (pc.pass * 40503u);
    h ^= (h >> 16u); h *= 0x45d9f3bu; h ^= (h >> 16u);
    return float(h & 0xFFFFu) / 65535.0;
}

#endif
