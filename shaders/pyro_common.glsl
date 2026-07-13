#ifndef PYRO_COMMON_GLSL
#define PYRO_COMMON_GLSL

// ── Bindless バッファ配列 ──────────────────────────────────────────────────
layout(set = 0, binding = 0) buffer StorageBuffers { uint data[]; } buffers[];

// ── PyroSimPC Push Constants (128 bytes) ───────────────────────────────────
layout(push_constant) uniform PC {
    uint  velIdxA;         // 0   vec4×CELLS (xyz=速度)
    uint  velIdxB;         // 4
    uint  densityIdxA;     // 8   float×CELLS
    uint  densityIdxB;     // 12
    uint  temperatureIdxA; // 16  float×CELLS
    uint  temperatureIdxB; // 20
    uint  fuelIdxA;        // 24  float×CELLS
    uint  fuelIdxB;        // 28
    uint  flameIdx;        // 32  float×CELLS
    uint  pressureIdxA;    // 36  float×CELLS (Red-Black GS、単一バッファ in-place)
    uint  gsColor;         // 40  0=red/1=black (pyro_pressure_gs.comp 専用)
    uint  divergenceIdx;   // 44  float×CELLS
    uint  colliderSDFIdx;  // 48  float×CELLS (Morton SDF, 0=無効)
    uint  emittersIdx;     // 52  EmitterGPU×emitterCount (0=無効)
    uint  gridRes;         // 56
    uint  emitterCount;    // 60
    float dt;              // 64
    float cellSize;        // 68
    float worldMin;        // 72
    float worldMax;        // 76
    float buoyancyAlpha;   // 80
    float buoyancyBeta;    // 84
    float ambientTemp;     // 88
    float vorticityEps;    // 92
    float densityDissipation; // 96
    float tempDissipation;    // 100
    float ignitionTemp;       // 104
    float burnRate;           // 108
    float heatRelease;        // 112
    float smokeYieldPerFuel;  // 116
    float flameBrightness;    // 120
    uint  curlIdx;            // 124 vec4×CELLS 渦度スクラッチ
    float velocityDissipation; // 128 速度減衰係数 [1/s]
    float maxVelocity;         // 132 速度magnitude上限 [m/s]
} pc;

// ── Buffer read/write マクロ (MoltenVK: buffers[] は main() でのみ展開) ────
#define readVec4(bufIdx, i) vec4( \
    uintBitsToFloat(buffers[(bufIdx)].data[(i)*4u     ]), \
    uintBitsToFloat(buffers[(bufIdx)].data[(i)*4u + 1u]), \
    uintBitsToFloat(buffers[(bufIdx)].data[(i)*4u + 2u]), \
    uintBitsToFloat(buffers[(bufIdx)].data[(i)*4u + 3u]))

#define writeVec4(bufIdx, i, v) { \
    uint _wb = (i)*4u; \
    buffers[(bufIdx)].data[_wb     ] = floatBitsToUint((v).x); \
    buffers[(bufIdx)].data[_wb + 1u] = floatBitsToUint((v).y); \
    buffers[(bufIdx)].data[_wb + 2u] = floatBitsToUint((v).z); \
    buffers[(bufIdx)].data[_wb + 3u] = floatBitsToUint((v).w); }

#define readUint(bufIdx, i)      buffers[(bufIdx)].data[(i)]
#define writeUint(bufIdx, i, v)  buffers[(bufIdx)].data[(i)] = (v)
#define readFloat(bufIdx, i)     uintBitsToFloat(buffers[(bufIdx)].data[(i)])
#define writeFloat(bufIdx, i, v) buffers[(bufIdx)].data[(i)] = floatBitsToUint(v)

// ── Emitter 形状定数 (Emitter.h の EmitterShape と一致) ────────────────────
#define EMITTER_SHAPE_AABB    0u
#define EMITTER_SHAPE_SPHERE  1u
#define EMITTER_SHAPE_ELLIPSE 2u

// ── Morton 符号 (Z-order curve, mpm_common.glsl と同一ロジック) ────────────
uint pyroMortonExpand(uint v) {
    v = (v | (v << 16u)) & 0x030000FFu;
    v = (v | (v <<  8u)) & 0x0300F00Fu;
    v = (v | (v <<  4u)) & 0x030C30C3u;
    v = (v | (v <<  2u)) & 0x09249249u;
    return v;
}
uint pyroMortonCompact(uint v) {
    v &= 0x09249249u;
    v = (v | (v >>  2u)) & 0x030C30C3u;
    v = (v | (v >>  4u)) & 0x0300F00Fu;
    v = (v | (v >>  8u)) & 0x030000FFu;
    v = (v | (v >> 16u)) & 0x000003FFu;
    return v;
}
uint pyroMortonEncodeI(ivec3 c) {
    return pyroMortonExpand(uint(c.x)) | (pyroMortonExpand(uint(c.y)) << 1u)
                                        | (pyroMortonExpand(uint(c.z)) << 2u);
}
ivec3 pyroMortonDecodeI(uint code) {
    return ivec3(pyroMortonCompact(code),
                 pyroMortonCompact(code >> 1u),
                 pyroMortonCompact(code >> 2u));
}

// セル中心グリッド座標 (ix,iy,iz) の境界クランプ付き読み取り
// グリッド外は最近傍セルへクランプ (Neumann/clamp-to-edge 境界)
#define CLAMP_CELL(ix_, iy_, iz_) ivec3( \
    clamp((ix_), 0, int(pc.gridRes) - 1), \
    clamp((iy_), 0, int(pc.gridRes) - 1), \
    clamp((iz_), 0, int(pc.gridRes) - 1))

#define SAMPLE_FLOAT_CLAMPED(bufIdx, ix_, iy_, iz_) \
    readFloat((bufIdx), pyroMortonEncodeI(CLAMP_CELL((ix_), (iy_), (iz_))))

#define SAMPLE_VEC3_CLAMPED(bufIdx, ix_, iy_, iz_) \
    readVec4((bufIdx), pyroMortonEncodeI(CLAMP_CELL((ix_), (iy_), (iz_)))).xyz

// セル (ix,iy,iz) (クランプ後) が障害物内部 (SDF<0) かどうか。SDF 未設定なら常に false。
#define CELL_IS_SOLID(ix_, iy_, iz_) ( pc.colliderSDFIdx != 0u && \
    readFloat(pc.colliderSDFIdx, pyroMortonEncodeI(CLAMP_CELL((ix_), (iy_), (iz_)))) < 0.0 )

// セル中心グリッド座標 (float) へのワールド座標変換
// セル (ix,iy,iz) の中心はワールド座標 worldMin + (ix+0.5, iy+0.5, iz+0.5)*cellSize
#define WORLD_TO_GRID(p) (((p) - pc.worldMin) / pc.cellSize - vec3(0.5))

// 三線形補間 (float フィールド)。gpos はセル格子座標系 (WORLD_TO_GRID 済み)
#define TRILERP_FLOAT(bufIdx, gpos, outVar) { \
    ivec3 _i0 = ivec3(floor(gpos)); \
    vec3  _f  = (gpos) - vec3(_i0); \
    float _c000 = SAMPLE_FLOAT_CLAMPED(bufIdx, _i0.x,   _i0.y,   _i0.z); \
    float _c100 = SAMPLE_FLOAT_CLAMPED(bufIdx, _i0.x+1, _i0.y,   _i0.z); \
    float _c010 = SAMPLE_FLOAT_CLAMPED(bufIdx, _i0.x,   _i0.y+1, _i0.z); \
    float _c110 = SAMPLE_FLOAT_CLAMPED(bufIdx, _i0.x+1, _i0.y+1, _i0.z); \
    float _c001 = SAMPLE_FLOAT_CLAMPED(bufIdx, _i0.x,   _i0.y,   _i0.z+1); \
    float _c101 = SAMPLE_FLOAT_CLAMPED(bufIdx, _i0.x+1, _i0.y,   _i0.z+1); \
    float _c011 = SAMPLE_FLOAT_CLAMPED(bufIdx, _i0.x,   _i0.y+1, _i0.z+1); \
    float _c111 = SAMPLE_FLOAT_CLAMPED(bufIdx, _i0.x+1, _i0.y+1, _i0.z+1); \
    float _x00 = mix(_c000, _c100, _f.x); \
    float _x10 = mix(_c010, _c110, _f.x); \
    float _x01 = mix(_c001, _c101, _f.x); \
    float _x11 = mix(_c011, _c111, _f.x); \
    float _y0  = mix(_x00, _x10, _f.y); \
    float _y1  = mix(_x01, _x11, _f.y); \
    outVar = mix(_y0, _y1, _f.z); }

// 三線形補間 (velocity vec3 フィールド)
#define TRILERP_VEC3(bufIdx, gpos, outVar) { \
    ivec3 _vi0 = ivec3(floor(gpos)); \
    vec3  _vf  = (gpos) - vec3(_vi0); \
    vec3 _vc000 = SAMPLE_VEC3_CLAMPED(bufIdx, _vi0.x,   _vi0.y,   _vi0.z); \
    vec3 _vc100 = SAMPLE_VEC3_CLAMPED(bufIdx, _vi0.x+1, _vi0.y,   _vi0.z); \
    vec3 _vc010 = SAMPLE_VEC3_CLAMPED(bufIdx, _vi0.x,   _vi0.y+1, _vi0.z); \
    vec3 _vc110 = SAMPLE_VEC3_CLAMPED(bufIdx, _vi0.x+1, _vi0.y+1, _vi0.z); \
    vec3 _vc001 = SAMPLE_VEC3_CLAMPED(bufIdx, _vi0.x,   _vi0.y,   _vi0.z+1); \
    vec3 _vc101 = SAMPLE_VEC3_CLAMPED(bufIdx, _vi0.x+1, _vi0.y,   _vi0.z+1); \
    vec3 _vc011 = SAMPLE_VEC3_CLAMPED(bufIdx, _vi0.x,   _vi0.y+1, _vi0.z+1); \
    vec3 _vc111 = SAMPLE_VEC3_CLAMPED(bufIdx, _vi0.x+1, _vi0.y+1, _vi0.z+1); \
    vec3 _vx00 = mix(_vc000, _vc100, _vf.x); \
    vec3 _vx10 = mix(_vc010, _vc110, _vf.x); \
    vec3 _vx01 = mix(_vc001, _vc101, _vf.x); \
    vec3 _vx11 = mix(_vc011, _vc111, _vf.x); \
    vec3 _vy0  = mix(_vx00, _vx10, _vf.y); \
    vec3 _vy1  = mix(_vx01, _vx11, _vf.y); \
    outVar = mix(_vy0, _vy1, _vf.z); }

// TRILERP_FLOAT と同じ補間に加え、使用した8隅の最小/最大値も返す
// (MacCormack移流のオーバーシュート抑制クランプ用)
#define TRILERP_FLOAT_MINMAX(bufIdx, gpos, outVar, outMin, outMax) { \
    ivec3 _mi0 = ivec3(floor(gpos)); \
    vec3  _mf  = (gpos) - vec3(_mi0); \
    float _mc000 = SAMPLE_FLOAT_CLAMPED(bufIdx, _mi0.x,   _mi0.y,   _mi0.z); \
    float _mc100 = SAMPLE_FLOAT_CLAMPED(bufIdx, _mi0.x+1, _mi0.y,   _mi0.z); \
    float _mc010 = SAMPLE_FLOAT_CLAMPED(bufIdx, _mi0.x,   _mi0.y+1, _mi0.z); \
    float _mc110 = SAMPLE_FLOAT_CLAMPED(bufIdx, _mi0.x+1, _mi0.y+1, _mi0.z); \
    float _mc001 = SAMPLE_FLOAT_CLAMPED(bufIdx, _mi0.x,   _mi0.y,   _mi0.z+1); \
    float _mc101 = SAMPLE_FLOAT_CLAMPED(bufIdx, _mi0.x+1, _mi0.y,   _mi0.z+1); \
    float _mc011 = SAMPLE_FLOAT_CLAMPED(bufIdx, _mi0.x,   _mi0.y+1, _mi0.z+1); \
    float _mc111 = SAMPLE_FLOAT_CLAMPED(bufIdx, _mi0.x+1, _mi0.y+1, _mi0.z+1); \
    float _mx00 = mix(_mc000, _mc100, _mf.x); \
    float _mx10 = mix(_mc010, _mc110, _mf.x); \
    float _mx01 = mix(_mc001, _mc101, _mf.x); \
    float _mx11 = mix(_mc011, _mc111, _mf.x); \
    float _my0  = mix(_mx00, _mx10, _mf.y); \
    float _my1  = mix(_mx01, _mx11, _mf.y); \
    outVar = mix(_my0, _my1, _mf.z); \
    outMin = min(_mc000, min(_mc100, min(_mc010, min(_mc110, min(_mc001, min(_mc101, min(_mc011, _mc111))))))); \
    outMax = max(_mc000, max(_mc100, max(_mc010, max(_mc110, max(_mc001, max(_mc101, max(_mc011, _mc111))))))); }

// TRILERP_VEC3 と同じ補間に加え、使用した8隅の成分ごとの最小/最大値も返す
#define TRILERP_VEC3_MINMAX(bufIdx, gpos, outVar, outMin, outMax) { \
    ivec3 _ni0 = ivec3(floor(gpos)); \
    vec3  _nf  = (gpos) - vec3(_ni0); \
    vec3 _nc000 = SAMPLE_VEC3_CLAMPED(bufIdx, _ni0.x,   _ni0.y,   _ni0.z); \
    vec3 _nc100 = SAMPLE_VEC3_CLAMPED(bufIdx, _ni0.x+1, _ni0.y,   _ni0.z); \
    vec3 _nc010 = SAMPLE_VEC3_CLAMPED(bufIdx, _ni0.x,   _ni0.y+1, _ni0.z); \
    vec3 _nc110 = SAMPLE_VEC3_CLAMPED(bufIdx, _ni0.x+1, _ni0.y+1, _ni0.z); \
    vec3 _nc001 = SAMPLE_VEC3_CLAMPED(bufIdx, _ni0.x,   _ni0.y,   _ni0.z+1); \
    vec3 _nc101 = SAMPLE_VEC3_CLAMPED(bufIdx, _ni0.x+1, _ni0.y,   _ni0.z+1); \
    vec3 _nc011 = SAMPLE_VEC3_CLAMPED(bufIdx, _ni0.x,   _ni0.y+1, _ni0.z+1); \
    vec3 _nc111 = SAMPLE_VEC3_CLAMPED(bufIdx, _ni0.x+1, _ni0.y+1, _ni0.z+1); \
    vec3 _nx00 = mix(_nc000, _nc100, _nf.x); \
    vec3 _nx10 = mix(_nc010, _nc110, _nf.x); \
    vec3 _nx01 = mix(_nc001, _nc101, _nf.x); \
    vec3 _nx11 = mix(_nc011, _nc111, _nf.x); \
    vec3 _ny0  = mix(_nx00, _nx10, _nf.y); \
    vec3 _ny1  = mix(_nx01, _nx11, _nf.y); \
    outVar = mix(_ny0, _ny1, _nf.z); \
    outMin = min(_nc000, min(_nc100, min(_nc010, min(_nc110, min(_nc001, min(_nc101, min(_nc011, _nc111))))))); \
    outMax = max(_nc000, max(_nc100, max(_nc010, max(_nc110, max(_nc001, max(_nc101, max(_nc011, _nc111))))))); }

#endif // PYRO_COMMON_GLSL
