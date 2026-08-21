#ifndef MPM_COMMON_GLSL
#define MPM_COMMON_GLSL

// ── Bindless バッファ配列 ──────────────────────────────────────────────────
layout(set = 0, binding = 0) buffer StorageBuffers { uint data[]; } buffers[];

// ── MPMSimPC Push Constants (180 bytes) ───────────────────────
// C++側 src/core/MPMSimPC.h と同一オフセット順であること。vec3系フィールド直前のスカラー数を
// 4の倍数に揃えてパディングなしで16byte境界に一致させている(順序変更時は要再検証)。
layout(push_constant) uniform PC {
    uint  posIdx;        // 0   vec4×N  (xyz=pos, w=Vp)
    uint  velIdx;        // 4   vec4×N  (xyz=vel, w=material id)
    uint  F0Idx;         // 8   vec4×N  F 列0 (xyz) + σ_xx (w)
    uint  F1Idx;         // 12  vec4×N  F 列1 (xyz) + σ_yy (w)
    uint  typeFlagIdx;   // 16  (reserved)
    uint  particleCount; // 20  ライブ粒子数
    uint  hashCells;     // 24  空間ハッシュ/MPMグリッドバッファの実要素数 (=cubeRes^3)
    uint  F2Idx;         // 28  F 列2 (xyz) + σ_zz (w)
    uint  materialsIdx;  // 32  MaterialParams SSBO (0=無効)
    float dt;            // 36
    float cellSize;      // 40  全軸共通のセルサイズ [m]
    uint  forceBufIdx;   // 44  Force配列(ForceGPU×forceCount)のbindless index (issue #30; 旧gravity)

    uvec3 gridRes; // 48  各軸の実セル数 (nx,ny,nz)
    float mu_lame; // 60  グローバルデフォルト μ

    float lambda_lame;   // 64  グローバルデフォルト λ
    float particleVolume; // 68  グローバルデフォルト Vp
    float M_friction;    // 72  グローバルデフォルト DP M
    float q_cohesion;    // 76  グローバルデフォルト DP q_c

    vec3  worldMin; // 80  ドメイン下限座標 [m]
    float q_max;    // 92  グローバルデフォルト VM q_max

    float flip_ratio;    // 96  0=PIC, 1=FLIP, -1=APIC
    uint  colliderIdx;   // 100 Collider SSBO (Phase 3)
    uint  colliderCount; // 104 コライダー数 (Phase 3)
    uint  B0Idx;         // 108 B 列0 (xyz, APIC) + σ_xy (w)

    vec3  worldMax; // 112 ドメイン上限座標 [m]
    uint  B1Idx;    // 124 B 列1 (xyz, APIC) + σ_xz (w)

    uint  B2Idx;         // 128 B 列2 (xyz, APIC) + σ_yz (w)
    uint  gridVelOldIdx; // 132 packed-half3×CELLS (8B/cell) FLIP用 v_old (旧NanoVDB SDF reserved)
    uint  gridMomIdx;    // 136 P2G(scatter)固定小数点atomicAdd蓄積専用、G2Pは読まない
    uint  gridMassIdx;   // 140 同上
    float restitution;   // 144
    float wall_friction; // 148
    uint  plasticModel;  // 152 グローバルモデル (Phase 1 まで有効)
    uint  materialCount; // 156 materials エントリ数
    float rho0;          // 160 グローバルデフォルト密度
    float p0_mcc;        // 164
    float xi_hard;       // 168
    uint  forceCount;    // 172 有効なForce数 (issue #30; 旧maxParticlesFrac予約枠)
    uint  gridVelIdx;    // 176 packed-half3×CELLS (8B/cell) GridUpdateが書きG2Pが読む v_new
} pc;

// ── Buffer read/write マクロ ──────────────────────────────────────────────
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

#define readUint(bufIdx, i)       buffers[(bufIdx)].data[(i)]
#define writeUint(bufIdx, i, v)   buffers[(bufIdx)].data[(i)] = (v)
#define readFloat(bufIdx, i)      uintBitsToFloat(buffers[(bufIdx)].data[(i)])
#define writeFloat(bufIdx, i, v)  buffers[(bufIdx)].data[(i)] = floatBitsToUint(v)

// ── mat3 SoA 読み書き ─────────────────────────────────────────────────────
// 3 本の vec4 バッファ xyz レーンに各列を格納 (w は別用途)
// MoltenVK: buffers[] を使う処理は関数ではなくマクロで展開する
#define readMat3(p, c0i, c1i, c2i) \
    mat3(readPackedXYZ((c0i), (p)), readPackedXYZ((c1i), (p)), readPackedXYZ((c2i), (p)))

// xyz のみ書き込み、w レーン（応力パック）を保持する
#define writeMat3xyz(p, M, c0i, c1i, c2i) { \
    writePackedXYZ((c0i), (p), (M)[0]); \
    writePackedXYZ((c1i), (p), (M)[1]); \
    writePackedXYZ((c2i), (p), (M)[2]); }

// 後方互換: xyz + w=0 を全書き込み
#define writeMat3(p, M, c0i, c1i, c2i) { \
    writePackedVec4((c0i), (p), vec4((M)[0], 0.0)); \
    writePackedVec4((c1i), (p), vec4((M)[1], 0.0)); \
    writePackedVec4((c2i), (p), vec4((M)[2], 0.0)); }

// ── 対称 Kirchhoff 応力 w レーン パック ──────────────────────────────────
// F0.w=σ_xx, F1.w=σ_yy, F2.w=σ_zz
// B0.w=σ_xy, B1.w=σ_xz, B2.w=σ_yz
//
// readStressW: 対称 mat3 を w レーンから再構成
// mat3(a,b,c,d,e,f,g,h,i) は列優先: col0=(a,b,c), col1=(d,e,f), col2=(g,h,i)
#define readStressW(p) mat3( \
    readPackedW(pc.F0Idx, (p)), \
    readPackedW(pc.B0Idx, (p)), \
    readPackedW(pc.B1Idx, (p)), \
    readPackedW(pc.B0Idx, (p)), \
    readPackedW(pc.F1Idx, (p)), \
    readPackedW(pc.B2Idx, (p)), \
    readPackedW(pc.B1Idx, (p)), \
    readPackedW(pc.B2Idx, (p)), \
    readPackedW(pc.F2Idx, (p)) )

// writeStressW: 対称 mat3 の 6 独立成分を w レーンに書き込み
#define writeStressW(p, tau) { \
    writePackedW(pc.F0Idx, (p), (tau)[0][0]); \
    writePackedW(pc.F1Idx, (p), (tau)[1][1]); \
    writePackedW(pc.F2Idx, (p), (tau)[2][2]); \
    writePackedW(pc.B0Idx, (p), (tau)[0][1]); \
    writePackedW(pc.B1Idx, (p), (tau)[0][2]); \
    writePackedW(pc.B2Idx, (p), (tau)[1][2]); }

// ── マテリアルパラメータ (GLSL 側, MaterialParams.h と std430 互換) ─────────
// model 定数
#define MAT_ELASTIC          0u
#define MAT_VON_MISES        1u
#define MAT_DRUCKER_PRAGER   2u
#define MAT_GRANULAR_POWDER  3u
#define MAT_FLUID            4u
#define MAT_VISCOPLASTIC_MUD 5u

struct MaterialParams {
    float    mu;           // 0
    float    lambda;       // 4
    float    rho0;         // 8
    uint     model;        // 12
    float    M_friction;   // 16
    float    q_cohesion;   // 20
    float    q_max;        // 24
    float    bulkK;        // 28
    float    fluidGamma;   // 32
    float    viscosity;    // 36
    float    hardening;    // 40
    float    xi;           // 44
    float    pad0;         // 48
    float    pad1;         // 52
    float    pad2;         // 56
    float    pad3;         // 60
};

// MoltenVK: buffers[] を直接関数で返せないため、materials バッファは
// main() 内でインライン展開して読む（関数内 buffers[] アクセス不可）
// 粒子インデックス p の material id: floatBitsToUint(readVec4(pc.velIdx, p).w)
// materials SSBO オフセット (1エントリ = 16 uint = 64 bytes):
//   base = matId * 16u
//   [0]=mu [1]=lambda [2]=rho0 [3]=model [4]=M_friction [5]=q_cohesion
//   [6]=q_max [7]=bulkK [8]=fluidGamma [9]=viscosity [10]=hardening [11]=xi

// ── Morton 符号 (Z-order curve) ───────────────────────────────────────────
uint mortonExpand(uint v) {
    v = (v | (v << 16u)) & 0x030000FFu;
    v = (v | (v <<  8u)) & 0x0300F00Fu;
    v = (v | (v <<  4u)) & 0x030C30C3u;
    v = (v | (v <<  2u)) & 0x09249249u;
    return v;
}

uint mortonCompact(uint v) {
    v &= 0x09249249u;
    v = (v | (v >>  2u)) & 0x030C30C3u;
    v = (v | (v >>  4u)) & 0x0300F00Fu;
    v = (v | (v >>  8u)) & 0x030000FFu;
    v = (v | (v >> 16u)) & 0x000003FFu;
    return v;
}

uint mortonEncodeI(ivec3 c) {
    return mortonExpand(uint(c.x)) | (mortonExpand(uint(c.y)) << 1u)
                                   | (mortonExpand(uint(c.z)) << 2u);
}

ivec3 mortonDecodeI(uint code) {
    return ivec3(mortonCompact(code),
                 mortonCompact(code >> 1u),
                 mortonCompact(code >> 2u));
}

// パーティクル位置 → Morton cell ID
uint cellIdFromPos(vec3 p) {
    vec3 local = clamp((p - pc.worldMin) / pc.cellSize,
                       vec3(0.0), vec3(pc.gridRes) - vec3(1.0));
    uvec3 g = uvec3(local);
    return mortonExpand(g.x) | (mortonExpand(g.y) << 1u) | (mortonExpand(g.z) << 2u);
}

// ── 2次 B-spline 重みと勾配 ───────────────────────────────────────────────
float bspline2(float d) {
    float ad = abs(d);
    if (ad < 0.5) return 0.75 - ad * ad;
    if (ad < 1.5) { float t = 1.5 - ad; return 0.5 * t * t; }
    return 0.0;
}

float bspline2g(float d) {
    float ad = abs(d);
    if (ad < 0.5) return -2.0 * d;
    if (ad < 1.5) return -sign(d) * (1.5 - ad);
    return 0.0;
}

#define FIXED_POINT_SCALE 65536.0
#define encodeFixed(v) uint(int((v) * FIXED_POINT_SCALE))
#define decodeFixed(u) (float(int(u)) / FIXED_POINT_SCALE)

// ── パーティクルバッファ half-float パック (12B/要素、実験用) ────────────────
// xyz(位置/速度/F列/B列)をpackHalf2x16で圧縮、wは既存のfloatBitsToUint往復を維持
// (Vp/material id/応力とも精度劣化なし)。xyz用とw用を分離しているのは、F0-2/B0-2で
// xyz(readMat3/writeMat3xyz)とw(readStressW/writeStressW)を別々に読み書きするため
#define writePackedXYZ(bufIdx, i, xyz) { \
    uint _pxb = (i) * 3u; \
    buffers[(bufIdx)].data[_pxb]      = packHalf2x16((xyz).xy); \
    buffers[(bufIdx)].data[_pxb + 1u] = packHalf2x16(vec2((xyz).z, 0.0)); }
#define writePackedW(bufIdx, i, w) buffers[(bufIdx)].data[(i) * 3u + 2u] = floatBitsToUint(w)
#define readPackedXYZ(bufIdx, i) vec3( \
    unpackHalf2x16(buffers[(bufIdx)].data[(i) * 3u]).x, \
    unpackHalf2x16(buffers[(bufIdx)].data[(i) * 3u]).y, \
    unpackHalf2x16(buffers[(bufIdx)].data[(i) * 3u + 1u]).x)
#define readPackedW(bufIdx, i) uintBitsToFloat(buffers[(bufIdx)].data[(i) * 3u + 2u])

#define writePackedVec4(bufIdx, i, v) { writePackedXYZ(bufIdx, i, (v).xyz); writePackedW(bufIdx, i, (v).w); }
#define readPackedVec4(bufIdx, i) vec4(readPackedXYZ(bufIdx, i), readPackedW(bufIdx, i))

// gridVel/gridVelOld専用: wを持たない分ストライド2(4B×2=8B/セル)でpackHalf2x16
#define writeGridVelPacked(bufIdx, i, v) { \
    uint _gvb = (i) * 2u; \
    buffers[(bufIdx)].data[_gvb]      = packHalf2x16((v).xy); \
    buffers[(bufIdx)].data[_gvb + 1u] = packHalf2x16(vec2((v).z, 0.0)); }
#define readGridVelPacked(bufIdx, i) vec3( \
    unpackHalf2x16(buffers[(bufIdx)].data[(i) * 2u]).x, \
    unpackHalf2x16(buffers[(bufIdx)].data[(i) * 2u]).y, \
    unpackHalf2x16(buffers[(bufIdx)].data[(i) * 2u + 1u]).x)

// ── 3×3 対称 Jacobi 固有値分解 ───────────────────────────────────────────
void jacobiEigen3(mat3 A, out vec3 D, out mat3 V) {
    V = mat3(1.0);
    for (int iter = 0; iter < 20; iter++) {
        float a01 = A[0][1], a02 = A[0][2], a12 = A[1][2];
        if (max(abs(a01), max(abs(a02), abs(a12))) < 1e-8) break;

        int p0, p1; float apq;
        if (abs(a01) >= max(abs(a02), abs(a12))) { p0=0; p1=1; apq=a01; }
        else if (abs(a02) >= abs(a12))            { p0=0; p1=2; apq=a02; }
        else                                       { p0=1; p1=2; apq=a12; }

        float app = A[p0][p0], aqq = A[p1][p1];
        float tau = (aqq - app) / (2.0 * apq);
        float t   = sign(tau) / (abs(tau) + sqrt(1.0 + tau*tau));
        float c   = inversesqrt(1.0 + t*t);
        float s   = t * c;

        A[p0][p0] = app - t*apq;
        A[p1][p1] = aqq + t*apq;
        A[p0][p1] = 0.0; A[p1][p0] = 0.0;

        for (int r = 0; r < 3; r++) {
            if (r == p0 || r == p1) continue;
            float arp = A[r][p0], arq = A[r][p1];
            A[r][p0] = c*arp - s*arq; A[p0][r] = A[r][p0];
            A[r][p1] = s*arp + c*arq; A[p1][r] = A[r][p1];
        }
        for (int r = 0; r < 3; r++) {
            float vrp = V[r][p0], vrq = V[r][p1];
            V[r][p0] = c*vrp - s*vrq;
            V[r][p1] = s*vrp + c*vrq;
        }
    }
    D = vec3(A[0][0], A[1][1], A[2][2]);
}

// ── 3×3 SVD (F^T*F の固有分解経由) ──────────────────────────────────────
void svd3(mat3 F, out mat3 U, out vec3 sigma, out mat3 V) {
    mat3 FtF = transpose(F) * F;
    vec3 sig2;
    jacobiEigen3(FtF, sig2, V);
    sigma = sqrt(max(sig2, vec3(0.0)));

    for (int i = 0; i < 2; i++) for (int j = i+1; j < 3; j++) {
        if (sigma[j] > sigma[i]) {
            float tmp = sigma[i]; sigma[i] = sigma[j]; sigma[j] = tmp;
            vec3 vtmp = vec3(V[0][i], V[1][i], V[2][i]);
            V[0][i]=V[0][j]; V[1][i]=V[1][j]; V[2][i]=V[2][j];
            V[0][j]=vtmp[0]; V[1][j]=vtmp[1]; V[2][j]=vtmp[2];
        }
    }

    U = mat3(0.0);
    for (int i = 0; i < 3; i++) {
        vec3 vi = vec3(V[0][i], V[1][i], V[2][i]);
        if (sigma[i] > 1e-8) {
            vec3 Fv = F * vi;
            U[0][i] = Fv[0] / sigma[i];
            U[1][i] = Fv[1] / sigma[i];
            U[2][i] = Fv[2] / sigma[i];
        } else {
            U[i][i] = 1.0;
        }
    }
    if (determinant(U) < 0.0) {
        U[0][2] = -U[0][2]; U[1][2] = -U[1][2]; U[2][2] = -U[2][2];
        sigma[2] = -sigma[2];
    }
}

// ── Hencky 弾性 Kirchhoff 応力 ────────────────────────────────────────────
mat3 henckyStress(mat3 F, float mu, float lam) {
    mat3 U; vec3 sigma; mat3 V;
    svd3(F, U, sigma, V);
    sigma = max(abs(sigma), vec3(1e-6));
    vec3 eps    = log(sigma);
    float trEps = eps.x + eps.y + eps.z;
    vec3 kp     = lam * trEps * vec3(1.0) + 2.0 * mu * eps;
    mat3 diag   = mat3(kp.x, 0, 0,  0, kp.y, 0,  0, 0, kp.z);
    return U * diag * transpose(U);
}

// ── 弱圧縮流体 Kirchhoff 応力 ─────────────────────────────────────────────
// J = det(F), K = 体積弾性率
// τ = K * (J - 1) * I (線形 EOS)
// 完全な Tait EOS: τ = (K/γ) * (J^(-γ) - 1) * I は γ=7 近似
mat3 fluidStressJ(float J, float bulkK) {
    float pressure = bulkK * (J - 1.0);
    return pressure * mat3(1.0);
}

#endif // MPM_COMMON_GLSL
