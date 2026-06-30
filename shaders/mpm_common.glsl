#ifndef MPM_COMMON_GLSL
#define MPM_COMMON_GLSL

// ── Bindless バッファ配列 ──────────────────────────────────────────────────
layout(set = 0, binding = 0) buffer StorageBuffers { uint data[]; } buffers[];

// ── MPMSimPC Push Constants (160 bytes, hash compat) ──────────────────────
layout(push_constant) uniform PC {
    uint  posIdx;        // 0   vec4×N  (xyz=pos, w=Vp)
    uint  velIdx;        // 4   vec4×N  (xyz=vel, w=material id)
    uint  F0Idx;         // 8   vec4×N  F 列0 (xyz) + σ_xx (w)
    uint  F1Idx;         // 12  vec4×N  F 列1 (xyz) + σ_yy (w)
    uint  typeFlagIdx;   // 16  (reserved)
    uint  cellCountIdx;  // 20  ← hash compat
    uint  cellOffsetIdx; // 24  ← hash compat
    uint  sortedIdxIdx;  // 28  ← hash compat
    uint  particleCount; // 32  ← hash compat (ライブ粒子数)
    uint  gridRes;       // 36  ← hash compat
    uint  F2Idx;         // 40  F 列2 (xyz) + σ_zz (w)
    uint  materialsIdx;  // 44  MaterialParams SSBO (0=無効)
    float dt;            // 48  ← hash compat
    float cellSize;      // 52  ← hash compat
    float worldMin;      // 56  ← hash compat
    float worldMax;      // 60  ← hash compat
    float gravity;       // 64
    float mu_lame;       // 68  グローバルデフォルト μ
    float lambda_lame;   // 72  グローバルデフォルト λ
    float particleVolume;// 76  グローバルデフォルト Vp
    float M_friction;    // 80  グローバルデフォルト DP M
    float q_cohesion;    // 84  グローバルデフォルト DP q_c
    float q_max;         // 88  グローバルデフォルト VM q_max
    float flip_ratio;    // 92  0=PIC, 1=FLIP, -1=APIC
    uint  colliderIdx;   // 96  Collider SSBO (Phase 3)
    uint  colliderCount; // 100 コライダー数 (Phase 3)
    uint  B0Idx;         // 104 B 列0 (xyz, APIC) + σ_xy (w)
    uint  B1Idx;         // 108 B 列1 (xyz, APIC) + σ_xz (w)
    uint  B2Idx;         // 112 B 列2 (xyz, APIC) + σ_yz (w)
    uint  nanoVDBIdx;    // 116 NanoVDB SDF バッファ (0=無効)
    uint  gridMomIdx;    // 120
    uint  gridMassIdx;   // 124
    float restitution;   // 128
    float wall_friction; // 132
    uint  plasticModel;  // 136 グローバルモデル (Phase 1 まで有効)
    uint  materialCount; // 140 materials エントリ数
    float rho0;          // 144 グローバルデフォルト密度
    float p0_mcc;        // 148
    float xi_hard;       // 152
    float maxParticlesFrac; // 156 予約
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
    mat3(readVec4((c0i), (p)).xyz, readVec4((c1i), (p)).xyz, readVec4((c2i), (p)).xyz)

// xyz のみ書き込み、w レーン（応力パック）を保持する
#define writeMat3xyz(p, M, c0i, c1i, c2i) { \
    buffers[(c0i)].data[(p)*4u   ] = floatBitsToUint((M)[0].x); \
    buffers[(c0i)].data[(p)*4u+1u] = floatBitsToUint((M)[0].y); \
    buffers[(c0i)].data[(p)*4u+2u] = floatBitsToUint((M)[0].z); \
    buffers[(c1i)].data[(p)*4u   ] = floatBitsToUint((M)[1].x); \
    buffers[(c1i)].data[(p)*4u+1u] = floatBitsToUint((M)[1].y); \
    buffers[(c1i)].data[(p)*4u+2u] = floatBitsToUint((M)[1].z); \
    buffers[(c2i)].data[(p)*4u   ] = floatBitsToUint((M)[2].x); \
    buffers[(c2i)].data[(p)*4u+1u] = floatBitsToUint((M)[2].y); \
    buffers[(c2i)].data[(p)*4u+2u] = floatBitsToUint((M)[2].z); }

// 後方互換: xyz + w=0 を全書き込み
#define writeMat3(p, M, c0i, c1i, c2i) { \
    writeVec4((c0i), (p), vec4((M)[0], 0.0)); \
    writeVec4((c1i), (p), vec4((M)[1], 0.0)); \
    writeVec4((c2i), (p), vec4((M)[2], 0.0)); }

// ── 対称 Kirchhoff 応力 w レーン パック ──────────────────────────────────
// F0.w=σ_xx, F1.w=σ_yy, F2.w=σ_zz
// B0.w=σ_xy, B1.w=σ_xz, B2.w=σ_yz
//
// readStressW: 対称 mat3 を w レーンから再構成
// mat3(a,b,c,d,e,f,g,h,i) は列優先: col0=(a,b,c), col1=(d,e,f), col2=(g,h,i)
#define readStressW(p) mat3( \
    uintBitsToFloat(buffers[pc.F0Idx].data[(p)*4u+3u]), \
    uintBitsToFloat(buffers[pc.B0Idx].data[(p)*4u+3u]), \
    uintBitsToFloat(buffers[pc.B1Idx].data[(p)*4u+3u]), \
    uintBitsToFloat(buffers[pc.B0Idx].data[(p)*4u+3u]), \
    uintBitsToFloat(buffers[pc.F1Idx].data[(p)*4u+3u]), \
    uintBitsToFloat(buffers[pc.B2Idx].data[(p)*4u+3u]), \
    uintBitsToFloat(buffers[pc.B1Idx].data[(p)*4u+3u]), \
    uintBitsToFloat(buffers[pc.B2Idx].data[(p)*4u+3u]), \
    uintBitsToFloat(buffers[pc.F2Idx].data[(p)*4u+3u]) )

// writeStressW: 対称 mat3 の 6 独立成分を w レーンに書き込み
#define writeStressW(p, tau) { \
    buffers[pc.F0Idx].data[(p)*4u+3u] = floatBitsToUint((tau)[0][0]); \
    buffers[pc.F1Idx].data[(p)*4u+3u] = floatBitsToUint((tau)[1][1]); \
    buffers[pc.F2Idx].data[(p)*4u+3u] = floatBitsToUint((tau)[2][2]); \
    buffers[pc.B0Idx].data[(p)*4u+3u] = floatBitsToUint((tau)[0][1]); \
    buffers[pc.B1Idx].data[(p)*4u+3u] = floatBitsToUint((tau)[0][2]); \
    buffers[pc.B2Idx].data[(p)*4u+3u] = floatBitsToUint((tau)[1][2]); }

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
                       vec3(0.0), vec3(float(pc.gridRes) - 1.0));
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
