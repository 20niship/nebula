#ifndef MPM_COMMON_GLSL
#define MPM_COMMON_GLSL

// ── Bindless バッファ配列 ──────────────────────────────────────────────────
layout(set = 0, binding = 0) buffer StorageBuffers { uint data[]; } buffers[];

// ── MPMSimPC Push Constants (200 bytes) ───────────────────────
// C++側 src/core/MPMSimPC.h と同一オフセット順であること。vec3系フィールド直前のスカラー数を
// 4の倍数に揃えてパディングなしで16byte境界に一致させている(順序変更時は要再検証)。
// ドメイン下限は常に原点固定 (worldMinは廃止、ワールド座標=ローカル座標)。
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

    float q_max;         // 80  グローバルデフォルト VM q_max
    float flip_ratio;    // 84  0=PIC, 1=FLIP, -1=APIC
    uint  colliderIdx;   // 88  Collider SSBO (Phase 3)
    uint  colliderCount; // 92  コライダー数 (Phase 3)
    uint  B0Idx;         // 96  B 列0 (xyz, APIC) + σ_xy (w)
    uint  B1Idx;         // 100 B 列1 (xyz, APIC) + σ_xz (w)
    uint  B2Idx;         // 104 B 列2 (xyz, APIC) + σ_yz (w)
    uint  reserved144;   // 108 旧NanoVDB SDF境界条件用、未使用化(MESH_SDFに統一)

    vec3  worldMax;   // 112 ドメイン上限座標 [m] (= domainSize、下限は常に原点)
    uint  gridMomIdx; // 124

    uint  gridMassIdx;    // 128
    float restitution;    // 132
    float wall_friction;  // 136
    uint  plasticModel;   // 140 グローバルモデル (Phase 1 まで有効)
    uint  materialCount;  // 144 materials エントリ数
    float rho0;           // 148 グローバルデフォルト密度
    float p0_mcc;         // 152
    float xi_hard;        // 156
    uint  forceCount;     // 160 有効なForce数 (issue #30; 旧maxParticlesFrac予約枠)

    uint  colliderForceIdx;  // 164 vec4×64 コライダーが受け取った力積(fixed-point)
    uint  colliderTorqueIdx; // 168 vec4×64 コライダーが受け取った反トルク積(fixed-point)

    uint  flipGridIdx;       // 172 float×3·CELLS: [0,C)=圧力, [C,2C)=右辺 b, [2C,3C)=ノード種別
    uint  pass;              // 176 赤黒 Gauss-Seidel の色

    uint  boxIdx;            // 180 アクティブ領域 (粒子が触れるノードの AABB)。0=無効

    uint  sortIdx;           // 184 粒子の処理順 (空間ブロック順)。0=無効 (src/core/MPMSimPC.h 参照)
    uint  sortBuckets;       // 188 ブロック数 B
    uint  sortCap;           // 192 粒子数の上限 cap

    uint  gridLinear;        // 196 0=Morton、1=線形 (src/core/MPMSimPC.h 参照)
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

// セル添字: 格子が 2 のべき乗の立方体に近ければ Morton (3 次元の局所性が良い)、そうでなければ線形 (x 最速) = x + nx*(y + ny*z)。
// Morton は軸長が 2^n に揃わないと立方体まで確保・走査するため、400x92x246 のような格子では実セルの約 15 倍を回してしまう。
// どちらを使うかは pc.gridLinear (ParticleGridEngine::init が決める、全カーネル共通の実行時フラグ)。
uint mortonEncodeI(ivec3 c) {
    if (pc.gridLinear != 0u) return uint(c.x) + pc.gridRes.x * (uint(c.y) + pc.gridRes.y * uint(c.z));
    return mortonExpand(uint(c.x)) | (mortonExpand(uint(c.y)) << 1u)
                                   | (mortonExpand(uint(c.z)) << 2u);
}

ivec3 mortonDecodeI(uint code) {
    if (pc.gridLinear != 0u) {
        uint x = code % pc.gridRes.x;
        uint t = code / pc.gridRes.x;
        return ivec3(x, t % pc.gridRes.y, t / pc.gridRes.y);
    }
    return ivec3(mortonCompact(code),
                 mortonCompact(code >> 1u),
                 mortonCompact(code >> 2u));
}

// 粒子の処理順 (空間ブロック順): 4x4x4 セルのブロックごとに粒子をまとめる。ブロックは線形 (x 最速)
uint sortBucketOf(vec3 xp) {
    ivec3 c  = clamp(ivec3(floor(xp / pc.cellSize)), ivec3(0), ivec3(pc.gridRes) - ivec3(1)) >> 2;
    uvec3 nb = (pc.gridRes + uvec3(3u)) >> 2;
    return uint(c.x) + nb.x * (uint(c.y) + nb.y * uint(c.z));
}

// パーティクル位置 → Morton cell ID
uint cellIdFromPos(vec3 p) {
    vec3 local = clamp(p / pc.cellSize,
                       vec3(0.0), vec3(pc.gridRes) - vec3(1.0));
    return mortonEncodeI(ivec3(local));
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

#ifndef FIXED_POINT_SCALE
#define FIXED_POINT_SCALE 65536.0
#endif
#define encodeFixed(v) uint(int((v) * FIXED_POINT_SCALE))
#define decodeFixed(u) (float(int(u)) / FIXED_POINT_SCALE)

// ── 3×3 対称 Jacobi 固有値分解 (循環法) ──────────────────────────────────
// 回転の組 (p,q) を (0,1) -> (0,2) -> (1,2) の順に固定した掃引を、非対角成分が trace に対して十分小さくなるまで繰り返す。
// 組が定数なので行列がレジスタに載る (旧実装は「最大の非対角要素」を探して実行時の添字で引いており、ローカルメモリに落ちて遅かった)。
// 回転自体 (tau, t, c, s と A/V の更新) は旧実装と同一。収束判定は対角成分の和 (F^T F では非負) に対する相対値。
#define JACOBI_ROT(P, Q, R) { \
    float apq = A[P][Q]; \
    if (apq != 0.0) { \
        float app = A[P][P], aqq = A[Q][Q]; \
        float tau = (aqq - app) / (2.0 * apq); \
        float t   = sign(tau) / (abs(tau) + sqrt(1.0 + tau*tau)); \
        float c   = inversesqrt(1.0 + t*t); \
        float s   = t * c; \
        A[P][P] = app - t*apq; \
        A[Q][Q] = aqq + t*apq; \
        A[P][Q] = 0.0; A[Q][P] = 0.0; \
        float arp = A[R][P], arq = A[R][Q]; \
        A[R][P] = c*arp - s*arq; A[P][R] = A[R][P]; \
        A[R][Q] = s*arp + c*arq; A[Q][R] = A[R][Q]; \
        for (int r = 0; r < 3; r++) { \
            float vrp = V[r][P], vrq = V[r][Q]; \
            V[r][P] = c*vrp - s*vrq; \
            V[r][Q] = s*vrp + c*vrq; \
        } \
    } \
}

void jacobiEigen3(mat3 A, out vec3 D, out mat3 V) {
    V = mat3(1.0);
    for (int sweep = 0; sweep < 8; sweep++) {
        float off = max(abs(A[0][1]), max(abs(A[0][2]), abs(A[1][2])));
        if (off <= 1e-7 * (A[0][0] + A[1][1] + A[2][2])) break;
        JACOBI_ROT(0, 1, 2)
        JACOBI_ROT(0, 2, 1)
        JACOBI_ROT(1, 2, 0)
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

// henckyStress()のsvd3再計算を省く版。塑性射影(projectVonMises/projectDruckerPrager)が
// 既に計算済みのU・log特異値epsをそのまま使う(塑性リターンマッピングはU/Vを変えないため
// 数学的に同一)。G2Pで1粒子あたりsvd3(=jacobiEigen3 20回イテレーション)呼び出しを半減する。
mat3 henckyStressFromEigen(mat3 U, vec3 eps, float mu, float lam) {
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
