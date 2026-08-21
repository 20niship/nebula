#ifndef COLLIDER_COMMON_GLSL
#define COLLIDER_COMMON_GLSL

// 解析コライダー共通 SDF + 境界条件ヘルパー。ColliderPrimitive GPUレイアウト(96 bytes=24 uint32_t)はsrc/core/Collider.h参照。
// MoltenVK制約: buffers[]はmain()でのみアクセス可(関数内不可)なので、buffers[]を使う読み取りはマクロ(MESH_SDF_TRILERP等)にしてmain()側でインライン展開させる。それ以外は純粋数学関数でよい。

#define COLL_PLANE    0u
#define COLL_SPHERE   1u
#define COLL_BOX      2u
#define COLL_CAPSULE  3u
#define COLL_MESH_SDF 4u
#define COLL_CYLINDER 5u

// クォータニオンでベクトルvを回転 (buffers[]不使用の純粋数学なので関数化してよい)
vec3 quatRotate(vec4 q, vec3 v) {
    vec3 u = q.xyz;
    float s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}
vec4 quatConj(vec4 q) { return vec4(-q.xyz, q.w); }

// ローカル空間1点のトライリニアSDFサンプル。bufIdx_はランタイム値でbuffers[]を使うためmain()内展開が必須。
#define MESH_SDF_TRILERP(bufIdx_, localP_, localMin_, cellSize_, res_, out_val_) \
{ \
    vec3 gpos = (localP_ - localMin_) / cellSize_; \
    ivec3 i0 = clamp(ivec3(floor(gpos)), ivec3(0), ivec3(int(res_) - 2)); \
    vec3 f = clamp(gpos - vec3(i0), vec3(0.0), vec3(1.0)); \
    uint rz = uint(res_); \
    uint base = (uint(i0.z) * rz + uint(i0.y)) * rz + uint(i0.x); \
    uint stepY = rz; \
    uint stepZ = rz * rz; \
    float c000 = readFloat(bufIdx_, base); \
    float c100 = readFloat(bufIdx_, base + 1u); \
    float c010 = readFloat(bufIdx_, base + stepY); \
    float c110 = readFloat(bufIdx_, base + stepY + 1u); \
    float c001 = readFloat(bufIdx_, base + stepZ); \
    float c101 = readFloat(bufIdx_, base + stepZ + 1u); \
    float c011 = readFloat(bufIdx_, base + stepZ + stepY); \
    float c111 = readFloat(bufIdx_, base + stepZ + stepY + 1u); \
    float c00 = mix(c000, c100, f.x); \
    float c10 = mix(c010, c110, f.x); \
    float c01 = mix(c001, c101, f.x); \
    float c11 = mix(c011, c111, f.x); \
    float c0 = mix(c00, c10, f.y); \
    float c1 = mix(c01, c11, f.y); \
    out_val_ = mix(c0, c1, f.z); \
}

// ローカル空間でのSDF値+法線(中心差分)。out_n_local_はワールドへ戻す前のローカル法線。
#define MESH_SDF_SAMPLE(bufIdx_, localP_, localMin_, cellSize_, res_, out_sdf_, out_n_local_) \
{ \
    MESH_SDF_TRILERP(bufIdx_, localP_, localMin_, cellSize_, res_, out_sdf_) \
    float eps = (cellSize_) * 0.5; \
    float sdx0, sdx1, sdy0, sdy1, sdz0, sdz1; \
    MESH_SDF_TRILERP(bufIdx_, (localP_) + vec3(eps, 0.0, 0.0), localMin_, cellSize_, res_, sdx1) \
    MESH_SDF_TRILERP(bufIdx_, (localP_) - vec3(eps, 0.0, 0.0), localMin_, cellSize_, res_, sdx0) \
    MESH_SDF_TRILERP(bufIdx_, (localP_) + vec3(0.0, eps, 0.0), localMin_, cellSize_, res_, sdy1) \
    MESH_SDF_TRILERP(bufIdx_, (localP_) - vec3(0.0, eps, 0.0), localMin_, cellSize_, res_, sdy0) \
    MESH_SDF_TRILERP(bufIdx_, (localP_) + vec3(0.0, 0.0, eps), localMin_, cellSize_, res_, sdz1) \
    MESH_SDF_TRILERP(bufIdx_, (localP_) - vec3(0.0, 0.0, eps), localMin_, cellSize_, res_, sdz0) \
    vec3 g = vec3(sdx1 - sdx0, sdy1 - sdy0, sdz1 - sdz0); \
    float glen = length(g); \
    out_n_local_ = (glen > 1e-6) ? (g / glen) : vec3(0.0, 1.0, 0.0); \
}

// ── 解析 SDF 関数 (buffers[] 不使用) ──────────────────────────────────────────

// 平面 SDF: out_n = 外向き単位法線
float planeSDF(vec3 p, vec3 cpos, vec3 cnrm, out vec3 out_n) {
    out_n = normalize(cnrm);
    return dot(p - cpos, out_n);
}

// 球 SDF
float sphereSDF(vec3 p, vec3 cpos, float cr, out vec3 out_n) {
    vec3  d    = p - cpos;
    float dist = length(d);
    out_n = (dist > 1e-8) ? (d / dist) : vec3(0.0, 1.0, 0.0);
    return dist - cr;
}

// 有向ボックス距離関数(解析幾何、SDFグリッド不使用): quatでローカル空間へ回転変換してから軸並行box距離公式を適用し、法線をワールドへ戻す
float boxSDF(vec3 p, vec3 cpos, vec3 halfExt, vec4 q, out vec3 out_n) {
    vec3  rel = quatRotate(quatConj(q), p - cpos);
    vec3  d   = abs(rel) - halfExt;
    float sdf = length(max(d, vec3(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
    vec3 nLocal;
    if (sdf < 0.0) {
        // 内部: 最近傍面の外向き法線
        vec3 diff = halfExt - abs(rel);
        if (diff.x < diff.y && diff.x < diff.z)
            nLocal = vec3(sign(rel.x + 1e-8), 0.0, 0.0);
        else if (diff.y < diff.z)
            nLocal = vec3(0.0, sign(rel.y + 1e-8), 0.0);
        else
            nLocal = vec3(0.0, 0.0, sign(rel.z + 1e-8));
    } else {
        vec3 dPos = max(d, vec3(0.0));
        float dLen = length(dPos);
        nLocal = (dLen > 1e-8) ? normalize(dPos * sign(rel + vec3(1e-8)))
                                : normalize(sign(rel + vec3(1e-8)));
    }
    out_n = quatRotate(q, nLocal);
    return sdf;
}

// 有向・平底円柱距離関数(解析幾何、SDFグリッド不使用): ローカルY軸=軸方向。boxSDFと同じ流儀でquat変換+piecewise法線
float cylinderSDF(vec3 p, vec3 cpos, float radius, float halfHeight, vec4 q, out vec3 out_n) {
    vec3 rel = quatRotate(quatConj(q), p - cpos);
    float r  = length(rel.xz);
    vec2  d  = vec2(r - radius, abs(rel.y) - halfHeight); // (半径方向の超過, 軸方向の超過)
    float sdf = min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)));
    vec2 rdir = (r > 1e-8) ? (rel.xz / r) : vec2(1.0, 0.0);
    vec3 nLocal;
    if (d.x < 0.0 && d.y < 0.0) {
        // 内部: 側面/端面のうち近い方
        if (d.x > d.y)
            nLocal = vec3(rdir.x, 0.0, rdir.y);
        else
            nLocal = vec3(0.0, sign(rel.y + 1e-8), 0.0);
    } else {
        vec2 dPos = max(d, vec2(0.0));
        float dLen = length(dPos);
        if (dLen > 1e-8) {
            vec2 rc = rdir * dPos.x;
            nLocal = normalize(vec3(rc.x, sign(rel.y + 1e-8) * dPos.y, rc.y));
        } else {
            nLocal = vec3(0.0, 1.0, 0.0);
        }
    }
    out_n = quatRotate(q, nLocal);
    return sdf;
}

// カプセル SDF: cpos=始点, axisVec=軸方向×長さ, cr=半径
float capsuleSDF(vec3 p, vec3 cpos, vec3 axisVec, float cr, out vec3 out_n) {
    float axisLen = length(axisVec);
    vec3  axisDir = (axisLen > 1e-8) ? (axisVec / axisLen) : vec3(0.0, 1.0, 0.0);
    float t       = clamp(dot(p - cpos, axisDir), 0.0, axisLen);
    vec3  closest = cpos + t * axisDir;
    vec3  d       = p - closest;
    float dist    = length(d);
    out_n = (dist > 1e-8) ? (d / dist) : vec3(0.0, 1.0, 0.0);
    return dist - cr;
}

// ── MPM グリッド / PBF 粒子への速度境界条件 ──────────────────────────────────
// v: 入出力速度, n: 外向き法線, v_coll: コライダー速度
// 法線方向の相対速度が負のとき: 反発 + 摩擦
void applyColliderBC(inout vec3 v, vec3 n, vec3 v_coll,
                     float restitution, float friction) {
    vec3  v_rel = v - v_coll;
    float v_n   = dot(v_rel, n);
    if (v_n < 0.0) {
        vec3  v_n_vec = v_n * n;
        vec3  v_t     = v_rel - v_n_vec;
        float vtLen   = length(v_t);
        // 動摩擦: Coulomb モデル
        if (vtLen > 1e-8) v_t *= max(0.0, 1.0 - friction * abs(v_n) / vtLen);
        v = v_coll + v_t - restitution * v_n_vec;
    }
}

#endif // COLLIDER_COMMON_GLSL
