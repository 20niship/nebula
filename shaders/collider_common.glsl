#ifndef COLLIDER_COMMON_GLSL
#define COLLIDER_COMMON_GLSL

// 解析コライダー共通 SDF + 境界条件ヘルパー
//
// ColliderPrimitive GPU レイアウト (64 bytes = 16 uint32_t):
//   [0]=type [1-3]=pos [4-6]=nrm/halfExt/axis [7]=radius
//   [8-10]=vel [11]=restitution [12]=friction [13-15]=pad
//
// MoltenVK 制約: buffers[] は main() でのみアクセス可（関数内不可）
//   → buffers[] を使う読み取りは各シェーダーの main() でインライン展開する
//   → このファイルは buffers[] を参照しない純粋数学関数のみ定義する

#define COLL_PLANE   0u
#define COLL_SPHERE  1u
#define COLL_BOX     2u
#define COLL_CAPSULE 3u

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

// AABB ボックス SDF (halfExt を cnrm として渡す)
float boxSDF(vec3 p, vec3 cpos, vec3 halfExt, out vec3 out_n) {
    vec3  rel = p - cpos;
    vec3  q   = abs(rel) - halfExt;
    float sdf = length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
    if (sdf < 0.0) {
        // 内部: 最近傍面の外向き法線
        vec3 diff = halfExt - abs(rel);
        if (diff.x < diff.y && diff.x < diff.z)
            out_n = vec3(sign(rel.x + 1e-8), 0.0, 0.0);
        else if (diff.y < diff.z)
            out_n = vec3(0.0, sign(rel.y + 1e-8), 0.0);
        else
            out_n = vec3(0.0, 0.0, sign(rel.z + 1e-8));
    } else {
        vec3 qPos = max(q, vec3(0.0));
        float qLen = length(qPos);
        out_n = (qLen > 1e-8) ? normalize(qPos * sign(rel + vec3(1e-8)))
                               : normalize(sign(rel + vec3(1e-8)));
    }
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
