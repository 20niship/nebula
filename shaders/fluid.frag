#version 450

layout(location = 0) in  float inSpeed;
layout(location = 0) out vec4  outColor;

// 速度からヒートマップ色を返す（青→シアン→緑→黄→赤）
vec3 speedToColor(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.0, 0.0, 1.0); // 青
    vec3 c1 = vec3(0.0, 1.0, 1.0); // シアン
    vec3 c2 = vec3(0.0, 1.0, 0.0); // 緑
    vec3 c3 = vec3(1.0, 1.0, 0.0); // 黄
    vec3 c4 = vec3(1.0, 0.0, 0.0); // 赤
    if (t < 0.25) return mix(c0, c1, t * 4.0);
    if (t < 0.50) return mix(c1, c2, (t - 0.25) * 4.0);
    if (t < 0.75) return mix(c2, c3, (t - 0.50) * 4.0);
    return mix(c3, c4, (t - 0.75) * 4.0);
}

void main() {
    // 円形ポイントスプライト
    vec2  uv = gl_PointCoord * 2.0 - 1.0;
    float d  = dot(uv, uv);
    if (d > 1.0) discard;

    // 速度 0〜5 m/s を [0,1] に正規化してヒートマップ適用
    const float maxSpeed = 5.0;
    float t = inSpeed / maxSpeed;

    vec3 heatColor = speedToColor(t);
    // 中心を明るく、縁を暗くするグラデーション
    float rim = 1.0 - d;
    outColor = vec4(heatColor * (0.6 + 0.4 * rim * rim), 1.0);
}
