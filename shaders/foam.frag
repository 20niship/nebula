#version 450

layout(location = 0) in  float inAlpha;
layout(location = 1) in  float inKind;
layout(location = 0) out vec4  outColor;

// kind: 1=spray(明るい白), 2=foam(白), 3=bubble(淡いシアン)
vec3 kindColor(float kind) {
    if (kind < 1.5) return vec3(1.0, 1.0, 1.0);
    if (kind < 2.5) return vec3(0.95, 0.97, 1.0);
    return vec3(0.7, 0.9, 1.0);
}

void main() {
    // 円形ポイントスプライト
    vec2  uv = gl_PointCoord * 2.0 - 1.0;
    float d  = dot(uv, uv);
    if (d > 1.0) discard;

    float rim = 1.0 - d;
    vec3 color = kindColor(inKind) * (0.6 + 0.4 * rim * rim);
    outColor = vec4(color, inAlpha * (0.5 + 0.5 * rim));
}
