#version 450

layout(location = 0) in  vec3 fragVel;
layout(location = 0) out vec4 outColor;

void main() {
    float speed = length(fragVel);
    float t = clamp(speed / 5.0, 0.0, 1.0);
    // 低速: 青 → 高速: 赤
    outColor = vec4(t, 0.3 * (1.0 - t), 1.0 - t, 1.0);
}
