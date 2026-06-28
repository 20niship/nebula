#version 450

layout(location = 0) in  vec3 inNormal;
layout(location = 1) in  vec3 inPos;
layout(location = 0) out vec4 outColor;

void main() {
    // シンプルなランバート拡散照明
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
    vec3 N        = normalize(inNormal);
    float diff    = max(dot(N, lightDir), 0.0) * 0.7 + 0.3; // ambient 0.3

    // 布の色 (空色に近いブルー)
    vec3 baseColor = vec3(0.3, 0.6, 0.9);
    outColor = vec4(baseColor * diff, 1.0);
}
