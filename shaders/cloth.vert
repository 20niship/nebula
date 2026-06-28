#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#include "common.glsl"

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outPos;

void main() {
    uint i = 0u + uint(gl_VertexIndex); // 布は常にバッファ先頭から開始
    vec4 p = readVec4(pc.posIdx, i);

    // Z-up 座標系。particle.vert と同じカメラ（斜め前方 45° 俯瞰、Y 反転）
    float mid  = (pc.worldMin + pc.worldMax) * 0.5;
    float span = pc.worldMax - pc.worldMin;

    vec3 camPos  = vec3(mid, mid - span * 2.0, mid + span * 1.2);
    vec3 target  = vec3(mid, mid, mid);
    vec3 worldUp = vec3(0.0, 0.0, 1.0);

    vec3 fwd = normalize(target - camPos);
    vec3 rgt = normalize(cross(fwd, worldUp));
    vec3 up  = cross(rgt, fwd);

    vec3  d  = p.xyz - camPos;
    float vx = dot(d, rgt);
    float vy = dot(d, up);
    float vz = dot(d, fwd);

    float fovY  = radians(45.0);
    float aspect = 1280.0 / 720.0;
    float f      = 1.0 / tan(fovY * 0.5);
    float near   = 0.1;
    float far    = span * 5.0;
    float ndc_z  = ((far + near) * vz - 2.0 * far * near) / ((far - near) * vz);

    gl_Position = vec4(f * vx / (aspect * vz), -f * vy / vz, ndc_z, 1.0);
    outPos      = p.xyz;
    outNormal   = vec3(0.0, 0.0, 1.0); // フラット法線 (簡易, Z-up)
}
