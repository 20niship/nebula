#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#include "common.glsl"

layout(location = 0) out vec3 fragVel;

void main() {
    uint i = uint(gl_VertexIndex);

    vec4 pos = readVec4(pc.posIdx, i);
    vec4 vel = readVec4(pc.velIdx, i);

    // 正射影投影: world [worldMin, worldMax] → NDC [-1, 1]、Z-up (軸ごとの実サイズを使用)
    vec3 range = pc.worldMax - pc.worldMin;
    float nx =  ((pos.x - pc.worldMin.x) / range.x) * 2.0 - 1.0;
    float ny = -(((pos.z - pc.worldMin.z) / range.z) * 2.0 - 1.0);

    gl_Position  = vec4(nx, ny, 0.0, 1.0);
    gl_PointSize = 5.0;

    fragVel = vel.xyz;
}
