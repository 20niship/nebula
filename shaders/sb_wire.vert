#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#include "common.glsl"

// LINE_LIST 描画: 各辺 1 本 = 頂点 2 個
// edgeData バッファ: [p0_idx, p1_idx, restLen_bits] × E  (pc.stretchEdgesIdx)
// gl_VertexIndex/2 = エッジ番号, %2 = 端点 (0=p0, 1=p1)

void main() {
    uint edgeIdx  = uint(gl_VertexIndex) / 2u;
    uint vertSlot = uint(gl_VertexIndex) % 2u;

    uint particleIdx = readUint(pc.stretchEdgesIdx, edgeIdx * 3u + vertSlot);
    vec3 p = readVec4(pc.posIdx, particleIdx).xyz;

    // particle.vert と同一カメラ
    vec3  mid  = (pc.worldMin + pc.worldMax) * 0.5;
    vec3  span = pc.worldMax - pc.worldMin;
    float maxSpan = max(span.x, max(span.y, span.z));

    vec3 camPos  = vec3(mid.x, mid.y - maxSpan * 2.0, mid.z + maxSpan * 1.2);
    vec3 target  = mid;
    vec3 worldUp = vec3(0.0, 0.0, 1.0);

    vec3 fwd = normalize(target - camPos);
    vec3 rgt = normalize(cross(fwd, worldUp));
    vec3 up  = cross(rgt, fwd);

    vec3  d  = p - camPos;
    float vx = dot(d, rgt);
    float vy = dot(d, up);
    float vz = dot(d, fwd);

    float fovY  = radians(45.0);
    float aspect = 1280.0 / 720.0;
    float f      = 1.0 / tan(fovY * 0.5);
    float near   = 0.1;
    float far    = maxSpan * 5.0;

    float ndc_z = ((far + near) * vz - 2.0 * far * near) / ((far - near) * vz);
    gl_Position = vec4(f * vx / (aspect * vz), -f * vy / vz, ndc_z, 1.0);
}
