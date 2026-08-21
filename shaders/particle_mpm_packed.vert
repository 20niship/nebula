#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#include "common.glsl"

// particle.vertの複製。pos/velがhalf-floatパック(12B/要素、実験用)のMPMエンジン専用。
// 他エンジンはparticle.vert(通常vec4)を使うためそちらは変更しない。

void main() {
    uint i = uint(gl_VertexIndex);
    uint _pb = i * 3u;
    vec2 _xy = unpackHalf2x16(buffers[pc.posIdx].data[_pb]);
    vec2 _z0 = unpackHalf2x16(buffers[pc.posIdx].data[_pb + 1u]);
    vec4 p = vec4(_xy.x, _xy.y, _z0.x, uintBitsToFloat(buffers[pc.posIdx].data[_pb + 2u]));

    // Z-up 座標系 (重力 = -Z)。斜め前方 45° から俯瞰するカメラ。
    // Vulkan NDC は Y+ が画面下のため Y を反転し、Z 減少 = 画面下になるよう補正。
    vec3  mid  = (pc.worldMin + pc.worldMax) * 0.5;
    vec3  span = pc.worldMax - pc.worldMin;
    float maxSpan = max(span.x, max(span.y, span.z));

    vec3 camPos  = vec3(mid.x, mid.y - maxSpan * 2.0, mid.z + maxSpan * 1.2);
    vec3 target  = mid;
    vec3 worldUp = vec3(0.0, 0.0, 1.0);

    vec3 fwd = normalize(target - camPos);
    vec3 rgt = normalize(cross(fwd, worldUp));
    vec3 up  = cross(rgt, fwd);

    vec3  d  = p.xyz - camPos;
    float vx = dot(d, rgt);
    float vy = dot(d, up);
    float vz = dot(d, fwd);

    float fovY   = radians(45.0);
    float aspect  = 1280.0 / 720.0;
    float f       = 1.0 / tan(fovY * 0.5);
    float near    = 0.1;
    float far     = maxSpan * 5.0;

    float ndc_z = ((far + near) * vz - 2.0 * far * near) / ((far - near) * vz);

    gl_Position  = vec4(f * vx / (aspect * vz), -f * vy / vz, ndc_z, 1.0);
    gl_PointSize = 2.0;
}
