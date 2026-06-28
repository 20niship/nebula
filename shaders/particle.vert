#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#include "common.glsl"

void main() {
    uint i = uint(gl_VertexIndex);
    vec4 p = readVec4(pc.posIdx, i);

    // Z-up 座標系 (重力 = -Z)。斜め前方 45° から俯瞰するカメラ。
    // Vulkan NDC は Y+ が画面下のため Y を反転し、Z 減少 = 画面下になるよう補正。
    float mid  = (pc.worldMin + pc.worldMax) * 0.5;
    float span = pc.worldMax - pc.worldMin;

    // 斜め前上方: Y 方向に 1.3 span 離れ、Z 方向に 0.8 span 上にある位置から
    // シミュレーション空間の中心 (mid, mid, mid) を見る
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

    float fovY   = radians(45.0);
    float aspect  = 1280.0 / 720.0;
    float f       = 1.0 / tan(fovY * 0.5);
    float near    = 0.1;
    float far     = span * 5.0;

    float ndc_z = ((far + near) * vz - 2.0 * far * near) / ((far - near) * vz);

    // Y 反転: Vulkan Y+ 下 → Z-up world で Z 大 = 画面上、Z 小 = 画面下
    gl_Position  = vec4(f * vx / (aspect * vz), -f * vy / vz, ndc_z, 1.0);
    gl_PointSize = 2.0;
}
