#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#include "common.glsl"

layout(location = 0) out float outSpeed;

void main() {
    // pc.boundaryStart を頂点オフセットとして使用
    uint i = uint(gl_VertexIndex) + pc.boundaryStart;

    // typeFlag==0 (死粒子・吸収済み) をクリップ外に追い出す（後方互換: typeFlagIdx==0 はスキップ）
    if (pc.typeFlagIdx != 0u && readUint(pc.typeFlagIdx, i) == 0u) {
        gl_Position  = vec4(0.0, 0.0, -10.0, 1.0);
        outSpeed     = 0.0;
        return;
    }

    vec4 p = readVec4(pc.posIdx, i);

    // 速度の大きさをフラグメントシェーダへ渡す
    vec4 vel = readVec4(pc.velIdx, i);
    outSpeed = length(vel.xyz);

    // Z-up 座標系。fluid_particle.vert (screw_fluid/fluid_pbf 等共通) をベースに、
    // fluid_wave_foam 専用でカメラをより近く・より斜めから見下ろす角度に変更した版
    // (他シーンの見た目に影響しないよう共有シェーダーは変更せず複製している)。
    vec3  mid  = (pc.worldMin + pc.worldMax) * 0.5;
    vec3  span = pc.worldMax - pc.worldMin;
    float maxSpan = max(span.x, max(span.y, span.z));

    // 元のカメラ (horizDist=maxSpan*2.0, vertDist=maxSpan*1.2, azimuth=0固定・
    // 常に -Y 方向から見下ろす) に対して、距離を半分にして約2倍近づけ、
    // Z軸(worldUp)まわりに azimuth 分だけ追加で回転させ斜め度を上げる。
    float horizDist = maxSpan * 1.0;
    float vertDist   = maxSpan * 0.6;
    float azimuth    = radians(130.0); // 追加のZ軸まわり回転角 (40°からさらに90°回転)
    vec3 camPos  = vec3(mid.x + sin(azimuth) * horizDist,
                         mid.y - cos(azimuth) * horizDist,
                         mid.z + vertDist);
    vec3 target  = mid;
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
    float far    = maxSpan * 5.0;

    float ndc_z = ((far + near) * vz - 2.0 * far * near) / ((far - near) * vz);

    gl_Position  = vec4(f * vx / (aspect * vz), -f * vy / vz, ndc_z, 1.0);
    gl_PointSize = 6.0;
}
