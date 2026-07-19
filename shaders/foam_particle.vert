#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#include "common.glsl"

// 泡 (spray/foam/bubble) 二次パーティクル描画 (issue #47)。
// 呼び出し側 (アプリ) は draw() 用の SimPC で posIdx=foamPosIdx / velIdx=foamVelIdx /
// typeFlagIdx=foamKindIdx / boundaryStart=0 を渡す（fluid_particle.vert と同じ
// クリップ・カメラ変換パターンを共有しつつ別バッファを読む）。

layout(location = 0) out float outAlpha;
layout(location = 1) out float outKind;

void main() {
    uint i = uint(gl_VertexIndex) + pc.boundaryStart;

    // kind==0 (死/未使用スロット) をクリップ外に追い出す（fluid_particle.vert と同じ墓場パターン）
    uint kind = readUint(pc.typeFlagIdx, i);
    if (kind == 0u) {
        gl_Position = vec4(0.0, 0.0, -10.0, 1.0);
        outAlpha    = 0.0;
        outKind     = 0.0;
        return;
    }

    vec4 p = readVec4(pc.posIdx, i); // xyz=位置, w=残り寿命
    vec4 v = readVec4(pc.velIdx, i); // xyz=速度(未使用), w=初期寿命

    float initLife = max(v.w, 1e-4);
    outAlpha = clamp(p.w / initLife, 0.0, 1.0);
    outKind  = float(kind);

    // Z-up 座標系。fluid_particle.vert と同じカメラ（斜め前方 45° 俯瞰、Y 反転）
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
    float aspect = 1280.0 / 720.0;
    float f      = 1.0 / tan(fovY * 0.5);
    float near   = 0.1;
    float far    = maxSpan * 5.0;

    float ndc_z = ((far + near) * vz - 2.0 * far * near) / ((far - near) * vz);

    gl_Position  = vec4(f * vx / (aspect * vz), -f * vy / vz, ndc_z, 1.0);
    gl_PointSize = 3.0; // 本流体より小さい点
}
