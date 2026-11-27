#version 450

layout(location = 0) flat in uint vMat;
layout(location = 1) in vec3 vWorld;
layout(location = 2) in float vSpeed;
layout(location = 0) out vec4 outColor;

// 材料IDごとの色: 0=ミルク(乳白), 1=クッキー(焦げ茶), 2以降=循環。
// 内部(刃の撹拌)が見えるよう、手前右(カメラ側 x>中心, y<中心)の1/4を切り欠いて描画する。
void main() {
    if(vWorld.x > 0.5 && vWorld.y < 0.5) discard;
    const vec3 palette[4] = vec3[4](vec3(0.95, 0.93, 0.88), vec3(0.35, 0.20, 0.10), vec3(0.85, 0.55, 0.15), vec3(0.45, 0.30, 0.20));
    vec3 c = palette[vMat & 3u];
    if(vMat == 0u) c = mix(c, vec3(0.1, 0.4, 1.0), clamp(vSpeed / 1.5, 0.0, 1.0)); // ミルクは速度で青く染め、撹拌の流れを可視化
    outColor = vec4(c, 1.0);
}
