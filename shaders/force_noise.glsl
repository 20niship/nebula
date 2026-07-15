#ifndef FORCE_NOISE_GLSL
#define FORCE_NOISE_GLSL

// Force システム (issue #30) の Turbulence/Noise が使うハッシュベースの
// 疑似乱数・ノイズ関数群。テクスチャ/バッファ不要のprocedural実装。

vec3 forceHash33(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float forceValueNoise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);

    float n000 = dot(forceHash33(i + vec3(0.0, 0.0, 0.0)), f - vec3(0.0, 0.0, 0.0));
    float n100 = dot(forceHash33(i + vec3(1.0, 0.0, 0.0)), f - vec3(1.0, 0.0, 0.0));
    float n010 = dot(forceHash33(i + vec3(0.0, 1.0, 0.0)), f - vec3(0.0, 1.0, 0.0));
    float n110 = dot(forceHash33(i + vec3(1.0, 1.0, 0.0)), f - vec3(1.0, 1.0, 0.0));
    float n001 = dot(forceHash33(i + vec3(0.0, 0.0, 1.0)), f - vec3(0.0, 0.0, 1.0));
    float n101 = dot(forceHash33(i + vec3(1.0, 0.0, 1.0)), f - vec3(1.0, 0.0, 1.0));
    float n011 = dot(forceHash33(i + vec3(0.0, 1.0, 1.0)), f - vec3(0.0, 1.0, 1.0));
    float n111 = dot(forceHash33(i + vec3(1.0, 1.0, 1.0)), f - vec3(1.0, 1.0, 1.0));

    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

// fBm (フラクショナルブラウン運動): octaves 回重ね合わせたスカラーノイズ (概ね[-1,1])
float forceFbmNoise3D(vec3 p, uint octaves) {
    float sum  = 0.0;
    float amp  = 0.5;
    float freq = 1.0;
    uint n = max(octaves, 1u);
    for (uint i = 0u; i < n; i++) {
        sum += amp * forceValueNoise3D(p * freq);
        freq *= 2.0;
        amp  *= 0.5;
    }
    return sum;
}

// curl-noise: 3つのベクトルポテンシャル場のfBmから回転(curl)を差分近似し、
// 非圧縮的な(divergence-freeな)乱流風ベクトル場を得る (Bridson et al.)
vec3 forceCurlNoise3D(vec3 p, uint octaves) {
    const float e = 0.1;
    vec3 offX = vec3(19.1, 0.0, 0.0);
    vec3 offY = vec3(0.0, 47.3, 0.0);
    vec3 offZ = vec3(0.0, 0.0, 71.7);

    float px1 = forceFbmNoise3D(p + offX + vec3(0.0, e, 0.0), octaves);
    float px2 = forceFbmNoise3D(p + offX - vec3(0.0, e, 0.0), octaves);
    float py3 = forceFbmNoise3D(p + offY + vec3(e, 0.0, 0.0), octaves);
    float py4 = forceFbmNoise3D(p + offY - vec3(e, 0.0, 0.0), octaves);

    float py1 = forceFbmNoise3D(p + offY + vec3(0.0, 0.0, e), octaves);
    float py2 = forceFbmNoise3D(p + offY - vec3(0.0, 0.0, e), octaves);
    float pz3 = forceFbmNoise3D(p + offZ + vec3(0.0, e, 0.0), octaves);
    float pz4 = forceFbmNoise3D(p + offZ - vec3(0.0, e, 0.0), octaves);

    float px3 = forceFbmNoise3D(p + offX + vec3(0.0, 0.0, e), octaves);
    float px4 = forceFbmNoise3D(p + offX - vec3(0.0, 0.0, e), octaves);
    float pz1 = forceFbmNoise3D(p + offZ + vec3(e, 0.0, 0.0), octaves);
    float pz2 = forceFbmNoise3D(p + offZ - vec3(e, 0.0, 0.0), octaves);

    float dPzdy = (pz3 - pz4) / (2.0 * e);
    float dPydz = (py1 - py2) / (2.0 * e);
    float dPxdz = (px3 - px4) / (2.0 * e);
    float dPzdx = (pz1 - pz2) / (2.0 * e);
    float dPydx = (py3 - py4) / (2.0 * e);
    float dPxdy = (px1 - px2) / (2.0 * e);

    return vec3(dPzdy - dPydz, dPxdz - dPzdx, dPydx - dPxdy);
}

#endif // FORCE_NOISE_GLSL
