# Nebula — GPU 物理シミュレータ

[![CI](https://github.com/20niship/nebula/actions/workflows/ci.yml/badge.svg)](https://github.com/20niship/nebula/actions/workflows/ci.yml)
[![Perf](https://github.com/20niship/nebula/actions/workflows/perf.yml/badge.svg)](https://github.com/20niship/nebula/actions/workflows/perf.yml)
[![Pages](https://github.com/20niship/nebula/actions/workflows/pages.yml/badge.svg)](https://github.com/20niship/nebula/actions/workflows/pages.yml)
[![Project page](https://img.shields.io/badge/project%20page-20niship.github.io%2Fnebula-blue?logo=github)](https://20niship.github.io/nebula/)
[![License: MIT](https://img.shields.io/github/license/20niship/nebula)](LICENSE)
![C++20](https://img.shields.io/badge/C%2B%2B-20-blue)
![Vulkan 1.2](https://img.shields.io/badge/Vulkan-1.2-red)
![Platforms](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey)

**Vulkan コンピュートシェーダー** だけで実装した、リアルタイム GPU 物理シミュレータ。布・ソフトボディ・流体・砂・雪・煙を、小さな C++20 ライブラリで扱えます。

[プロジェクトページ](https://20niship.github.io/nebula/) · [Examples](#サンプル) · [English](README.md)

https://github.com/user-attachments/assets/fc7bfe51-39a6-4694-a515-f3288ba808cb

https://github.com/user-attachments/assets/312909e0-fa6e-499f-b63e-fa1554af9ced

<sub>注: 上の動画 (海岸の波・都市風・雪崩) は Blender でレンダリングしています。実行方法の詳細は [examples/demo/README.md](examples/demo/README.md) を参照してください。</sub>

---

## nebula の特長

- **クロスプラットフォーム** — ベンダー固有拡張を使わない素の Vulkan 1.2 なので、NVIDIA や Ubuntu に縛られません。NVIDIA (Linux)・Apple GPU (macOS / MoltenVK)・CI の Mesa lavapipe (CPU の Vulkan ドライバ) で動作を確認しています。
- **高速** — 物理計算・近傍探索・描画のすべてを GPU 上で完結。同一シーンで流体・砂は NVIDIA Warp (Newton) の約 2〜6 倍高速です ([数値は下記](#パフォーマンス-nebula-vs-nvidia-warp))。
- **複数のソルバー** — XPBD 布、XPBD 四面体ソフトボディ、Position Based Fluids、MPM (PIC/FLIP/APIC、弾性・Von Mises・Drucker-Prager・流体)、グリッド型 Pyro (煙・炎)、布↔流体の連成。
- **依存が少ない** — CUDA も Python 実行環境も重量級フレームワークも不要。パッケージマネージャの Vulkan・GLFW・GLM だけで、VMA・argparse・doctest・tinyobjloader は CMake / git submodule が取得します。

---

## クイックスタート

Ubuntu / macOS で、clone から実行まで:

```bash
# 1. 依存パッケージ
sudo apt install shaderc vulkan-headers vulkan-loader glfw3 libglm-dev   # Ubuntu
brew install shaderc vulkan-headers vulkan-loader molten-vk glfw glm     # macOS

# 2. clone + ビルド
git clone --recursive https://github.com/20niship/nebula && cd nebula
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 3. 好きなサンプルを実行 (すべて --help 対応)
./build/mpm_avalanche
```

Taskfile・手動ビルド・ツール類の詳細は [ビルド詳細](#ビルド詳細) を参照。

---

## パフォーマンス: nebula vs NVIDIA Warp

[`benchmarks/warp/`](benchmarks/warp) は、同じシーンを nebula（Vulkan compute）と、Warp 上に作られた NVIDIA の物理エンジン [Newton](https://github.com/newton-physics/newton) で動かし、1 フレームあたりの時間を比較する。フレーム dt・粒子数・グリッドセル・初期配置・重力は `scenes.py` で共有しているが、ソルバ自体は異なる（nebula: 陽的 MLS-MPM / XPBD、Newton: implicit MPM / XPBD）。

RTX 3060 での計測結果（Warp 1.17.0 / Newton 1.7.0、warmup 30 + 計測 100 フレーム、3 回の中央値）:

| シーン | 粒子数 | nebula [ms/frame] | Warp (Newton) [ms/frame] | Warp CUDA graph [ms/frame] |
|---|---:|---:|---:|---:|
| 流体（水柱の崩壊, MPM FLIP） | 54,872 | 11.34 | 26.71 | — |
| 粉体（砂柱の崩壊, MPM APIC） | 45,000 | 10.87 | 61.31 | — |
| Softbody（XPBD 四面体キューブ） | 4,096 | 84.35 | 17.31 | 5.92 |

粒子数・グリッド解像度を変えたときのスケーリング（小さいほど速い）:

<p align="center">
  <img src="https://github.com/user-attachments/assets/fd63cdcb-4850-4952-9232-1e2106b8a4e6" alt="nebula vs NVIDIA Warp (Newton): scaling with particle count and grid resolution" width="900">
</p>

MPM の 2 シーンでは nebula の方が速く、グリッドを細かくしても nebula の時間はほぼ横ばいだが Newton は増える。Softbody では nebula は粒子間衝突パスが O(N²) のため粒子数とともに急に遅くなり、Newton の方が速い。これは同じシーンを各エンジンの標準的なソルバで解いた時間の比較であり、同一アルゴリズムの比較ではない。注意点は [`benchmarks/warp/README.md`](benchmarks/warp/README.md) を参照。

### 再現方法

```sh
# nebula 側（headless 実行ファイル。既定 OFF、CI ではビルドしない）
cmake -S . -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release -DNEBULA_BUILD_BENCH_WARP=ON
cmake --build build-bench --target nebula_bench

# Warp 側（uv で pyproject.toml から環境を構築）
cd benchmarks/warp && uv sync
uv run python run_benchmark.py   # 比較表 -> out/results.md
uv run python run_sweep.py       # 粒子数/グリッドのスイープ -> out/sweep_log.csv, out/sweep_overview.png
```

### 実装例: エンジンを headless で駆動する

`benchmarks/warp/nebula_bench.cpp` は、ウィンドウなしで `MPMEngine` を動かして 1 フレームごとに計測している:

```cpp
HeadlessCtx ctx;
ctx.init();

MPMEngine e;
e.domainSize   = glm::vec3(10.0f);
e.cellSize     = 10.0f / 64.0f;
e.maxParticles = pos.size();            // 0 以外にすると自動シードを行わない
e.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADER_DIR);
e.numSubsteps = 20;
e.flip_ratio  = 0.95f;                  // FLIP。-1 = APIC, 0 = PIC
e.addForce(GravityForce::FromDirection({0, -1, 0}, 9.8f));
e.setMaterials({presetWater(1000.0f, 2e5f)});
e.appendParticles(pos, vel);            // pos.w = 粒子体積

VkCommandBuffer cmd = ctx.beginCmd();
e.step(cmd, 1.0f / 60.0f);
ctx.submitCmd(cmd);                     // GPU 完了まで待つ
```

Warp 側（`benchmarks/warp/warp_bench.py`）は、同じシーンを `newton.ModelBuilder` で組み、`SolverImplicitMPM` / `SolverXPBD` を回している。

---

---

## サンプル

`examples/` の 1 サンプルにつき 1 つの GIF を載せています。各サンプルは重力・解像度・反復回数などをコマンドライン引数 (`--help`) で指定できます。`pyro_*` は headless 実行で `.pvox` を書き出し、`tools/pyro_raymarch.py` でレンダリングします。

### 布 (XPBD)

<table>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/981c832f-1f84-4dd2-8450-532a23a7d1cf" alt="cloth_3d" width="360"></p><p align="center"><b><code>cloth_3d</code></b><br>伸び・曲げ拘束と自己衝突を持つ 3D 布</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/12299196-82ec-4357-b23f-ed5fa5ddb349" alt="cloth_scene" width="360"></p><p align="center"><b><code>cloth_scene</code></b><br>複数布シーン。2 枚重ね、または 4 隅を回転させて絞る (<code>--scene 5</code> / <code>--scene 7</code>)</p></td></tr>
</table>

### ソフトボディ (XPBD)

<table>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/b5477e95-d4f6-4aac-946e-9afbbf6f8741" alt="xpbd_softbody" width="360"></p><p align="center"><b><code>xpbd_softbody</code></b><br>XPBD 四面体ソフトボディ。Bunny とゼリー立方体の落下・衝突</p></td><td></td></tr>
</table>

### 流体・煙 (PBF)

<table>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/7bdf1dd9-9d68-469e-9db3-44f7a2f4613d" alt="fluid_pbf" width="360"></p><p align="center"><b><code>fluid_pbf</code></b><br>PBF のダムブレイク。移動ソースと任意 OBJ 境界</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/6714e4e6-9a08-48cd-ae86-66376b896fbb" alt="fluid_sphere" width="360"></p><p align="center"><b><code>fluid_sphere</code></b><br>球状 OBJ 境界の中に閉じ込めた PBF 流体</p></td></tr>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/b0911aab-ed9f-4a47-8a79-653f943a5c64" alt="fluid_absorb" width="360"></p><p align="center"><b><code>fluid_absorb</code></b><br>楕円の水たまりを、横切る円柱ポートが吸収</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/50753770-ae62-4557-a9f7-0d67d871a14f" alt="fluid_milk_crown" width="360"></p><p align="center"><b><code>fluid_milk_crown</code></b><br>表面張力 (Akinci 凝集)。水たまりへ落ちる液滴</p></td></tr>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/2bb095f7-2001-4895-8135-d4006f1894bb" alt="fluid_wave_foam" width="360"></p><p align="center"><b><code>fluid_wave_foam</code></b><br>波パドルが Bunny 障害物に当たり泡を生成</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/9c3b2a7d-7b3b-422b-aed3-f940774d2738" alt="screw_fluid" width="360"></p><p align="center"><b><code>screw_fluid</code></b><br>4 枚羽根プロペラが PBF 流体をかき混ぜる (キネマティック境界)</p></td></tr>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/7e3b28b0-562b-41c6-bb54-93abea75f008" alt="smoke" width="360"></p><p align="center"><b><code>smoke</code></b><br>浮力と渦度閉じ込めによるパーティクル煙</p></td><td></td></tr>
</table>

### MPM (流体・砂・雪・弾性体)

<table>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/4b78ac16-f645-4733-9ebe-a2adde83fc75" alt="mpm_elastic" width="360"></p><p align="center"><b><code>mpm_elastic</code></b><br>弾塑性ブロック。PIC/FLIP/APIC と Elastic/Von Mises/Drucker-Prager を切替</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/9e831969-e84e-448f-b36d-8f8c86600a4a" alt="mpm_multimaterial" width="360"></p><p align="center"><b><code>mpm_multimaterial</code></b><br>弾性ゼリーと Drucker-Prager 砂の混在</p></td></tr>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/35baada6-4d39-4d27-9f46-7c04404f3f56" alt="mpm_geolayer" width="360"></p><p align="center"><b><code>mpm_geolayer</code></b><br>3 層の地層 (岩・粘土・土) が球コライダで崩壊</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/21fe2dc3-7e4b-445e-9ef9-dce6337c7704" alt="mpm_avalanche" width="360"></p><p align="center"><b><code>mpm_avalanche</code></b><br>山岳地形を流れる Drucker-Prager 雪崩 (8 万粒子)</p></td></tr>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/1fa2ff6d-faf7-4cf6-8dc1-c22d3d14156d" alt="mpm_snow_impact" width="360"></p><p align="center"><b><code>mpm_snow_impact</code></b><br>移動する箱コライダに打たれる Von Mises 塑性の雪ブロック</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/8c071b55-3006-4914-83a7-8ce8e7cd01f5" alt="mpm_tire_sand" width="360"></p><p align="center"><b><code>mpm_tire_sand</code></b><br>転がるタイヤが砂を巻き上げる</p></td></tr>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/385d6789-6010-4984-8c14-a40eb6e80aa7" alt="fluid_spume" width="360"></p><p align="center"><b><code>fluid_spume</code></b><br>MPMEngine 上の FLIP 液体シーン (geyser / outfall / culvert / slab / dual。`--flip` で非圧縮 FLIPEngine)</p></td><td></td></tr>
</table>

### Pyro (グリッド煙・炎)

<table>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/db93238e-eb6c-4ba9-acc8-c6b87a1ebe09" alt="pyro_basic" width="360"></p><p align="center"><b><code>pyro_basic</code></b><br>複数ソースと移動球障害物のグリッド燃焼・煙 (headless)</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/3d188b26-538c-43a5-8e09-027f5435a696" alt="pyro_explosion" width="360"></p><p align="center"><b><code>pyro_explosion</code></b><br>地表爆発のキノコ雲 (headless)</p></td></tr>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/bb22bbb4-3361-4610-bfad-76d2e88b60a9" alt="pyro_cow_blast" width="360"></p><p align="center"><b><code>pyro_cow_blast</code></b><br>牛形 STL 障害物に当たる高密度爆風 (headless)</p></td><td></td></tr>
</table>

---

## リファレンス

<details>
<summary>実装されているアルゴリズム</summary>

### 1. XPBD 布シミュレーション（`SimulationEngine` / `ClothSceneEngine`）

Müller et al. の **Extended Position Based Dynamics (XPBD)** を実装。1フレームを `numSubsteps` 個のサブステップに分割し、各サブステップで以下を実行する。

1. **Predict**: 重力・風を積分して予測位置 `predP` を計算
2. **SDF境界衝突**: ワールド境界（床・壁）との衝突を位置投影で解決
3. **ストレッチ／ベンド拘束**: コンプライアンス `stretchCompliance` / `bendCompliance` 付きの距離拘束・曲げ拘束を、**グラフ2色彩色**（隣接エッジが同時に更新されないよう色分け）した並列 Gauss-Seidel で `solverIterations` 回反復
4. **自己衝突**: 空間ハッシュグリッド（下記）で近傍粒子を検索し、密度ベースの反発で布の貫通を防止
5. **速度更新**: `(predP - P) / dt` から速度を再計算し、線形減衰を適用

`ClothSceneEngine` はこのソルバーに複数枚の布と、時間変化するピン拘束（`PinAnimated`）を追加し、4隅を回転させて布を「絞る」演出などを実現する。

### 2. XPBD 四面体ソフトボディ（`SoftBodyEngine`）

四面体メッシュ（`tools/gen_softbody.py` が生成する独自 `.sb` 形式）に対して、エッジ拘束（伸び）と四面体拘束（体積保存）の2種類をXPBDで解く。エッジ・四面体それぞれをグラフ彩色して並列反復し、複数インスタンス間の粒子衝突（`particleCollisionRadius`）も解決する。

### 3. Position Based Fluids（`FluidEngine`）

Macklin & Müller (2013) の **PBF** をほぼ論文準拠で実装：

1. Predict + SDF境界衝突
2. **空間ハッシュ構築**（count → local prefix scan → global scan → base加算 → sort の5パス、カウンティングソート方式）で近傍粒子を求める
3. **密度拘束**: Poly6カーネルで密度を評価し、**CFM緩和**（式11, `cfmEpsilon`）付きのラグランジュ乗数 λ を解いて位置補正 `Δp` を計算。**人工圧力（s_corr, 式13）** でクラスタリング（Tensile Instability）を抑制。`pbfIterations` 回反復
4. SDF再適用、速度更新
5. **渦度閉じ込め**（式15-16、`vorticityEnabled`/`vorticityEpsilon`）: 数値散逸で失われる渦を強調
6. **XSPH粘性**（`viscosityC`）: 近傍粒子の速度を平均化して粘性流体らしい滑らかさを出す

境界はOBJメッシュ（`tinyobjloader`経由）やプロシージャル生成（円柱・矩形・4枚羽根プロペラ等）から粒子化して読み込め、**キネマティック境界**（回転スクリュー、移動円柱）は毎フレームGPUへ位置・速度をアップロードして反映する。`smoke` シーンは同じソルバーを転用し、`pbfIterations=0`（非圧縮拘束なし）＋浮力＋渦度閉じ込めのみで軽量な煙表現を行う。`fluid_absorb` は円柱・カプセル等の形状に確率的な粒子吸収（`AbsorberDesc`）を適用する専用パスを追加している。

### 4. Material Point Method（`MPMEngine`）

ハイブリッド Lagrangian-Eulerian の **MPM**。密なグリッドではなく、流体と同じ**空間ハッシュ**でMorton順にソートした疎グリッドを毎フレーム再構築する点が特徴：

1. グリッド・ハッシュのゼロクリア → 空間ハッシュ構築（5パス、流体と共通ロジック）
2. **P2G (Particle-to-Grid)**: 各粒子の質量・運動量をB-スプライン重みでグリッドノードへ散布。APIC（Affine Particle-in-Cell）の運動量勾配 `B` 行列を使うことで角運動量を保存
3. **グリッド更新**: 質量で正規化し、重力・壁境界条件を適用
4. **境界条件**: NanoVDB風の任意形状SDF（地形・移動障害物）＋解析コライダー（平面/球/ボックス/カプセル、移動速度付き）
5. **G2P (Grid-to-Particle)**: グリッド速度を粒子へ再収集（**PIC/FLIP/APIC** を `flip_ratio` で連続的に切替: 0=PIC、-1=APIC、0〜1=FLIPブレンド）、変形勾配 `F` を更新し、マテリアルモデルに応じた応力を計算して粒子位置を更新

**マテリアルモデル**（`MaterialParams` テーブル、粒子ごとに材質IDで切替可能）:
- `ELASTIC`: Hencky弾性/Fixed-Corotated（ゼリー等）
- `VON_MISES`: 降伏応力 `q_max` を持つ金属的塑性
- `DRUCKER_PRAGER`: 摩擦角 `M_friction` ・粘着力 `q_cohesion` を持つ砂・土のモデル（雪崩・地層崩壊で使用）
- `GRANULAR_POWDER` / `FLUID` / `VISCOPLASTIC_MUD`: 粉体・弱圧縮流体・粘塑性泥のプリセットも用意

`mpm_geolayer` は3層、`mpm_multimaterial` は上下2層など、同一シーン内で複数マテリアルを粒子ごとに混在させられる。

### 5. グリッドベース煙・炎（Pyro）ソルバー（`PyroEngine`）

MPM/PBFのような粒子ベースではなく、Houdini Pyroに近い **Eulerianグリッド法**。Morton順の密グリッド上で density / temperature / fuel / flame / velocity をダブルバッファ（A/B）で直接解く：

1. **Source注入**（密度・温度・燃料・流入速度）
2. **燃焼反応**: 発火温度 `ignitionTemp` を超えた燃料を `burnRate` で消費し、`heatRelease`（温度上昇）・`smokeYieldPerFuel`（煙生成）・`flameBrightness`（発光）を生成
3. **浮力**: 温度による上昇（`buoyancyAlpha`）と密度による重さ（`buoyancyBeta`）
4. **渦度閉じ込め**: curl（渦度）を計算し、その勾配方向に閉じ込め力を加えて乱流のディテールを維持
5. **障害物SDF境界条件**: 任意メッシュ（STL）から構築したSDFで速度をゼロ化
6. **圧力投影**（非圧縮化）: 発散を計算し、**Jacobi反復**（`numJacobiIters`回）でポアソン方程式を解いて速度場から発散を除去
7. **移流**: semi-Lagrangian法でA→Bバッファへ全チャンネルを移流

可視化用のパーティクル/メッシュを持たないヘッドレス実行で、`.pvox` 独自バイナリ形式へ density/temperature/fuel/flame/velocity/sdf の全チャンネルをダンプし、Python側 (`tools/pyro_raymarch.py`) でレイマーチングレンダリングする。障害物STLは `MeshSDF.h` の `buildMeshSDF()` で任意タイミングに再構築でき、移動・回転する障害物にも対応（`pyro_basic`）。

### 6. 布↔流体 双方向連成（`MultiPhysicsEngine`）

XPBD布ソルバーとPBF流体ソルバーを1つのバッファ・1つのGPUパイプラインに統合し、両者の間に**連成力パス**（`kCouplingCloth_`）を追加。`enableCoupling` で流体が布を押す/布が流体を押す相互作用のON/OFFを切替できる。

### 共通基盤: 空間ハッシュグリッド

Cloth・Fluid・MPM・String2Dのすべてで共有される近傍探索アルゴリズム。ワールドを`grid_res`³のセルに分割し、以下の5パスの**カウンティングソート**でセルごとの粒子リストを構築する: `hash_count`（セルごとの粒子数を数える）→ `hash_scan_local`（ワークグループ内prefix sum）→ `hash_scan_global`（ワークグループ間の1回のprefix sum）→ `hash_add_base`（グローバルオフセット加算）→ `hash_sort`（粒子をセル順に並べ替え）。GPU上でO(N)のBlelloch風スキャンにより近傍探索を実現している。

---

</details>

---

## アーキテクチャ

```
src/
  core/          Vulkanコンテキスト、コンピュートパイプライン、属性バッファ、
                 マテリアルパラメータ、コライダー、MeshSDF、Sourceエミッタ
  engine/        SimulationEngine（XPBD布）、ClothSceneEngine、SoftBodyEngine、
                 FluidEngine（PBF）、MPMEngine、PyroEngine、MultiPhysicsEngine
  graphics/      レンダリングパイプライン、布レンダラー
shaders/         GLSL コンピュート・グラフィクスシェーダー（SPIR-V にコンパイル）
examples/        シーンごとのエントリーポイント（18シーン）
tests/           doctest ユニットテスト、ヘッドレス実行ヘルパー (HeadlessCtx)
tools/           .sb ソフトボディ生成、STL地形/牛/球体生成、Pyroレイマーチ可視化 (Python)
(アセットは tools/gen_assets.py がビルドディレクトリへ生成)
```

**主要技術:**
- Vulkan 1.2（macOS では MoltenVK 経由）
- `glslc` でコンパイルする GLSL コンピュートシェーダー
- GPU メモリ管理: VulkanMemoryAllocator（VMA）
- ウィンドウ: GLFW
- 数学ライブラリ: GLM
- 境界メッシュ読み込み: tinyobjloader

---

## 動作要件

| ツール | バージョン |
|---|---|
| CMake | ≥ 3.22 |
| C++ コンパイラ | C++20（clang / GCC） |
| Vulkan SDK | 1.3.280.1（LunarG または Homebrew） |
| Task | 最新版 |
| GLFW, GLM | Homebrew または apt 経由 |
| Python 3 | `tools/` スクリプト・`pyro_raymarch.py`・`tools/capture_sim.py` 用 |

**現状 macOS のみ対応**（MoltenVK / Metal 使用）。Linux 対応は可能ですが未検証です。

macOS (MoltenVK) と Linux (Ubuntu 24.04。CI は Mesa lavapipe、ベンチは NVIDIA GPU) で開発。Windows は未検証。

---

## ビルド詳細

### 1. 依存パッケージのインストール

```bash
# macOS
brew install shaderc vulkan-headers vulkan-loader molten-vk glfw glm go-task

# Ubuntu
sudo apt update
sudo apt install shaderc vulkan-headers vulkan-loader glfw3 libglm-dev golang
```

### 2. リポジトリのクローンとサブモジュールの初期化

```bash
git clone <repo-url>
cd nebula
git submodule update --init --recursive
```

### 3. ビルドと実行

```bash
# 全ターゲットをビルド
task build

# 各シーンを実行（Taskfile に登録済みのもの）
task run:cloth     # cloth_3d（布）
task run:fluid     # fluid_pbf（流体）
task run:multi     # multi_physics（布 + 流体連成）

# それ以外のシーン（MPM系・Pyro系・softbody 等）はビルド後に直接実行
./build/mpm_avalanche
./build/pyro_explosion --help   # 各シーンは argparse による --help 対応
```

### Manual CMake build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/multi_physics
```

---

## Task コマンド一覧

```bash
task build              # 全ターゲットをビルド
task build:cloth        # cloth_3d のみビルド
task build:fluid        # fluid_pbf のみビルド
task build:multi        # multi_physics のみビルド
task build:smoke        # smoke のみビルド
task build:cloth-scene  # cloth_scene のみビルド
task build:absorb       # fluid_absorb のみビルド
task build:softbody     # xpbd_softbody のみビルド
task test               # ユニットテストを実行
task shaders            # シェーダーのみ再コンパイル
task capture            # テストケース別シミュレーションを実行しグリッド動画を生成 (sim_captures/simulation_results.mp4)
task clean              # ビルドディレクトリを削除
```

上記に含まれない実行ファイル（`mpm_*`, `pyro_*`, `screw_fluid`, `string_2d` 等）は `task build` で一括ビルドされ、`./build/<target>` から直接起動する。

---

## シミュレーションパラメータ（cloth_3d の例）

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `--cloth-n` | 128 | グリッド解像度（N×N 粒子） |
| `--world-size` | 10.0 | シミュレーション空間サイズ |
| `--grid-res` | 64 | 空間ハッシュグリッドの解像度 |
| `--dt` | 1/60 | タイムステップ（秒） |

その他のシーン固有パラメータ（マテリアル定数、渦度閉じ込め強度、Jacobi反復回数など）は各実行ファイルに `--help` を付けて確認できる。

---

## ライセンス

[LICENSE](LICENSE) を参照してください。
