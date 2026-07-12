# Nebula — GPU 物理シミュレータ

**Vulkan コンピュートシェーダー** で実装した、リアルタイム GPU 物理シミュレータです。XPBD 布・XPBD ソフトボディ・Position Based Fluids（PBF）・Material Point Method（MPM）・グリッドベース煙火炎（Pyro）・布↔流体の双方向連成まで、すべて GPU 上で動作します。

https://github.com/user-attachments/assets/fc7bfe51-39a6-4694-a515-f3288ba808cb

---

## シーン一覧

| 実行ファイル | エンジン | 内容 |
|---|---|---|
| `cloth_3d` | SimulationEngine | ストレッチ・ベンド拘束と自己衝突を持つ 3D 布（自由落下） |
| `cloth_scene` | ClothSceneEngine | 複数布シーン（`--scene 5`=布2枚重ね / `--scene 7`=4隅を軸周りに回転させて絞る） |
| `string_2d` | （専用実装） | 自己衝突検証用の 2D 糸（XPBD 距離拘束、3秒で自動終了するハーネス） |
| `xpbd_softbody` | SoftBodyEngine | XPBD 四面体ソフトボディ（Stanford Bunny + ジェリーキューブ9個の落下・衝突） |
| `fluid_pbf` | FluidEngine | Position Based Fluids（渦度保存・XSPH粘性・CFM緩和）。dam-break / source-flow の2シナリオ、任意OBJ境界の動的ロード対応 |
| `fluid_sphere` | FluidEngine | 球体OBJを境界として読み込み、内部にPBF流体を閉じ込める |
| `fluid_absorb` | FluidEngine | 楕円形の水たまりを、X方向に移動する円柱型「吸収ポート」が通過して吸収していくデモ |
| `screw_fluid` | FluidEngine | 手続き生成した4枚羽根プロペラを回転させ、PBF流体を撹拌するスクリューデモ |
| `smoke` | FluidEngine（流用） | 浮力＋渦度閉じ込めによる粒子ベースの煙（非圧縮拘束なし） |
| `multi_physics` | MultiPhysicsEngine | 布とPBF流体を同一ワールドに配置し、双方向連成をON/OFF可能 |
| `mpm_elastic` | MPMEngine | 弾性/塑性ブロック落下。PIC/FLIP/APIC転写モードと弾性/Von Mises/Drucker-Prager塑性モデルを切替可能 |
| `mpm_multimaterial` | MPMEngine | 弾性体（ゼリー）+ Drucker-Prager砂の混在シミュレーション |
| `mpm_geolayer` | MPMEngine | 地層崩壊（硬岩=弾性 / 弱粘土=Von Mises / 緩い土=Drucker-Prager の3層＋球コライダーで崩す） |
| `mpm_avalanche` | MPMEngine | 山岳地形（尾根・クーロワール）上のDrucker-Prager雪崩（8万粒子、速度急増検知UI付き） |
| `mpm_snow_impact` | MPMEngine | Von Mises塑性の雪ブロックに、水平移動する箱コライダーを衝突させる |
| `pyro_basic` | PyroEngine | 複数の炎・煙ソース＋移動球体障害物によるグリッドベース燃焼・煙の基本デモ（ヘッドレス） |
| `pyro_explosion` | PyroEngine | 地表付近での爆発燃焼から「キノコ雲」形状の形成を狙うデモ（ヘッドレス） |
| `pyro_cow_blast` | PyroEngine | 牛モデルSTLに超高密度・高速の煙バースト（爆風）を浴びせるデモ（ヘッドレス） |

全ウィンドウ表示シーンで **ImGuiパネルによるリアルタイムパラメータ調整**（重力・コンプライアンス・風・ソルバー反復数など）が可能です。`pyro_*` はウィンドウ/ImGuiを持たないヘッドレス実行で、`.pvox` ボクセルファイルにダンプし `tools/pyro_raymarch.py`（Python）側でレイマーチ可視化します。

---

## 実装されているアルゴリズム

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
assets/          bunny.obj, cow_obstacle.stl, sphere_obstacle.stl 等のアセット
```

**主要技術:**
- Vulkan 1.2（macOS では MoltenVK 経由）
- `glslc` でコンパイルする GLSL コンピュートシェーダー
- GPU メモリ管理: VulkanMemoryAllocator（VMA）
- UI / ウィンドウ: ImGui + GLFW
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
| Python 3 | `tools/` スクリプト・`pyro_raymarch.py`・`capture_sim.py` 用 |

**現状 macOS のみ対応**（MoltenVK / Metal 使用）。Linux 対応は可能ですが未検証です。

---

## セットアップ

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
