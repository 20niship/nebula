# Nebula — GPU 物理シミュレータ

**Vulkan コンピュートシェーダー** と **XPBD（拡張位置ベース動力学）** で実装した、リアルタイム GPU 物理シミュレータです。布・流体（PBF）・煙・布↔流体の双方向連成をすべて GPU 上で動作させます。

<div><video controls src="assets/sample.mp4" muted="false"></video></div>


---

## 機能一覧

| シーン | 内容 |
|---|---|
| `cloth_3d` | ストレッチ・ベンド拘束と自己衝突を持つ 3D 布 |
| `string_2d` | 自己衝突検証用の 2D 糸シミュレーション |
| `fluid_pbf` | 渦度保存・粘性付き位置ベース流体（PBF） |
| `fluid_sphere` | 球体境界内の PBF 流体 |
| `smoke` | 浮力つき煙パーティクルシミュレーション |
| `screw_fluid` | PBF 流体中の回転スクリュー |
| `cloth_scene` | 複数布シーン（2 枚布 / 4 隅ねじれ） |
| `multi_physics` | 布↔流体の双方向連成 |

全シーンで **ImGui パネルによるリアルタイムパラメータ調整**（重力・コンプライアンス・風・ソルバー反復数など）が可能です。

---

## アーキテクチャ

```
src/
  core/          Vulkan コンテキスト、コンピュートパイプライン、属性バッファ
  engine/        XPBD 布エンジン、PBF 流体エンジン、マルチフィジクスエンジン
  graphics/      レンダリングパイプライン、布レンダラー
shaders/         GLSL コンピュート・グラフィクスシェーダー（SPIR-V にコンパイル）
examples/        シーンごとのエントリーポイント
tests/           doctest ユニットテスト
```

**主要技術:**
- Vulkan 1.2（macOS では MoltenVK 経由）
- `glslc` でコンパイルする GLSL コンピュートシェーダー
- GPU メモリ管理: VulkanMemoryAllocator（VMA）
- UI / ウィンドウ: ImGui + GLFW
- 数学ライブラリ: GLM

---

## 動作要件

| ツール | バージョン |
|---|---|
| CMake | ≥ 3.22 |
| C++ コンパイラ | C++20（clang / GCC） |
| Vulkan SDK | 1.3.280.1（LunarG または Homebrew） |
| Task | 最新版 |
| GLFW, GLM | Homebrew または apt 経由 |

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

# 各シーンを実行
task run:cloth     # cloth_3d（布）
task run:fluid     # fluid_pbf（流体）
task run:multi     # multi_physics（布 + 流体連成）
```

### CMake で直接ビルドする場合

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/multi_physics
```

---

## Task コマンド一覧

```bash
task build           # 全ターゲットをビルド
task build:cloth     # cloth_3d のみビルド
task build:fluid     # fluid_pbf のみビルド
task build:multi     # multi_physics のみビルド
task test            # ユニットテストを実行
task shaders         # シェーダーのみ再コンパイル
task capture         # 全テストケースの動画を生成 → sim_captures/
task clean           # ビルドディレクトリを削除
```

---

## シミュレーションパラメータ（cloth_3d の例）

| オプション | デフォルト | 説明 |
|---|---|---|
| `--cloth-n` | 128 | グリッド解像度（N×N 粒子） |
| `--world-size` | 10.0 | シミュレーション空間サイズ |
| `--grid-res` | 64 | 空間ハッシュグリッドの解像度 |
| `--dt` | 1/60 | タイムステップ（秒） |

---

## ライセンス

[LICENSE](LICENSE) を参照してください。
