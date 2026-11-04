# nebula 高品質デモ (Blender レンダリング)

nebula で時間をかけてシミュレーションし、結果を Blender (`bpy`) でフォトリアルにレンダリングするデモ集。
シミュレーションは headless 実行 (ウィンドウ不要)、レンダリングは Python から `bpy` を直接呼ぶ。

| デモ | 実行ファイル | ソルバー | 規模 | レンダリング |
|---|---|---|---|---|
| 1. 1000 万粒子の流体 | `demo_fluid_10m` | `MPMEngine` (FLUID, FLIP/PIC) | 10,007,808 粒子 / 256×128×128 格子 | 粒子の密度場から等値面メッシュを再構成 → 透過 + 吸収ボリュームの水 (Cycles) |
| 2. 市街地の気流 | `demo_city_wind` | `PyroEngine` (オイラー格子, 開境界) | 256×96×128 格子 / ビル 20 棟 | 速度場から追跡した流線 (風速で色分け) + ビル群 (Cycles) |
| 3. 雪崩 | `demo_avalanche_mesh` | `MPMEngine` (Drucker-Prager) | 約 600 万粒子 / 高解像度地形 + 樹木 44 本 + 岩 | 密度場から数百万三角形の雪面メッシュを再構成 + 地形メッシュ + 樹木 (Cycles) |

## 使い方

```bash
# 1. nebula 側 (headless)。出力は out/<demo>/ (git 管理外)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DNEBULA_BUILD_DEMOS=ON && cmake --build build -j
./build/examples/demo/demo_fluid_10m      --out examples/demo/out/fluid        # 約 1 時間 (RTX 3060)
./build/examples/demo/demo_city_wind      --out examples/demo/out/wind         # 数分
./build/examples/demo/demo_avalanche_mesh --out examples/demo/out/avalanche

# 2. Blender 側 (uv が bpy を取得する。Python 3.11 固定)
cd examples/demo
uv run python render/render_fluid.py     out/fluid     --frames 0 240 --out out/fluid_png
uv run python render/render_wind.py      out/wind      --frames 0 119 --out out/wind_png
uv run python render/render_avalanche.py out/avalanche --frames 0 288 --out out/avalanche_png
```

`--scale` (流体・雪崩) で粒子数を縮小して素早く確認できる。`--frames A B --step N` で一部のフレームだけレンダリングできる。
連番 PNG は `ffmpeg -framerate 24 -i out/fluid_png/frame_%05d.png -c:v libx264 -pix_fmt yuv420p fluid.mp4` で動画化する。
画像・動画は git に含めず、`gh pr comment --attach` で配布する。

## データ形式

- `.nbp` (`NbpDump.h`): 粒子フレーム。位置は領域 AABB に対する `uint16` 量子化 (1000 万粒子で 1 フレーム約 70MB) + 速さ `uint8`。`render/nbp.py` が読む。
- `.pvox` (`PyroEngine::dumpFrame`): Pyro の格子データ。
- `city.txt` / `obstacles.txt` / `terrain.f32`: Blender 側でシーンを再現するためのビル配置・樹木と岩・地形高さ。

## 実装メモ

### 1. 流体 (1000 万粒子)
- 粒子間隔 = セルの半分 (8 粒子/セル)。`--scale` は間隔とセルを同時に変えるため常に 8 粒子/セル。
- 弱圧縮 EOS (`K(J-1)`) なので体積保存は音速 `sqrt(K/ρ)` に依存する。K=1e6 (音速約 30 m/s) とし、CFL を満たすよう 1 フレーム 192 サブステップ。
  K が小さい、または FLIP 比 0.95 では水柱が薄いシート状に潰れた (FLIP 0.5 で安定)。
- 1000 万粒子では 1 粒子あたりの圧力インパルスが小さいため、固定小数点スケール (既定 65536) を 1e8 に上げている。
- 水面は粒子ヒストグラムのガウス平滑 + marching cubes (`render/nbp.py`) で再構成する。

### 2. 気流
- `PyroSimPC::openBoundaryMask` を追加した (本 PR)。立っている面は領域外の圧力を 0 とみなす開境界になり、風が出入りできる。既定 0 は従来どおり全面閉。
- Gauss-Seidel の圧力ソルバは大域的な圧力場の収束が遅い (N² 反復) ため、最初の 45 ステップだけ領域全体を一様流に寄せ、街区周りの局所補正だけ解かせる。
  以降は上流のファン (エミッタ) が風速を維持する。

### 3. 雪崩
- 雪は Drucker-Prager (摩擦 M=0.45 ≈ 24°) で、斜面 (約 33°) に積もった厚さ 1.8〜3 m の層。地形コライダの摩擦を小さくして滑り出させる。
- 樹木は円柱、岩は球の解析コライダ。地形は `LocalMeshSDF` (ハイトフィールドから生成)。
