# nebula 高品質デモ (Blender レンダリング)

nebula で時間をかけてシミュレーションし、結果を Blender (`bpy`) でフォトリアルにレンダリングするデモ集。
シミュレーションは headless 実行 (ウィンドウ不要)、レンダリングは Python から `bpy` を直接呼ぶ。

| デモ | 実行ファイル | ソルバー | 規模 | レンダリング |
|---|---|---|---|---|
| 1. 1000 万粒子の流体 | `demo_fluid_10m` | `MPMEngine` (FLUID, FLIP/PIC) | 10,007,808 粒子 / 256×128×128 格子 | 粒子の密度場から等値面メッシュを再構成 → 透過 + 吸収ボリュームの水 (Cycles) |
| 2. 市街地の気流 | `demo_city_wind` | `PyroEngine` (オイラー格子, 開境界) | 256×96×128 格子 / ビル 20 棟 | 速度場から追跡した流線 (風速で色分け) + ビル群 (Cycles) |
| 3. 雪崩 | `demo_avalanche_mesh` | `MPMEngine` (Drucker-Prager) | 約 1070 万粒子 / 国土地理院 DEM 由来の地形 + 樹木 37 本 + 岩 16 個 | 密度場から数百万三角形の雪面メッシュを再構成 + DEM 地形メッシュ + 黒松メッシュ (Cycles) |

## 使い方

```bash
# 1. nebula 側 (headless)。出力は out/<demo>/ (git 管理外)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DNEBULA_BUILD_DEMOS=ON && cmake --build build -j
./build/examples/demo/demo_fluid_10m      --out examples/demo/out/fluid        # 約 1 時間 (RTX 3060)
./build/examples/demo/demo_city_wind      --out examples/demo/out/wind         # 数分
# 雪崩は地形 DEM が先に必要 (下記「雪崩の入力アセット」。DEM は git 管理外)
(cd examples/demo && python dem/fetch_dem.py && python dem/make_patch.py 122 88 58 42)
./build/examples/demo/demo_avalanche_mesh --out examples/demo/out/avalanche --scale 1.3 --frames 400 --substeps 48 --ground-friction 0.1 --friction 0.3 --cohesion 100   # 約 45 分

# 2. Blender 側 (uv が bpy を取得する。Python 3.11 固定)
cd examples/demo
uv run python render/render_fluid.py     out/fluid     --frames 0 240 --out out/fluid_png
uv run python render/render_wind.py      out/wind      --frames 0 119 --out out/wind_png
uv run python render/render_avalanche.py out/avalanche --frames 0 330 --out out/avalanche_png
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
- 雪は Drucker-Prager (摩擦 M=0.3, 凝集 100 Pa) で、斜面 (約 31°) に積もった厚さ 3〜4.5 m の層。地形コライダの摩擦を小さく (0.1) して滑り出させる。
- 地形は国土地理院 DEM (白馬周辺, z14 の `dem_png`) の一部を縦横同率で縮小した模型スケール (59×43 m)。勾配は実地形のまま。
- 樹木 (幹の円柱, 半径は樹高の約 3%)・岩 (球) は地形 SDF に焼き込む。`MPMEngine` の解析コライダは最大 64 個で、それを超える障害物を置くため。
- 樹木は雪崩の通り道の下部に間隔を空けて、岩は斜面中腹に置く。雪が幹や岩にぶつかって割れて流れる様子が見える。

#### 雪崩の入力アセット (git 管理外)
- DEM: `dem/fetch_dem.py` が国土地理院タイルを取得して `assets_local/dem.npy` に保存し、`dem/make_patch.py` が `assets_local/dem_patch.f32` を作る。
- 樹木メッシュ: virtual-tokyo リポジトリの黒松 USD (`veg_kuromatsu_*`) を Blender の `wm.usd_import` で読む。場所は環境変数 `VT_ASSETS` (既定 `~/Desktop/programs/virtual-tokyo/assets/modules`)。
- どちらもリポジトリには置かない (`assets_local/` は `.gitignore`)。
