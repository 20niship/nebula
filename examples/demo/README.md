# nebula 高品質デモ (Blender レンダリング)

nebula で時間をかけてシミュレーションし、結果を Blender (`bpy`) でフォトリアルにレンダリングするデモ集。
シミュレーションは headless 実行 (ウィンドウ不要)、レンダリングは Python から `bpy` を直接呼ぶ。

| デモ | 実行ファイル | ソルバー | 規模 | レンダリング |
|---|---|---|---|---|
| 1. 市街地の気流 | `demo_city_wind` | `PyroEngine` (オイラー格子, 開境界) | 512×256×256 格子 (セル 0.4 m, 領域 205×102×102 m) / ビル 28 棟 | 流れに乗せたトレーサ粒子の軌跡 (風速で色分け) + ビル群 (Cycles) |
| 2. 雪崩 | `demo_avalanche_mesh` | `MPMEngine` (Drucker-Prager) | 約 1070 万粒子 / 国土地理院 DEM 由来の地形 + 樹木 37 本 + 岩 16 個 | 密度場から数百万三角形の雪面メッシュを再構成 + DEM 地形メッシュ + 黒松メッシュ (Cycles) |
| 3. 海岸の波 (issue #137) | `demo_seashore` | `FLIPEngine` (非圧縮 FLIP + 泡) | 既定 約 610 万粒子 (`--spacing 0.055` で約 1000 万) + 泡 最大 400 万 | 水面等値面 + 泡の小球インスタンス + 岩 usdc (Cycles, 横動画) |
| 4. 回転する車の風洞 | `demo_car_wind` | `PyroEngine` (オイラー格子, 回転コライダー) | 既定 1024×256×256 格子 (セル 2 cm, 領域 20.5×5.1×5.1 m) / 全長 4.4 m の手続き生成スポーツカー | 風速で色分けしたトレーサの軌跡 + 車メッシュ (Cycles) |

## 使い方

```bash
# 1. nebula 側 (headless)。出力は out/<demo>/ (git 管理外)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DNEBULA_BUILD_DEMOS=ON && cmake --build build -j
./build/examples/demo/demo_city_wind      --out examples/demo/out/wind         # 約 16 分 (出力はトレーサ軌跡 tracers.bin 約 210MB。GPU メモリ約 3.4GB)
# 雪崩は地形 DEM が先に必要 (下記「雪崩の入力アセット」。DEM は git 管理外)
(cd examples/demo && python dem/fetch_dem.py && python dem/make_patch.py 122 88 58 42)
./build/examples/demo/demo_avalanche_mesh --out examples/demo/out/avalanche --scale 1.3 --frames 400 --substeps 48 --ground-friction 0.1 --friction 0.3 --cohesion 100   # 約 45 分

# 2. Blender 側 (uv が bpy を取得する。Python 3.11 固定)
cd examples/demo
uv run python render/render_wind.py      out/wind      --frames 0 659 --step 2 --out out/wind_png
uv run python render/render_avalanche.py out/avalanche --frames 0 330 --out out/avalanche_png
```

`--scale` (雪崩) で粒子数を縮小して素早く確認できる。`--frames A B --step N` で一部のフレームだけレンダリングできる。
連番 PNG は `ffmpeg -framerate 24 -i out/wind_png/frame_%05d.png -c:v libx264 -pix_fmt yuv420p wind.mp4` で動画化する。
長時間のレンダリングは `render/supervise.py` で包むと、異常終了や停止 (出力が増えない) のときに自動で再起動し、出力済みフレームをスキップして続きから再開する (`render_*.py` は出力済みフレームを飛ばす)。

```bash
uv run python render/supervise.py --watch out/avalanche_png -- \
    python render/render_avalanche.py out/avalanche --frames 0 330 --out out/avalanche_png
```

3 つのデモ動画は `python render/combine.py out/seashore.mp4 out/city_wind.mp4 out/avalanche.mp4 -o out/demo_all.mp4` で横に並べた 1 本にできる。

画像・動画は git に含めず、`gh pr comment --attach` で配布する。

### 海岸の波 (岩場に砕ける波と飛沫)

`fluid_wave_foam` (単振動パドル + FLIP + 泡) をベースに、砂浜の斜面と岩 40 個を SDF コライダーにした headless デモ。
岩は virtual-tokyo PR #6 の高ポリ岩 (usdc)。アセット・動画・画像は git に入れない (`assets_local/`, `out/` は ignore 済み)。

```bash
cd examples/demo
# 1. アセット取得 + シーン生成 (gh のログインが必要 / uv が usd-core, scipy を取得する)
uv run seashore/fetch_rocks.py          # assets_local/seashore/rocks/*.usdc, *.npz
uv run seashore/make_scene.py --seed 3  # scene.sdf / scene.json / terrain.npz
# 2. シミュレーション (既定 約 610 万粒子, 600 フレーム (25 秒)。`--spacing 0.1` なら約 130 万粒子で数分)
../../build/examples/demo/demo_seashore --out out/seashore
#    高解像度: --spacing 0.055 (約 1000 万粒子, --max-diffuse 6000000 など)
# 3. まず点群のままプレビューで確認 (Blender 不要)
uv run seashore/preview_points.py out/seashore --frames 0 600 --step 12 --out out/seashore_preview
# 4. 良さそうなら Blender (横動画 1920x1080)
uv run python seashore/render_seashore.py out/seashore --frames 150 150 --out out/seashore_png --samples 64
```

- 座標は Y-up (x: 沖→岸)。パドルが x≈1..5 で振動し、約 5 秒後に波が岩場 (x 22..38) に届く。泡は `--foam-delay` (2 秒) までは生成しない (初期水面ノイズでの誤生成を避ける)。
- 格子の固定小数点スケール (`--fp-scale`, 既定 65536) は大きすぎると格子で桁あふれして粒子が爆発する (1e6, および 524288 で確認。131072 まで安定)。8 粒子/セルならセル質量は解像度によらず一定なので、粒子数を増やしても上げない (質量が小さい MPM の 1000 万粒子デモとは逆)。
- 泡の生成係数 `--foam-rate` も粒子間隔から自動 (`100*(spacing/0.2)^2`)。固定のままだと 80 万粒子で常に上限に張り付いた。
- 岩の SDF はシミュ格子とは独立に 0.15 m セル (`make_scene.py --sdf-cell`)。レンダでは岩を実メッシュで描く。
- usdc の色テクスチャは取得元に無いため、レンダ側で手続き的な岩マテリアルに差し替えている。
- Blender の泡は `foam_*.nbf` の点を Geometry Nodes で小球インスタンス化している (`--foam-radius`)。

### 回転する車の風洞 (スポーツカー)

一定角速度 (既定 0.5 rad/s) で自転するスポーツカーに、固定向き (+x) の風 (既定 20 m/s) を当てる。車体の流線型に沿った流れと、リアウイング 2 枚の間のスロットを抜ける流れを見る。
車は Python で手続き生成する (全長 4.4 m の車体・キャビン・車輪・フロントスプリッター・2 枚構成のリアウイング + 端板 + 支柱)。メッシュのダウンロードもライセンスも不要で、ウイングの隙間 (`--gap`) を自由に変えられる。
アセットは `assets_local/car/` (git 管理外)。

```bash
cd examples/demo
uv run car/make_car.py                    # car.sdf (シミュレーション用, 2cm) / car.npz (Blender 用メッシュ) / 形状確認の断面 PNG
../../build/examples/demo/demo_car_wind --out out/car      # 約 1024×256×256 格子, 720 ステップ (dt=1/120)
uv run python car/render_car.py out/car --frames 0 600 --step 4 --out out/car_png --samples 48
```

- 回転は `PyroEngine::setRotatingCollider` (車のローカル SDF を毎サブステップ GPU でトリリニア補間して回し、固体セルと表面付近に壁面速度 ω×r を課す)。車体の自転で流れが引きずられる効果が入る。静止コライダー (都市風デモなど) の挙動は変えていない。
- 格子は 1024×256×256 (2^26 セル, 約 6GB)。Pyro のバッファは Morton 符号の最大値 + 1 セルで、x>=y>=z の 2^n 直方体ならちょうど実セル数になる。1024×512×256 は約 12GB で 12GB の GPU に載らない。
- 流線は毎フレーム引き直さず、トレーサ粒子を格子速度に乗せて流した軌跡を描く (都市風デモと同じ。`tracers.bin` の形式も同じ)。
- 車の SDF は節点間隔 2 cm。ウイングのスロットは既定 8 cm = 4 セル。

## データ形式

- `.nbp` (`NbpDump.h`): 粒子フレーム。位置は領域 AABB に対する `uint16` 量子化 (1000 万粒子で 1 フレーム約 70MB) + 速さ `uint8`。`render/nbp.py` が読む。
- `.pvox` (`PyroEngine::dumpFrame`): Pyro の格子データ (512×256×256 では 1 枚約 600MB。`demo_city_wind --dump-every N` で出す。既定は出さない)。
- `tracers.bin` (`demo_city_wind`): トレーサ粒子の軌跡。`"TRC1" | u32 steps | u32 M | f32 dt | f32 domain[3] | steps × M × { f32 xyz, u32 gen }` (660 ステップ × 20000 粒子 × 16 byte ≈ 210MB)。`gen` は湧き直すと増える (描画側で軌跡を切る)。`render/render_wind.py` が読む。
- `city.txt` / `domain.txt` / `obstacles.txt` / `terrain.f32`: Blender 側でシーンを再現するためのビル配置・領域・樹木と岩・地形高さ。

## 実装メモ

### 1. 気流
- `PyroSimPC::openBoundaryMask` を追加した (本 PR)。立っている面は領域外の圧力を 0 とみなす開境界になり、風が出入りできる。既定 0 は従来どおり全面閉。
- Gauss-Seidel の圧力ソルバは大域的な圧力場の収束が遅い (N² 反復) ため、最初の 45 ステップだけ領域全体を一様流に寄せ、街区周りの局所補正だけ解かせる。
  以降は上流のファン (エミッタ) が風速を維持する。
- Pyro のバッファは「Morton 符号の最大値 + 1」セル (1 セル約 88 byte)。従来は 2^n の立方体 (`cubeRes^3`) を確保しており、512³ は約 12GB で載らなかった。
  軸が x>=y>=z の順に長い 2^n 直方体 (512×256×256) なら上位ビットが立たず、符号の範囲がちょうど実セル数 (3350 万セル, 約 3GB) になる。風下 x を最長、高さ y と幅 z を短くしている。
  短い軸が x/y の直方体でも動くが、符号に穴が空く分だけ大きくなる (立方体を超えることはない)。
- 流線ではなくトレーサの軌跡 (pathline) を描く。瞬間速度場から毎フレーム流線を引くと、(1) 種を毎フレーム作り直していた、(2) 後流の渦で軌跡が敏感に変わる、の両方でちらつく。256×128×128 で実測すると、速度場自体は 0.2 秒で約 3.5% しか変わらないが、500 ステップ追跡した流線の終点は大きくずれた。
  トレーサは毎ステップ格子速度 (トリリニア, RK2) で流し、風上の断面から湧かせて、風下・建物内に入ったら湧き直す。軌跡は時間方向に連続するのでちらつかない。

### 2. 雪崩
- 雪は Drucker-Prager (摩擦 M=0.3, 凝集 100 Pa) で、斜面 (約 31°) に積もった厚さ 3〜4.5 m の層。地形コライダの摩擦を小さく (0.1) して滑り出させる。
- 地形は国土地理院 DEM (白馬周辺, z14 の `dem_png`) の一部を縦横同率で縮小した模型スケール (59×43 m)。勾配は実地形のまま。
- 樹木 (幹の円柱, 半径は樹高の約 3%)・岩 (球) は地形 SDF に焼き込む。`MPMEngine` の解析コライダは最大 64 個で、それを超える障害物を置くため。
- 樹木は雪崩の通り道の下部に間隔を空けて、岩は斜面中腹に置く。雪が幹や岩にぶつかって割れて流れる様子が見える。

#### 雪崩の入力アセット (git 管理外)
- DEM: `dem/fetch_dem.py` が国土地理院タイルを取得して `assets_local/dem.npy` に保存し、`dem/make_patch.py` が `assets_local/dem_patch.f32` を作る。
- 樹木メッシュ: virtual-tokyo リポジトリの黒松 USD (`veg_kuromatsu_*`) を Blender の `wm.usd_import` で読む。場所は環境変数 `VT_ASSETS` (既定 `~/Desktop/programs/virtual-tokyo/assets/modules`)。
- どちらもリポジトリには置かない (`assets_local/` は `.gitignore`)。
