# nebula vs NVIDIA Warp (Newton) ベンチマーク

同じ条件のシーンを nebula (Vulkan compute) と NVIDIA Warp 製の物理エンジン [Newton](https://github.com/newton-physics/newton) で走らせ、1 フレームあたりの時間を表にまとめる。

| シーン | nebula | Warp (Newton) |
|---|---|---|
| 流体 (水柱の崩壊, 約 55k 粒子) | 陽的 MLS-MPM, FLIP(0.95), 弱圧縮 EOS | `SolverImplicitMPM` (implicit MPM) |
| 粉体 (砂柱の崩壊, 45k 粒子) | 陽的 MLS-MPM, APIC, Drucker-Prager | `SolverImplicitMPM` (摩擦係数 tan 30°) |
| Softbody (四面体キューブ落下, 4096 頂点) | XPBD (辺 + 体積拘束) | `SolverXPBD` + `add_soft_grid` |

シーン定義 (フレーム dt / ドメイン / グリッドセル / 初期粒子配置 / 重力 / 粒子数) は [`scenes.py`](scenes.py) が唯一のソース。nebula へは CLI 引数で渡し、Warp 側 ([`warp_bench.py`](warp_bench.py)) は直接参照する。

## 実行

```sh
# nebula 側 (headless 実行ファイル。既定 OFF、CI 非対象)
cmake -S . -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release -DNEBULA_BUILD_BENCH_WARP=ON
cmake --build build-bench --target nebula_bench

# Warp 側の環境は pyproject.toml (uv)。uv.lock と .venv は gitignore
cd benchmarks/warp
uv sync
uv run python run_benchmark.py                              # 結果: out/results.{md,json}
uv run python run_benchmark.py --scenes sand --repeats 1    # 一部だけ
```

nebula の softbody 用 `.sb` メッシュは初回に `tools/gen_softbody.py --cube` で `out/` に生成される。Vulkan デバイスが複数ある環境 (例: llvmpipe) では `VK_ICD_FILENAMES` 等で GPU を選ぶこと。選ばれたデバイス名は結果表の `GPU:` に出る。

## スイープ (粒子数・グリッド解像度)

```sh
uv run python run_sweep.py        # out/sweep_log.csv, out/sweep_results.json, out/sweep_overview.png
```

- 粒子数スイープ: MPM 粒子数を 1/8〜4 倍 (グリッドセルは固定)、softbody は一辺の頂点数を 8〜24。
- グリッドスイープ: 解像度を 32〜128 cells/軸 (粒子数は固定。MPM のみ)。
- 各点 40 フレーム (warmup 10) の 1 回計測なので、小さい粒子数ではノイズが乗る。
- ログ・グラフは `out/` (gitignore) に出力する。グラフはリポジトリに commit しない。

## 計測方法

- warmup (既定 30 フレーム、GPU クロック上昇待ちも兼ねる) の後、100 フレームの wall-clock。毎フレーム GPU 完了まで同期する。
- プロセス単位で 3 回繰り返して中央値を採用する。
- Warp は通常実行と、1 フレームを CUDA graph にキャプチャして再生する場合の 2 通りを測る。Newton の implicit MPM は sparse グリッド (既定) だとキャプチャできないため、流体・粉体では通常実行のみ (表では「—」)。

## 注意

- **ソルバが違う。** nebula は陽的 MPM / XPBD、Newton は implicit MPM / XPBD で、Newton の MPM は非圧縮寄り・大きい時間刻みで解く。そのため substeps はソルバごとに適切な値 (nebula 20 / Newton 2 など) にしてあり、結果の動きも一致しない (表の「終了時重心」は崩壊・落下していることの確認用)。同じ「シーン」を各エンジンの標準的な解き方で解いた時間の比較であり、同一アルゴリズムの比較ではない。
- Softbody の終了時重心は nebula (反発して跳ねている) と Newton (床で潰れている) で大きく異なる。剛性・減衰の対応づけは概算 (nebula は compliance、Newton は k_mu/k_lambda) で、時間比較が目的なので挙動の一致までは追っていない。
- 流体の体積弾性率は 2e5 Pa にしてある (`presetWater` の 5e3 では 3 m の水柱が崩れず縦に潰れる)。
- Softbody は `add_soft_grid` の四面体分割が nebula の `.sb` と一致しない (頂点数は同じ 4096)。nebula 側は粒子間衝突半径を 0 にしてあるが、同じ O(N²) ループは回る。
- **Warp のバージョン:** warp-lang 1.18 以降の PyPI wheel は CUDA 13 ドライバ必須。CUDA 12.x ドライバでも動くよう `pyproject.toml` で 1.17 に固定している。CUDA 13 ドライバなら pin を外してよい。
