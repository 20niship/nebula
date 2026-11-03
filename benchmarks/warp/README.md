# nebula vs NVIDIA Warp ベンチマーク

同一条件のシミュレーションを nebula (Vulkan compute) と NVIDIA Warp (CUDA) で走らせ、1 フレームあたりの時間を表にまとめる。

| シーン | 内容 |
|---|---|
| 流体 | 水柱 (dam break)。MPM + FLIP(0.95)、弱圧縮 EOS、約 55k 粒子 |
| 粉体 | 砂柱の崩壊。MPM + APIC、Drucker-Prager、45k 粒子 |
| Softbody | 四面体ジェリーキューブ落下。XPBD (edge + tet 体積拘束)、4096 頂点 / 16875 tet |

シーン定義は [`scenes.py`](scenes.py) が唯一のソース (dt / substeps / 材料 / 初期配置 / 重力)。nebula へは CLI 引数で渡し、Warp は直接参照する。

## 実行

```sh
# nebula 側 (headless 実行ファイル。既定 OFF、CI 非対象)
cmake -S . -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release -DNEBULA_BUILD_BENCH_WARP=ON
cmake --build build-bench --target nebula_bench

# Warp 側
pip install -r benchmarks/warp/requirements.txt

python3 benchmarks/warp/run_benchmark.py          # 結果: benchmarks/warp/out/results.{md,json}
python3 benchmarks/warp/run_benchmark.py --scenes sand --repeats 1   # 一部だけ
```

`.sb` メッシュは初回に `tools/gen_softbody.py --cube` で `out/` に生成される。Vulkan デバイスが複数ある環境 (例: llvmpipe) では `VK_ICD_FILENAMES` 等で GPU を選ぶこと。選ばれたデバイス名は結果表の `GPU:` に出る。

## 計測方法

- 計測は warmup (既定 30 フレーム、GPU クロック上昇待ちも兼ねる) の後、100 フレームの wall-clock。毎フレーム GPU 完了まで同期する。
- プロセス単位で 3 回繰り返して中央値を採用する。
- Warp は 2 通り計測する: 通常の `wp.launch` と、1 フレームを CUDA graph にキャプチャして再生する場合。nebula は 1 フレームを 1 つの command buffer に記録して submit するので、後者が起動オーバーヘッド面では近い。
- 「重心差」は終了時の粒子重心の差で、同じ物理を解いているかの目安。

## 注意

- Warp 側は `warp.sim` / `warp.fem` を使わず、nebula のシェーダ (`mpm_p2g/grid_update/g2p.comp`, `predict/solve_stretch/sb_solve_volume/sdf_collision_velocity.comp`) を Warp カーネルへ移植している。同一アルゴリズムの実装比較であり、Warp の既製ソルバの性能ではない。
- 粒子-格子の atomic は nebula が固定小数点 int、Warp が float。演算順も異なるため結果は bit 一致しない。粉体の崩壊は特にカオス的で、フレームが進むと重心差が広がる。
- Softbody は粒子間衝突半径を 0 にしてあり補正は発生しないが、nebula と同じ O(N²) ループは両者とも回している。
- Warp 1.19 以降は CUDA 13 ドライバ必須。ドライバが 12.x の環境では `pip install "warp-lang<1.11"` を使う。
