#!/usr/bin/env python3
"""nebula と NVIDIA Warp を同一条件で走らせ、比較表 (results.md / results.json) を出力する。

使い方:
  cmake -S . -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release -DNEBULA_BUILD_BENCH_WARP=ON
  cmake --build build-bench --target nebula_bench
  (cd benchmarks/warp && uv sync && uv run python run_benchmark.py)
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import scenes as S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run_json(cmd, cwd):
    """cmd を実行し、stdout の `BENCH_JSON {...}` 行をすべて dict で返す。"""
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"失敗: {' '.join(map(str, cmd))}\n{p.stdout}\n{p.stderr}")
    return [json.loads(l.split(" ", 1)[1]) for l in p.stdout.splitlines() if l.startswith("BENCH_JSON ")]


def ensure_mesh(out_dir):
    path = out_dir / S.SB_FILENAME
    if not path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(ROOT / "tools/gen_softbody.py"), "--cube", "--grid", str(S.SB_GRID),
                        "--size", str(S.SB_SIZE), "-o", str(path)], check=True, capture_output=True)
    return path


def fmt_table(results, meta):
    rows = ["| シーン | 粒子数 | nebula [ms/frame] | Warp (Newton) [ms/frame] | Warp CUDA graph [ms/frame] | vs Warp | vs Warp graph | substeps (nebula / Warp) | 終了時重心 y [m] (nebula / Warp) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        n, w, g = r["nebula"], r["warp"], r.get("warp_graph")
        gms = f"{g['ms_per_frame']:.2f}" if g else "—"
        gratio = f"{g['ms_per_frame'] / n['ms_per_frame']:.2f}x" if g else "—"
        rows.append(f"| {S.SCENES[name]['title']} | {n['n']:,} | {n['ms_per_frame']:.2f} | {w['ms_per_frame']:.2f} | "
                    f"{gms} | {w['ms_per_frame'] / n['ms_per_frame']:.2f}x | {gratio} | "
                    f"{S.SCENES[name]['substeps']} / {w['substeps']} | {n['centroid'][1]:.2f} / {w['centroid'][1]:.2f} |")
    head = [f"- GPU: {meta['gpu']}", f"- Warp: {meta['warp']}, Newton: {meta['newton']}",
            f"- 共通条件: フレーム dt=1/60 s, ドメイン {S.DOMAIN:g} m, グリッドセル {S.CELL:.4f} m, 重力 {S.GRAVITY} m/s², 同一の粒子数・初期配置",
            f"- 計測: warmup {meta['warmup']} + 計測 {meta['frames']} フレーム, {meta['repeats']} 回の中央値",
            "- vs Warp / vs Warp graph: Warp の時間 ÷ nebula の時間 (1 より大きいと nebula が速い)。graph は CUDA graph にキャプチャできたシーンのみ",
            "- 終了時重心 y: ソルバが違う (nebula 陽的 MLS-MPM / Newton implicit MPM) ため値は一致しない。両者とも崩壊・落下していることの確認用", ""]
    return "\n".join(head + rows) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nebula-bench", default=str(ROOT / "build-bench/benchmarks/warp/nebula_bench"))
    ap.add_argument("--scenes", nargs="+", default=list(S.SCENES), choices=list(S.SCENES))
    ap.add_argument("--frames", type=int, default=S.FRAMES)
    ap.add_argument("--warmup", type=int, default=S.WARMUP)
    ap.add_argument("--repeats", type=int, default=3, help="プロセス単位の繰り返し回数 (中央値を採用)")
    ap.add_argument("--out-dir", default=str(HERE / "out"))
    a = ap.parse_args()

    if not Path(a.nebula_bench).exists():
        sys.exit(f"nebula_bench が無い: {a.nebula_bench}\n-DNEBULA_BUILD_BENCH_WARP=ON でビルドすること")
    try:
        import warp as wp
        wp.config.quiet = True
        wp.init()
        import newton
        warp_ver, newton_ver = wp.__version__, newton.__version__
    except ImportError:
        sys.exit("warp が無い: uv sync (benchmarks/warp)")

    out = Path(a.out_dir)
    ensure_mesh(out)
    results, gpu = {}, None
    for name in a.scenes:
        runs = {"nebula": [], "warp": [], "warp_graph": []}
        for i in range(a.repeats):
            print(f"[{name}] run {i + 1}/{a.repeats}", file=sys.stderr)
            for r in run_json([a.nebula_bench] + S.nebula_args(name, out, a.warmup, a.frames), ROOT):
                gpu = r.get("device", gpu)
                runs["nebula"].append(r)
            for r in run_json([sys.executable, "warp_bench.py", "--scene", name, "--warmup", str(a.warmup), "--frames", str(a.frames)], HERE):
                runs[r["engine"]].append(r)
        # 中央値の run を代表として採用
        results[name] = {}
        for k, v in runs.items():
            if v:  # CUDA graph にできないシーンでは warp_graph が空
                med = sorted(v, key=lambda r: r["ms_per_frame"])[len(v) // 2]
                results[name][k] = dict(med, all_ms=[r["ms_per_frame"] for r in v])

    meta = dict(gpu=gpu or "unknown", warp=warp_ver, newton=newton_ver, warmup=a.warmup, frames=a.frames, repeats=a.repeats)
    (out / "results.json").write_text(json.dumps(dict(meta=meta, results=results), indent=2, ensure_ascii=False))
    md = fmt_table(results, meta)
    (out / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
