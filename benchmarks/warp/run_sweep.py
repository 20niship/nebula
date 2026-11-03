#!/usr/bin/env python3
"""粒子数 / グリッド解像度を振って nebula と Warp (Newton) の速度を計測し、ログとグラフを出力する。

  uv run python run_sweep.py                  # out/sweep_log.csv, out/sweep_results.json, out/sweep_overview.png
  uv run python run_sweep.py --scenes fluid   # 一部のシーンだけ

スイープ:
  particles : MPM 粒子数を 1/8〜4 倍 (流体・粉体) / softbody の一辺の頂点数を 8〜24 (グリッドセルは固定)
  grid      : グリッド解像度を 32〜128 cells/軸 (粒子数は固定。MPM のみ)
グラフ・ログは out/ (gitignore) に出力する。リポジトリには commit しない。
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import scenes as S
from run_benchmark import ROOT, HERE, ensure_mesh, run_json

PARTICLE_SCALES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]  # MPM 粒子数の倍率
SB_GRIDS = [8, 12, 16, 20, 24]  # softbody の一辺の頂点数
GRID_RES = [32, 48, 64, 96, 128]  # グリッド解像度 [cells/軸]

ENGINES = ["nebula", "warp", "warp_graph"]
LABELS = {"nebula": "nebula (Vulkan)", "warp": "Warp / Newton", "warp_graph": "Warp / Newton (CUDA graph)"}
STYLE = {"nebula": ("#1f77b4", "o", "-"), "warp": ("#d95f02", "s", "--"), "warp_graph": ("#7570b3", "^", ":")}


def sweep_points(name):
    """(sweep 名, 値, configure 引数) の一覧。"""
    if S.SCENES[name]["kind"] == "mpm":
        pts = [("particles", f, dict(particle_scale=f)) for f in PARTICLE_SCALES]
        pts += [("grid", r, dict(cell=S.DOMAIN / r)) for r in GRID_RES]
    else:
        pts = [("particles", g, dict(sb_grid=g)) for g in SB_GRIDS]
    return pts


def measure(name, kw, a, out):
    """1 点分: 現在の configure 済み状態で nebula / Warp を実行して結果 dict を返す。"""
    flags = []
    if "cell" in kw:
        flags += ["--cell-size", repr(kw["cell"])]
    if "particle_scale" in kw:
        flags += ["--particle-scale", repr(kw["particle_scale"])]
    if "sb_grid" in kw:
        flags += ["--sb-grid", str(kw["sb_grid"])]
    ensure_mesh(out)
    res = {}
    for r in run_json([a.nebula_bench] + S.nebula_args(name, out, a.warmup, a.frames), ROOT):
        res[r["engine"]] = r
    for r in run_json([sys.executable, "warp_bench.py", "--scene", name, "--warmup", str(a.warmup),
                       "--frames", str(a.frames)] + flags, HERE):
        res[r["engine"]] = r
    return res


def plot(rows, path, gpu):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenes = list(dict.fromkeys(r["scene"] for r in rows))
    kinds = [("particles", "Number of particles", "Particle-count sweep"),
             ("grid", "Grid resolution [cells per axis]", "Grid-resolution sweep")]
    fig, axes = plt.subplots(2, len(scenes), figsize=(5.2 * len(scenes), 8.6), squeeze=False)
    for j, sc in enumerate(scenes):
        for i, (kind, xlabel, ttl) in enumerate(kinds):
            ax = axes[i][j]
            sel = [r for r in rows if r["scene"] == sc and r["sweep"] == kind]
            if not sel:
                ax.set_visible(False)
                continue
            for eng in ENGINES:
                pts = sorted((r["x"], r["ms_per_frame"]) for r in sel if r["engine"] == eng)
                if pts:
                    c, m, ls = STYLE[eng]
                    ax.plot(*zip(*pts), color=c, marker=m, linestyle=ls, linewidth=1.8, label=LABELS[eng])
            ax.set_xscale("log", base=2 if kind == "grid" else 10)
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Time per frame [ms] (lower is faster)")
            ax.set_title(f"{S.SCENES[sc]['title_en']}\n{ttl}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle(f"nebula vs NVIDIA Warp (Newton): simulation speed scaling  [{gpu}]", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nebula-bench", default=str(ROOT / "build-bench/benchmarks/warp/nebula_bench"))
    ap.add_argument("--scenes", nargs="+", default=list(S.SCENES), choices=list(S.SCENES))
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out-dir", default=str(HERE / "out"))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, gpu = [], "unknown"
    with open(out / "sweep_log.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["scene", "sweep", "x", "particles", "cell_size", "engine", "ms_per_frame"])
        for name in a.scenes:
            for sweep, x, kw in sweep_points(name):
                S.configure(**kw)
                res = measure(name, kw, a, out)
                for eng, r in res.items():
                    gpu = r.get("device", gpu) if eng == "nebula" else gpu
                    # x 軸: particles スイープは粒子数、grid スイープは cells/軸
                    xv = r["n"] if sweep == "particles" else round(S.DOMAIN / S.CELL)
                    row = dict(scene=name, sweep=sweep, x=xv, particles=r["n"], cell_size=S.CELL, engine=eng,
                               ms_per_frame=r["ms_per_frame"])
                    rows.append(row)
                    wr.writerow(row.values())
                    f.flush()
                    print(f"{name:9s} {sweep:9s} x={xv:<8} {eng:11s} {r['ms_per_frame']:9.2f} ms/frame", flush=True)
                S.configure(cell=S.DOMAIN / 64.0, particle_scale=1.0, sb_grid=16)  # 既定に戻す
    (out / "sweep_results.json").write_text(json.dumps(rows, indent=2))
    plot(rows, out / "sweep_overview.png", gpu)
    print(f"graph: {out / 'sweep_overview.png'}")


if __name__ == "__main__":
    main()
