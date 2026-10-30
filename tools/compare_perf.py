#!/usr/bin/env python3
"""
2つのビルド (base=main, new=PR) の MPM シーンについて速度と精度(PPMスクリーンショット差分)を比較する。

使い方:
  python3 tools/compare_perf.py --base <base_build_dir> --new <new_build_dir> [--frames 100] [--reps 3] [--sims tc_snow tc_tire ...]

速度: perf_sim.py と同じ PERF_RESULT (スクリーンショット無し/vsync無し) を reps 回測り中央値を比較。
精度: 同一フレーム数のPPMを保存して画素単位で比較 (最大差/平均差/完全一致フレーム数)。
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

# capture_sim.py のシーン定義と同じ引数 (capture_sim は PIL 依存のため import しない)
SIMS = [
    {"id": "tc_snow", "exe": "mpm_snow_impact", "extra_args": ["--particles", "85184", "--substeps", "25", "--box-speed", "6.0", "--box-scale", "0.5"]},
    {"id": "tc_tire", "exe": "mpm_tire_sand", "extra_args": ["--sand-nx", "150", "--sand-nz", "150", "--sand-layers", "4", "--substeps", "25", "--launch-frame", "20"]},
    {"id": "tc_avalanche", "exe": "mpm_avalanche", "extra_args": ["--max-n", "80000", "--substeps", "30", "--flip-ratio", "-1.0"]},
]

PERF_RE = re.compile(r"PERF_RESULT frames=(\d+) elapsed_s=([\d.]+) ms_per_frame=([\d.]+)")


def run(build: Path, sim: dict, frames: int, shot_dir: str = ""):
    env = {**os.environ, **sim.get("env", {}), "NEBULA_NO_VSYNC": "1"}
    cmd = [str(build / sim["exe"]), "--n-shots", str(frames), *sim.get("extra_args", [])]
    if shot_dir:
        cmd += ["--screenshot-dir", shot_dir]
    out = subprocess.run(cmd, env=env, cwd=build, capture_output=True, text=True, timeout=600).stdout
    m = PERF_RE.search(out)
    return float(m.group(3)) if m else None


def diff_dirs(a: Path, b: Path):
    files = sorted(a.glob("frame*.ppm"))
    identical, max_d, sum_d, n = 0, 0, 0, 0
    for f in files:
        x, y = f.read_bytes(), (b / f.name).read_bytes()
        if x == y:
            identical += 1
            n += len(x)
            continue
        d = [abs(p - q) for p, q in zip(x, y)]
        max_d, sum_d, n = max(max_d, max(d)), sum_d + sum(d), n + len(d)
    return len(files), identical, max_d, (sum_d / n if n else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=lambda p: Path(p).resolve())
    ap.add_argument("--new", required=True, type=lambda p: Path(p).resolve())
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sims", nargs="*", default=["tc_snow", "tc_tire", "tc_avalanche"])
    args = ap.parse_args()

    print(f"{'sim':14} {'base ms/f':>10} {'new ms/f':>10} {'speedup':>8}  frames identical maxdiff meandiff")
    for sim in (s for s in SIMS if s["id"] in args.sims):
        t = {}
        for name, b in (("base", args.base), ("new", args.new)):
            # 交互実行ではなく順次。ウォームアップ1回捨て
            run(b, sim, 10)
            t[name] = statistics.median(run(b, sim, args.frames) for _ in range(args.reps))
        with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
            run(args.base, sim, args.frames, da)
            run(args.new, sim, args.frames, db)
            n, same, mx, mean = diff_dirs(Path(da), Path(db))
        print(f"{sim['id']:14} {t['base']:10.3f} {t['new']:10.3f} {t['base'] / t['new']:7.2f}x  {n:6} {same:9} {mx:7} {mean:8.4f}")


if __name__ == "__main__":
    main()
