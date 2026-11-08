#!/usr/bin/env python3
# examples/ を 1 本ずつ実行して GIF を gif_out/ へ出力。git に上げず gh pr comment --attach で上げる。
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_sim as cs  # noqa: E402

# capture_sim.SIMS にまだ無い example (exe 名 -> 追加引数 / kind)
EXTRA = [
    {"id": "g_fluid_sphere", "exe": "fluid_sphere", "title": "fluid_sphere", "env": {}, "extra_args": []},
    {"id": "g_milk", "exe": "fluid_milk_crown", "title": "fluid_milk_crown", "env": {}, "extra_args": []},
    {"id": "g_pyro_basic", "exe": "pyro_basic", "kind": "pyro", "title": "pyro_basic", "env": {},
     "extra_args": [], "render_mode": "volume", "render_kwargs": {"absorption": 3.0, "exposure": 1.2}},
]


def pick():
    """example 1 本につき 1 シム (capture_sim の最初のエントリ) を選ぶ。"""
    seen, out = set(), []
    for s in [s for s in cs.SIMS if s["exe"]] + EXTRA:
        if s["exe"] not in seen:
            seen.add(s["exe"])
            out.append(s)
    return out


def to_gif(frames, dst: Path, step: int, fps: int, width: int):
    frames = frames[::step]
    if not frames:
        return False
    lst = dst.with_suffix(".txt")
    lst.write_text("".join(f"file '{f}'\nduration {1 / fps}\n" for f in frames))
    vf = f"scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-vf", vf, "-loop", "0", str(dst)], check=True)
    lst.unlink()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "gif_out"))
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--timeout", type=float, default=240)
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    cs.OUT_DIR = out / "frames"
    cs.OUT_DIR.mkdir(exist_ok=True)
    for s in pick():
        if a.only and s["exe"] not in a.only:
            continue
        run = cs.run_pyro_sim if s.get("kind") == "pyro" else cs.run_sim
        frames, _ = run(s, a.frames, a.timeout)
        ok = to_gif(frames, out / f"{s['exe']}.gif", a.step, a.fps, a.width)
        print(f"{s['exe']}: {'ok' if ok else 'NO FRAMES'}")


if __name__ == "__main__":
    main()
