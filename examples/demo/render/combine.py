#!/usr/bin/env python3
# 複数の動画を同じ高さに揃えて横に並べる。短い動画は最終フレームで止める。
import argparse
import json
import subprocess


def duration(path: str) -> float:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path])
    return float(json.loads(out)["format"]["duration"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args()

    longest = max(duration(v) for v in a.videos)
    n = len(a.videos)
    chains = [
        f"[{i}:v]fps={a.fps},scale=-2:{a.height},tpad=stop_mode=clone:stop_duration={longest:.3f}[v{i}]"
        for i in range(n)
    ]
    graph = ";".join(chains) + ";" + "".join(f"[v{i}]" for i in range(n)) + f"hstack=inputs={n}:shortest=0,trim=duration={longest:.3f}[o]"
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for v in a.videos:
        cmd += ["-i", v]
    cmd += ["-filter_complex", graph, "-map", "[o]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", a.out]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
