#!/opt/homebrew/bin/python3
"""
task perf: 純粋なシミュレーション実行時間の計測スクリプト (task capture の代替)

## なぜこのスクリプトが必要か
`capture_sim.py` (task capture) の計測時間には、シミュレーション本体とは無関係な
以下のオーバーヘッドが丸ごと含まれており、GPU/エンジン側の性能比較には使えない:

- 非Pyro系 (`run_sim()`): `subprocess.Popen` + `while proc.poll() is None: time.sleep(2)`
  というポーリングループでプロセス終了を検知するため、実行時間が**2秒粒度**に丸められる。
  実際の実行が2秒未満のケース (このプロジェクトのほぼ全シーン) では、計測値は
  シミュレーション性能ではなくポーリング粒度そのものを表してしまう。
- Pyro系 (`run_pyro_sim()`): `.pvox` ダンプ後に `tools/pyro_raymarch.py` で
  numpyベースのソフトウェアレイマーチングを行いPNGへ変換しており、これは
  スクリプト自身のコメントにもある通り「1フレームあたり数百ms〜1秒」かかる。
  GPU物理計算 (実測: 数ms/frame) の1000倍以上のオーダーで、計測のほぼ全てを占める。

本スクリプトは、C++側 (`src/App.cpp` の `BaseApp::saveScreenshot()`, および
`examples/pyro_cow_blast.cpp`/`pyro_explosion.cpp`) に追加した `PERF_RESULT` 行
(画面キャプチャ/pvoxダンプ無しでのフレームループ実測、非Pyro系はさらに
`NEBULA_NO_VSYNC=1` で vsync による上限も排除) を `subprocess.run()` で
ブロッキング実行して直接パースするため、上記2つの問題をどちらも回避できる。

使い方:
  python3 perf_sim.py [--frames N] [--perf-json PATH] [--commit-sha SHA]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from capture_sim import SIMS, BUILD_DIR, _resolve_commit_sha  # noqa: E402  (シーン定義を再利用)

N_FRAMES = 100  # task capture (300) より少なめ。ポーリング/レイマーチング起因の
                # ノイズが無いため、少ないフレーム数でも十分安定した値が得られる。

PERF_RESULT_RE = re.compile(r"PERF_RESULT frames=(\d+) elapsed_s=([\d.]+) ms_per_frame=([\d.]+)")


def run_one(sim: dict, n_frames: int) -> dict:
    exe = BUILD_DIR / sim["exe"]
    if not exe.exists():
        return {"status": "missing"}

    is_pyro = sim.get("kind") == "pyro"
    env = os.environ.copy()
    env.update(sim.get("env", {}))

    if is_pyro:
        # --dump-every 0: .pvox ダンプ (GPU→CPU readback + ディスク書き込み) を無効化
        cmd = [str(exe), "--n-frames", str(n_frames), "--dump-every", "0"]
    else:
        # --screenshot-dir を渡さない (デフォルト空文字) ことで PPM 保存を無効化。
        # NEBULA_NO_VSYNC=1 で vsync によるフレームレート上限も外す。
        env["NEBULA_NO_VSYNC"] = "1"
        env["MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS"] = "0"
        cmd = [str(exe), "--n-shots", str(n_frames)]
    if "SIM_BOUNDARY_OBJ" in sim.get("env", {}):
        cmd += ["--boundary-obj", sim["env"]["SIM_BOUNDARY_OBJ"]]
    cmd += sim.get("extra_args", [])

    try:
        proc = subprocess.run(cmd, env=env, cwd=str(BUILD_DIR), capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}

    match = None
    for line in reversed(proc.stdout.splitlines()):
        match = PERF_RESULT_RE.search(line)
        if match:
            break
    if not match:
        return {"status": "no_perf_result", "returncode": proc.returncode, "stderr_tail": proc.stderr[-1500:]}

    frames, elapsed_s, ms_per_frame = match.groups()
    return {
        "status": "ok",
        "frames": int(frames),
        "elapsed_s": float(elapsed_s),
        "ms_per_frame": float(ms_per_frame),
        "returncode": proc.returncode,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=N_FRAMES, help=f"各シムの実行フレーム数 (デフォルト: {N_FRAMES})")
    parser.add_argument("--perf-json", type=str, default=None, help="計測結果をJSONで書き出すパス")
    parser.add_argument("--commit-sha", type=str, default="", help="perf-json に埋め込む commit sha (未指定なら git rev-parse HEAD)")
    cli = parser.parse_args()

    print(f"=== task perf: {cli.frames} frames/case, キャプチャI/O無し, vsync無効 ===")
    results = {}
    for sim in SIMS:
        if sim["exe"] is None:
            continue
        print(f"\n--- {sim['title']} ({sim['exe']}) ---")
        r = run_one(sim, cli.frames)
        results[sim["id"]] = {"title": sim["title"], **r}
        if r["status"] == "ok":
            print(f"  {r['ms_per_frame']:.4f} ms/frame  (total {r['elapsed_s']:.3f}s / {r['frames']}frames)")
        else:
            print(f"  FAILED: {r}")

    print("\n=== summary (ms/frame) ===")
    for r in results.values():
        val = f"{r['ms_per_frame']:.4f}" if r["status"] == "ok" else f"[{r['status']}]"
        print(f"  {r['title']:45s} {val}")

    if cli.perf_json:
        out = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "commit": _resolve_commit_sha(cli.commit_sha),
            "frames": cli.frames,
            "mode": "perf_sim (no capture I/O, no vsync)",
            "sims": results,
        }
        Path(cli.perf_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n書き出し: {cli.perf_json}")


if __name__ == "__main__":
    main()
