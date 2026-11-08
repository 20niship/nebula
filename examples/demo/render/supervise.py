#!/usr/bin/env python3
# render_*.py は出力済みフレームをスキップするので、同じコマンドを再実行するだけで続きから始まる。
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def newest(watch: Path) -> float:
    """監視ディレクトリで最後にファイルが更新された時刻 (無ければ 0)。"""
    return max((p.stat().st_mtime for p in watch.glob("*.png")), default=0.0)


def run_once(cmd, watch: Path, stall: float) -> int:
    # 子 (と、その子の Blender 等) をまとめて止められるよう、別セッションで起動する
    proc = subprocess.Popen(cmd, start_new_session=True)
    last, last_change = newest(watch), time.time()
    while proc.poll() is None:
        time.sleep(5)
        cur = newest(watch)
        if cur != last:
            last, last_change = cur, time.time()
        elif time.time() - last_change > stall:
            print(f"[supervise] {stall:.0f} 秒間フレームが増えないため停止とみなして再起動", flush=True)
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return -1
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", required=True, help="進捗を見る出力ディレクトリ (frame_*.png が増えるか監視する)")
    ap.add_argument("--retries", type=int, default=20, help="連続で失敗してよい回数 (フレームが進めば数え直す)")
    ap.add_argument("--stall", type=float, default=300.0, help="この秒数フレームが増えなければ停止とみなす")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- の後ろに実行するコマンド")
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("実行するコマンドを -- の後ろに指定する")
    watch = Path(a.watch)
    watch.mkdir(parents=True, exist_ok=True)

    fails = 0
    while True:
        before = len(list(watch.glob("frame_*.png")))
        rc = run_once(cmd, watch, a.stall)
        if rc == 0:
            print("[supervise] 完了", flush=True)
            return 0
        after = len(list(watch.glob("frame_*.png")))
        fails = 0 if after > before else fails + 1  # 進んだ失敗は回数に数えない
        print(f"[supervise] 異常終了 (rc={rc}), 出力 {before} -> {after} フレーム, 連続失敗 {fails}/{a.retries}", flush=True)
        if fails >= a.retries:
            print("[supervise] 再試行の上限に達したので中止", flush=True)
            return 1
        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
