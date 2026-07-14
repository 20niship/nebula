#!/opt/homebrew/bin/python3
"""
compare_bench.py — 2つの capture_sim.py --perf-json 出力を比較し、
Markdown テーブルを標準出力へ書き出す (PR コメント用)。

指標は ms_per_frame (1フレームあたりの実行時間, 小さいほど速い)。
CI (lavapipe ソフトウェアレンダラ) 上の絶対値はノイズが乗りやすいため、
変化率が閾値 (THRESHOLD_PCT) を超えた場合のみ speedup/slowdown として強調する。

使い方:
  python3 tools/compare_bench.py baseline.json current.json > comment.md
"""

import argparse
import json
import sys
from pathlib import Path

THRESHOLD_PCT = 5.0  # これ未満の変化はノイズとみなす


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        text = path.read_text().strip()
        if not text:
            return {}
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def fmt_ms(ms) -> str:
    return f"{ms:.1f}ms" if ms is not None else "N/A"


def build_table(baseline: dict, current: dict) -> str:
    base_sims = baseline.get("sims", {})
    cur_sims  = current.get("sims", {})

    ids = list(cur_sims.keys())
    for sid in base_sims:
        if sid not in ids:
            ids.append(sid)

    lines = []
    lines.append("| Test | Baseline (ms/frame) | Current (ms/frame) | Δ% |")
    lines.append("|------|---------------------|---------------------|----|")

    n_slower = n_faster = n_same = n_na = 0

    for sid in ids:
        base = base_sims.get(sid)
        cur  = cur_sims.get(sid)
        title = (cur or base or {}).get("title", sid)

        if base is None or cur is None:
            lines.append(f"| {title} | {fmt_ms(base['ms_per_frame']) if base else 'N/A'} "
                         f"| {fmt_ms(cur['ms_per_frame']) if cur else 'N/A'} | N/A (new/removed) |")
            n_na += 1
            continue

        base_ms = base["ms_per_frame"]
        cur_ms  = cur["ms_per_frame"]
        pct = (cur_ms - base_ms) / base_ms * 100.0 if base_ms else 0.0

        if pct > THRESHOLD_PCT:
            mark = "🔴"
            delta_cell = f"**+{pct:.1f}% {mark}**"
            n_slower += 1
        elif pct < -THRESHOLD_PCT:
            mark = "🟢"
            delta_cell = f"**{pct:.1f}% {mark}**"
            n_faster += 1
        else:
            mark = "⚪"
            delta_cell = f"{pct:+.1f}% {mark}"
            n_same += 1

        lines.append(f"| {title} | {fmt_ms(base_ms)} | {fmt_ms(cur_ms)} | {delta_cell} |")

    summary = f"🔴 slower: {n_slower}  🟢 faster: {n_faster}  ⚪ unchanged: {n_same}"
    if n_na:
        summary += f"  ⚠️ new/removed: {n_na}"

    return "\n".join(lines) + "\n\n" + summary


def main():
    parser = argparse.ArgumentParser(description="perf JSON 2ファイルを比較して Markdown を出力")
    parser.add_argument("baseline", type=Path, help="main ブランチの baseline JSON")
    parser.add_argument("current",  type=Path, help="今回計測した JSON")
    cli = parser.parse_args()

    baseline = load_json(cli.baseline)
    current  = load_json(cli.current)

    if not current.get("sims"):
        print("⚠️ 今回の計測結果 (current) が空か読み込めませんでした。capture_sim.py の実行に失敗している可能性があります。")
        sys.exit(0)

    print("### 📊 Simulation Capture Performance\n")

    if not baseline.get("sims"):
        print("baseline (`benchmarks/baseline.json`) がまだ存在しません。"
              "このブランチが main にマージされると、次回 PR 以降の baseline として記録されます。\n")
        print("| Test | Current (ms/frame) |")
        print("|------|---------------------|")
        for sid, entry in current.get("sims", {}).items():
            print(f"| {entry.get('title', sid)} | {fmt_ms(entry['ms_per_frame'])} |")
        sys.exit(0)

    print(build_table(baseline, current))
    print(f"\n<sub>baseline commit: `{baseline.get('commit', 'unknown')}` "
          f"/ current commit: `{current.get('commit', 'unknown')}` "
          f"/ 閾値: ±{THRESHOLD_PCT}% (CI はソフトウェアレンダラ実行のため絶対値は参考値、相対比較のみ有効)</sub>")


if __name__ == "__main__":
    main()
