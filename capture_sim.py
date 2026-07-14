#!/opt/homebrew/bin/python3
"""
全シミュレーション キャプチャ → 1本の動画生成スクリプト

各シミュレーションを順次実行し、フレームごとに PPM として保存。
全シム完了後、対応フレームを 4×5 グリッドに合成して MP4 動画を書き出す。

テストケース一覧:
  TC1:  流体 ダムブレイク                                      fluid_pbf
  TC2:  流体 移動ソースフロー                                   fluid_pbf
  TC3:  高粘性ゼリー流体                                       fluid_pbf
  TC4:  布シミュレーション（2隅固定）                          cloth_3d
  TC5:  布2枚 + 布-布衝突                                      cloth_scene --scene 5
  TC6:  煙パーティクル                                         smoke
  TC7:  4隅アニメーションピン・布ねじれ                        cloth_scene --scene 7
  TC8:  回転スクリュー + PBF 流体                              screw_fluid
  TC9:  楕円水たまり + 移動円柱吸収ポート                      fluid_absorb
  TC10: XPBD 四面体ソフトボディ                               xpbd_softbody
  TC-A: MPM Elastic — PIC  (flip_ratio=0.00)                  mpm_elastic
  TC-B: MPM Elastic — APIC (flip_ratio=-1.00)                 mpm_elastic
  TC-E: MPM Mountain Avalanche — Drucker-Prager                mpm_avalanche
  TC-F: MPM マルチマテリアル — 弾性体 + Drucker-Prager 砂      mpm_multimaterial
  TC-G: MPM 雪衝突 — 移動箱SDF (粒子50倍・高速・半サイズ箱・固定フレーム自動衝突) mpm_snow_impact
  TC-H: MPM 地層崩壊 — 硬岩/弱粘土/緩土 3層                  mpm_geolayer
  TC-K: Pyro 牛への爆風 — 流速ヒートマップ表示                 pyro_cow_blast
  TC-L: Pyro 爆発 (キノコ雲) — smoke/fire ボリューム表示       pyro_explosion
使い方:
  python3 capture_sim.py [--frames N] [--fps F] [--perf-json PATH] [--commit-sha SHA]
"""

import argparse
import json
import subprocess
import time
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).parent / "tools"))
import pyro_raymarch  # noqa: E402  (.pvox -> PNG レンダリングを直接呼び出す)

BUILD_DIR = Path(__file__).parent / "build"
OUT_DIR   = Path(__file__).parent / "sim_captures"
OUT_DIR.mkdir(exist_ok=True)

# キャプチャ設定
N_FRAMES  = 300    # 各シムの保存フレーム数
VIDEO_FPS = 60     # 出力動画 FPS
THUMB_W   = 480    # 4列 × 480 = 1920px (ffmpeg scale と一致)
THUMB_H   = 270    # 16:9
GRID_COLS = 4
GRID_ROWS = 5      # 4×5=20 セル; TC1–TC10 + TC-A,B,E,F,G,H,K,L (TC-C/D/I/J除外、18使用 + 空き2)

# テストケース定義
# exe=None のエントリは空きセル（グリッドのパディング用）
SIMS = [
    # ── PBF 流体 / 布 / 煙 / ソフトボディ ───────────────────────────────────
    {
        "id": "tc1", "exe": "fluid_pbf",
        "title": "TC1: Dam Break",
        "env": {}, "extra_args": ["--scenario", "dam-break"],
        "params": "N~110K | dam-break (left-top half)",
    },
    {
        "id": "tc2", "exe": "fluid_pbf",
        "title": "TC2: Moving Source Flow",
        "env": {},
        "extra_args": [
            "--scenario", "source-flow",
            "--world-size", "40",
            "--nx", "192", "--ny", "3", "--nz", "192",
            "--grid-res", "128",
        ],
        "params": "N~110K | moving AABB source | world=40",
    },
    {
        "id": "tc3", "exe": "fluid_pbf",
        "title": "TC3: Jelly (High Viscosity)",
        "env": {"SIM_VISCOSITY_C": "0.5"}, "extra_args": [],
        "params": "N~110K | rho0=2097 | visc=0.50",
    },
    {
        "id": "tc4", "exe": "cloth_3d",
        "title": "TC4: Cloth Drape",
        "env": {}, "extra_args": [],
        "params": "verts=16384 | 2-corner pinned",
    },
    {
        "id": "tc5", "exe": "cloth_scene",
        "title": "TC5: Two-Cloth Collision",
        "env": {}, "extra_args": ["--scene", "5", "--cloth-n", "32"],
        "params": "verts=2×1024 | top-row pinned | cloth-cloth SDF",
    },
    {
        "id": "tc6", "exe": "smoke",
        "title": "TC6: Smoke",
        "env": {}, "extra_args": ["--max-particles", "4096", "--pps", "48"],
        "params": "max=4096 | buoyancy+vorticity | sphere emitter",
    },
    {
        "id": "tc7", "exe": "cloth_scene",
        "title": "TC7: Cloth Twist",
        "env": {}, "extra_args": ["--scene", "7", "--cloth-n", "32"],
        "params": "verts=1024 | 4-corner animated pins | ±20°/s twist",
    },
    {
        "id": "tc8", "exe": "screw_fluid",
        "title": "TC8: Rotating Screw",
        "env": {}, "extra_args": ["--ang-vel", "2.0", "--viscosity", "0.01"],
        "params": "N~27K | screw ω=2rad/s | visc=0.01",
    },
    {
        "id": "tc9", "exe": "fluid_absorb",
        "title": "TC9: Fluid Absorb",
        "env": {}, "extra_args": [],
        "params": "N~1.4K | ellipse puddle | moving cylinder absorber",
    },
    {
        "id": "tc10", "exe": "xpbd_softbody",
        "title": "TC10: XPBD Softbody",
        "env": {}, "extra_args": [],
        "params": "bunny + cube | XPBD volumetric | substeps=15",
    },
    # ── MPM / FLIP ──────────────────────────────────────────────────────────
    {
        "id": "tc_pic", "exe": "mpm_elastic",
        "title": "TC-A: MPM Elastic — PIC",
        "env": {},
        "extra_args": [
            "--nx", "20", "--ny", "20", "--nz", "20",
            "--grid-res", "64", "--substeps", "25", "--flip-ratio", "0.0",
        ],
        "params": "nx=ny=nz=20 | flip=0.00 | 散逸大",
    },
    {
        "id": "tc_apic", "exe": "mpm_elastic",
        "title": "TC-B: MPM Elastic — APIC",
        "env": {},
        "extra_args": [
            "--nx", "20", "--ny", "20", "--nz", "20",
            "--grid-res", "64", "--substeps", "25", "--flip-ratio", "-1.0",
        ],
        "params": "nx=ny=nz=20 | flip=-1.00 | 角運動量保存",
    },
    {
        "id": "tc_avalanche", "exe": "mpm_avalanche",
        "title": "TC-E: MPM Avalanche — Snow",
        "env": {},
        "extra_args": [
            "--max-n", "80000",
            "--grid-res", "128", "--substeps", "30", "--flip-ratio", "-1.0",
        ],
        "params": "N=80K | Drucker-Prager snow | 地形 SDF コライダー",
    },
    # ── 追加 MPM シーン ────────────────────────────────────────────────────
    {
        "id": "tc_multi", "exe": "mpm_multimaterial",
        "title": "TC-F: MPM Multi-Material",
        "env": {},
        "extra_args": [
            "--n", "20",
            "--grid-res", "64", "--substeps", "25",
        ],
        "params": "N=8000 | 下半分=Hencky弾性体 | 上半分=Drucker-Prager砂",
    },
    {
        "id": "tc_snow", "exe": "mpm_snow_impact",
        "title": "TC-G: MPM Snow Impact",
        "env": {},
        "extra_args": [
            "--pn", "44",
            "--grid-res", "64", "--substeps", "25",
            "--box-speed", "6.0",
            "--box-scale", "0.5",
        ],
        "params": "N~85K | VON_MISES雪 | 箱速6m/s・半サイズ | 固定フレームで自動衝突",
    },
    {
        "id": "tc_geolayer", "exe": "mpm_geolayer",
        "title": "TC-H: MPM Geo-Layer Collapse",
        "env": {},
        "extra_args": [
            "--grid-res", "64", "--substeps", "25",
        ],
        "params": "N=1920 | 下=硬岩ELASTIC | 中=弱粘土VON_MISES | 上=緩土D-P",
    },
    # ── Pyro (グリッドベース煙・火炎) ────────────────────────────────────────
    {
        "id": "tc_cow", "exe": "pyro_cow_blast", "kind": "pyro",
        "title": "TC-K: Pyro Cow Blast",
        "env": {}, "extra_args": ["--grid-res", "32"],
        "params": "超高密度爆風 | 低ポリ牛SDF障害物 | 流速ヒートマップ",
        "render_mode": "heatmap",
        "render_kwargs": {"threshold": 0.5, "speed_max": 8.0, "axis": "z"},
    },
    {
        "id": "tc_explosion", "exe": "pyro_explosion", "kind": "pyro",
        "title": "TC-L: Pyro Explosion (Mushroom Cloud)",
        "env": {}, "extra_args": ["--grid-res", "32"],
        "params": "地表爆発 | fuel燃焼+強浮力+渦度閉じ込め | smoke/fireボリューム",
        "render_mode": "volume",
        "render_kwargs": {"absorption": 3.0, "exposure": 1.2, "axis": "z"},
    },
]


# ── ユーティリティ ─────────────────────────────────────────────────────────────

def _load_font(size: int):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_overlay(draw, x, y, title, params, fps_str, font_big, font_small):
    lines = [title, fps_str, params]
    fonts = [font_big, font_small, font_small]
    bg_h  = 14 + 16 + 16 + 8
    bg_w  = max(len(l) * 7 + 12 for l in lines) + 8
    draw.rectangle([x, y, x + bg_w, y + bg_h], fill=(0, 0, 0, 160))
    cy = y + 4
    for line, fnt in zip(lines, fonts):
        draw.text((x + 6, cy), line, fill=(255, 255, 255), font=fnt)
        cy += 15


def make_grid_frame(frame_idx: int, sim_frames: dict, fps_map: dict) -> Image.Image:
    label_h  = 52
    canvas_w = GRID_COLS * THUMB_W
    canvas_h = GRID_ROWS * (THUMB_H + label_h)
    canvas   = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))
    draw     = ImageDraw.Draw(canvas, "RGBA")

    font_big   = _load_font(14)
    font_small = _load_font(12)

    for cell_i, sim in enumerate(SIMS):
        if cell_i >= GRID_COLS * GRID_ROWS:
            break
        col = cell_i % GRID_COLS
        row = cell_i // GRID_COLS
        cx  = col * THUMB_W
        cy  = row * (THUMB_H + label_h)

        if sim["exe"] is None:
            canvas.paste(Image.new("RGB", (THUMB_W, THUMB_H), (24, 24, 32)), (cx, cy))
            continue

        frames = sim_frames.get(sim["id"], [])
        if frames and frame_idx < len(frames):
            try:
                img   = Image.open(frames[frame_idx])
                img.load()
                thumb = img.resize((THUMB_W, THUMB_H), Image.Resampling.LANCZOS)
                canvas.paste(thumb, (cx, cy))
            except Exception as e:
                print(f"  [warn] frame load failed: {e}")
                canvas.paste(Image.new("RGB", (THUMB_W, THUMB_H), (40, 20, 20)), (cx, cy))
        else:
            canvas.paste(Image.new("RGB", (THUMB_W, THUMB_H), (30, 30, 30)), (cx, cy))

        fps_str = fps_map.get(sim["id"], "FPS: --")
        _draw_overlay(draw, cx, cy, sim["title"], sim["params"], fps_str, font_big, font_small)

    return canvas


def run_sim(sim: dict, n_frames: int) -> list[str]:
    exe   = BUILD_DIR / sim["exe"]
    title = sim["title"]

    print(f"\n=== {title} (exe: {sim['exe']}) ===")
    if not exe.exists():
        print(f"  ERROR: {exe} not found, skipping.")
        return []

    ppm_dir = OUT_DIR / sim["id"]
    ppm_dir.mkdir(exist_ok=True)
    for f in ppm_dir.glob("*.ppm"):
        f.unlink()

    env = os.environ.copy()
    env["MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS"] = "0"
    env.update(sim.get("env", {}))

    cmd = [
        str(exe),
        "--n-shots",        str(n_frames),
        "--screenshot-dir", str(ppm_dir),
    ]
    cmd += sim.get("extra_args", [])
    # SIM_BOUNDARY_OBJ が指定されている場合は --boundary-obj へ変換
    if "SIM_BOUNDARY_OBJ" in sim.get("env", {}):
        cmd += ["--boundary-obj", sim["env"]["SIM_BOUNDARY_OBJ"]]

    proc = subprocess.Popen(cmd, env=env, cwd=str(BUILD_DIR))
    print(f"  PID {proc.pid} — {n_frames} フレーム待機中 …")

    while proc.poll() is None:
        captured = len(list(ppm_dir.glob("frame*.ppm")))
        print(f"  [{int(time.time() % 10000)}] {captured}/{n_frames} frames", end="\r", flush=True)
        time.sleep(2)
    print()
    print(f"  終了 (rc={proc.returncode})")

    frames = sorted(ppm_dir.glob("frame*.ppm"))
    print(f"  キャプチャ完了: {len(frames)} frames")
    return [str(p) for p in frames]


def run_pyro_sim(sim: dict, n_frames: int) -> list[str]:
    """Pyro (ヘッドレス) サンプル用ランナー。--n-shots/--screenshot-dir ではなく
    --n-frames/--dump-every/--out で .pvox をダンプさせ、tools/pyro_raymarch.py の
    関数を直接呼び出して PNG (frameNNNN.png) に変換する。"""
    exe   = BUILD_DIR / sim["exe"]
    title = sim["title"]

    print(f"\n=== {title} (exe: {sim['exe']}) ===")
    if not exe.exists():
        print(f"  ERROR: {exe} not found, skipping.")
        return []

    sim_dir  = OUT_DIR / sim["id"]
    pvox_dir = sim_dir / "pvox"
    pvox_dir.mkdir(parents=True, exist_ok=True)
    for f in pvox_dir.glob("*.pvox"):
        f.unlink()
    for f in sim_dir.glob("frame*.png"):
        f.unlink()

    env = os.environ.copy()
    env.update(sim.get("env", {}))

    cmd = [
        str(exe),
        "--n-frames",   str(n_frames),
        "--dump-every", "1",
        "--out",        str(pvox_dir),
    ]
    cmd += sim.get("extra_args", [])

    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(BUILD_DIR))
    print(f"  シミュレーション完了 (rc={proc.returncode}, {time.time()-t0:.1f}s)")

    pvox_files = sorted(pvox_dir.glob("frame_*.pvox"))
    print(f"  .pvox: {len(pvox_files)} frames — PNG へレンダリング中 …")

    # レイマーチングは1フレームあたり数百ms〜1秒程度かかるため (numpyのファンシー
    # インデックス由来のオーバーヘッド)、グリッド合成用サムネイル解像度に合わせて
    # 直接描画しステップ数も抑える (300フレーム×2シムで数分程度に収める)。
    mode          = sim.get("render_mode", "volume")
    render_kwargs = sim.get("render_kwargs", {})
    frames = []
    for i, pvox_path in enumerate(pvox_files):
        vox = pyro_raymarch.load_pvox(str(pvox_path))
        if mode == "heatmap":
            img = pyro_raymarch.render_heatmap(
                vox, render_kwargs.get("axis", "z"), THUMB_W, THUMB_H, 32,
                render_kwargs.get("threshold", 1.0), render_kwargs.get("speed_max", 6.0),
                cow_color=(0.55, 0.42, 0.30), background=(0.04, 0.04, 0.06))
        else:
            img = pyro_raymarch.render(
                vox, render_kwargs.get("axis", "z"), THUMB_W, THUMB_H, 32,
                render_kwargs.get("absorption", 2.0), render_kwargs.get("exposure", 1.5))
        out_png = sim_dir / f"frame{i + 1:04d}.png"
        Image.fromarray(img, mode="RGB").save(out_png)
        frames.append(str(out_png))
        if (i + 1) % 30 == 0:
            print(f"    render {i + 1}/{len(pvox_files)}")

    print(f"  レンダリング完了: {len(frames)} frames")
    return frames


def _resolve_commit_sha(cli_sha: str) -> str:
    if cli_sha:
        return cli_sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).parent),
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def write_perf_json(path: Path, n_frames: int, commit_sha: str, timing: dict):
    """各シムの実行時間 (1フレームあたり ms) を JSON で書き出す。
    FPS ではなく ms_per_frame を主指標とする (値が小さいほど速い)。"""
    sims = {}
    for sim in SIMS:
        if sim["exe"] is None:
            continue
        entry = timing.get(sim["id"])
        if entry is None:
            continue
        frames, elapsed_s = entry
        if frames == 0:
            continue
        sims[sim["id"]] = {
            "title": sim["title"],
            "frames": frames,
            "elapsed_s": round(elapsed_s, 3),
            "ms_per_frame": round(elapsed_s / frames * 1000.0, 3),
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit_sha,
        "frames": n_frames,
        "sims": sims,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  perf JSON 出力: {path}")


def encode_video(frame_dir: Path, out_path: Path, fps: int):
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i",         str(frame_dir / "frame%04d.png"),
            "-vcodec",    "libx264",
            "-pix_fmt",   "yuv420p",
            "-preset",    "fast",
            "-vf",        "scale=1920:-2",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        print(f"  動画保存: {out_path}")
    else:
        try:
            import imageio
            writer = imageio.get_writer(str(out_path), fps=fps)
            for p in sorted(frame_dir.glob("frame*.png")):
                writer.append_data(imageio.imread(str(p)))
            writer.close()
            print(f"  動画保存 (imageio): {out_path}")
        except ImportError:
            print("  [warn] ffmpeg / imageio が未インストールです。")
            print(f"  PNG フレーム: {frame_dir}")


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="全シミュレーション キャプチャ → 1本の動画生成")
    parser.add_argument("--frames", type=int, default=N_FRAMES,
                        help=f"各シムのフレーム数 (default: {N_FRAMES})")
    parser.add_argument("--fps",    type=int, default=VIDEO_FPS,
                        help=f"出力動画 FPS (default: {VIDEO_FPS})")
    parser.add_argument("--perf-json", type=str, default=None,
                        help="各シムの実行時間 (ms/frame) を書き出す JSON ファイルパス")
    parser.add_argument("--commit-sha", type=str, default="",
                        help="perf-json に埋め込む commit sha (未指定なら git rev-parse HEAD)")
    cli = parser.parse_args()

    n_frames  = cli.frames
    video_fps = cli.fps

    print("=== Simulation Capture Pipeline ===")
    print(f"  N_FRAMES={n_frames}  VIDEO_FPS={video_fps}  THUMB={THUMB_W}×{THUMB_H}")
    print(f"  Grid={GRID_COLS}×{GRID_ROWS}  Canvas={GRID_COLS*THUMB_W}×{GRID_ROWS*(THUMB_H+52)}")
    print(f"  出力先: {OUT_DIR}")

    # Step 1: 各シムを順次実行してフレームを収集
    sim_frames: dict = {}
    fps_map:    dict = {}
    timing:     dict = {}  # id -> (n_frames_captured, elapsed_s)
    for sim in SIMS:
        if sim["exe"] is None:
            continue
        t0      = time.time()
        frames  = run_pyro_sim(sim, n_frames) if sim.get("kind") == "pyro" else run_sim(sim, n_frames)
        elapsed = time.time() - t0
        sim_frames[sim["id"]] = frames
        if frames:
            fps_map[sim["id"]] = f"RealFPS: {len(frames) / elapsed:.1f}"
            timing[sim["id"]]  = (len(frames), elapsed)

    if cli.perf_json:
        write_perf_json(Path(cli.perf_json), n_frames, _resolve_commit_sha(cli.commit_sha), timing)

    max_frames = max((len(f) for f in sim_frames.values()), default=0)
    if max_frames == 0:
        print("\n[ERROR] どのシムからもフレームを取得できませんでした。")
        sys.exit(1)

    # Step 2: グリッドフレームを合成
    grid_dir = OUT_DIR / "grid_frames"
    if grid_dir.exists():
        shutil.rmtree(grid_dir)
    grid_dir.mkdir()

    print(f"\n=== {max_frames} フレームをグリッド合成中 ===")
    for i in range(max_frames):
        canvas  = make_grid_frame(i, sim_frames, fps_map)
        out_png = grid_dir / f"frame{i + 1:04d}.png"
        canvas.save(out_png)
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{max_frames}")

    # Step 3: 動画エンコード
    # (ffmpeg/imageio が使えない環境でも、perf-json 出力や計測結果に影響しないよう
    #  失敗しても処理を継続する)
    print("\n=== 動画エンコード ===")
    video_path = OUT_DIR / "simulation_results.mp4"
    try:
        encode_video(grid_dir, video_path, video_fps)
    except Exception as e:
        print(f"  [warn] 動画エンコードに失敗しました (フレーム/perf-json は生成済み): {e}")

    print("\n=== 完了 ===")
    print(f"  出力: {video_path}")


if __name__ == "__main__":
    main()
