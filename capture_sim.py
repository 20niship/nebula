#!/opt/homebrew/bin/python3
"""
テストケース別シミュレーション実行 → グリッド動画生成スクリプト

各シミュレーションを順次実行し、フレームごとにPPMとして保存。
全シム完了後、対応フレームを3×3グリッドに合成してMP4動画を書き出す。

テストケース一覧:
  TC1: 流体ダムブレイク（ボリューム上半分から落下）   fluid_pbf
  TC2: 流体 + OBJメッシュ境界                         fluid_pbf + SIM_BOUNDARY_OBJ
  TC3: 高粘性ゼリー流体                               fluid_pbf + SIM_VISCOSITY_C=0.5
  TC4: 布シミュレーション（2隅固定で垂れる）          cloth_3d
  TC5: 布2枚 + ボックスSDF・布-布衝突                 cloth_scene --scene 5
  TC6: 煙パーティクルシミュレーション                  smoke
  TC7: 4隅アニメーションピン・布ねじれ                cloth_scene --scene 7
  TC9: 楕円水たまり + 移動円柱吸収ポート              fluid_absorb
"""

import subprocess
import time
import os
import shutil
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

BUILD_DIR = Path(__file__).parent / "build"
OUT_DIR = Path(__file__).parent / "sim_captures"
OUT_DIR.mkdir(exist_ok=True)

# キャプチャ設定
N_FRAMES = 600  # 各シムの保存フレーム数
SHOT_SECS = 30.0  # 何シム秒分を録る
VIDEO_FPS = 60  # 出力動画FPS
THUMB_W = 640
THUMB_H = 360
GRID_COLS = 3
GRID_ROWS = 3  # 3×3 = 9セル

# テストケース定義
SIMS = [
    {
        "id": "tc1",
        "exe": "fluid_pbf",
        "title": "TC1: Dam Break",
        "env": {},
        "extra_args": ["--scenario", "dam-break"],
        "params": "N~110K | dam-break (left-top half)",
    },
    {
        "id": "tc2",
        "exe": "fluid_pbf",
        "title": "TC2: Moving Source Flow",
        "env": {},
        "extra_args": [
            "--scenario",
            "source-flow",
            "--world-size",
            "40",
            "--nx",
            "192",
            "--ny",
            "3",
            "--nz",
            "192",
            "--grid-res",
            "128",
        ],
        "params": "N~110K | moving AABB source | world=40 | source travels left→right",
    },
    {
        "id": "tc3",
        "exe": "fluid_pbf",
        "title": "TC3: Jelly (High Viscosity)",
        "env": {"SIM_VISCOSITY_C": "0.5"},
        "params": "N~110K | rho0=2097 | visc=0.50",
    },
    {
        "id": "tc4",
        "exe": "cloth_3d",
        "title": "TC4: Cloth Drape",
        "env": {},
        "params": "verts=16384 | 2-corner pinned",
    },
    {
        "id": "tc5",
        "exe": "cloth_scene",
        "title": "TC5: Two-Cloth Collision",
        "env": {},
        "extra_args": ["--scene", "5", "--cloth-n", "32", "--world-size", "15"],
        "params": "verts=2×1024 | top-row pinned | 45° rotated | cloth-cloth SDF",
    },
    {
        "id": "tc6",
        "exe": "smoke",
        "title": "TC6: Smoke",
        "env": {},
        "extra_args": ["--max-particles", "4096", "--pps", "48"],
        "params": "max=4096 | buoyancy+vorticity | sphere emitter",
    },
    {
        "id": "tc7",
        "exe": "cloth_scene",
        "title": "TC7: Cloth Twist",
        "env": {},
        "extra_args": ["--scene", "7", "--cloth-n", "32"],
        "params": "verts=1024 | 4-corner animated pins | ±20°/s twist",
    },
    {
        "id": "tc8",
        "exe": "screw_fluid",
        "title": "TC8: Rotating Screw",
        "env": {},
        "extra_args": ["--ang-vel", "2.0", "--viscosity", "0.01"],
        "params": "N~27K | screw ω=2rad/s | visc=0.01",
    },
    {
        "id": "tc9",
        "exe": "fluid_absorb",
        "title": "TC9: Fluid Absorb",
        "env": {},
        "extra_args": [],
        "params": "N~1.4K | ellipse puddle a=5m b=3m | moving cylinder absorber r=1.2m rate=0.8",
    },
]


def _load_font(size):
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
    bg_h = 14 + 16 + 16 + 8
    bg_w = max(len(l) * 7 + 12 for l in lines) + 8
    draw.rectangle([x, y, x + bg_w, y + bg_h], fill=(0, 0, 0, 160))
    cy = y + 4
    for line, fnt in zip(lines, fonts):
        draw.text((x + 6, cy), line, fill=(255, 255, 255), font=fnt)
        cy += 15


def make_grid_frame(frame_idx, sim_frames, fps_map):
    label_h = 52
    canvas_w = GRID_COLS * THUMB_W
    canvas_h = GRID_ROWS * (THUMB_H + label_h)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas, "RGBA")

    font_big = _load_font(14)
    font_small = _load_font(12)

    for cell_i, sim in enumerate(SIMS):
        if cell_i >= GRID_COLS * GRID_ROWS:
            break
        col = cell_i % GRID_COLS
        row = cell_i // GRID_COLS
        cx = col * THUMB_W
        cy = row * (THUMB_H + label_h)

        if sim["exe"] is None:
            na = Image.new("RGB", (THUMB_W, THUMB_H), (30, 30, 40))
            na_draw = ImageDraw.Draw(na)
            na_draw.text((20, THUMB_H // 2 - 20), sim["title"], fill=(120, 120, 140), font=font_big)
            na_draw.text((20, THUMB_H // 2 + 2), sim["params"], fill=(80, 80, 100), font=font_small)
            canvas.paste(na, (cx, cy))
        else:
            frames = sim_frames.get(sim["id"], [])
            if frames and frame_idx < len(frames):
                try:
                    img = Image.open(frames[frame_idx])
                    img.load()
                    thumb = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
                    canvas.paste(thumb, (cx, cy))
                except Exception as e:
                    print(f"  [warn] frame load failed: {e}")
                    canvas.paste(Image.new("RGB", (THUMB_W, THUMB_H), (40, 20, 20)), (cx, cy))
            else:
                canvas.paste(Image.new("RGB", (THUMB_W, THUMB_H), (30, 30, 30)), (cx, cy))

            fps_str = fps_map.get(sim["id"], "FPS: --")
            _draw_overlay(draw, cx, cy, sim["title"], sim["params"], fps_str, font_big, font_small)

    return canvas


def run_sim(sim):
    exe = BUILD_DIR / sim["exe"]
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
    env.update(sim["env"])

    # スクリーンショット引数は CLI で直接渡す（env var 読み込みより確実）
    cmd = [
        str(exe),
        "--n-shots",
        str(N_FRAMES),
        "--screenshot-dir",
        str(ppm_dir),
    ]
    # シナリオ固有の追加引数
    cmd += sim.get("extra_args", [])
    # SIM_BOUNDARY_OBJ が指定されている場合は --boundary-obj へ変換
    if "SIM_BOUNDARY_OBJ" in sim["env"]:
        cmd += ["--boundary-obj", sim["env"]["SIM_BOUNDARY_OBJ"]]

    proc = subprocess.Popen(cmd, env=env, cwd=str(BUILD_DIR))
    print(f"  PID {proc.pid} — waiting for process to finish ({N_FRAMES} frames)...")

    # シミュレーションは全フレーム保存後に自動終了するのでそのまま待つ
    while proc.poll() is None:
        frames = sorted(ppm_dir.glob("frame*.ppm"))
        print(f"  [{int(time.time() % 10000)}] {len(frames)}/{N_FRAMES} frames", end="\r", flush=True)
        time.sleep(2)
    print()
    print(f"  Process exited (rc={proc.returncode})")

    frames = sorted(ppm_dir.glob("frame*.ppm"))
    print(f"  Captured: {len(frames)} frames")
    return [str(p) for p in frames]


def encode_video(frame_dir, out_path):
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(VIDEO_FPS),
            "-i",
            str(frame_dir / "frame%04d.png"),
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-vf",
            "scale=1920:-1",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        print(f"  Video saved: {out_path}")
    else:
        try:
            import imageio

            writer = imageio.get_writer(str(out_path), fps=VIDEO_FPS)
            for p in sorted(frame_dir.glob("frame*.png")):
                writer.append_data(imageio.imread(str(p)))
            writer.close()
            print(f"  Video saved (imageio): {out_path}")
        except ImportError:
            print("  [warn] ffmpeg not found and imageio not installed.")
            print(f"  PNG frames are available in: {frame_dir}")


def main():
    print("=== Simulation Capture Pipeline ===")
    print(f"  N_FRAMES={N_FRAMES}, SHOT_SECS={SHOT_SECS}, VIDEO_FPS={VIDEO_FPS}")

    # Step 1: 各シムを実行してフレームを収集
    sim_frames = {}
    fps_map = {}
    for sim in SIMS:
        if sim["exe"] is None:
            continue
        t0 = time.time()
        frames = run_sim(sim)
        elapsed = time.time() - t0
        sim_frames[sim["id"]] = frames
        if frames:
            fps = len(frames) / elapsed
            fps_map[sim["id"]] = f"FPS: {fps:.1f}"

    max_frames = max((len(f) for f in sim_frames.values()), default=0)
    if max_frames == 0:
        print("\n[ERROR] No frames captured from any simulation.")
        sys.exit(1)

    # Step 2: グリッドフレームを生成
    grid_dir = OUT_DIR / "grid_frames"
    if grid_dir.exists():
        shutil.rmtree(grid_dir)
    grid_dir.mkdir()

    print(f"\n=== Compositing {max_frames} grid frames ===")
    for i in range(max_frames):
        canvas = make_grid_frame(i, sim_frames, fps_map)
        out_png = grid_dir / f"frame{i + 1:04d}.png"
        canvas.save(out_png)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{max_frames}")

    # Step 3: 動画エンコード
    print("\n=== Encoding video ===")
    video_path = OUT_DIR / "simulation_results.mp4"
    encode_video(grid_dir, video_path)

    print("\n=== Done ===")
    print(f"  Output: {video_path}")


if __name__ == "__main__":
    main()
