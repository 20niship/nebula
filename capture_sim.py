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
  TC-C: MPM Elastic — FLIP (flip_ratio=0.95)                  mpm_elastic
  TC-D: MPM Fountain — FLIP (flip_ratio=0.95)                 mpm_fountain
  TC-E: MPM Mountain Avalanche — Drucker-Prager                mpm_avalanche
  TC-F: MPM マルチマテリアル — 弾性体 + Drucker-Prager 砂      mpm_multimaterial
  TC-G: MPM 雪衝突 — 移動箱SDF (粒子50倍・高速・半サイズ箱)   mpm_snow_impact
  TC-H: MPM 地層崩壊 — 硬岩/弱粘土/緩土 3層                  mpm_geolayer
  TC-I: MPM 砂柱崩壊 — Drucker-Prager 粒状体                  mpm_granular
  TC-J: MPM STL落下 — 球体STL障害物 SDF                       mpm_stl_drop

使い方:
  python3 capture_sim.py [--frames N] [--fps F]
"""

import argparse
import subprocess
import time
import os
import shutil
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

BUILD_DIR = Path(__file__).parent / "build"
OUT_DIR   = Path(__file__).parent / "sim_captures"
OUT_DIR.mkdir(exist_ok=True)

# キャプチャ設定
N_FRAMES  = 300    # 各シムの保存フレーム数
VIDEO_FPS = 60     # 出力動画 FPS
THUMB_W   = 480    # 4列 × 480 = 1920px (ffmpeg scale と一致)
THUMB_H   = 270    # 16:9
GRID_COLS = 4
GRID_ROWS = 5      # 4×5=20 セル; TC1–TC10 + TC-A–TC-J

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
        "id": "tc_flip", "exe": "mpm_elastic",
        "title": "TC-C: MPM Elastic — FLIP",
        "env": {},
        "extra_args": [
            "--nx", "20", "--ny", "20", "--nz", "20",
            "--grid-res", "64", "--substeps", "25", "--flip-ratio", "0.95",
        ],
        "params": "nx=ny=nz=20 | flip=0.95 | 散逸最小",
    },
    {
        "id": "tc_fountain", "exe": "mpm_fountain",
        "title": "TC-D: MPM Fountain — FLIP",
        "env": {},
        "extra_args": [
            "--max-n", "32768", "--emit-n", "512",
            "--grid-res", "64", "--substeps", "25", "--flip-ratio", "0.95",
        ],
        "params": "N≤32768 | emit=512/step | 球コライダー",
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
            "--auto-launch", "1",
        ],
        "params": "N~85K | VON_MISES雪 | 箱速6m/s・半サイズ | 自動衝突",
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
    {
        "id": "tc_granular", "exe": "mpm_granular",
        "title": "TC-I: MPM Granular — Sand Column",
        "env": {},
        "extra_args": [
            "--nx", "8", "--ny", "32", "--nz", "8",
            "--grid-res", "64", "--substeps", "25",
        ],
        "params": "N=2048 | Drucker-Prager砂柱崩壊 | E=50kPa rho=1600",
    },
    {
        "id": "tc_stl", "exe": "mpm_stl_drop",
        "title": "TC-J: MPM STL Drop",
        "env": {},
        "extra_args": [
            "--grid-res", "64", "--substeps", "25",
        ],
        "params": "N~16K | 球体STL SDF障害物 | パーティクル落下",
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
    for sim in SIMS:
        if sim["exe"] is None:
            continue
        t0      = time.time()
        frames  = run_sim(sim, n_frames)
        elapsed = time.time() - t0
        sim_frames[sim["id"]] = frames
        if frames:
            fps_map[sim["id"]] = f"RealFPS: {len(frames) / elapsed:.1f}"

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
    print("\n=== 動画エンコード ===")
    video_path = OUT_DIR / "simulation_results.mp4"
    encode_video(grid_dir, video_path, video_fps)

    print("\n=== 完了 ===")
    print(f"  出力: {video_path}")


if __name__ == "__main__":
    main()
