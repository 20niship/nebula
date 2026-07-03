#!/opt/homebrew/bin/python3
"""
pyro_raymarch.py — pyro_basic が出力する .pvox ボクセルダンプを
emission-absorption 方式でレイマーチングし、PNG として書き出す簡易ビューア。

C++側 (PyroEngine) はボクセルデータの出力のみを行い、可視化はこのスクリプトで行う。

.pvox フォーマット (PyroEngine::dumpFrame と対応):
  header: magic"PVX1"(4B), nx,ny,nz(u32×3), worldSize(f32), simTime(f32),
          numChannels(u32), [name(16B ascii固定長), components(u32)]×numChannels
  body:   float32 データ (チャンネル毎、線形 x+y*nx+z*nx*ny 順)

使い方:
  python3 tools/pyro_raymarch.py frame_0100.pvox --out out.png
  python3 tools/pyro_raymarch.py sim_captures/pyro/*.pvox --out-dir renders/
"""

import argparse
import glob
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def load_pvox(path):
    with open(path, "rb") as f:
        data = f.read()

    magic = data[0:4]
    if magic != b"PVX1":
        raise ValueError(f"{path}: unknown magic {magic!r}")

    nx, ny, nz = struct.unpack_from("<3I", data, 4)
    world_size, sim_time = struct.unpack_from("<2f", data, 16)
    (num_channels,) = struct.unpack_from("<I", data, 24)

    off = 28
    channels = {}
    channel_order = []
    for _ in range(num_channels):
        name = data[off:off + 16].split(b"\x00")[0].decode()
        off += 16
        (comp,) = struct.unpack_from("<I", data, off)
        off += 4
        channel_order.append((name, comp))

    for name, comp in channel_order:
        n = nx * ny * nz * comp
        arr = np.frombuffer(data, dtype="<f4", count=n, offset=off).copy()
        off += n * 4
        if comp == 1:
            arr = arr.reshape(nz, ny, nx).transpose(2, 1, 0)  # -> (nx,ny,nz)
        else:
            arr = arr.reshape(nz, ny, nx, comp).transpose(2, 1, 0, 3)  # -> (nx,ny,nz,comp)
        channels[name] = arr

    return {
        "dims": (nx, ny, nz),
        "world_size": world_size,
        "sim_time": sim_time,
        **channels,
    }


def sample_trilinear(field, gx, gy, gz):
    """field: (nx,ny,nz[,c]).  gx/gy/gz: 同形状のグリッド座標 (0..dim-1 の連続値)。"""
    nx, ny, nz = field.shape[0], field.shape[1], field.shape[2]

    def clampi(v, n):
        return np.clip(v, 0, n - 1)

    x0 = np.floor(gx).astype(np.int32); x1 = clampi(x0 + 1, nx); x0 = clampi(x0, nx)
    y0 = np.floor(gy).astype(np.int32); y1 = clampi(y0 + 1, ny); y0 = clampi(y0, ny)
    z0 = np.floor(gz).astype(np.int32); z1 = clampi(z0 + 1, nz); z0 = clampi(z0, nz)
    fx = (gx - np.floor(gx))[..., None] if field.ndim == 4 else (gx - np.floor(gx))
    fy = (gy - np.floor(gy))[..., None] if field.ndim == 4 else (gy - np.floor(gy))
    fz = (gz - np.floor(gz))[..., None] if field.ndim == 4 else (gz - np.floor(gz))

    c000 = field[x0, y0, z0]; c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]; c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]; c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]; c111 = field[x1, y1, z1]

    x00 = c000 * (1 - fx) + c100 * fx
    x10 = c010 * (1 - fx) + c110 * fx
    x01 = c001 * (1 - fx) + c101 * fx
    x11 = c011 * (1 - fx) + c111 * fx
    y0v = x00 * (1 - fy) + x10 * fy
    y1v = x01 * (1 - fy) + x11 * fy
    return y0v * (1 - fz) + y1v * fz


def setup_rays(vox, axis, width, height):
    """axis: レイの進行方向 ("x","y","z"); 画像平面は残り2軸。
    戻り値: (dims dict, grid_pos テンプレート (h,w,3), ray_axis index, ds_world)"""
    nx, ny, nz = vox["dims"]
    world_size = vox["world_size"]
    dims = {"x": nx, "y": ny, "z": nz}
    axes = {
        "z": (0, 1, 2),  # image = (X,Y), march over Z (front view)
        "x": (2, 1, 0),  # image = (Z,Y), march over X (side view)
        "y": (0, 2, 1),  # image = (X,Z), march over Y (top view)
    }
    ia, ib, ray_axis = axes[axis]
    axis_names = ["x", "y", "z"]

    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    uu, vv = np.meshgrid(u, 1.0 - v, indexing="xy")  # v反転: 画像上が+方向

    grid_pos = np.zeros((height, width, 3), dtype=np.float32)
    grid_pos[..., ia] = uu * (dims[axis_names[ia]] - 1)
    grid_pos[..., ib] = vv * (dims[axis_names[ib]] - 1)

    return dims, grid_pos, ray_axis, world_size


def blackbody_color(temp, flame):
    """温度・発光量から擬似的な炎色 (黒→赤→橙→黄→白) を作る。temp/flame は正規化前の生値。"""
    t = np.clip(temp / 4.0, 0.0, 1.0)
    r = np.clip(1.5 * t, 0.0, 1.0)
    g = np.clip(1.5 * t - 0.4, 0.0, 1.0)
    b = np.clip(1.8 * t - 1.0, 0.0, 1.0)
    glow = np.clip(flame, 0.0, 3.0) / 3.0
    r = np.clip(r + glow * 0.6, 0.0, 1.0)
    g = np.clip(g + glow * 0.3, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def render(vox, axis, width, height, steps, absorption, exposure):
    """emission-absorption 方式: smoke/fire らしい見た目のボリュームレンダリング。"""
    density = vox["density"]
    temperature = vox["temperature"]
    flame = vox["flame"]

    dims, grid_pos, ray_axis, world_size = setup_rays(vox, axis, width, height)
    axis_names = ["x", "y", "z"]

    color = np.zeros((height, width, 3), dtype=np.float32)
    transmittance = np.ones((height, width), dtype=np.float32)

    ds_world = world_size / steps
    for s in range(steps):
        t = (s + 0.5) / steps
        grid_pos[..., ray_axis] = t * (dims[axis_names[ray_axis]] - 1)

        gx, gy, gz = grid_pos[..., 0], grid_pos[..., 1], grid_pos[..., 2]
        d = sample_trilinear(density, gx, gy, gz)
        temp = sample_trilinear(temperature, gx, gy, gz)
        fl = sample_trilinear(flame, gx, gy, gz)

        sigma_a = absorption * d
        alpha = 1.0 - np.exp(-sigma_a * ds_world)
        emit = (0.85 * d[..., None] * np.array([0.6, 0.6, 0.65])) + blackbody_color(temp, fl) * (temp[..., None] + fl[..., None])

        color += transmittance[..., None] * alpha[..., None] * emit
        transmittance *= (1.0 - alpha)

    # 背景 (簡易アンビエント光): 純吸収性の煙は黒背景では見えないため、
    # 一様な背景光を仮定してシルエットを可視化する。
    background = np.array([0.05, 0.06, 0.09], dtype=np.float32)
    color += transmittance[..., None] * background

    color = 1.0 - np.exp(-exposure * color)  # 簡易トーンマッピング
    img = np.clip(color * 255.0, 0, 255).astype(np.uint8)
    return img


def jet_colormap(t):
    """t: [0,1] の numpy 配列 → (..., 3) RGB。MATLAB 'jet' 風の簡易近似 (matplotlib非依存)。"""
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def render_heatmap(vox, axis, width, height, steps, threshold, speed_max, cow_color, background):
    """CFD 数値解析風の可視化: 流速(velocity)の大きさをヒートマップ色で表示し、
    閾値 (threshold) 未満の領域は透明のままにする。障害物 (sdf<0) は不透明な単色で描画する。"""
    velocity = vox["velocity"]
    has_sdf = "sdf" in vox
    sdf = vox["sdf"] if has_sdf else None

    dims, grid_pos, ray_axis, world_size = setup_rays(vox, axis, width, height)
    axis_names = ["x", "y", "z"]

    color = np.zeros((height, width, 3), dtype=np.float32)
    transmittance = np.ones((height, width), dtype=np.float32)
    cow_color = np.array(cow_color, dtype=np.float32)

    step_alpha = 1.0 - (1e-3) ** (1.0 / steps)  # steps回でほぼ完全不透明になる速度の減衰係数

    for s in range(steps):
        t = (s + 0.5) / steps
        grid_pos[..., ray_axis] = t * (dims[axis_names[ray_axis]] - 1)

        gx, gy, gz = grid_pos[..., 0], grid_pos[..., 1], grid_pos[..., 2]

        if has_sdf:
            sd = sample_trilinear(sdf, gx, gy, gz)
            solid = sd < 0.0
            solid_alpha = solid.astype(np.float32)
            color += transmittance[..., None] * solid_alpha[..., None] * cow_color
            transmittance *= (1.0 - solid_alpha)

        vel = sample_trilinear(velocity, gx, gy, gz)
        speed = np.linalg.norm(vel, axis=-1)

        excess = np.clip((speed - threshold) / max(speed_max - threshold, 1e-6), 0.0, 1.0)
        above = speed > threshold
        if has_sdf:
            above = above & ~solid
        alpha = np.where(above, step_alpha * (0.3 + 0.7 * excess), 0.0)
        heat_color = jet_colormap(np.clip(speed / speed_max, 0.0, 1.0))

        color += transmittance[..., None] * alpha[..., None] * heat_color
        transmittance *= (1.0 - alpha)

    color += transmittance[..., None] * np.array(background, dtype=np.float32)
    img = np.clip(color * 255.0, 0, 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser(description="Ray-march a PyroEngine .pvox voxel dump into a PNG")
    # 注意: "inputs out.png" のような2つの位置引数は argparse の nargs="+" が
    # 貪欲にすべて inputs へ吸収してしまうため使えない。出力は --out/--out-dir で明示する。
    ap.add_argument("inputs", nargs="+", help=".pvox file(s) (glob patterns allowed)")
    ap.add_argument("--out", default=None, help="Output PNG path (single-input mode)")
    ap.add_argument("--out-dir", default=None, help="Output directory (multi-input mode)")
    ap.add_argument("--axis", choices=["x", "y", "z"], default="z", help="View axis (default: z / front view)")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--steps", type=int, default=96, help="Ray march step count")
    ap.add_argument("--mode", choices=["volume", "heatmap"], default="volume",
                    help="volume: smoke/fire emission-absorption render. "
                         "heatmap: CFD-style velocity-magnitude heatmap with threshold cutoff")
    ap.add_argument("--absorption", type=float, default=2.0, help="[volume] Density absorption coefficient")
    ap.add_argument("--exposure", type=float, default=1.5, help="[volume] Tone-mapping exposure")
    ap.add_argument("--threshold", type=float, default=1.0, help="[heatmap] Speed threshold to start rendering")
    ap.add_argument("--speed-max", type=float, default=6.0, help="[heatmap] Speed mapped to the top of the colormap")
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        matches = sorted(glob.glob(pat))
        paths.extend(matches if matches else [pat])

    if len(paths) == 1 and args.out and not args.out_dir:
        out_paths = [args.out]
    else:
        out_dir = Path(args.out_dir or "pyro_renders")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_paths = [str(out_dir / (Path(p).stem + ".png")) for p in paths]

    for path, out_path in zip(paths, out_paths):
        vox = load_pvox(path)
        if args.mode == "heatmap":
            img = render_heatmap(vox, args.axis, args.width, args.height, args.steps,
                                  args.threshold, args.speed_max,
                                  cow_color=(0.55, 0.42, 0.30), background=(0.04, 0.04, 0.06))
        else:
            img = render(vox, args.axis, args.width, args.height, args.steps, args.absorption, args.exposure)
        Image.fromarray(img, mode="RGB").save(out_path)
        print(f"{path}  (t={vox['sim_time']:.3f}s)  ->  {out_path}")


if __name__ == "__main__":
    main()
