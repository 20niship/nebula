"""nebula の .nbp 粒子フレーム (NbpDump.h) を読む。"""
import struct

import numpy as np


def load_nbp(path):
    """(pos[n,3] float32 [m], speed[n] float32 [m/s], lo, hi) を返す。"""
    with open(path, "rb") as f:
        assert f.read(4) == b"NBP1", path
        (n,) = struct.unpack("<I", f.read(4))
        lo = np.array(struct.unpack("<3f", f.read(12)), np.float32)
        hi = np.array(struct.unpack("<3f", f.read(12)), np.float32)
        (smax,) = struct.unpack("<f", f.read(4))
        q = np.frombuffer(f.read(n * 6), np.uint16).reshape(n, 3)
        s = np.frombuffer(f.read(n), np.uint8)
    pos = lo + q.astype(np.float32) * ((hi - lo) / 65535.0)
    return pos, s.astype(np.float32) * (smax / 255.0), lo, hi


def density_grid(pos, lo, hi, voxel, sigma=1.5):
    """粒子数ヒストグラムをガウシアン平滑した密度場 (nx,ny,nz) を返す。"""
    from scipy.ndimage import gaussian_filter

    shape = np.ceil((hi - lo) / voxel).astype(int) + 1
    idx = np.clip(((pos - lo) / voxel).astype(np.int64), 0, shape - 1)
    flat = (idx[:, 0] * shape[1] + idx[:, 1]) * shape[2] + idx[:, 2]
    g = np.bincount(flat, minlength=int(shape.prod())).reshape(shape).astype(np.float32)
    return gaussian_filter(g, sigma)


def surface_mesh(pos, lo, hi, voxel, sigma=1.5, level=0.5):
    """密度場の等値面 (marching cubes) を (verts[m,3], faces[k,3]) で返す。level は内部密度に対する比。"""
    from skimage.measure import marching_cubes

    lo = np.maximum(lo, pos.min(0) - 4 * voxel)  # 粒子の AABB に切り詰める (雪崩のように領域の一部にしか粒子が無い場合に高速)
    hi = np.minimum(hi, pos.max(0) + 4 * voxel)
    g = density_grid(pos, lo, hi, voxel, sigma)
    inner = np.percentile(g[g > 1e-3], 90)
    v, f, _, _ = marching_cubes(np.pad(g, 1), level * inner, spacing=(voxel,) * 3)
    return (v + lo - voxel).astype(np.float32), f.astype(np.int32)
