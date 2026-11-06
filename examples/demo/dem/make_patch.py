# dem.npy (GSI z14, 7.66 m/px) の一部を 0.1 m 格子の模型スケール地形にして terrain 形式 (nx,nz,step,float32[nz][nx]) で保存。ローカル専用。
import sys, numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
H = np.load("assets_local/dem.npy"); px = 7.66
x0, y0, W, D = [int(v) for v in sys.argv[1:5]]
s = 0.1326                       # 縮尺 (縦横同率なので勾配は保たれる)
step = 0.1
nx, nz = int(W * px * s / step), int(D * px * s / step)
u = x0 + np.arange(nx) * step / (px * s); v = y0 + np.arange(nz) * step / (px * s)
h = map_coordinates(H, np.meshgrid(v, u, indexing="ij"), order=3, mode="nearest")
h = (h - h.min()) * s + 0.4
rng = np.random.default_rng(3)
n = gaussian_filter(rng.standard_normal(h.shape), 12) * 12
h = gaussian_filter(h, 2) + 0.12 * n / n.std()
h = h.astype(np.float32)
with open("assets_local/dem_patch.f32", "wb") as f:
    np.array([nx, nz], np.int32).tofile(f); np.float32(step).tofile(f); h.tofile(f)
g = np.degrees(np.arctan(-np.gradient(h, step, axis=1)))
print(nx, nz, "size", nx*step, nz*step, "hmax", h.max())
for a in range(0, nx, nx // 8): print(a * step, "slope", g[:, a:a + nx // 8].mean().round(1), "h", h[:, a].mean().round(1))
