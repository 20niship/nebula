# 国土地理院 dem_png (z14) を 4x4 取得して標高 [m] の npy に結合。ローカル専用。
import io, urllib.request, numpy as np
from PIL import Image
Z, X0, Y0, N = 14, 14463, 6392, 4
H = np.zeros((N*256, N*256), np.float32)
for j in range(N):
    for i in range(N):
        u = f"https://cyberjapandata.gsi.go.jp/xyz/dem_png/{Z}/{X0+i}/{Y0+j}.png"
        a = np.array(Image.open(io.BytesIO(urllib.request.urlopen(u).read())).convert("RGB"), np.int64)
        v = a[..., 0]*65536 + a[..., 1]*256 + a[..., 2]
        h = np.where(v == 2**23, np.nan, np.where(v < 2**23, v, v - 2**24) * 0.01)
        H[j*256:(j+1)*256, i*256:(i+1)*256] = h
np.save("assets_local/dem.npy", H)
print(H.shape, np.nanmin(H), np.nanmax(H), np.isnan(H).sum())
