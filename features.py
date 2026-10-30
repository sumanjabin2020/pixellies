"""Step 1: feature extraction. Run `python features.py` to build features.csv."""
import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

SIZE = 256
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def prepare(img):
    """Same treatment for every image: shrink so the SHORTER side is 256 (area averaging),
    then take the centre 256x256. Smaller images are enlarged instead."""
    h, w = img.shape[:2]
    scale = SIZE / min(h, w)
    if scale != 1:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        img = cv2.resize(img, (max(SIZE, round(w * scale)), max(SIZE, round(h * scale))), interpolation=interp)
    h, w = img.shape[:2]
    t, l = (h - SIZE) // 2, (w - SIZE) // 2
    return img[t:t + SIZE, l:l + SIZE].copy()


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"could not read {path}")
    return prepare(img)


def color_features(img):
    f = {}
    for i, n in enumerate("bgr"):
        ch = img[:, :, i].astype(np.float32) / 255.0
        m, s = ch.mean(), ch.std()
        hist, _ = np.histogram(ch, bins=32, range=(0, 1))
        p = hist / hist.sum(); p = p[p > 0]
        f.update({f"{n}_mean": m, f"{n}_std": s, f"{n}_skew": ((ch - m) ** 3).mean() / (s ** 3 + 1e-8),
                  f"{n}_entropy": -(p * np.log2(p)).sum()})
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
    f["sat_mean"], f["sat_std"] = sat.mean(), sat.std()
    return f


def edge_features(gray):
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    g = np.hypot(gx, gy)
    return {"edge_density": (cv2.Canny(gray, 100, 200) > 0).mean(),
            "laplacian_var": cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0,
            "grad_mean": g.mean(), "grad_std": g.std()}


def noise_features(img, gray):
    res = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    res2 = gray.astype(np.float32) - cv2.GaussianBlur(gray, (5, 5), 1.0).astype(np.float32)
    f = {"noise_std": res.std(), "noise_abs_mean": np.abs(res).mean(),
         "noise_kurt": kurtosis(res.ravel()), "noise_skew": skew(res.ravel()),
         "noise2_std": res2.std(), "noise2_kurt": kurtosis(res2.ravel())}
    chans = [img[:, :, i].astype(np.float32) - cv2.medianBlur(img[:, :, i], 3).astype(np.float32) for i in range(3)]
    for i, n in enumerate("bgr"):
        f[f"noise_{n}_std"] = chans[i].std()
    for (a, b), n in (((0, 1), "bg"), ((1, 2), "gr"), ((0, 2), "br")):   # colour-channel noise correlation
        f[f"noise_corr_{n}"] = np.corrcoef(chans[a].ravel(), chans[b].ravel())[0, 1]
    patches = res.reshape(8, 32, 8, 32).std(axis=(1, 3)).ravel()          # noise level per 32x32 block
    f.update({"patch_noise_cv": patches.std() / (patches.mean() + 1e-8),
              "patch_noise_min": patches.min(), "patch_noise_max": patches.max()})
    return f, res


def _fft_mag(a):
    win = np.outer(np.hanning(SIZE), np.hanning(SIZE))
    return np.abs(np.fft.fftshift(np.fft.fft2((a - a.mean()) * win)))


def frequency_features(gray, res):
    mag = _fft_mag(gray.astype(np.float32) / 255.0)
    logm = np.log1p(mag)
    y, x = np.indices(mag.shape)
    r = np.hypot(y - SIZE // 2, x - SIZE // 2)
    rn = r / (SIZE // 2)
    f = {}
    for b in range(16):
        f[f"fft_band_{b}"] = logm[(rn >= b / 16) & (rn < (b + 1) / 16)].mean()
    high = logm[(rn >= 0.5) & (rn < 1.0)]
    f["fft_high_ratio"] = high.sum() / logm.sum()
    f["fft_high_peak"] = high.max() / (high.mean() + 1e-8)
    ri = r.astype(int).ravel()                                            # power-law slope of the spectrum
    prof = np.bincount(ri, weights=(mag ** 2).ravel()) / np.maximum(np.bincount(ri), 1)
    k = np.arange(3, 127)
    f["fft_slope"] = np.polyfit(np.log10(k), np.log10(prof[k] + 1e-12), 1)[0]
    ang = ((np.arctan2(y - SIZE // 2, x - SIZE // 2) + np.pi) / (2 * np.pi) * 16).astype(int) % 16
    m = rn >= 0.4
    a = np.bincount(ang[m], weights=logm[m]) / np.bincount(ang[m])
    f["fft_aniso"] = a.std() / a.mean()
    rm = np.log1p(_fft_mag(res))                                          # spectrum of the noise residual
    rh = rm[rn >= 0.5]
    f["resfft_peak"] = rh.max() / (rh.mean() + 1e-8)
    f["resfft_high_ratio"] = rh.sum() / rm.sum()
    return f


def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = color_features(img)
    f.update(edge_features(gray))
    nf, res = noise_features(img, gray)
    f.update(nf)
    f.update(frequency_features(gray, res))
    return f


def build_csv(data_dir="data", out_file="features.csv"):
    rows = []
    for label, folder in [(0, "real"), (1, "ai")]:
        path = os.path.join(data_dir, folder)
        files = [x for x in sorted(os.listdir(path)) if x.lower().endswith(EXTENSIONS)]
        print(f"{folder}: {len(files)} images")
        for i, name in enumerate(files):
            try:
                f = extract_features(load_image(os.path.join(path, name)))
            except ValueError as e:
                print("skipped:", e); continue
            f["label"], f["file"] = label, name
            rows.append(f)
            if (i + 1) % 500 == 0:
                print(f"  {folder}: {i + 1}/{len(files)}")
    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    print(f"saved {len(df)} rows, {df.shape[1] - 2} features -> {out_file}")


if __name__ == "__main__":
    build_csv()
