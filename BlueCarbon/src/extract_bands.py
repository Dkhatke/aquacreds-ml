# src/extract_bands.py

import os
import glob
import rasterio
import numpy as np
import pandas as pd

# ======================================================
# CONFIG
# ======================================================

ROOT = r"C:\AquaCreds new model\BlueCarbon\Sentinel Tiles\Dec"

BANDS = {
    "B2_Blue": "*B02_10m.jp2",
    "B3_Green": "*B03_10m.jp2",
    "B4_Red": "*B04_10m.jp2",
    "B8_NIR": "*B08_10m.jp2",
    "B11_SWIR1": "*B11_20m.jp2",
    "B12_SWIR2": "*B12_20m.jp2"
}

def load_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        arr[arr == 0] = np.nan
        return arr

def compute_indices(B2, B3, B4, B8, B11, B12):
    eps = 1e-10

    return {
        "NDVI": np.nanmean((B8 - B4) / (B8 + B4 + eps)),
        "EVI": np.nanmean(2.5 * ((B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + eps))),
        "SAVI": np.nanmean(1.5 * ((B8 - B4) / (B8 + B4 + 0.5 + eps))),
        "NDWI": np.nanmean((B3 - B8) / (B3 + B8 + eps)),
        "MNDWI": np.nanmean((B3 - B11) / (B3 + B11 + eps)),
        "MSI": np.nanmean(B11 / (B8 + eps)),
        "NDMI": np.nanmean((B8 - B11) / (B8 + B11 + eps)),
    }

def extract_from_safe(safe_path):

    extracted = {}

    for key, pattern in BANDS.items():

        # ensure only JP2 files are loaded
        files = [
            f for f in glob.glob(os.path.join(safe_path, "**", pattern), recursive=True)
            if f.endswith(".jp2")
        ]

        if not files:
            print(f"⚠ Missing band {key} in {safe_path}")
            extracted[key] = np.nan
            continue

        band = load_band(files[0])
        extracted[key] = np.nanmean(band)

    idx = compute_indices(
        extracted["B2_Blue"],
        extracted["B3_Green"],
        extracted["B4_Red"],
        extracted["B8_NIR"],
        extracted["B11_SWIR1"],
        extracted["B12_SWIR2"],
    )

    extracted.update(idx)
    return extracted

# ======================================================
# LOOP THROUGH SAFE FOLDERS
# ======================================================

rows = []

safe_folders = glob.glob(os.path.join(ROOT, "**", "*.SAFE"), recursive=True)
print(f"Found {len(safe_folders)} SAFE tiles.")

for safe in safe_folders:
    tile_id = os.path.basename(safe)
    print(f"Processing → {tile_id}")
    data = extract_from_safe(safe)
    data["Tile_ID"] = tile_id
    rows.append(data)

df = pd.DataFrame(rows)
df.to_csv("sentinel_band_values.csv", index=False)

print("\n✓ Saved: sentinel_band_values.csv")
print("✓ Extraction complete")
