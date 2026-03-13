# src/multi_month_extract_bands.py
"""
Multi-month Sentinel-2 Band Extractor
-------------------------------------
Automatically scans:
- Sept
- Oct
- Nov
- Dec

Extracts:
- B02, B03, B04, B08, B11, B12
Computes:
- NDVI, EVI, SAVI, NDWI, MNDWI, MSI, NDMI

Outputs:
- sentinel_band_values_master.csv
"""

import os
import glob
import rasterio
import numpy as np
import pandas as pd

# ======================================================
# CONFIG
# ======================================================
BASE_ROOT = r"C:\AquaCreds new model\BlueCarbon\Sentinel Tiles"

MONTHS = ["Dec", "Nov", "Oct", "Sept"]

BANDS = {
    "B2_Blue": "*B02_10m.jp2",
    "B3_Green": "*B03_10m.jp2",
    "B4_Red": "*B04_10m.jp2",
    "B8_NIR": "*B08_10m.jp2",
    "B11_SWIR1": "*B11_20m.jp2",
    "B12_SWIR2": "*B12_20m.jp2"
}


# ======================================================
# Band loader
# ======================================================
def load_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        arr[arr == 0] = np.nan
        return arr


# ======================================================
# Index computation
# ======================================================
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


# ======================================================
# Extract from one SAFE folder
# ======================================================
def extract_from_safe(safe_path):

    extracted = {}

    for band_name, pattern in BANDS.items():

        # JP2 only (avoid thumbnails)
        files = [
            f for f in glob.glob(os.path.join(safe_path, "**", pattern), recursive=True)
            if f.endswith(".jp2")
        ]

        if not files:
            print(f"⚠ Missing {band_name} in {safe_path}")
            extracted[band_name] = np.nan
            continue

        band = load_band(files[0])
        extracted[band_name] = np.nanmean(band)

    # Compute indices
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
# MAIN MULTI-MONTH EXTRACTOR
# ======================================================
all_rows = []

print("===========================================")
print("   Multi-Month Sentinel-2 Extraction")
print("===========================================")

for month in MONTHS:
    month_path = os.path.join(BASE_ROOT, month)

    print(f"\n📁 Checking month: {month}")

    safe_folders = glob.glob(os.path.join(month_path, "**", "*.SAFE"), recursive=True)

    print(f"   → Found {len(safe_folders)} SAFE tiles")

    for safe in safe_folders:
        tile_id = os.path.basename(safe)
        print(f"      🔍 Extracting: {tile_id}")

        data = extract_from_safe(safe)
        data["Tile_ID"] = tile_id
        data["Month"] = month

        all_rows.append(data)


# Save combined CSV
df = pd.DataFrame(all_rows)

OUT_FILE = "sentinel_band_values_master.csv"
df.to_csv(OUT_FILE, index=False)

print("\n===========================================")
print(f"✓ COMPLETED — Saved: {OUT_FILE}")
print("===========================================")
