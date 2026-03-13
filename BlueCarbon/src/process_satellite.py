# src/process_satellite.py
"""
process_satellite.py
---------------------
Extracts all important vegetation & moisture indices from
Sentinel-2 SAFE folders.

Outputs:
- NDVI
- NDWI
- NDMI
- NBR
- EVI
- SAVI

Fully resamples 20m → 10m to maintain spatial alignment.
"""

import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from glob import glob


# ===============================================================
# Safe band loader with resampling
# ===============================================================
def load_and_resample(path, ref_ds):
    """
    Loads a band and resamples it to match a reference dataset (typically 10m band).
    """
    with rasterio.open(path) as ds:
        arr = ds.read(
            out_shape=(
                ds.count,
                ref_ds.height,
                ref_ds.width
            ),
            resampling=Resampling.bilinear
        )[0]

    arr = arr.astype("float32")
    arr[arr == 0] = np.nan
    return arr


def load_band(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype("float32")
        arr[arr == 0] = np.nan
        return arr, ds.transform, ds


# ===============================================================
# Normalized index calculator
# ===============================================================
def norm_index(b1, b2):
    return (b1 - b2) / (b1 + b2 + 1e-10)


# ===============================================================
# MAIN TILE PROCESSOR
# ===============================================================
def process_tile(safe_folder):
    """
    Extracts NDVI, NDWI, NDMI, NBR, EVI, SAVI from a Sentinel-2 SAFE folder.
    Returns dictionary of mean values.
    """

    print(f"\n📡 Processing Sentinel-2 tile: {safe_folder}")

    # -------------------------------
    # Locate Granule folder
    # -------------------------------
    granule_dir = os.path.join(safe_folder, "GRANULE")
    granules = os.listdir(granule_dir)

    if not granules:
        raise RuntimeError("No GRANULE directory found inside SAFE folder.")

    granule = os.path.join(granule_dir, granules[0])

    img10 = os.path.join(granule, "IMG_DATA", "R10m")
    img20 = os.path.join(granule, "IMG_DATA", "R20m")

    # -------------------------------
    # Find band files
    # -------------------------------
    def find(pattern, folder):
        files = glob(os.path.join(folder, pattern))
        return files[0] if files else None

    B04 = find("*B04_10m.jp2", img10)   # red
    B08 = find("*B08_10m.jp2", img10)   # nir
    B03 = find("*B03_10m.jp2", img10)   # green
    B11 = find("*B11_20m.jp2", img20)   # swir1
    B12 = find("*B12_20m.jp2", img20)   # swir2

    required = [("B04", B04), ("B08", B08), ("B03", B03),
                ("B11", B11), ("B12", B12)]

    for name, p in required:
        if p is None:
            raise FileNotFoundError(f"Missing {name} band in: {safe_folder}")

    # -------------------------------
    # Load 10m bands (reference resolution)
    # -------------------------------
    red, transform, ref_ds = load_band(B04)
    nir, _, _ = load_band(B08)
    green, _, _ = load_band(B03)

    # -------------------------------
    # Resample 20m → 10m
    # -------------------------------
    swir1 = load_and_resample(B11, ref_ds)
    swir2 = load_and_resample(B12, ref_ds)

    # -------------------------------
    # Compute spectral indices
    # -------------------------------
    indices = {}

    indices["NDVI"] = np.nanmean(norm_index(nir, red))
    indices["NDWI"] = np.nanmean(norm_index(green, nir))
    indices["NDMI"] = np.nanmean(norm_index(nir, swir1))
    indices["NBR"] = np.nanmean(norm_index(nir, swir2))

    # EVI
    indices["EVI"] = np.nanmean(
        2.5 * ((nir - red) / (nir + 6*red - 7.5*green + 1 + 1e-10))
    )

    # SAVI
    L = 0.5
    indices["SAVI"] = np.nanmean(
        ((nir - red) / (nir + red + L)) * (1 + L)
    )

    # -------------------------------
    # Return ML-ready band means too
    # -------------------------------
    indices.update({
        "B2_Blue": np.nanmean(find_and_load(img10, "*B02_10m.jp2", ref_ds)),
        "B3_Green": np.nanmean(green),
        "B4_Red": np.nanmean(red),
        "B8_NIR": np.nanmean(nir),
        "B11_SWIR1": np.nanmean(swir1),
        "B12_SWIR2": np.nanmean(swir2),
    })

    print("✓ Computed indices:", indices)
    return indices


# Helper for loading B02
def find_and_load(folder, pattern, ref_ds):
    files = glob(os.path.join(folder, pattern))
    if not files:
        return np.nan
    arr = load_and_resample(files[0], ref_ds)
    return arr
