# src/features_builder.py
"""
features_builder.py
-------------------
Builds ML-ready feature dataset from raw Sentinel band CSV files.

Input CSV must contain:
    B2_Blue, B3_Green, B4_Red, B8_NIR, B11_SWIR1, B12_SWIR2
Optional:
    label, Area_ha

Output:
    features.csv → includes all spectral indices (NDVI, EVI, SAVI, NDWI, etc.)
"""

import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================
RAW_PATH = "sentinel_band_values.csv"

OUT_DIR = "data/processed"
OUT_PATH = os.path.join(OUT_DIR, "features.csv")

os.makedirs(OUT_DIR, exist_ok=True)

REQUIRED_COLS = {
    "B2_Blue", "B3_Green", "B4_Red",
    "B8_NIR", "B11_SWIR1", "B12_SWIR2"
}


# ============================================================
# INDEX CALCULATIONS
# ============================================================
def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-10

    # Vegetation Indices
    df["NDVI"] = (df["B8_NIR"] - df["B4_Red"]) / (df["B8_NIR"] + df["B4_Red"] + eps)

    df["EVI"] = 2.5 * ((df["B8_NIR"] - df["B4_Red"]) /
                       (df["B8_NIR"] + 6*df["B4_Red"] - 7.5*df["B2_Blue"] + 1 + eps))

    L = 0.5
    df["SAVI"] = ((df["B8_NIR"] - df["B4_Red"]) /
                  (df["B8_NIR"] + df["B4_Red"] + L + eps)) * (1 + L)

    # Water Indices
    df["NDWI"] = (df["B3_Green"] - df["B8_NIR"]) / (df["B3_Green"] + df["B8_NIR"] + eps)
    df["MNDWI"] = (df["B3_Green"] - df["B11_SWIR1"]) / (df["B3_Green"] + df["B11_SWIR1"] + eps)

    # Moisture / Stress Indices
    df["MSI"] = df["B11_SWIR1"] / (df["B8_NIR"] + eps)
    df["NDMI"] = (df["B8_NIR"] - df["B11_SWIR1"]) / (df["B8_NIR"] + df["B11_SWIR1"] + eps)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_feature_builder(raw_path: str = RAW_PATH, out_path: str = OUT_PATH):
    print("📥 Loading raw dataset:", raw_path)

    df = pd.read_csv(raw_path)

    # Validate columns
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")

    # Default Area_ha if not present
    if "Area_ha" not in df.columns:
        df["Area_ha"] = 1.0

    # Compute spectral indices
    print("📐 Computing spectral indices...")
    df = compute_indices(df)

    # Save output
    df.to_csv(out_path, index=False)
    print(f"✅ Features saved to: {out_path}")

    return df


# ============================================================
# RUN SCRIPT
# ============================================================
if __name__ == "__main__":
    run_feature_builder()
