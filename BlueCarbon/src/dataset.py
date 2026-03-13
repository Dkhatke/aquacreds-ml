# src/dataset.py
"""
dataset.py
----------
Prepares Sentinel-2 multispectral band data + derived indices
for training the RandomForest classification model.
"""

import pandas as pd
from src.api import compute_indices


# These must match extract_bands.py output
REQUIRED_COLS = {
    "B2_Blue",
    "B3_Green",
    "B4_Red",
    "B8_NIR",
    "B11_SWIR1",
    "B12_SWIR2",
    "label"
}


def prepare_dataset(csv_path: str) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if "Area_ha" not in df.columns:
        df["Area_ha"] = 1.0

    rows = []

    for _, row in df.iterrows():

        indices = compute_indices(
            row["B2_Blue"], row["B3_Green"], row["B4_Red"],
            row["B8_NIR"], row["B11_SWIR1"], row["B12_SWIR2"]
        )

        rows.append({
            "B2": float(row["B2_Blue"]),
            "B3": float(row["B3_Green"]),
            "B4": float(row["B4_Red"]),
            "B8": float(row["B8_NIR"]),
            "B11": float(row["B11_SWIR1"]),
            "B12": float(row["B12_SWIR2"]),

            # derived
            "NDVI": indices["NDVI"],
            "EVI": indices["EVI"],
            "SAVI": indices["SAVI"],
            "NDWI": indices["NDWI"],
            "MNDWI": indices["MNDWI"],
            "MSI": indices["MSI"],
            "NDMI": indices["NDMI"],

            "Area_ha": float(row["Area_ha"]),
            "label": str(row["label"])
        })

    return pd.DataFrame(rows)
