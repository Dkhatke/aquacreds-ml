# src/ndvi_extraction.py
"""
Utility for computing NDVI from Sentinel-2 bands.
If bands are different resolution (10m vs 20m), caller must resample.
"""

import rasterio
import numpy as np


def compute_ndvi(nir_band_path: str, red_band_path: str):
    """
    Computes NDVI = (NIR - RED) / (NIR + RED)
    """

    with rasterio.open(nir_band_path) as nir_ds, rasterio.open(red_band_path) as red_ds:
        nir = nir_ds.read(1).astype("float32")
        red = red_ds.read(1).astype("float32")

    nir[nir == 0] = np.nan
    red[red == 0] = np.nan

    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi
