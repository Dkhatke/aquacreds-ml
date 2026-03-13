# src/api.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
from io import BytesIO
from PIL import Image
import json

from src.biomass import estimate_all


app = FastAPI(title="BlueCarbon API — ML + Carbon Credits")

# =========================================================
# CORS
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# LOAD RANDOM FOREST MODEL
# =========================================================
RF_MODEL_PATH = "models/classifier.pkl"
RF_ENCODER_PATH = "models/label_encoder.pkl"

try:
    rf_model = joblib.load(RF_MODEL_PATH)
    label_encoder = joblib.load(RF_ENCODER_PATH)
    print("✓ RandomForest loaded")
except Exception as e:
    rf_model = None
    label_encoder = None
    print(f"⚠ RandomForest NOT found: {e}")

# =========================================================
# LOAD CNN MODEL
# =========================================================
CNN_MODEL_PATH = "models/eco_model.h5"
CLASSES_PATH = "models/classes.json"

try:
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    CLASS_NAMES = json.load(open(CLASSES_PATH))
    print("✓ CNN loaded")
except Exception as e:
    cnn_model = None
    CLASS_NAMES = []
    print(f"⚠ CNN NOT found: {e}")


# =========================================================
# REQUEST BODY FOR BAND INPUT
# =========================================================
class BandInput(BaseModel):
    Tile_ID: str = "UNKNOWN"
    B2_Blue: float
    B3_Green: float
    B4_Red: float
    B8_NIR: float
    B11_SWIR1: float
    B12_SWIR2: float
    Area_ha: float = 1.0


# =========================================================
# COMPUTE INDICES (Matches entire ML pipeline)
# =========================================================
def compute_indices(B2, B3, B4, B8, B11, B12):
    eps = 1e-10
    return {
        "NDVI": (B8 - B4) / (B8 + B4 + eps),
        "EVI": 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1 + eps)),
        "SAVI": ((B8 - B4) / (B8 + B4 + 0.5 + eps)) * 1.5,
        "NDWI": (B3 - B8) / (B3 + B8 + eps),
        "MNDWI": (B3 - B11) / (B3 + B11 + eps),
        "MSI": B11 / (B8 + eps),
        "NDMI": (B8 - B11) / (B8 + B11 + eps),
    }


# =========================================================
# MAIN ENDPOINT — RANDOM FOREST BAND PREDICTION
# =========================================================
@app.post("/predict-bands")
def predict_bands(tile: BandInput):

    if rf_model is None:
        raise HTTPException(500, "RandomForest model missing.")

    # Compute vegetation indices
    indices = compute_indices(
        tile.B2_Blue, tile.B3_Green, tile.B4_Red,
        tile.B8_NIR, tile.B11_SWIR1, tile.B12_SWIR2
    )

    # Build feature vector in EXACT ML order
    X = np.array([
        tile.B2_Blue, tile.B3_Green, tile.B4_Red,
        tile.B8_NIR, tile.B11_SWIR1, tile.B12_SWIR2,
        indices["NDVI"], indices["EVI"], indices["SAVI"],
        indices["NDWI"], indices["MNDWI"], indices["MSI"],
        indices["NDMI"], tile.Area_ha
    ]).reshape(1, -1)

    # Predict ecosystem class
    try:
        prob = rf_model.predict_proba(X)[0]
        pred_idx = np.argmax(prob)
        eco_class = label_encoder.inverse_transform([pred_idx])[0]
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    # Compute biomass, carbon, credits
    biomass = estimate_all(
        eco_class,
        indices["NDVI"],
        indices["EVI"],
        tile.Area_ha,
    )

    # JSON Response Format (Matches MongoDB MRV schema)
    return {
        "Tile_ID": tile.Tile_ID,
        "ml_result": {
            "class": eco_class,
            "ecosystem_class": biomass["ecosystem_class"],
            "canopy_percent": biomass["canopy_percent"],
            "indices": indices,
            "biomass": biomass["biomass"],
            "credit_suggestion": biomass["credit_suggestion"],
            "ndvi_satellite": biomass["ndvi_satellite"],
            "satellite_score": biomass["satellite_score"],
            "carbon_stock_tCO2e": biomass["carbon_stock_tCO2e"]
        }
    }
