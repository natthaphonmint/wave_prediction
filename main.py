from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import os
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_PATH = os.path.expanduser("~/wave_backend/wave_model")

LOCATIONS = {
    "Patong":              "patong",
    "Cha-am":              "chaam",
    "Pattaya":             "pattaya",
    "Hua Sai-Pak Phanang": "huasai",
}

DURATIONS = {
    "6 Hours":  6,
    "12 Hours": 12,
    "24 Hours": 24,
}

FEATURE_NAMES = [
    "swh_combined", "u10", "v10", "wind_gust",
    "swh_wind", "swh_swell", "wave_period",
    "wave_direction", "temp_2m", "sst"
]

# โหลดโมเดลและ scaler ทั้งหมด
print("📦 กำลังโหลดโมเดลทั้งหมด...")
MODELS  = {}
SCALERS = {}

for loc_key in LOCATIONS.values():
    for dur in DURATIONS.values():
        key        = f"{loc_key}_{dur}h"
        model_path  = f"{BASE_PATH}/models/{loc_key}_{dur}h.h5"
        scaler_path = f"{BASE_PATH}/scaler/{loc_key}_{dur}h_scaler.pkl"

        MODELS[key]  = tf.keras.models.load_model(model_path, compile=False) 
        with open(scaler_path, "rb") as f:
            SCALERS[key] = pickle.load(f)

        print(f"   ✅ โหลด {key}")

print("🎉 โหลดครบทุกโมเดล!")

class PredictInput(BaseModel):
    location: str
    duration: str
    swh_combined:   float
    u10:            float
    v10:            float
    wind_gust:      float
    swh_wind:       float
    swh_swell:      float
    wave_period:    float
    wave_direction: float
    temp_2m:        float
    sst:            float

def get_safety_level(h: float):
    if h <= 0.10:   return 1, "Calm"
    elif h <= 0.50: return 2, "Smooth"
    elif h <= 1.25: return 3, "Slight"
    elif h <= 2.50: return 4, "Moderate"
    elif h <= 4.00: return 5, "Rough"
    elif h <= 6.00: return 6, "Very Rough"
    elif h <= 9.00: return 7, "High"
    elif h <= 14.0: return 8, "Very High"
    else:           return 9, "Phenomenal"

@app.post("/predict")
def predict(data: PredictInput):
    if data.location not in LOCATIONS:
        raise HTTPException(status_code=400, detail="Location not found")
    if data.duration not in DURATIONS:
        raise HTTPException(status_code=400, detail="Duration not found")

    loc_key  = LOCATIONS[data.location]
    dur      = DURATIONS[data.duration]
    key      = f"{loc_key}_{dur}h"
    model    = MODELS[key]
    scaler   = SCALERS[key]

    X = np.array([[
        data.swh_combined, data.u10, data.v10, data.wind_gust,
        data.swh_wind, data.swh_swell, data.wave_period,
        data.wave_direction, data.temp_2m, data.sst
    ]])

    X_scaled = scaler.transform(X)
    X_lstm   = np.tile(X_scaled, (dur, 1)).reshape(1, dur, len(FEATURE_NAMES))

    prediction  = model.predict(X_lstm, verbose=0)
    wave_height = round(float(max(prediction[0][0], 0.01)), 2)
    level, status = get_safety_level(wave_height)

    return {
        "wave_height":   wave_height,
        "safety_level":  level,
        "safety_status": status,
    }

@app.get("/")
def root():
    return {"message": "🌊 Wave Prediction API Running!"}
