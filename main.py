from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle, os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE_PATH = os.path.expanduser("~/wave_backend/wave_model")
MODELS = {}
SCALERS = {}
LOCATION_MAP = {"Patong":"patong","Cha-am":"chaam","Pattaya":"pattaya","Hua Sai-Pak Phanang":"huasai"}
DURATION_MAP = {"6 Hours":6,"12 Hours":12,"24 Hours":24}
FEATURE_NAMES = ["u10","v10","wind_gust","swh_wind","swh_swell","wave_period","wave_direction","temp_2m","sst"]

print("Loading models...")
for loc in ["patong","chaam","pattaya","huasai"]:
    for dur in [6,12,24]:
        key = f"{loc}_{dur}h"
        mp = os.path.join(BASE_PATH,"models",f"{key}.h5")
        sp = os.path.join(BASE_PATH,"scaler",f"{key}_scaler.pkl")
        if os.path.exists(mp) and os.path.exists(sp):
            MODELS[key] = tf.keras.models.load_model(mp, compile=False)
            with open(sp,"rb") as f: SCALERS[key] = pickle.load(f)
            print(f"  OK {key}")

class PredictInput(BaseModel):
    location: str; duration: str; swh_combined: float; u10: float; v10: float
    wind_gust: float; swh_wind: float; swh_swell: float; wave_period: float
    wave_direction: float; temp_2m: float; sst: float

def get_safety(h):
    for threshold, level, status in [(0.10,1,"Calm"),(0.50,2,"Smooth"),(1.25,3,"Slight"),(2.50,4,"Moderate"),(4.00,5,"Rough"),(6.00,6,"Very Rough"),(9.00,7,"High"),(14.0,8,"Very High")]:
        if h <= threshold: return level, status
    return 9, "Phenomenal"

@app.get("/")
def root(): return {"message": "Wave Prediction API Running"}

@app.post("/predict")
def predict(data: PredictInput):
    loc_key = LOCATION_MAP.get(data.location,"patong")
    dur = DURATION_MAP.get(data.duration,6)
    key = f"{loc_key}_{dur}h"
    features = np.array([[data.swh_combined,data.u10,data.v10,data.wind_gust,data.swh_wind,data.swh_swell,data.wave_period,data.wave_direction,data.temp_2m,data.sst]])
    scaled = SCALERS[key].transform(features)
    X = np.tile(scaled,(1,dur,1)).reshape(1,dur,10)
    prediction = float(MODELS[key].predict(X,verbose=0)[0][0])
    level,status = get_safety(prediction)
    shap_values = {}
    shap_path = os.path.join(BASE_PATH,"shap",f"{loc_key}_{dur}h_shap.pkl")
    if os.path.exists(shap_path):
        with open(shap_path,"rb") as f: shap_data = pickle.load(f)
        if isinstance(shap_data, dict):
            shap_values = {k: float(list(v.values())[0]) if isinstance(v,dict) else float(v) for k,v in shap_data.items()}
        else:
            arr = np.array(shap_data).flatten()
            for i,name in enumerate(FEATURE_NAMES):
                shap_values[name] = float(arr[i]) if i < len(arr) else 0.0
    return {"wave_height":prediction,"safety_level":level,"safety_status":status,"shap_values":shap_values}
