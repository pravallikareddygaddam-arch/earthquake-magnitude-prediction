from pathlib import Path
from typing import Any
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/earthquake_model.pkl"
PREPROCESSOR_PATH = ROOT / "models/preprocessor.pkl"

app = FastAPI(
    title="Earthquake Magnitude Prediction API",
    version="1.0.0",
    description="Lab 10: Online inference API for the Earthquake Magnitude Prediction MLOps project."
)

model = None
preprocessor = None

def load_artifacts():
    global model, preprocessor
    if model is None or preprocessor is None:
        if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(
                "Model artifacts are missing. Run Labs 3-6 to create "
                "models/earthquake_model.pkl and models/preprocessor.pkl."
            )
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Raw earthquake input features. Supply the same source columns used by the training dataset, excluding Magnitude and ID."
    )

class BatchPredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)

def prepare_features(records):
    frame = pd.DataFrame(records)
    frame = frame.drop(columns=["Magnitude", "ID"], errors="ignore")
    if "Date" in frame:
        date = pd.to_datetime(frame["Date"], errors="coerce")
        frame["Year"], frame["Month"], frame["Day"] = date.dt.year, date.dt.month, date.dt.day
        frame = frame.drop(columns=["Date"])
    if "Time" in frame:
        parts = frame["Time"].astype(str).str.extract(
            r"(?P<Hour>\d{1,2}):(?P<Minute>\d{1,2}):(?P<Second>\d{1,2})"
        )
        for col in ["Hour", "Minute", "Second"]:
            frame[col] = pd.to_numeric(parts[col], errors="coerce")
        frame = frame.drop(columns=["Time"])
    _, prep = load_artifacts()
    expected = list(getattr(prep, "feature_names_in_", []))
    if expected:
        missing = [c for c in expected if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        frame = frame[expected]
    return prep.transform(frame)

@app.get("/")
def root():
    return {"message": "Earthquake Magnitude Prediction API", "docs": "/docs"}

@app.get("/health")
def health():
    try:
        load_artifacts()
        return {"status": "healthy", "model_loaded": True, "preprocessor_loaded": True}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.get("/metadata")
def metadata():
    try:
        loaded_model, prep = load_artifacts()
        return {
            "model": type(loaded_model).__name__,
            "target": "Magnitude",
            "model_loaded": True,
            "preprocessor_loaded": True,
            "input_feature_count": len(getattr(prep, "feature_names_in_", []))
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        matrix = prepare_features([request.features])
        prediction = float(load_artifacts()[0].predict(matrix)[0])
        return {"predicted_magnitude": round(prediction, 3), "target": "Magnitude"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    try:
        matrix = prepare_features(request.records)
        predictions = load_artifacts()[0].predict(matrix)
        return {
            "count": len(predictions),
            "predictions": [
                {"predicted_magnitude": round(float(value), 3), "target": "Magnitude"}
                for value in predictions
            ]
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
