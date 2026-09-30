from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"

def train_model():
    X_train = np.load(PROCESSED / "X_train_final.npy")
    y_train = np.load(PROCESSED / "y_train.npy")

    params = {
        "n_estimators": 250,
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    MODELS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)

    joblib.dump(model, MODELS / "earthquake_model.pkl")
    with open(ARTIFACTS / "model_metadata.json", "w") as f:
        json.dump({
            "model": "RandomForestRegressor",
            **params,
            "target": "Magnitude"
        }, f, indent=2)

    print("Training completed successfully!")
    return model

if __name__ == "__main__":
    train_model()
