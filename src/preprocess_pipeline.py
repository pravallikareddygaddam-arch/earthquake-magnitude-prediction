from pathlib import Path
import sys
import json
import numpy as np
import joblib
from preprocess import load_and_prepare, build_preprocessor
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"

def run_production_preprocessing():
    print("[INFO] Starting Production Preprocessing Pipeline...")
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    print("[INFO] Transforming test data...")
    X_test_final = preprocessor.transform(X_test)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    np.save(PROCESSED / "X_train_final.npy", X_train_final)
    np.save(PROCESSED / "X_test_final.npy", X_test_final)
    np.save(PROCESSED / "y_train.npy", y_train.to_numpy(dtype=float))
    np.save(PROCESSED / "y_test.npy", y_test.to_numpy(dtype=float))
    joblib.dump(preprocessor, MODELS / "preprocessor.pkl")

    report = {
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "target": "Magnitude",
        "random_state": 42
    }
    with open(ARTIFACTS / "preprocessing_summary_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("[SUCCESS] Production preprocessing completed successfully!")
    print("[SUCCESS] Preprocessor saved as models/preprocessor.pkl")

if __name__ == "__main__":
    run_production_preprocessing()
