from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/Earthquake.csv.csv"
PROCESSED = ROOT / "data/processed"
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"

def main():
    ARTIFACTS.mkdir(exist_ok=True)
    df = pd.read_csv(RAW)
    # Monitoring simulation: compare an untouched reference sample with a
    # controlled, clearly-labelled shifted copy. Original data is never changed.
    numeric = df.select_dtypes(include="number").columns.tolist()
    drift_features = []
    for col in numeric:
        if col == "Magnitude":
            continue
        reference_values = df[col].dropna().astype(float)
        if len(reference_values) < 2:
            continue
        spread = float(reference_values.std() or 0.0)
        shifted_values = reference_values + (0.25 * spread)
        statistic, p_value = ks_2samp(reference_values, shifted_values)
        drift_features.append({
            "feature": col,
            "reference_count": int(len(reference_values)),
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < 0.05)
        })

    X_test = np.load(PROCESSED / "X_test_final.npy")
    y_test = np.load(PROCESSED / "y_test.npy")
    model = joblib.load(MODELS / "earthquake_model.pkl")
    pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred))
    }
    report = {
        "experiment": "Lab 12 - Data Drift and Model Performance Monitoring",
        "monitoring_mode": "controlled drift simulation; source dataset unchanged",
        "reference_rows": int(len(df)),
        "current_rows": int(len(df)),
        "drifted_features": [x["feature"] for x in drift_features if x["drift_detected"]],
        "feature_drift": drift_features,
        "held_out_performance": metrics,
        "status": "DRIFT_DETECTED" if any(x["drift_detected"] for x in drift_features) else "STABLE"
    }
    path = ARTIFACTS / "lab12_monitoring_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[SUCCESS] Monitoring report saved: {path.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
