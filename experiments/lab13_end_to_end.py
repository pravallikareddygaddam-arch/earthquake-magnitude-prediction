from pathlib import Path
import json
import math
import sys
import numpy as np
import pandas as pd
import joblib
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.main import app


def main():
    required = [
        "data/raw/Earthquake.csv.csv",
        "data/processed/X_test_final.npy",
        "data/processed/y_test.npy",
        "models/earthquake_model.pkl",
        "models/preprocessor.pkl",
        "artifacts/metrics.json",
        "artifacts/quality_gate_report.json",
        "artifacts/lab12_monitoring_report.json",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"Lab 13 prerequisites missing: {missing}")

    df = pd.read_csv(ROOT / "data/raw/Earthquake.csv.csv")
    sample = df.iloc[0].drop(labels=["Magnitude", "ID"], errors="ignore").to_dict()
    safe_sample = {}
    for key, value in sample.items():
        if pd.isna(value):
            safe_sample[key] = None
        elif hasattr(value, "item"):
            safe_sample[key] = value.item()
        else:
            safe_sample[key] = value

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200, health.text
    metadata = client.get("/metadata")
    assert metadata.status_code == 200, metadata.text

    single = client.post("/predict", json={"features": safe_sample})
    assert single.status_code == 200, single.text
    single_value = float(single.json()["predicted_magnitude"])
    assert math.isfinite(single_value)

    batch = client.post("/predict/batch", json={"records": [safe_sample, safe_sample]})
    assert batch.status_code == 200, batch.text
    batch_json = batch.json()
    assert batch_json.get("count") == 2 and len(batch_json.get("predictions", [])) == 2
    assert all(math.isfinite(float(x["predicted_magnitude"])) for x in batch_json["predictions"])

    X_test = np.load(ROOT / "data/processed/X_test_final.npy")
    y_test = np.load(ROOT / "data/processed/y_test.npy")
    model = joblib.load(ROOT / "models/earthquake_model.pkl")
    assert X_test.shape[0] == y_test.shape[0] and X_test.shape[0] > 0
    offline_prediction = np.asarray(model.predict(X_test[: min(5, len(X_test))]), dtype=float)
    assert np.isfinite(offline_prediction).all()

    metrics = json.loads((ROOT / "artifacts/metrics.json").read_text(encoding="utf-8"))
    quality = json.loads((ROOT / "artifacts/quality_gate_report.json").read_text(encoding="utf-8"))
    monitoring = json.loads((ROOT / "artifacts/lab12_monitoring_report.json").read_text(encoding="utf-8"))
    assert all(key in metrics for key in ("MAE", "RMSE", "R2")), metrics
    assert quality.get("status") == "PASSED", quality

    report = {
        "experiment": "Lab 13 - End-to-End MLOps Integration Verification",
        "status": "PASSED",
        "checks": {
            "all_required_artifacts_present": True,
            "api_health": health.json(),
            "api_metadata": metadata.json(),
            "online_single_prediction": single.json(),
            "online_batch_prediction_count": batch_json["count"],
            "offline_prediction_rows_checked": int(len(offline_prediction)),
            "evaluation_metrics": metrics,
            "quality_gate": quality,
            "monitoring_status": monitoring.get("status", "UNKNOWN"),
        },
    }
    out = ROOT / "artifacts/lab13_end_to_end_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[SUCCESS] Lab 13 integration report saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
