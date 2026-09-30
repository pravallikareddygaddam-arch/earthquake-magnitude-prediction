from pathlib import Path
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

def run_once():
    X_train = np.load(PROCESSED / "X_train_final.npy")
    X_test = np.load(PROCESSED / "X_test_final.npy")
    y_train = np.load(PROCESSED / "y_train.npy")
    y_test = np.load(PROCESSED / "y_test.npy")

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return float(r2_score(y_test, model.predict(X_test)))

def validate_reproducibility():
    print("Running Reproducibility Validation...")
    r1 = run_once()
    r2 = run_once()
    report = {
        "execution_1_r2": r1,
        "execution_2_r2": r2,
        "difference": abs(r1-r2),
        "reproducible": bool(np.isclose(r1, r2))
    }
    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "reproducibility_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if report["reproducible"]:
        print("SUCCESS: Pipeline is 100% reproducible. Report saved.")
    else:
        raise RuntimeError("Reproducibility validation failed.")

if __name__ == "__main__":
    validate_reproducibility()
