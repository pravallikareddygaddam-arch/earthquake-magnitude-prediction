from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

def validate_outputs():
    required = [
        PROCESSED / "X_train_final.npy",
        PROCESSED / "X_test_final.npy",
        PROCESSED / "y_train.npy",
        PROCESSED / "y_test.npy",
        ROOT / "models" / "preprocessor.pkl"
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing production outputs: " + ", ".join(missing))

    X_train = np.load(PROCESSED / "X_train_final.npy")
    X_test = np.load(PROCESSED / "X_test_final.npy")
    y_train = np.load(PROCESSED / "y_train.npy")
    y_test = np.load(PROCESSED / "y_test.npy")

    report = {
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "y_train_shape": list(y_train.shape),
        "y_test_shape": list(y_test.shape),
        "feature_count_match": bool(X_train.shape[1] == X_test.shape[1]),
        "row_count_match_train": bool(X_train.shape[0] == y_train.shape[0]),
        "row_count_match_test": bool(X_test.shape[0] == y_test.shape[0])
    }

    if not all(report.values()):
        raise ValueError("Production output validation failed.")

    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "production_model_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("[SUCCESS] Production output validation PASSED.")

if __name__ == "__main__":
    validate_outputs()
