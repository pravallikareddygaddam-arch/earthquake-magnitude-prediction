from pathlib import Path
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"

def evaluate():
    X_test = np.load(PROCESSED / "X_test_final.npy")
    y_test = np.load(PROCESSED / "y_test.npy")
    model = joblib.load(MODELS / "earthquake_model.pkl")

    pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred))
    }

    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savetxt(
        ARTIFACTS / "predictions.csv",
        np.column_stack((y_test, pred)),
        delimiter=",",
        header="Actual_Magnitude,Predicted_Magnitude",
        comments=""
    )

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, pred, s=12)
    lo = min(y_test.min(), pred.min())
    hi = max(y_test.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual Magnitude")
    plt.ylabel("Predicted Magnitude")
    plt.title("Actual vs Predicted Earthquake Magnitude")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "actual_vs_predicted.png")
    plt.close()

    print(f"MAE  : {metrics['MAE']:.4f}")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"R2   : {metrics['R2']:.4f}")
    return metrics

if __name__ == "__main__":
    evaluate()
