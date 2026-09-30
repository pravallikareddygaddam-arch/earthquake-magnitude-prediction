from pathlib import Path
import sys
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import load_and_prepare

print("[INFO] Starting Lab 6: Batch / Offline Inference")

model = joblib.load(ROOT / "models" / "earthquake_model.pkl")
preprocessor = joblib.load(ROOT / "models" / "preprocessor.pkl")

X, y = load_and_prepare()
pred = model.predict(preprocessor.transform(X))

out = X.copy()
out["Predicted_Magnitude"] = pred
out["Actual_Magnitude"] = y.values

outdir = ROOT / "outputs"
outdir.mkdir(exist_ok=True)
path = outdir / "batch_predictions.csv"
out.to_csv(path, index=False)

print("[SUCCESS] Batch inference completed successfully!")
print("Predictions saved to:", path)
print("Number of predictions:", len(out))
