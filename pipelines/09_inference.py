
import os
import sys

import joblib
import numpy as np
import pandas as pd
import yaml


def run_inference():
    print("[INFO] Starting Earthquake Magnitude Inference...")

    try:
        with open("config.yaml", "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        raw_path = "data/raw/Earthquake.csv.csv"
        model_path = "models/earthquake_model.pkl"
        preprocessor_path = "models/preprocessor.pkl"
        output_path = "artifacts/earthquake_predictions.csv"

        for path in [raw_path, model_path, preprocessor_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")

        df = pd.read_csv(raw_path)
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        target = "Magnitude"
        X = df.drop(columns=[target], errors="ignore").copy()

        # Convert Date into Year, Month and Day.
        if "Date" in X.columns:
            date_parts = pd.to_datetime(
                X["Date"], format="%m/%d/%Y", errors="coerce"
            )
            X["Year"] = date_parts.dt.year
            X["Month"] = date_parts.dt.month
            X["Day"] = date_parts.dt.day
            X.drop(columns=["Date"], inplace=True)

        # Convert Time into Hour, Minute and Second.
        if "Time" in X.columns:
            time_parts = pd.to_datetime(
                X["Time"], format="%H:%M:%S", errors="coerce"
            )
            X["Hour"] = time_parts.dt.hour
            X["Minute"] = time_parts.dt.minute
            X["Second"] = time_parts.dt.second
            X.drop(columns=["Time"], inplace=True)

        # Match the exact input columns used during training.
        expected_features = getattr(
            preprocessor, "feature_names_in_", None
        )

        if expected_features is not None:
            missing = [
                col for col in expected_features if col not in X.columns
            ]
            if missing:
                raise ValueError(
                    f"Missing required input columns: {missing}"
                )
            X = X[list(expected_features)]

        X_processed = preprocessor.transform(X)
        predictions = model.predict(X_processed)

        results = df.copy()
        results["Predicted_Magnitude"] = np.round(predictions, 3)

        os.makedirs("artifacts", exist_ok=True)
        results.to_csv(output_path, index=False)

        print("\n--- Earthquake Prediction Results ---")
        print(
            results[
                ["Magnitude", "Predicted_Magnitude"]
            ].head(10).to_string(index=False)
        )
        print(f"\n[SUCCESS] Predictions saved to {output_path}")

        return True

    except Exception as exc:
        print(f"[ERROR] Inference failed: {exc}")
        return False


if __name__ == "__main__":
    sys.exit(0 if run_inference() else 1)
