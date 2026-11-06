
from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def main():
    # Load configuration
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    processed_dir = Path(config["data"]["processed_dir"])
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed earthquake data
    X_train = np.load(processed_dir / "X_train_final.npy")
    X_test = np.load(processed_dir / "X_test_final.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    print("[INFO] Training data:", X_train.shape)
    print("[INFO] Testing data:", X_test.shape)

    # Validate the data before training
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Training and testing features must be 2D arrays.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and testing feature counts do not match.")

    if len(X_train) != len(y_train):
        raise ValueError("Training feature and target row counts do not match.")

    if len(X_test) != len(y_test):
        raise ValueError("Testing feature and target row counts do not match.")

    # Configure MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Create a candidate regression model
    model_config = config["model"]

    model = RandomForestRegressor(
        n_estimators=model_config["n_estimators"],
        max_depth=model_config["max_depth"],
        random_state=model_config["random_state"],
        n_jobs=-1,
    )

    # Train and log the candidate
    with mlflow.start_run(
        run_name="Earthquake_RandomForest_Candidate"
    ) as run:

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions))

        print("\n[RESULTS]")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")

        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": model_config["n_estimators"],
            "max_depth": model_config["max_depth"],
            "random_state": model_config["random_state"],
            "target": config["data"]["target_column"],
        })

        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        })

        # Trust only the specific tree type reported by skops.
        # Use this only for the model trained locally above.
        mlflow.sklearn.log_model(
            sk_model=model,
            name="earthquake_model_candidate",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

        print("\n[INFO] MLflow run ID:", run.info.run_id)

    # Save the candidate separately from the Production model
    candidate_path = artifacts_dir / "earthquake_model_candidate.pkl"
    joblib.dump(model, candidate_path)

    # Save evaluation metrics
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    metrics_path = artifacts_dir / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n[SUCCESS] Candidate model trained and logged.")
    print("Candidate model:", candidate_path)
    print("Metrics:", metrics_path)
    print("Existing Production model was not changed.")


if __name__ == "__main__":
    main()
