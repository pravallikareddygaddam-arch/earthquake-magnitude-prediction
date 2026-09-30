from pathlib import Path
import json
import numpy as np
import joblib

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"


# ============================================================
# LAB 4 - MLFLOW EXPERIMENT TRACKING
# ============================================================

def train_with_mlflow():

    print("--- Starting MLflow Run: RandomForestRegressor ---")

    # --------------------------------------------------------
    # MLflow Experiment
    # --------------------------------------------------------

    mlflow.set_experiment("Earthquake_Magnitude_Prediction")

    # --------------------------------------------------------
    # Load processed data
    # --------------------------------------------------------

    X_train = np.load(PROCESSED / "X_train_final.npy")
    X_test = np.load(PROCESSED / "X_test_final.npy")
    y_train = np.load(PROCESSED / "y_train.npy")
    y_test = np.load(PROCESSED / "y_test.npy")

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")

    # --------------------------------------------------------
    # Model parameters
    # --------------------------------------------------------

    params = {
        "n_estimators": 250,
        "random_state": 42,
        "n_jobs": -1
    }

    # --------------------------------------------------------
    # Start MLflow Run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="RandomForestRegressor"
    ) as run:

        print(f"MLflow Run ID: {run.info.run_id}")

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        model = RandomForestRegressor(**params)

        model.fit(X_train, y_train)

        print("[INFO] RandomForestRegressor training completed.")

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        pred = model.predict(X_test)

        # ----------------------------------------------------
        # Evaluation metrics
        # ----------------------------------------------------

        metrics = {
            "mae": mean_absolute_error(
                y_test,
                pred
            ),

            "rmse": np.sqrt(
                mean_squared_error(
                    y_test,
                    pred
                )
            ),

            "r2": r2_score(
                y_test,
                pred
            )
        }

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_params(params)

        mlflow.log_param(
            "model_family",
            "RandomForestRegressor"
        )

        mlflow.log_param(
            "problem_type",
            "regression"
        )

        mlflow.log_param(
            "target",
            "Magnitude"
        )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metrics(metrics)

        print(
            f"MAE  : {metrics['mae']:.4f}"
        )

        print(
            f"RMSE : {metrics['rmse']:.4f}"
        )

        print(
            f"R2   : {metrics['r2']:.4f}"
        )

        # ----------------------------------------------------
        # Create MLflow metrics report
        # ----------------------------------------------------

        ARTIFACTS.mkdir(
            parents=True,
            exist_ok=True
        )

        report_path = (
            ARTIFACTS /
            "mlflow_metrics.json"
        )

        with open(
            report_path,
            "w"
        ) as f:

            json.dump(
                {
                    key: float(value)
                    for key, value in metrics.items()
                },
                f,
                indent=2
            )

        # ----------------------------------------------------
        # Log metrics report
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(report_path)
        )

        # ----------------------------------------------------
        # Log preprocessing pipeline if available
        # ----------------------------------------------------

        preprocessor_path = (
            MODELS /
            "preprocessor.pkl"
        )

        if preprocessor_path.exists():

            print(
                "[INFO] Logging preprocessing pipeline..."
            )

            mlflow.log_artifact(
                str(preprocessor_path),
                artifact_path="preprocessing_pipeline"
            )

        else:

            print(
                "[WARNING] "
                "models/preprocessor.pkl not found."
            )

        # ----------------------------------------------------
        # Log RandomForest model
        #
        # MLflow/skops requires explicit trust for the
        # internal sklearn Tree type used by RandomForest.
        # ----------------------------------------------------

        print(
            "[INFO] Logging model to MLflow..."
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # ----------------------------------------------------
        # Save local model
        # ----------------------------------------------------

        MODELS.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = (
            MODELS /
            "random_forest_model.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"[INFO] Local model saved: {model_path}"
        )

        # ----------------------------------------------------
        # Final message
        # ----------------------------------------------------

        print(
            "\n[SUCCESS] "
            "Run 'RandomForestRegressor' successfully tracked!"
        )

        print(
            f"[SUCCESS] R2 = {metrics['r2']:.4f} | "
            f"RMSE = {metrics['rmse']:.4f}"
        )

        print(
            f"[SUCCESS] MLflow Run ID: {run.info.run_id}"
        )

        return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_with_mlflow()