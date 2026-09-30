import os
import numpy as np
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

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

PROCESSED = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")


# ============================================================
# LAB 6 - MODEL REGISTRY & LIFECYCLE
# ============================================================

def train_and_register_model():

    print("[INFO] --- Starting MLflow Run with Model Registry ---")

    # --------------------------------------------------------
    # MLflow experiment
    # --------------------------------------------------------

    mlflow.set_experiment(
        "Earthquake_Magnitude_Prediction"
    )

    # --------------------------------------------------------
    # Load processed data
    # --------------------------------------------------------

    X_train = np.load(
        os.path.join(
            PROCESSED,
            "X_train_final.npy"
        )
    )

    X_test = np.load(
        os.path.join(
            PROCESSED,
            "X_test_final.npy"
        )
    )

    y_train = np.load(
        os.path.join(
            PROCESSED,
            "y_train.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            PROCESSED,
            "y_test.npy"
        )
    )

    print(
        f"[INFO] X_train shape: {X_train.shape}"
    )

    print(
        f"[INFO] X_test shape: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Model parameters
    # --------------------------------------------------------

    params = {
        "n_estimators": 250,
        "random_state": 42,
        "n_jobs": -1
    }

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="RandomForestRegressor_Registry"
    ) as run:

        print(
            f"[INFO] MLflow Run ID: "
            f"{run.info.run_id}"
        )

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        model = RandomForestRegressor(
            **params
        )

        model.fit(
            X_train,
            y_train
        )

        print(
            "[INFO] Model training completed."
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = {
            "mae": mean_absolute_error(
                y_test,
                predictions
            ),

            "rmse": np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions
                )
            ),

            "r2": r2_score(
                y_test,
                predictions
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

        mlflow.log_metrics(
            metrics
        )

        print(
            f"[SUCCESS] MAE  = "
            f"{metrics['mae']:.4f}"
        )

        print(
            f"[SUCCESS] RMSE = "
            f"{metrics['rmse']:.4f}"
        )

        print(
            f"[SUCCESS] R2   = "
            f"{metrics['r2']:.4f}"
        )

        # ----------------------------------------------------
        # Log preprocessing pipeline
        # ----------------------------------------------------

        preprocessor_path = os.path.join(
            MODELS,
            "preprocessor.pkl"
        )

        if os.path.exists(
            preprocessor_path
        ):

            print(
                "[INFO] Logging preprocessing pipeline..."
            )

            mlflow.log_artifact(
                preprocessor_path,
                artifact_path="preprocessing_pipeline"
            )

        else:

            print(
                "[WARNING] "
                "models/preprocessor.pkl not found."
            )

        # ----------------------------------------------------
        # Register model
        #
        # IMPORTANT:
        # RandomForest contains sklearn Tree objects.
        # MLflow/skops requires explicit trust.
        # ----------------------------------------------------

        print(
            "[INFO] Registering model in MLflow..."
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=(
                "Earthquake_Magnitude_Production_Model"
            ),
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # ----------------------------------------------------
        # Save local model
        # ----------------------------------------------------

        os.makedirs(
            MODELS,
            exist_ok=True
        )

        import joblib

        local_model_path = os.path.join(
            MODELS,
            "random_forest_model.pkl"
        )

        joblib.dump(
            model,
            local_model_path
        )

        print(
            f"[INFO] Local model saved: "
            f"{local_model_path}"
        )

        # ----------------------------------------------------
        # Final confirmation
        # ----------------------------------------------------

        print(
            "\n[SUCCESS] Model successfully "
            "registered in MLflow!"
        )

        print(
            "[SUCCESS] Registered Model Name: "
            "Earthquake_Magnitude_Production_Model"
        )

        print(
            f"[SUCCESS] R2 = "
            f"{metrics['r2']:.4f} | "
            f"RMSE = "
            f"{metrics['rmse']:.4f}"
        )

        print(
            f"[SUCCESS] MLflow Run ID: "
            f"{run.info.run_id}"
        )

        return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_and_register_model()
    