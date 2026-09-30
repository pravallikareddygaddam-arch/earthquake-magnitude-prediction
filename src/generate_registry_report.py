from pathlib import Path
import json
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODEL_NAME = "Earthquake_Magnitude_Production_Model"

def generate_report():
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    production = [v for v in versions if v.current_stage == "Production"]

    if not production:
        raise RuntimeError("No Production model version found in the registry.")

    version = max(production, key=lambda v: int(v.version))
    run = client.get_run(version.run_id)

    report = {
        "registry_status": "READY_FOR_DEPLOYMENT",
        "model_lineage": {
            "registered_name": MODEL_NAME,
            "version": version.version,
            "current_stage": version.current_stage,
            "run_id": version.run_id,
            "artifact_uri": version.source
        },
        "performance": {
            "mae": run.data.metrics.get("mae"),
            "rmse": run.data.metrics.get("rmse"),
            "r2": run.data.metrics.get("r2")
        },
        "hyperparameters": dict(run.data.params),
        "preprocessing_dependency": "models/preprocessor.pkl"
    }

    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "production_model_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("[SUCCESS] Production registry report saved.")

if __name__ == "__main__":
    generate_report()
