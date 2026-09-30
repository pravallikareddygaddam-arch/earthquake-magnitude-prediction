from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "Earthquake_Magnitude_Production_Model"
METRIC_TO_OPTIMIZE = "r2"

def metric(version):
    run = MlflowClient().get_run(version.run_id)
    return float(run.data.metrics.get(METRIC_TO_OPTIMIZE, float("-inf")))

def automate_lifecycle():
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    if not versions:
        print("[WARNING] No registered model versions found.")
        return

    for v in versions:
        if v.current_stage == "None":
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=v.version,
                stage="Staging",
                archive_existing_versions=False
            )

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    staging = [v for v in versions if v.current_stage == "Staging"]

    if not staging:
        print("[WARNING] No Staging model version found.")
        return

    challenger = max(staging, key=metric)
    production = [v for v in versions if v.current_stage == "Production"]

    if not production:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=challenger.version,
            stage="Production",
            archive_existing_versions=False
        )
        print(f"[SUCCESS] Version {challenger.version} promoted to Production.")
    else:
        champion = max(production, key=metric)
        if metric(challenger) > metric(champion):
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=champion.version,
                stage="Archived",
                archive_existing_versions=False
            )
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=challenger.version,
                stage="Production",
                archive_existing_versions=False
            )
            print(f"[SUCCESS] Version {challenger.version} promoted to Production.")
        else:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=challenger.version,
                stage="Archived",
                archive_existing_versions=False
            )
            print("[INFO] Existing Production model retained.")

    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        if v.current_stage == "Staging":
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=v.version,
                stage="Archived",
                archive_existing_versions=False
            )

if __name__ == "__main__":
    automate_lifecycle()
