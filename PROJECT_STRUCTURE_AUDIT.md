# Earthquake Magnitude Prediction — organized project audit

This folder contains the earthquake project only. It was reconstructed from `earthquake_magnitude_prediction_all_13_labs.zip`. No Telco churn project files were included.

## Main structure

- `.github/workflows/` — GitHub Actions for Labs 8, 9, and 13
- `api/` — FastAPI serving application
- `artifacts/` — model metrics, experiment reports, predictions, and plots
- `data/raw/` — earthquake source dataset
- `data/processed/` — processed train/test datasets and metadata
- `docs/` — lab documentation
- `experiments/` — Labs 1, 2, 6, 7, 10, 11, and 12 experiment scripts
- `models/` — trained earthquake model and preprocessor
- `pipelines/` — Labs 3–10 pipelines and all-labs runner
- `src/` — reusable training, preprocessing, evaluation, and validation modules
- `tests/` — test package

## Important verification note

The archive includes a `pipelines/run_all_13_labs.py` orchestrator and per-lab scripts/workflows. This confirms the implementation files are present, but it does not prove that every lab passes on your computer. Run the orchestrator and inspect `artifacts/lab13_execution_report.json` for actual results.

## Files included

- `.dockerignore`
- `.github/workflows/lab13_end_to_end.yml`
- `.github/workflows/lab8_ci.yml`
- `.github/workflows/lab9_failure_tests.yml`
- `.gitignore`
- `Dockerfile`
- `README.md`
- `api/__init__.py`
- `api/main.py`
- `artifacts/.gitkeep`
- `artifacts/actual_vs_predicted.png`
- `artifacts/earthquake_model_candidate.pkl`
- `artifacts/earthquake_predictions.csv`
- `artifacts/lab01_dataset_audit.json`
- `artifacts/lab02_data_quality_report.json`
- `artifacts/lab12_monitoring_report.json`
- `artifacts/lab9_failure_report.json`
- `artifacts/metrics.json`
- `artifacts/mlflow_metrics.json`
- `artifacts/model_metadata.json`
- `artifacts/predictions.csv`
- `artifacts/preprocessing_summary_report.json`
- `artifacts/production_model_report.json`
- `artifacts/quality_gate_report.json`
- `artifacts/reproducibility_report.json`
- `config.yaml`
- `data/processed/.gitkeep`
- `data/processed/X_test_final.npy`
- `data/processed/X_test_processed.csv`
- `data/processed/X_train_final.npy`
- `data/processed/X_train_processed.csv`
- `data/processed/dataset_metadata.json`
- `data/processed/y_test.csv`
- `data/processed/y_test.npy`
- `data/processed/y_train.csv`
- `data/processed/y_train.npy`
- `data/raw/.gitkeep`
- `data/raw/Earthquake.csv.csv`
- `docs/LAB7_PIPELINE.md`
- `docs/LAB8_CI_REPORT.md`
- `docs/LAB9_FAILURE_HANDLING_REPORT.md`
- `docs/MLFLOW_CI_SETUP.md`
- `earthquake_magnitude_prediction.ipynb`
- `experiments/lab01_dataset_audit.py`
- `experiments/lab02_data_quality.py`
- `experiments/lab06_registry_and_batch.py`
- `experiments/lab07_evaluation_quality_gate.py`
- `experiments/lab10_api_smoke_test.py`
- `experiments/lab11_docker_check.py`
- `experiments/lab12_monitoring.py`
- `logs/.gitkeep`
- `models/.gitkeep`
- `models/earthquake_model.pkl`
- `models/preprocessor.pkl`
- `models/random_forest_model.pkl`
- `outputs/.gitkeep`
- `pipelines/.gitkeep`
- `pipelines/01_load_data.py`
- `pipelines/02_validate_data.py`
- `pipelines/03_preprocessing.py`
- `pipelines/04_validate_outputs.py`
- `pipelines/05_train_registry.py`
- `pipelines/06_model_evaluation.py`
- `pipelines/07_quality_gate.py`
- `pipelines/08_automate_lifecycle.py`
- `pipelines/09_inference.py`
- `pipelines/10_failure_injection.py`
- `pipelines/run_all_13_labs.py`
- `pipelines/run_lab3_baseline.py`
- `pipelines/run_lab4_tracking.py`
- `pipelines/run_lab5_pipeline.py`
- `pipelines/run_lab6_batch_inference.py`
- `pipelines/run_lab6_registry.py`
- `requirements-deploy.txt`
- `requirements.txt`
- `src/.gitkeep`
- `src/automate_lifecycle.py`
- `src/evaluate.py`
- `src/generate_registry_report.py`
- `src/preprocess.py`
- `src/preprocess_pipeline.py`
- `src/train.py`
- `src/train_mlflow.py`
- `src/train_registry.py`
- `src/validate_data.py`
- `src/validate_outputs.py`
- `src/validate_reproducibility.py`
- `tests/__init__.py`


## Corrective additions

- `experiments/lab08_ci_validation.py`: local CI and project integrity validation.
- `experiments/lab13_end_to_end.py`: distinct online/offline integration test.
- `pipelines/run_all_13_labs.py`: corrected sequential runner for 13 distinct labs.
- `.github/workflows/lab8_ci.yml` and `.github/workflows/lab13_end_to_end.yml`: updated automation.
- `docs/13_LAB_EXECUTION_GUIDE.md`: setup and execution guide.

## Execution verification boundary

The source tree and Python syntax were checked. Labs 1, 2, 8, 10, and the standalone Lab 13 integration check passed in the available validation environment. The full sequential run was not claimed because MLflow and Docker were unavailable in that environment. Run `python pipelines/run_all_13_labs.py` on the target machine after installing requirements and starting Docker Desktop; inspect `artifacts/lab13_execution_report.json`.
