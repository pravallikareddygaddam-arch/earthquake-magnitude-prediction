# 13-Lab Execution Guide

Run all 13 distinct experiments from the project root:

```powershell
python pipelines/run_all_13_labs.py
```

The runner executes Labs 1 through 13 in order, stops at the first failure, and writes `artifacts/lab13_execution_report.json`. Lab 13 is a separate integration check of the trained model, preprocessor, API health/metadata, single prediction, batch prediction, offline predictions, evaluation report, quality gate, and monitoring report. It is not a repeat of Lab 10.

## Requirements

- Python 3.11 (recommended; matches GitHub Actions and Docker)
- Install dependencies: `python -m pip install -r requirements.txt`
- Docker Desktop running for Lab 11
- Dataset at `data/raw/Earthquake.csv.csv`
- Internet access for dependency installation and MLflow model registry operations if required by the configured MLflow setup

## Individual lab commands

```powershell
python experiments/lab01_dataset_audit.py
python experiments/lab02_data_quality.py
python pipelines/run_lab3_baseline.py
python pipelines/run_lab4_tracking.py
python pipelines/run_lab5_pipeline.py
python experiments/lab06_registry_and_batch.py
python experiments/lab07_evaluation_quality_gate.py
python experiments/lab08_ci_validation.py
python pipelines/10_failure_injection.py
python experiments/lab10_api_smoke_test.py
python experiments/lab11_docker_check.py
python experiments/lab12_monitoring.py
python experiments/lab13_end_to_end.py
```

A script being present is not proof that a lab has passed. Use the generated JSON execution report and GitHub Actions run as execution evidence.
