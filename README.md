# Earthquake Magnitude Prediction — Complete 13-Lab MLOps Project

This repository applies a 13-experiment MLOps workflow to the Earthquake Magnitude Prediction regression task.

- **Dataset:** `data/raw/Earthquake.csv.csv`
- **Target:** `Magnitude`
- **Estimator:** `RandomForestRegressor`
- **Model tracking and registry:** MLflow
- **Serving:** FastAPI + Swagger UI
- **Containerization:** Docker
- **Automation:** GitHub Actions

> The supplied ZIP already contained the Labs 3–9 implementation (including the Lab 8 and Lab 9 workflows). Labs 1–2 and 10–13 have been added in this updated project. Existing project code and dataset are retained.

## 13 experiment map

| Lab | Experiment | Main implementation / evidence |
|---|---|---|
| 1 | Dataset audit and understanding | `experiments/lab01_dataset_audit.py` → `artifacts/lab01_dataset_audit.json` |
| 2 | Data quality and schema validation | `experiments/lab02_data_quality.py` → `artifacts/lab02_data_quality_report.json` |
| 3 | Baseline ML pipeline | `pipelines/run_lab3_baseline.py` |
| 4 | MLflow experiment tracking and reproducibility | `pipelines/run_lab4_tracking.py` |
| 5 | Production data validation and preprocessing pipeline | `pipelines/run_lab5_pipeline.py` |
| 6 | Model registry, lifecycle and batch/offline inference | `pipelines/run_lab6_registry.py`, `pipelines/run_lab6_batch_inference.py` |
| 7 | Regression evaluation and quality gate | `src/evaluate.py`, `pipelines/07_quality_gate.py` |
| 8 | Continuous integration | `experiments/lab08_ci_validation.py`, `.github/workflows/lab8_ci.yml` |
| 9 | Failure injection and recovery validation | `pipelines/10_failure_injection.py`, `.github/workflows/lab9_failure_tests.yml` |
| 10 | Online model serving API | `api/main.py`, `experiments/lab10_api_smoke_test.py` |
| 11 | Docker image build and container health test | `Dockerfile`, `experiments/lab11_docker_check.py` |
| 12 | Data drift and model performance monitoring | `experiments/lab12_monitoring.py` |
| 13 | End-to-end integration verification | `experiments/lab13_end_to_end.py`, `pipelines/run_all_13_labs.py`, `.github/workflows/lab13_end_to_end.yml` |

## Setup (Windows PowerShell)

Run from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run the complete 13-lab sequence

Make sure Docker Desktop is running before Lab 11. Then:

```powershell
python pipelines/run_all_13_labs.py
```

The sequence runs 13 distinct experiments, stops at the first failed experiment, and writes:

`artifacts/lab13_execution_report.json`

The full run includes training, MLflow tracking/registry operations, CI validation, failure/recovery tests, API smoke tests, Docker build and container health validation, drift monitoring, and a distinct end-to-end integration verification. It may take several minutes.

## Run individual experiments

```powershell
python experiments/lab01_dataset_audit.py
python experiments/lab02_data_quality.py
python pipelines/run_lab3_baseline.py
python pipelines/run_lab4_tracking.py
python pipelines/run_lab5_pipeline.py
python experiments/lab06_registry_and_batch.py
python experiments/lab07_evaluation_quality_gate.py
python pipelines/10_failure_injection.py
python experiments/lab10_api_smoke_test.py
python experiments/lab11_docker_check.py
python experiments/lab12_monitoring.py
```

## Run the API and open Swagger

The model and preprocessor must exist in `models/`.

```powershell
docker build -t earthquake-magnitude-api .
docker rm -f earthquake-api 2>$null
docker run -d --name earthquake-api -p 8000:8000 earthquake-magnitude-api
```

Open:

- Swagger UI: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Metadata: `http://localhost:8000/metadata`

### Example Swagger request

Open `POST /predict`, choose **Try it out**, and send a `features` object containing all model input columns (excluding `Magnitude` and `ID`). For example, use a row from `data/raw/Earthquake.csv.csv` and remove those two fields. The API derives year/month/day and hour/minute/second from `Date` and `Time`.

Batch predictions use `POST /predict/batch` with a `records` array of feature objects.

## Lab 12 monitoring

The monitoring experiment compares a reference numeric feature distribution with a controlled shifted copy to demonstrate drift detection. It does not overwrite or alter the source earthquake dataset. It also reports MAE, RMSE and R² on the held-out test data.

Output: `artifacts/lab12_monitoring_report.json`.

## GitHub Actions

The repository contains separate workflows for Labs 8, 9 and 13. Use the **Actions** tab and select **Lab 13 - Earthquake End-to-End MLOps** to run the integrated CI workflow manually.

## Validation note

Project structure and Python syntax were checked. Labs 1, 2, 8, 10, and the standalone Lab 13 integration check were executed successfully in the available validation environment. The complete sequential 13-lab run was not executed in that environment because MLflow and Docker were unavailable there; install `requirements.txt` and run with Docker Desktop active to verify Labs 3–7, 9, and 11–13 end to end. The generated `artifacts/lab13_execution_report.json` from a full run is the authoritative pass/fail record.

## Notes

- The original project artifacts and source files are preserved.
- `mlflow.db`, `mlruns/`, virtual environments, and generated binaries should not be committed unless specifically required for your submission.
- Docker Desktop must be running for Lab 11 and for the full local 13-lab run.
- The Lab 13 report records actual pass/fail results; adding scripts does not itself mean all experiments have already been executed successfully.


See `docs/13_LAB_EXECUTION_GUIDE.md` for the exact commands and prerequisites.
