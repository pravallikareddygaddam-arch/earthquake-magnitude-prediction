# Earthquake Magnitude Prediction — MLOps Labs 3–6

This project is the Earthquake Magnitude Prediction project from the supplied Lab 3–6 template, organized using the same MLOps structure used in the Telco Customer Churn project.

**Task:** Regression  
**Target:** `Magnitude`  
**Dataset:** `data/raw/Earthquake.csv.csv`

## Project structure

```text
earthquake-magnitude-prediction-labs3-6/
├── data/
│   ├── raw/
│   │   └── Earthquake.csv.csv
│   └── processed/
├── pipelines/
│   ├── run_lab3_baseline.py
│   ├── run_lab4_tracking.py
│   ├── run_lab5_pipeline.py
│   ├── run_lab6_registry.py
│   └── run_lab6_batch_inference.py
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── train_mlflow.py
│   ├── validate_reproducibility.py
│   ├── validate_data.py
│   ├── preprocess_pipeline.py
│   ├── validate_outputs.py
│   ├── train_registry.py
│   ├── automate_lifecycle.py
│   └── generate_registry_report.py
├── models/
├── artifacts/
├── outputs/
├── logs/
├── requirements.txt
└── .gitignore
```

## Run

From the project root with the virtual environment activated:

```powershell
python pipelines/run_lab3_baseline.py
python pipelines/run_lab4_tracking.py
python pipelines/run_lab5_pipeline.py
python pipelines/run_lab6_registry.py
python pipelines/run_lab6_batch_inference.py
```

For MLflow UI:

```powershell
mlflow server --port 5000
```

Then open `http://127.0.0.1:5000`.

Generated MLflow files, model binaries, processed data, plots, reports, and batch outputs are ignored by Git.
