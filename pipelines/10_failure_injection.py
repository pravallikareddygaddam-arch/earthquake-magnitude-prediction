
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_failure_tests():
    print("[INFO] Starting Earthquake Failure Tests...")

    raw_path = PROJECT_ROOT / "data/raw/Earthquake.csv.csv"
    model_path = PROJECT_ROOT / "models/earthquake_model.pkl"
    report_path = PROJECT_ROOT / "artifacts/lab9_failure_report.json"

    results = []

    # Test 1: Missing dataset
    print("\n[TEST 1] Missing dataset")
    backup_path = raw_path.with_suffix(".csv.lab9_backup")

    try:
        os.replace(raw_path, backup_path)

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "pipelines/09_inference.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        passed = result.returncode != 0
        results.append({
            "test": "missing_dataset",
            "expected": "FAIL_SAFELY",
            "passed": passed
        })

        print("[PASS]" if passed else "[FAIL]")

    finally:
        if backup_path.exists():
            os.replace(backup_path, raw_path)

    # Test 2: Missing model
    print("\n[TEST 2] Missing model")
    backup_model = model_path.with_name("earthquake_model.lab9_backup")

    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        os.replace(model_path, backup_model)

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "pipelines/09_inference.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        passed = result.returncode != 0
        results.append({
            "test": "missing_model",
            "expected": "FAIL_SAFELY",
            "passed": passed
        })

        print("[PASS]" if passed else "[FAIL]")

    finally:
        if backup_model.exists():
            os.replace(backup_model, model_path)

    # Test 3: Dataset schema
    print("\n[TEST 3] Dataset schema")
    try:
        df = pd.read_csv(raw_path)
        required_columns = {"Date", "Time", "Latitude", "Longitude", "Magnitude"}
        missing_columns = required_columns - set(df.columns)

        passed = not missing_columns
        results.append({
            "test": "dataset_schema",
            "expected": "REQUIRED_COLUMNS_PRESENT",
            "missing_columns": sorted(missing_columns),
            "passed": passed
        })

        print("[PASS]" if passed else f"[FAIL] Missing: {missing_columns}")

    except Exception as exc:
        results.append({
            "test": "dataset_schema",
            "passed": False,
            "error": str(exc)
        })
        print(f"[FAIL] {exc}")

    report = {
        "stage": "Lab 9 - Earthquake Failure Handling",
        "total_tests": len(results),
        "passed_tests": sum(item["passed"] for item in results),
        "overall_status": (
            "PASSED" if all(item["passed"] for item in results)
            else "FAILED"
        ),
        "results": results
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\n[INFO] Report saved to {report_path}")
    print(f"[INFO] Overall status: {report['overall_status']}")

    return report["overall_status"] == "PASSED"


if __name__ == "__main__":
    sys.exit(0 if run_failure_tests() else 1)
