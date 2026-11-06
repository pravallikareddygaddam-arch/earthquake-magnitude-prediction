from pathlib import Path
import json
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

# Each lab is a separate, named experiment. Lab 13 is a distinct integration
# verification; it does not simply repeat the Lab 10 API smoke test.
STAGES = [
    ("Lab 1 - Dataset audit", "experiments/lab01_dataset_audit.py"),
    ("Lab 2 - Data quality and schema validation", "experiments/lab02_data_quality.py"),
    ("Lab 3 - Baseline ML pipeline", "pipelines/run_lab3_baseline.py"),
    ("Lab 4 - MLflow experiment tracking and reproducibility", "pipelines/run_lab4_tracking.py"),
    ("Lab 5 - Production data validation and preprocessing", "pipelines/run_lab5_pipeline.py"),
    ("Lab 6 - Model registry and batch/offline inference", "experiments/lab06_registry_and_batch.py"),
    ("Lab 7 - Model evaluation and quality gate", "experiments/lab07_evaluation_quality_gate.py"),
    ("Lab 8 - Continuous integration validation", "experiments/lab08_ci_validation.py"),
    ("Lab 9 - Failure injection and recovery", "pipelines/10_failure_injection.py"),
    ("Lab 10 - Online inference API", "experiments/lab10_api_smoke_test.py"),
    ("Lab 11 - Docker build and container health", "experiments/lab11_docker_check.py"),
    ("Lab 12 - Drift and model performance monitoring", "experiments/lab12_monitoring.py"),
    ("Lab 13 - End-to-end integration verification", "experiments/lab13_end_to_end.py"),
]


def main():
    results = []
    started = datetime.now(timezone.utc).isoformat()
    for index, (name, script) in enumerate(STAGES, start=1):
        print("\n" + "=" * 78, flush=True)
        print(f"{name} ({index}/13)", flush=True)
        print("=" * 78, flush=True)
        path = ROOT / script
        if not path.is_file():
            code, detail = 2, f"Script not found: {script}"
            print(f"[FAILED] {detail}", flush=True)
        else:
            completed = subprocess.run([PYTHON, str(path)], cwd=ROOT)
            code, detail = completed.returncode, None
        results.append({
            "lab": index,
            "experiment": name,
            "script": script,
            "status": "PASSED" if code == 0 else "FAILED",
            "exit_code": code,
            **({"detail": detail} if detail else {}),
        })
        if code != 0:
            print(f"\n[HALT] {name} failed. Fix this lab before continuing.", flush=True)
            break

    passed = sum(item["status"] == "PASSED" for item in results)
    report = {
        "project": "Earthquake Magnitude Prediction MLOps",
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_labs": len(results),
        "passed_labs": passed,
        "total_labs": 13,
        "overall_status": "PASSED" if len(results) == 13 and passed == 13 else "INCOMPLETE_OR_FAILED",
        "experiments": results,
    }
    out = ROOT / "artifacts/lab13_execution_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print("FINAL 13-LAB EXECUTION REPORT")
    print("=" * 78)
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {out.relative_to(ROOT)}")
    return 0 if report["overall_status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
