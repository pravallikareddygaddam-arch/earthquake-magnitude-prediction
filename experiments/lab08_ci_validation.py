from pathlib import Path
import compileall
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    checks = {}
    required_files = [
        "data/raw/Earthquake.csv.csv",
        "config.yaml",
        "requirements.txt",
        ".github/workflows/lab8_ci.yml",
        ".github/workflows/lab9_failure_tests.yml",
        ".github/workflows/lab13_end_to_end.yml",
        "pipelines/run_all_13_labs.py",
        "api/main.py",
    ]
    missing = [item for item in required_files if not (ROOT / item).is_file()]
    checks["required_project_files_exist"] = not missing

    dataset_path = ROOT / "data/raw/Earthquake.csv.csv"
    df = pd.read_csv(dataset_path) if dataset_path.exists() else pd.DataFrame()
    required_columns = {"Date", "Time", "Latitude", "Longitude", "Magnitude"}
    checks["dataset_not_empty"] = not df.empty
    checks["dataset_required_columns"] = required_columns.issubset(df.columns)
    checks["magnitude_numeric"] = (
        "Magnitude" in df.columns
        and bool(pd.to_numeric(df["Magnitude"], errors="coerce").notna().all())
    )

    compile_ok = compileall.compile_dir(str(ROOT / "src"), quiet=1)
    compile_ok = compileall.compile_dir(str(ROOT / "pipelines"), quiet=1) and compile_ok
    compile_ok = compileall.compile_dir(str(ROOT / "experiments"), quiet=1) and compile_ok
    compile_ok = compileall.compile_dir(str(ROOT / "api"), quiet=1) and compile_ok
    checks["python_syntax_compilation"] = bool(compile_ok)

    workflow = (ROOT / ".github/workflows/lab8_ci.yml").read_text(encoding="utf-8")
    checks["github_actions_workflow_configured"] = all(
        token in workflow for token in ("actions/checkout", "actions/setup-python", "workflow_dispatch")
    )

    report = {
        "experiment": "Lab 8 - Continuous Integration",
        "checks": checks,
        "missing_files": missing,
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "status": "PASSED" if all(checks.values()) else "FAILED",
    }
    out = ROOT / "artifacts/lab08_ci_validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[INFO] Report saved: {out.relative_to(ROOT)}")
    if report["status"] != "PASSED":
        raise SystemExit("[FAILED] Lab 8 CI validation")
    print("[SUCCESS] Lab 8 CI validation passed.")


if __name__ == "__main__":
    main()
