from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/Earthquake.csv.csv"
OUT = ROOT / "artifacts/lab02_data_quality_report.json"
REQUIRED = {"Date", "Time", "Latitude", "Longitude", "Magnitude"}

def main():
    df = pd.read_csv(DATA)
    missing_columns = sorted(REQUIRED - set(df.columns))
    checks = {
        "file_exists": DATA.exists(),
        "dataset_not_empty": not df.empty,
        "required_columns_present": not missing_columns,
        "target_has_values": bool(df["Magnitude"].notna().any()) if "Magnitude" in df else False,
        "target_numeric": bool(pd.to_numeric(df["Magnitude"], errors="coerce").notna().all()) if "Magnitude" in df else False,
        "latitude_in_range": bool(df["Latitude"].dropna().between(-90, 90).all()) if "Latitude" in df else False,
        "longitude_in_range": bool(df["Longitude"].dropna().between(-180, 180).all()) if "Longitude" in df else False
    }
    report = {
        "experiment": "Lab 2 - Data Quality and Schema Validation",
        "row_count": int(len(df)),
        "missing_required_columns": missing_columns,
        "checks": checks,
        "status": "PASSED" if all(checks.values()) else "FAILED"
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASSED":
        raise SystemExit("[FAILED] Lab 2 data quality checks")
    print(f"[SUCCESS] Lab 2 report: {OUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
