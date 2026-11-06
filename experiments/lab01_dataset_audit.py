from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/Earthquake.csv.csv"
OUT = ROOT / "artifacts/lab01_dataset_audit.json"

def main():
    df = pd.read_csv(DATA)
    report = {
        "experiment": "Lab 1 - Dataset Audit and Understanding",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": df.columns.tolist(),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "target": "Magnitude",
        "target_min": float(df["Magnitude"].min()),
        "target_max": float(df["Magnitude"].max()),
        "target_mean": float(df["Magnitude"].mean())
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[SUCCESS] Lab 1 report: {OUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
