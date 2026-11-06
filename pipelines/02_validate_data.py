
from pathlib import Path
import pandas as pd
import yaml

REQUIRED_COLUMNS = [
    "Date",
    "Time",
    "Latitude",
    "Longitude",
    "Type",
    "Depth",
    "Magnitude",
]

def main():
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_path = Path(config["data"]["raw_path"])

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    print(f"[INFO] Dataset loaded: {data_path}")
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Columns: {len(df.columns)}")

    missing = [column for column in REQUIRED_COLUMNS
               if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["Magnitude"].isna().all():
        raise ValueError("Magnitude has no valid values.")

    if not pd.api.types.is_numeric_dtype(df["Magnitude"]):
        raise ValueError("Magnitude must be numeric.")

    print("[SUCCESS] Earthquake dataset validation PASSED.")

if __name__ == "__main__":
    main()
