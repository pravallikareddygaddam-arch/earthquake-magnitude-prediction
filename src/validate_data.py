from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Earthquake.csv.csv"

def validate():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA)

    if "Magnitude" not in df.columns:
        raise ValueError("Missing required column: Magnitude")
    if df.empty:
        raise ValueError("Dataset is empty")
    if df["Magnitude"].isna().all():
        raise ValueError("Magnitude has no usable values")

    print(f"[INFO] Validating dataset (Records: {len(df)})...")
    print("[SUCCESS] Schema Validation PASSED. Dataset is usable.")
    print(f"[INFO] Columns: {len(df.columns)}")
    print("[INFO] Target: Magnitude")
    return True

if __name__ == "__main__":
    validate()
