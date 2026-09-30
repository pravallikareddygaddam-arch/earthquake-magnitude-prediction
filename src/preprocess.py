from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Earthquake.csv.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

def load_and_prepare():
    if not RAW.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW}")
    df = pd.read_csv(RAW)
    if "Magnitude" not in df.columns:
        raise KeyError("Target column 'Magnitude' was not found.")
    df = df.drop_duplicates().reset_index(drop=True)
    work = df.copy()

    if "Date" in work.columns:
        d = pd.to_datetime(work["Date"], errors="coerce")
        work["Year"] = d.dt.year
        work["Month"] = d.dt.month
        work["Day"] = d.dt.day

    if "Time" in work.columns:
        parts = work["Time"].astype(str).str.extract(
            r"(?P<Hour>\d{1,2}):(?P<Minute>\d{1,2}):(?P<Second>\d{1,2})"
        )
        for c in ["Hour", "Minute", "Second"]:
            work[c] = pd.to_numeric(parts[c], errors="coerce")

    drop = ["Magnitude"] + [c for c in ["ID", "Date", "Time"] if c in work.columns]
    X = work.drop(columns=drop)
    y = pd.to_numeric(work["Magnitude"], errors="coerce")
    valid = y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)

def build_preprocessor(X):
    nums = X.select_dtypes(include=np.number).columns.tolist()
    cats = X.select_dtypes(include="object").columns.tolist()
    transformers = []

    if nums:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            nums
        ))

    if cats:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]),
            cats
        ))

    return ColumnTransformer(transformers=transformers)

def preprocess_and_save():
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    Xtr = preprocessor.fit_transform(X_train)
    Xte = preprocessor.transform(X_test)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    np.save(PROCESSED / "X_train_final.npy", Xtr)
    np.save(PROCESSED / "X_test_final.npy", Xte)
    np.save(PROCESSED / "y_train.npy", y_train.to_numpy(dtype=float))
    np.save(PROCESSED / "y_test.npy", y_test.to_numpy(dtype=float))

    pd.DataFrame(Xtr).to_csv(PROCESSED / "X_train_processed.csv", index=False)
    pd.DataFrame(Xte).to_csv(PROCESSED / "X_test_processed.csv", index=False)
    pd.DataFrame({"Magnitude": y_train}).to_csv(PROCESSED / "y_train.csv", index=False)
    pd.DataFrame({"Magnitude": y_test}).to_csv(PROCESSED / "y_test.csv", index=False)

    joblib.dump(preprocessor, MODELS / "preprocessor.pkl")

    metadata = {
        "dataset_name": "Earthquake Magnitude Prediction",
        "target": "Magnitude",
        "train_shape": list(Xtr.shape),
        "test_shape": list(Xte.shape),
        "feature_columns": list(X.columns),
        "random_state": 42
    }
    with open(PROCESSED / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, default=str)

    print("Preprocessing completed successfully!")
    print("X_train:", Xtr.shape, "X_test:", Xte.shape)
    return Xtr, Xte, y_train, y_test

if __name__ == "__main__":
    preprocess_and_save()
