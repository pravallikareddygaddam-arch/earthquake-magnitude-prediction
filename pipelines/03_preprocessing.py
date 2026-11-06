
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_path = Path(config["data"]["raw_path"])
    output_dir = Path(config["data"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor_path = Path(config["paths"]["preprocessor"])
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    target = config["data"]["target_column"]
    id_column = config["data"].get("id_column")

    # Extract date and time features
    date = pd.to_datetime(df["Date"], errors="coerce")
    time = pd.to_datetime(
        df["Time"], format="%H:%M:%S", errors="coerce"
    )

    df["Year"] = date.dt.year
    df["Month"] = date.dt.month
    df["Day"] = date.dt.day
    df["Hour"] = time.dt.hour
    df["Minute"] = time.dt.minute
    df["Second"] = time.dt.second

    # Clean the regression target
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).copy()

    drop_columns = [target, "Date", "Time"]
    if id_column and id_column in df.columns:
        drop_columns.append(id_column)

    X = df.drop(columns=drop_columns, errors="ignore")
    y = df[target].astype(float)

    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    # Fit only on training data and transform both datasets
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    # Convert sparse matrices to dense numeric arrays
    if hasattr(X_train_final, "toarray"):
        X_train_final = X_train_final.toarray()

    if hasattr(X_test_final, "toarray"):
        X_test_final = X_test_final.toarray()

    X_train_final = np.asarray(X_train_final, dtype=np.float32)
    X_test_final = np.asarray(X_test_final, dtype=np.float32)

    # Save the fitted preprocessor and transformed data
    joblib.dump(preprocessor, preprocessor_path)

    np.save(output_dir / "X_train_final.npy", X_train_final)
    np.save(output_dir / "X_test_final.npy", X_test_final)
    np.save(output_dir / "y_train.npy", y_train.to_numpy())
    np.save(output_dir / "y_test.npy", y_test.to_numpy())

    print("[SUCCESS] Earthquake preprocessing completed.")
    print("Training rows:", len(y_train))
    print("Testing rows:", len(y_test))
    print("Transformed training shape:", X_train_final.shape)
    print("Transformed testing shape:", X_test_final.shape)
    print("Training dtype:", X_train_final.dtype)
    print("Testing dtype:", X_test_final.dtype)
    print("Preprocessor saved to:", preprocessor_path)


if __name__ == "__main__":
    main()
