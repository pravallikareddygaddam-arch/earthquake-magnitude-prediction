from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from api.main import app

def main():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200, health.text
    metadata = client.get("/metadata")
    assert metadata.status_code == 200, metadata.text

    sample = pd.read_csv(ROOT / "data/raw/Earthquake.csv.csv").iloc[0].drop(
        labels=["Magnitude", "ID"], errors="ignore"
    ).to_dict()
    # Convert numpy scalar values and NaN values to JSON-safe values.
    features = {}
    for key, value in sample.items():
        if pd.isna(value):
            features[key] = None
        elif hasattr(value, "item"):
            features[key] = value.item()
        else:
            features[key] = value

    prediction = client.post("/predict", json={"features": features})
    assert prediction.status_code == 200, prediction.text
    result = prediction.json()
    assert "predicted_magnitude" in result, result

    print("[SUCCESS] Lab 10 API health, metadata and prediction endpoints passed.")
    print("Health:", health.json())
    print("Metadata:", metadata.json())
    print("Sample prediction:", result)

if __name__ == "__main__":
    main()
