"""Smoke-check the actual Telco deployment artifact without a running server."""
import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent

def sample_customer():
    schema = json.loads((ROOT / "streamlit/data_schema.json").read_text())
    return {
        **{key: stats["median"] for key, stats in schema["numerical"].items()},
        **{key: max(info["value_counts"], key=info["value_counts"].get)
           for key, info in schema["categorical"].items()},
    }

def main():
    model = joblib.load(ROOT / "api/models/telco_churn_best_model.pkl")
    features = pd.DataFrame([sample_customer()])
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0, list(model.classes_).index(1)])
    assert prediction in (0, 1) and 0 <= probability <= 1
    print(json.dumps({"churn_prediction": prediction, "churn_probability": probability}))

if __name__ == "__main__":
    main()
