"""
Predict churn for new customers using the saved pipeline.

Usage:
    python scripts/predict.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from joblib import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocessing  # noqa: F401 — required for unpickling custom transformers

MODEL_PATH = os.path.join("results", "churn_pipeline.pkl")
INPUT_PATH = os.path.join("data", "new_customers.csv")
OUTPUT_PATH = os.path.join("results", "predictions.csv")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\nRun training first: python scripts/train.py"
        )

    model = load(MODEL_PATH)
    data = pd.read_csv(INPUT_PATH)

    if "customerID" not in data.columns:
        raise ValueError("new_customers.csv must contain a customerID column")

    customer_ids = data["customerID"]
    features = data.drop(columns=["customerID", "Churn"], errors="ignore")

    churn_proba = model.predict_proba(features)[:, 1]
    churn_pred = model.predict(features)

    output = pd.DataFrame(
        {
            "customerID": customer_ids,
            "churn_pred": churn_pred,
            "churn_proba": churn_proba.round(4),
        }
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"[predict] saved predictions to {OUTPUT_PATH}")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
