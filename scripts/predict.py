import os
import pandas as pd
from joblib import load

MODEL_PATH = os.path.join("results", "churn_pipeline.pkl")
INPUT_PATH = os.path.join("data", "new_customers.csv")
OUTPUT_PATH = os.path.join("results", "predictions.csv")


def main():
    model = load(MODEL_PATH)
    data = pd.read_csv(INPUT_PATH)
    if "customerID" not in data.columns:
        raise ValueError("new_customers.csv must contain customerID column")

    customer_ids = data["customerID"]
    features = data.drop(columns=["customerID"])

    churn_proba = model.predict_proba(features)[:, 1]
    churn_pred = model.predict(features)

    output = pd.DataFrame(
        {
            "customerID": customer_ids,
            "churn_pred": churn_pred,
            "churn_proba": churn_proba,
        }
    )
    os.makedirs("results", exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
