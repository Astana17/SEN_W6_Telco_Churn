"""
predict.py — Apply trained pipeline to new customers.

Loads:
    results/churn_pipeline.pkl
    results/threshold.json
    data/new_customers.csv

Writes:
    results/predictions.csv  (columns: customerID, churn_pred, churn_proba)

Run:
    python scripts/predict.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'data' / 'new_customers.csv'
MODEL_PATH = ROOT / 'results' / 'churn_pipeline.pkl'
THRESHOLD_PATH = ROOT / 'results' / 'threshold.json'
OUTPUT_PATH = ROOT / 'results' / 'predictions.csv'

# preprocessing module must be importable so Pipeline can unpickle FeatureEngineer
sys.path.insert(0, str(ROOT / 'scripts'))

# ---------------------------------------------------------------------------
# Load model and threshold
# ---------------------------------------------------------------------------
print("Loading pipeline …")
pipeline = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH) as f:
    threshold = json.load(f)['threshold']
print(f"  Threshold : {threshold:.4f}")

# ---------------------------------------------------------------------------
# Load new customers
# ---------------------------------------------------------------------------
print(f"\nLoading {DATA_PATH} …")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# Stash customerID
if 'customerID' in df.columns:
    customer_ids = df['customerID'].copy()
    df = df.drop(columns=['customerID'])
else:
    customer_ids = pd.Series(range(len(df)), name='customerID')

# ---------------------------------------------------------------------------
# Only the type coercion that happens before the Pipeline in train.py
# FeatureEngineer and all sklearn transformers run inside the pipeline
# ---------------------------------------------------------------------------
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
probas = pipeline.predict_proba(df)[:, 1]
preds = (probas >= threshold).astype(int)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out = pd.DataFrame({
    'customerID': customer_ids.values,
    'churn_pred': preds,
    'churn_proba': probas.round(4),
})
out.to_csv(OUTPUT_PATH, index=False)
print(f"\nPredictions saved → {OUTPUT_PATH}")
print(out.to_string(index=False))

# ---------------------------------------------------------------------------
# Save predict report
# ---------------------------------------------------------------------------
report_path = ROOT / 'results' / 'predict_report.txt'
n_total = len(out)
n_at_risk = int(out['churn_pred'].sum())
n_safe = n_total - n_at_risk

report_lines = []
report_lines.append("=" * 50)
report_lines.append("ОТЧЁТ О ПРЕДСКАЗАНИЯХ")
report_lines.append("=" * 50)
report_lines.append(f"\nВсего клиентов: {n_total}")
report_lines.append(f"Под риском оттока (churn_pred=1): {n_at_risk}")
report_lines.append(f"Лояльные клиенты (churn_pred=0): {n_safe}")
report_lines.append(f"\nПорог срабатывания: {threshold:.4f}")
report_lines.append("\n--- Детали по каждому клиенту ---")
for _, row in out.iterrows():
    risk = "РИСК УХОДА" if row['churn_pred'] == 1 else "ОК"
    report_lines.append(
        f"  {row['customerID']:15s}  вероятность={row['churn_proba']:.3f}  [{risk}]"
    )
report_lines.append("\n--- Рекомендация ---")
if n_at_risk > 0:
    report_lines.append(f"Передать {n_at_risk} клиентов в отдел удержания для обзвона.")
else:
    report_lines.append("Клиентов под риском не обнаружено.")

with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"\nPredict report saved → {report_path}")
