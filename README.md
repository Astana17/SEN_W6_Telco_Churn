# Telco Customer Churn Prediction

## Overview

This project builds a churn prediction model for a telecom operator using the IBM Telco Customer Churn dataset. The marketing team needs to know which customers are likely to cancel before they leave.

A missed churner (false negative) is roughly **5× more expensive** than calling a loyal customer (false positive). The model therefore prioritises **recall** for the churn class, while still keeping precision usable for outbound calls.

## Dataset

- Source: IBM Telco Customer Churn dataset
- Rows: 7,043 customers
- Target: `Churn` (`Yes` / `No` → 1 / 0)
- Class balance: ~73% non-churn, ~27% churn

`TotalCharges` loads as `object` because some rows contain empty strings. These are converted with `pd.to_numeric(..., errors='coerce')` and imputed inside the sklearn pipeline.

## Why not accuracy?

A `DummyClassifier(strategy='most_frequent')` already reaches about **73% accuracy** by always predicting "No Churn". That baseline is useless for finding churners, so accuracy is not used as the main metric. Instead we track recall, precision, F1, and ROC-AUC for the churn class.

## Feature engineering (inside Pipeline)

All feature engineering lives in a custom `FeatureEngineer` transformer so training and inference stay identical.

| Feature | Justification |
|---|---|
| `tenure_bucket` | Groups customers into tenure stages (0–12, 13–24, 25–48, 49+ months). Short-tenure customers churn much more often than long-tenure customers. |
| `charges_per_month_of_tenure` | `TotalCharges / max(tenure, 1)`. Captures average spend rate and helps separate new low-spend churners from stable long-term customers. |
| `n_services` | Count of active services. A service counts as active when its value is neither `No`, `No internet service`, nor `No phone service`. This correctly treats DSL/Fiber as active internet and avoids the naive `== "Yes"` trap on `InternetService`. |

## Pipeline architecture

A single leakage-free sklearn `Pipeline`:

1. `FeatureEngineer` — derived features
2. `ColumnTransformer` — numeric imputation + scaling, categorical one-hot encoding
3. Classifier — Logistic Regression / Random Forest / Gradient Boosting

All fitting happens inside cross-validation folds or on the training split only.

## Model selection

- **Split:** 80% train / 20% test, stratified, `random_state=42`
- **Baselines:** `DummyClassifier` (`most_frequent`, `stratified`)
- **Candidates:** Logistic Regression (`class_weight='balanced'`), Random Forest, Gradient Boosting
- **CV:** 5-fold stratified cross-validation
- **Tuning:** `GridSearchCV` on the best model (`scoring='recall'`), wrapping the full pipeline

### Cross-validation results (mean ± std)

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Dummy (most_frequent) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| Dummy (stratified) | 0.278 ± 0.025 | 0.275 ± 0.025 | 0.276 ± 0.025 | 0.507 ± 0.017 |
| **Logistic Regression** | **0.797 ± 0.035** | **0.518 ± 0.016** | **0.628 ± 0.021** | **0.846 ± 0.011** |
| Random Forest | 0.464 ± 0.031 | 0.640 ± 0.033 | 0.538 ± 0.030 | 0.828 ± 0.011 |
| Gradient Boosting | 0.522 ± 0.025 | 0.662 ± 0.036 | 0.583 ± 0.026 | 0.848 ± 0.011 |

All three candidate models beat both dummy baselines on recall and ROC-AUC.

**Best model:** Logistic Regression after `GridSearchCV` (`C=10`, `solver=liblinear`).

## Threshold tuning

The default 0.5 cutoff is not a business decision. Out-of-fold churn probabilities were generated on the training set with `cross_val_predict(..., method='predict_proba')`. The precision-recall curve was used to pick the threshold with **recall ≥ 0.80** and the **highest precision**.

- **Frozen threshold:** 0.4971 (chosen on validation only, never on the test set)

## Final test results (single evaluation)

| Metric | Value |
|---|---|
| Recall (churn) | **0.7995** |
| Precision (churn) | **0.5008** |
| F1 (churn) | 0.6159 |
| ROC-AUC | 0.8410 |

### Confusion matrix (test set)

| | Predicted No Churn | Predicted Churn |
|---|---|---|
| Actual No Churn | 737 | 298 |
| Actual Churn | 75 | 299 |

Business summary is in `results/results.md`.

## Repository structure

```
project
│   README.md
│   requirements.txt
│
├── data
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── new_customers.csv
│
├── notebook
│   └── EDA.ipynb
│
├── scripts
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
└── results
    ├── plots/
    ├── predictions.csv
    ├── churn_pipeline.pkl
    └── results.md
```

## Running the project

```bash
pip install -r requirements.txt
python scripts/train.py
python scripts/predict.py
```

Or with Make:

```bash
make setup
```

## Deliverables

- `README.md` — this file
- `requirements.txt`
- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- `data/new_customers.csv` — 5 rows from raw data, `Churn` removed
- `notebook/EDA.ipynb`
- `scripts/preprocessing.py` — custom transformers
- `scripts/train.py`
- `scripts/predict.py`
- `results/churn_pipeline.pkl`
- `results/predictions.csv` — columns `customerID`, `churn_pred`, `churn_proba`
- `results/results.md`
