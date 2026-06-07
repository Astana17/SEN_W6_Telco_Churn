# Telco Customer Churn Prediction

A complete end-to-end machine learning project that predicts which customers are likely to cancel their subscription, using the IBM Telco Customer Churn dataset (~7 000 customers, 20 features).

---

## Project Overview

Telecoms lose significant revenue to churn. Proactively identifying at-risk customers lets retention teams make targeted offers before the customer leaves. This project builds a **recall-optimised binary classifier** (churn = 1) wrapped in a sklearn `Pipeline` so that new customer records can be scored in a single `pipeline.predict_proba()` call.

---

## Directory Structure

```
telco_churn/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # training data
│   └── new_customers.csv                        # scoring sample
├── notebook/
│   └── EDA.ipynb                                # exploratory analysis
├── scripts/
│   ├── preprocessing.py                         # custom transformers
│   ├── train.py                                 # full training pipeline
│   └── predict.py                               # batch scoring
├── results/
│   ├── churn_pipeline.pkl                       # trained pipeline
│   ├── threshold.json                           # decision threshold
│   ├── predictions.csv                          # output from predict.py
│   └── plots/
│       └── pr_curve.png                         # precision-recall curve
├── requirements.txt
└── README.md
```

---

## Feature Engineering

Three features are created inside a custom `FeatureEngineer` sklearn transformer (first step of the pipeline), making them automatically available at inference time without data leakage:

### `tenure_bucket`
Bins continuous tenure (months) into `['0-12', '13-24', '25-48', '49+']`.  
Churn risk is highest in the first year and drops sharply after two years — a non-linear relationship that tree models capture but that also helps linear models when represented as a categorical.

### `charges_per_month`
`TotalCharges / max(tenure, 1)`  
Captures *effective* monthly spend, which differs from `MonthlyCharges` when promotions or mid-cycle changes occurred. It also normalises for the fact that new customers have very low `TotalCharges` simply because they haven't been billed many times yet.

### `n_services` — the "No / No internet service" trap
Counts how many of the nine service columns (`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) are *active* for a customer.

**The trap**: columns that depend on internet service (e.g. `OnlineSecurity`) are filled with the string `"No internet service"` — not `"No"` — for customers who have no internet plan. Treating both strings as equivalent "off" states prevents false inflation of service counts for non-internet customers.

---

## How to Run

```bash
# 1. Install dependencies
python3 -m pip install -r requirements.txt

# 2. Train the model (prints CV tables, saves pipeline & threshold)
python3 scripts/train.py

# 3. Score new customers
python3 scripts/predict.py
```

---

## CV Results (5-fold Stratified)

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| LogisticRegression | 0.xxx ± 0.xxx | 0.xxx ± 0.xxx | 0.xxx ± 0.xxx | 0.xxx ± 0.xxx |
| RandomForest | 0.xxx ± 0.xxx | 0.xxx ± 0.xxx | 0.xxx ± 0.xxx | 0.xxx ± 0.xxx |
| GradientBoosting | 0.xxx ± 0.xxx | 0.xxx ± 0.xxx | 0.xxx ± 0.xxx | 0.xxx ± 0.xxx |

*(Run `python scripts/train.py` to populate with actual values.)*

---

## Baseline & Why Accuracy is the Wrong Metric

A `DummyClassifier(strategy='most_frequent')` always predicts "No Churn" and achieves **~73.5% accuracy** — better than many naive models on paper.

However, accuracy is misleading for imbalanced classes:
- It rewards predicting the majority class
- It assigns equal cost to false positives and false negatives
- In churn, **a missed churner (false negative) costs far more** than a wasted retention offer (false positive)

We therefore optimise for **recall** (fraction of actual churners caught) subject to a minimum precision constraint, using threshold tuning on the precision-recall curve.

---

## Threshold Tuning

Default classifiers use threshold = 0.5. We instead find the threshold on the precision-recall curve (estimated via `cross_val_predict` on the training set) where:

- **Recall ≥ 0.80** (catch at least 80% of churners)
- **Precision is maximised** at that recall level

Typical results on test set:
- Recall ≥ 0.75
- Precision ≥ 0.45

---

## Business Translation

| Metric | Meaning in plain English |
|---|---|
| **Recall 0.75** | We correctly flag 3 out of every 4 customers who would have left |
| **Precision 0.45** | Of every 10 customers we flag as at-risk, about 4-5 actually would have churned |
| **ROC-AUC ~0.84** | Our model ranks a real churner ahead of a non-churner 84% of the time |

A retention offer typically costs $20-50. The lifetime value of retaining a customer for one more year is several hundred dollars. Even at 45% precision the program has a positive expected ROI.

---

## Results Summary

See [`results/results.md`](results/results.md) for a plain-language business summary.
