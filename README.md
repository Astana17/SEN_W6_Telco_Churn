# Telco Customer Churn Prediction

## Overview

This project builds a churn prediction model for a telecom operator using the IBM Telco Customer Churn dataset. The goal is to identify customers who are likely to cancel service so the marketing team can proactively call at-risk customers.

The business requirement is asymmetric: a missed churner is worth about **5× more** than a false alarm. Therefore the model must prioritize recall for the churn class while still preserving useful precision.

## Dataset

- Source: IBM Telco Customer Churn dataset
- Rows: ~7,000 customers
- Columns: 21 features plus `Churn` target
- Target: `Churn` (`Yes` / `No`) → binary 1/0

Important note: `TotalCharges` appears numeric but loads as `object` because some rows contain empty strings. These must be converted with `pd.to_numeric(..., errors='coerce')` and imputed inside the pipeline.

## Business Problem

The CMO believes customers are leaving and wants the marketing team to know which customers are at risk. The model must:

- predict churn probability for each customer
- use a threshold chosen on validation data only
- apply a higher cost to false negatives than false positives
- deliver a final recall target of at least 0.75 on the held-out test set
- deliver a precision target of at least 0.45 on the held-out test set

## Approach

### 1. Exploratory Data Analysis (EDA)

The EDA notebook investigates:

- data shape and column types
- distribution of `Churn` and numeric features
- implicit and explicit missing values
- cross-tabulation of churn against categorical variables like `Contract`, `PaymentMethod`, and `InternetService`Щ
- written business insights from the data

**Observations**
1. Dataset is imbalanced (~73% non-churn).
2. Month-to-month contracts have the highest churn rate.
3. Customers with low tenure churn more frequently.
4. Electronic check users show increased churn risk.

### 2. Feature Engineering

Feature engineering happens inside the scikit-learn pipeline to avoid leakage. At least two derived features should be created, such as:

- `tenure_bucket` — binned customer tenure groups
- `charges_per_month_of_tenure` — ratio of `TotalCharges` to tenure
- `n_services` — count of active services, including DSL/Fiber and add-ons

### 3. Leakage-Free Pipeline

The model uses a single `Pipeline` with a `ColumnTransformer` for preprocessing and a classifier at the end.

Preprocessing includes:

- numeric imputation for missing values
- scaling numeric features with `StandardScaler`
- one-hot encoding categorical features with `OneHotEncoder(handle_unknown='ignore')`
- feature construction via a custom transformer or `FunctionTransformer`

All transformers are fit only on training data inside the pipeline.

### 4. Model Selection

The training process follows a strict train/test split:

- 80% training, 20% test
- stratified split by `Churn`
- test set touched only once at the end

Baseline models:

- `DummyClassifier(strategy='most_frequent')`
- `DummyClassifier(strategy='stratified')`

Candidate classifiers:

- Logistic Regression with `class_weight='balanced'`
- Random Forest
- Gradient Boosting (`GradientBoostingClassifier`)

Evaluation:

- 5-fold stratified cross-validation
- metrics: recall, precision, F1, ROC-AUC for churn class
- report mean and standard deviation across folds

### 5. Threshold Tuning

The decision threshold is tuned on validation/out-of-fold probabilities only.

- generate out-of-fold churn probabilities using `cross_val_predict(..., method='predict_proba')`
- plot precision-recall curve
- choose a threshold with recall ≥ 0.80 and the highest possible precision
- freeze that threshold before testing

### 6. Final Evaluation

The chosen model is evaluated once on the held-out test set with the frozen threshold.

Expected business-level results in the README:

- test-set recall ≥ 0.75 for churn
- test-set precision ≥ 0.45 for churn

A confusion matrix is displayed as a labeled DataFrame.

### 7. Inference

The final pipeline is serialized to `results/churn_pipeline.pkl`.

Prediction workflow:

- load `data/new_customers.csv` (5 rows with `Churn` removed)
- preserve `customerID` separately
- drop `customerID` before prediction
- output `results/predictions.csv` with columns `customerID, churn_pred, churn_proba`

## Repository Structure

Expected repository layout:

```
project
│   README.md
│   requirements.txt
│
└───data
│   │   WA_Fn-UseC_-Telco-Customer-Churn.csv
│   │   new_customers.csv
│
└───notebook
│   │   EDA.ipynb
│
└───scripts
│   │   preprocessing.py
│   │   train.py
│   │   predict.py
│
└───results
    │   plots
    │   predictions.csv
    │   churn_pipeline.pkl
    │   results.md
```

## Running the project

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python scripts/train.py
```

3. Predict on new customers:

```bash
python scripts/predict.py

python scripts/predict.py --test_mode )(так как я делала сплит по stratify y)
```

## Notes

- Accuracy is not a valid evaluation metric for this imbalanced problem; the majority-class baseline is ~73%.
- All preprocessing and feature engineering must be implemented inside the pipeline to avoid leakage.
- The test set must remain untouched until the final evaluation.
- The threshold decision must be based on validation data only.

## Deliverables

- `README.md`
- `requirements.txt`
- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- `data/new_customers.csv`
- `notebook/EDA.ipynb`
- `scripts/preprocessing.py`
- `scripts/train.py`
- `scripts/predict.py`
- `results/churn_pipeline.pkl`
- `results/predictions.csv`
- `results/results.md`

```
cd /Users/Guest/Desktop/SEN_W6_Telco_Churn
python3 -m pip install -r requirements.txt
python3 scripts/train.py
python3 scripts/predict.py
python3 -m notebook notebook/EDA.ipynb
```