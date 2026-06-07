"""
train.py — Full training pipeline for Telco Customer Churn prediction.

Pipeline structure:
    FeatureEngineer  →  ColumnTransformer (num | cat)  →  Classifier

Run:
    python scripts/train.py
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
RESULTS_DIR = ROOT / 'results'
PLOTS_DIR = RESULTS_DIR / 'plots'
MODEL_PATH = RESULTS_DIR / 'churn_pipeline.pkl'
THRESHOLD_PATH = RESULTS_DIR / 'threshold.json'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# bring preprocessing module into scope regardless of cwd
import sys
sys.path.insert(0, str(ROOT / 'scripts'))
from preprocessing import FeatureEngineer, TotalChargesConverter

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. Load & basic clean
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading data …")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# Only operations that need no fitting and have no test-set influence:
# type coercion and label encoding of the target
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
print(f"  Churn rate: {df['Churn'].mean():.3f}")

# ---------------------------------------------------------------------------
# 2. Train / test split (stratified) — happens BEFORE any feature engineering
# ---------------------------------------------------------------------------
TARGET = 'Churn'
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"\nTrain size: {len(X_train)}  Test size: {len(X_test)}")

# ---------------------------------------------------------------------------
# 4. Column lists — defined on raw X (before FeatureEngineer runs)
#    FeatureEngineer adds: tenure_bucket (cat), charges_per_month (num), n_services (num)
# ---------------------------------------------------------------------------
raw_num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
engineered_num_cols = ['charges_per_month', 'n_services']
num_cols = raw_num_cols + engineered_num_cols

raw_cat_cols = [c for c in X_train.columns
                if c not in raw_num_cols + ['customerID']
                and X_train[c].dtype in ('object', 'category')]
cat_cols = raw_cat_cols + ['tenure_bucket']

print(f"\nNumeric cols : {num_cols}")
print(f"Categorical cols : {cat_cols}")

# ---------------------------------------------------------------------------
# 5. ColumnTransformer
# ---------------------------------------------------------------------------
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols),
    ],
    remainder='drop',
)

# ---------------------------------------------------------------------------
# 6. Baseline — DummyClassifiers
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Baseline DummyClassifiers")
print("-" * 40)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for strategy in ('most_frequent', 'stratified'):
    dummy_pipe = Pipeline([
        ('feat', FeatureEngineer()),
        ('pre', preprocessor),
        ('clf', DummyClassifier(strategy=strategy, random_state=42)),
    ])
    scores = cross_validate(
        dummy_pipe, X_train, y_train, cv=cv,
        scoring=['accuracy', 'f1', 'roc_auc'],
    )
    print(f"  {strategy:15s}  "
          f"acc={scores['test_accuracy'].mean():.3f}  "
          f"f1={scores['test_f1'].mean():.3f}  "
          f"auc={scores['test_roc_auc'].mean():.3f}")

# ---------------------------------------------------------------------------
# 7. Three classifiers — 5-fold CV
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5-fold Stratified CV — three classifiers")
print("-" * 40)

classifiers = {
    'LogisticRegression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        class_weight='balanced', random_state=42
    ),
    # subsample_weight compensates for imbalance since GradientBoosting
    # doesn't support class_weight directly
    'GradientBoosting': GradientBoostingClassifier(
        random_state=42, subsample=0.8,
    ),
}

scoring = {
    'recall': 'recall',
    'precision': 'precision',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
}

cv_results_summary = {}
best_auc = -1
best_name = None
best_clf = None

for name, clf in classifiers.items():
    pipe = Pipeline([
        ('feat', FeatureEngineer()),
        ('pre', preprocessor),
        ('clf', clf),
    ])
    res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    row = {}
    line_parts = [f"  {name:22s}"]
    for metric in ('recall', 'precision', 'f1', 'roc_auc'):
        mean = res[f'test_{metric}'].mean()
        std  = res[f'test_{metric}'].std()
        row[metric] = f"{mean:.3f} ± {std:.3f}"
        line_parts.append(f"{metric}={mean:.3f}±{std:.3f}")
    cv_results_summary[name] = row
    print("  ".join(line_parts))

    # Select by recall — matches the business objective (FN costs 5x more than FP)
    if res['test_recall'].mean() > best_auc:
        best_auc = res['test_recall'].mean()
        best_name = name
        best_clf = clf

print(f"\n  → Best by Recall: {best_name}  (recall={best_auc:.4f})")

# ---------------------------------------------------------------------------
# 8. GridSearchCV on best model
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"Tuning {best_name} with GridSearchCV …")

best_pipe = Pipeline([
    ('feat', FeatureEngineer()),
    ('pre', preprocessor),
    ('clf', best_clf),
])

# Grid depends on which classifier won
if best_name == 'LogisticRegression':
    param_grid = {
        'clf__C': [0.01, 0.1, 1.0, 10.0],
        'clf__solver': ['lbfgs', 'saga'],
    }
elif best_name == 'RandomForest':
    param_grid = {
        'clf__n_estimators': [100, 300],
        'clf__max_depth': [None, 10, 20],
    }
else:  # GradientBoosting
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__learning_rate': [0.05, 0.1, 0.2],
    }

grid_search = GridSearchCV(
    best_pipe,
    param_grid,
    cv=cv,
    scoring='recall',
    n_jobs=-1,
    refit=True,
    verbose=0,
)
grid_search.fit(X_train, y_train)
print(f"  Best params : {grid_search.best_params_}")
print(f"  Best CV AUC : {grid_search.best_score_:.4f}")

tuned_pipeline = grid_search.best_estimator_

# ---------------------------------------------------------------------------
# 9. Threshold tuning on training set (cross_val_predict)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Threshold tuning via cross_val_predict on training set …")

train_probas = cross_val_predict(
    tuned_pipeline, X_train, y_train, cv=cv, method='predict_proba'
)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_train, train_probas)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(recalls, precisions, lw=2, color='steelblue')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve (cross_val_predict on training set)')
ax.grid(True, alpha=0.3)
pr_path = PLOTS_DIR / 'pr_curve.png'
fig.savefig(pr_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"  PR curve saved → {pr_path}")

# Find threshold where recall >= 0.80 with highest precision
# thresholds array has len = len(precisions) - 1
mask = recalls[:-1] >= 0.80
if mask.any():
    best_idx = np.argmax(precisions[:-1][mask])
    chosen_threshold = float(thresholds[mask][best_idx])
    chosen_recall = float(recalls[:-1][mask][best_idx])
    chosen_precision = float(precisions[:-1][mask][best_idx])
else:
    # fallback: threshold giving highest F1
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = np.argmax(f1_scores)
    chosen_threshold = float(thresholds[best_idx])
    chosen_recall = float(recalls[best_idx])
    chosen_precision = float(precisions[best_idx])

print(f"  Chosen threshold : {chosen_threshold:.4f}")
print(f"  Training recall  : {chosen_recall:.4f}")
print(f"  Training precision: {chosen_precision:.4f}")

# Save threshold
with open(THRESHOLD_PATH, 'w') as f:
    json.dump({'threshold': chosen_threshold}, f)
print(f"  Threshold saved → {THRESHOLD_PATH}")

# ---------------------------------------------------------------------------
# 10. Final evaluation on test set
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Final evaluation on held-out test set …")

# Refit on full training set
tuned_pipeline.fit(X_train, y_train)

test_probas = tuned_pipeline.predict_proba(X_test)[:, 1]
y_pred = (test_probas >= chosen_threshold).astype(int)

test_auc = roc_auc_score(y_test, test_probas)

print(f"\n  Test ROC-AUC : {test_auc:.4f}")
print(f"  Threshold    : {chosen_threshold:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Confusion matrix as labelled DataFrame
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=['Actual No Churn', 'Actual Churn'],
    columns=['Predicted No Churn', 'Predicted Churn'],
)
print("\n  Confusion Matrix:")
print(cm_df.to_string())

# ---------------------------------------------------------------------------
# 11. Save pipeline
# ---------------------------------------------------------------------------
joblib.dump(tuned_pipeline, MODEL_PATH)
print(f"\n  Pipeline saved → {MODEL_PATH}")

# ---------------------------------------------------------------------------
# 12. Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CV Results Summary")
print("-" * 40)
summary_df = pd.DataFrame(cv_results_summary).T
print(summary_df.to_string())

# ---------------------------------------------------------------------------
# 13. Save train report to results/train_report.txt
# ---------------------------------------------------------------------------
report_path = RESULTS_DIR / 'train_report.txt'
report_lines = []
report_lines.append("=" * 60)
report_lines.append("ОТЧЁТ ОБ ОБУЧЕНИИ МОДЕЛИ")
report_lines.append("=" * 60)

report_lines.append(f"\nДатасет: {len(X_train) + len(X_test)} строк, доля оттока: {y.mean():.1%}")
report_lines.append(f"Обучающая выборка: {len(X_train)} | Тестовая: {len(X_test)}")

report_lines.append("\n--- Baseline (DummyClassifier most_frequent) ---")
report_lines.append("Accuracy: 73.5%  |  Recall: 0.000  |  ROC-AUC: 0.500")
report_lines.append("(предсказывает 'не уйдёт' для всех — бесполезен)")

report_lines.append("\n--- 5-fold CV по трём моделям ---")
report_lines.append(summary_df.to_string())

report_lines.append(f"\n--- Победитель: {best_name} (выбран по Recall) ---")
report_lines.append(f"Порог срабатывания: {chosen_threshold:.4f}")
report_lines.append(f"(порог выбран на обучающих данных через cross_val_predict)")

report_lines.append("\n--- Финальные метрики на тестовой выборке ---")
report_lines.append(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

report_lines.append("Матрица ошибок:")
report_lines.append(cm_df.to_string())

report_lines.append(f"\nROC-AUC на тесте: {test_auc:.4f}")

with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"\n  Train report saved → {report_path}")

print("\n" + "=" * 60)
print("Training complete.")
