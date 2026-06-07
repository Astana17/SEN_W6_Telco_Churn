"""
Train churn models with a leakage-free sklearn Pipeline.

Usage:
    python scripts/train.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import ThresholdClassifier, build_model_pipeline

DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join("results", "churn_pipeline.pkl")
RESULTS_PATH = os.path.join("results", "results.md")
PLOTS_DIR = os.path.join("results", "plots")
CV_RESULTS_PATH = os.path.join("results", "cv_results.csv")
RANDOM_STATE = 42
RECALL_TARGET = 0.80
CV_FOLDS = 5


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = (df["Churn"] == "Yes").astype(int)
    return X, y


def choose_threshold(y_true, y_probs, min_recall=0.80):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    candidates = [
        (thr, prec, rec)
        for thr, prec, rec in zip(thresholds, precision[:-1], recall[:-1])
        if rec >= min_recall
    ]
    if candidates:
        return max(candidates, key=lambda item: item[1])[0]
    return max(
        zip(thresholds, precision[:-1], recall[:-1]),
        key=lambda item: item[1] * item[2],
    )[0]


def run_cross_validation(pipeline, X, y, model_name):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["recall", "precision", "f1", "roc_auc"]
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    row = {"model": model_name}
    for metric in scoring:
        row[f"{metric}_mean"] = round(scores[f"test_{metric}"].mean(), 4)
        row[f"{metric}_std"] = round(scores[f"test_{metric}"].std(), 4)
    return row


def get_candidate_pipelines():
    return {
        "Dummy (most_frequent)": build_model_pipeline(
            DummyClassifier(strategy="most_frequent")
        ),
        "Dummy (stratified)": build_model_pipeline(
            DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
        ),
        "Logistic Regression": build_model_pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )
        ),
        "Random Forest": build_model_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "Gradient Boosting": build_model_pipeline(
            GradientBoostingClassifier(random_state=RANDOM_STATE)
        ),
    }


def get_param_grid(model_name):
    if model_name == "Logistic Regression":
        return {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["lbfgs", "liblinear"],
        }
    if model_name == "Random Forest":
        return {
            "model__n_estimators": [200, 300],
            "model__max_depth": [8, 12, None],
            "model__min_samples_leaf": [1, 5],
        }
    if model_name == "Gradient Boosting":
        return {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 4],
        }
    return None


def save_pr_curve(y_true, y_probs, threshold, output_path):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.axvline(RECALL_TARGET, color="gray", linestyle="--", label=f"recall target={RECALL_TARGET}")
    plt.scatter(
        [recall_score(y_true, (y_probs >= threshold).astype(int))],
        [precision_score(y_true, (y_probs >= threshold).astype(int), zero_division=0)],
        color="red",
        zorder=5,
        label=f"threshold={threshold:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve (out-of-fold validation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def format_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        matrix,
        index=["Actual No Churn", "Actual Churn"],
        columns=["Predicted No Churn", "Predicted Churn"],
    )


def write_results_md(
    best_model_name,
    threshold,
    precision,
    recall,
    f1,
    roc_auc,
    cm_df,
    n_test,
    n_flagged,
    n_true_churners_flagged,
):
    flagged_rate = n_flagged / n_test
    precision_among_flagged = n_true_churners_flagged / n_flagged if n_flagged else 0.0

    with open(RESULTS_PATH, "w", encoding="utf-8") as out:
        out.write("# Training results\n\n")
        out.write(f"- best model: {best_model_name}\n")
        out.write(f"- frozen threshold: {threshold:.4f}\n")
        out.write(f"- test precision (churn): {precision:.4f}\n")
        out.write(f"- test recall (churn): {recall:.4f}\n")
        out.write(f"- test f1 (churn): {f1:.4f}\n")
        out.write(f"- test roc_auc: {roc_auc:.4f}\n\n")

        out.write("## Confusion matrix (test set)\n\n")
        out.write("| | Predicted No Churn | Predicted Churn |\n")
        out.write("|---|---|---|\n")
        out.write(
            f"| Actual No Churn | {cm_df.iloc[0, 0]} | {cm_df.iloc[0, 1]} |\n"
        )
        out.write(
            f"| Actual Churn | {cm_df.iloc[1, 0]} | {cm_df.iloc[1, 1]} |\n\n"
        )

        out.write("## Business interpretation\n\n")
        out.write(
            f"At threshold **{threshold:.2f}**, the model flags about "
            f"**{flagged_rate * 1000:.0f} customers per 1,000**, "
            f"of whom roughly **{precision_among_flagged * 100:.0f}%** are real churners. "
            f"On the held-out test sample ({n_test:,} customers), that means "
            f"**{n_flagged:,} marketing calls**, catching **{n_true_churners_flagged:,}** "
            f"actual churners while missing the rest.\n\n"
        )
        out.write(
            "Because a missed churner costs about 5x more than calling a loyal customer, "
            "we prioritised recall on validation data and then chose the highest precision "
            f"among thresholds with recall >= {RECALL_TARGET:.2f}.\n"
        )


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"[train] train={len(X_train):,}  test={len(X_test):,}")
    print(f"[train] churn rate train={y_train.mean():.3f}  test={y_test.mean():.3f}")

    cv_rows = []
    pipelines = get_candidate_pipelines()
    for name, pipeline in pipelines.items():
        print(f"[train] CV: {name}")
        cv_rows.append(run_cross_validation(pipeline, X_train, y_train, name))

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(CV_RESULTS_PATH, index=False)
    print(f"\n[train] CV results saved to {CV_RESULTS_PATH}")
    print(cv_df.to_string(index=False))

    model_candidates = [
        name
        for name in pipelines
        if not name.startswith("Dummy")
    ]
    best_row = cv_df[cv_df["model"].isin(model_candidates)].sort_values(
        ["recall_mean", "roc_auc_mean"], ascending=False
    ).iloc[0]
    best_model_name = best_row["model"]
    print(f"\n[train] best model by CV recall: {best_model_name}")

    tuned_pipeline = pipelines[best_model_name]
    param_grid = get_param_grid(best_model_name)
    if param_grid:
        search = GridSearchCV(
            tuned_pipeline,
            param_grid=param_grid,
            scoring="recall",
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        tuned_pipeline = search.best_estimator_
        print(f"[train] GridSearch best params: {search.best_params_}")
        print(f"[train] GridSearch best CV recall: {search.best_score_:.4f}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    train_probs = cross_val_predict(
        tuned_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    threshold = choose_threshold(y_train, train_probs, min_recall=RECALL_TARGET)
    print(f"[train] frozen threshold from OOF validation: {threshold:.4f}")

    save_pr_curve(
        y_train,
        train_probs,
        threshold,
        os.path.join(PLOTS_DIR, "precision_recall_curve.png"),
    )

    tuned_pipeline.fit(X_train, y_train)
    test_probs = tuned_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (test_probs >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_probs)
    cm_df = format_confusion_matrix(y_test, y_pred)

    print("\n[train] final test evaluation (single pass)")
    print(f"precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}  roc_auc={roc_auc:.4f}")
    print(cm_df)

    n_flagged = int(y_pred.sum())
    n_true_churners_flagged = int(((y_pred == 1) & (y_test == 1)).sum())

    write_results_md(
        best_model_name,
        threshold,
        precision,
        recall,
        f1,
        roc_auc,
        cm_df,
        len(X_test),
        n_flagged,
        n_true_churners_flagged,
    )

    cm_df.to_csv(os.path.join("results", "confusion_matrix.csv"))
    final_model = ThresholdClassifier(estimator=tuned_pipeline, threshold=threshold)
    dump(final_model, MODEL_PATH)

    meta = {
        "best_model": best_model_name,
        "threshold": round(threshold, 4),
        "test_precision": round(precision, 4),
        "test_recall": round(recall, 4),
        "test_f1": round(f1, 4),
        "test_roc_auc": round(roc_auc, 4),
    }
    with open(os.path.join("results", "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[train] saved model to {MODEL_PATH}")
    print(f"[train] saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
