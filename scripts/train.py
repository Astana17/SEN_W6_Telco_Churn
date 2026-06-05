import os
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
                             precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import ThresholdClassifier, make_pipeline

DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join("results", "churn_pipeline.pkl")
RESULTS_PATH = os.path.join("results", "results.md")


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = (df["Churn"] == "Yes").astype(int)
    return X, y


def choose_threshold(y_true, y_probs, min_recall=0.8):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    candidates = [
        (thr, p, r)
        for thr, p, r in zip(thresholds, precision[:-1], recall[:-1])
        if r >= min_recall
    ]
    if candidates:
        best = max(candidates, key=lambda item: item[1])
        return best[0]
    best = max(zip(thresholds, precision[:-1], recall[:-1]), key=lambda item: item[1] * item[2])
    return best[0]


def format_results(y_true, y_pred, y_proba):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    matrix = confusion_matrix(y_true, y_pred)
    return precision, recall, f1, roc_auc, matrix


def main():
    os.makedirs("results", exist_ok=True)
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", make_pipeline()),
            ("classifier", LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42)),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_probs = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    threshold = choose_threshold(y_train, train_probs, min_recall=0.8)
    pipeline.fit(X_train, y_train)

    proba_test = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (proba_test >= threshold).astype(int)

    precision, recall, f1, roc_auc, matrix = format_results(y_test, y_test_pred, proba_test)

    with open(RESULTS_PATH, "w", encoding="utf-8") as out:
        out.write("# Training results\n")
        out.write("\n")
        out.write(f"- threshold: {threshold:.4f}\n")
        out.write(f"- precision: {precision:.4f}\n")
        out.write(f"- recall: {recall:.4f}\n")
        out.write(f"- f1-score: {f1:.4f}\n")
        out.write(f"- roc_auc: {roc_auc:.4f}\n")
        out.write("\n")
        out.write("## Confusion matrix\n")
        out.write("\n")
        out.write("| | Predicted negative | Predicted positive |\n")
        out.write("|---|---|---|\n")
        out.write(f"| Actual negative | {matrix[0,0]} | {matrix[0,1]} |\n")
        out.write(f"| Actual positive | {matrix[1,0]} | {matrix[1,1]} |\n")

    thresholded_model = ThresholdClassifier(estimator=pipeline, threshold=threshold)
    dump(thresholded_model, MODEL_PATH)

    print("Training finished.")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved results to {RESULTS_PATH}")
    print(f"precision={precision:.4f}, recall={recall:.4f}, threshold={threshold:.4f}")


if __name__ == "__main__":
    main()
