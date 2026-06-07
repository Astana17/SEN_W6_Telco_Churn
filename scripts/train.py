"""
train.py
─────────────────────────────────────────────────────────────────────────────
Запуск:
    python scripts/train.py
    python scripts/train.py --model rf
    python scripts/train.py --model all --recall_target 0.80

Читает из data/processed/:
    X_train.csv, y_train.csv, X_test.csv, y_test.csv

Сохраняет в results/:
    lr_model.pkl / rf_model.pkl / gb_model.pkl
    lr_meta.json / rf_meta.json / gb_meta.json
    comparison.csv
    pr_curve_*.png
    feat_imp_*.png
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")


# ─── загрузка ─────────────────────────────────────────────────────────────────

def load_data(data_dir: str):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()
    print(f"[train] X_train={X_train.shape}  X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─── модели ───────────────────────────────────────────────────────────────────

def get_models() -> dict:
    return {
        "lr": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        ),
    }

MODEL_NAMES = {"lr": "Logistic Regression", "rf": "Random Forest", "gb": "Gradient Boosting"}


# ─── порог ────────────────────────────────────────────────────────────────────

def find_threshold(y_true, y_prob, recall_target: float) -> float:
    """
    Ищет наименьший порог при котором recall >= recall_target,
    среди таких кандидатов выбирает максимальный F1.
    Если ни один не даёт нужного recall — просто максимизирует F1.
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
    # последний элемент precision/recall не имеет порога
    p, r, t = precision_arr[:-1], recall_arr[:-1], thresholds
    f1 = 2 * p * r / (p + r + 1e-9)

    mask = r >= recall_target
    idx  = np.argmax(f1[mask]) if mask.any() else np.argmax(f1)
    return float(t[mask][idx] if mask.any() else t[idx])


# ─── кросс-валидация ──────────────────────────────────────────────────────────

def run_cv(model, X, y, n_splits: int, seed: int) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = {}
    for metric in ("recall", "roc_auc", "f1"):
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        out[f"cv_{metric}_mean"] = round(float(scores.mean()), 4)
        out[f"cv_{metric}_std"]  = round(float(scores.std()),  4)
    print(f"[train] CV  recall={out['cv_recall_mean']:.4f}±{out['cv_recall_std']:.4f}  "
          f"auc={out['cv_roc_auc_mean']:.4f}±{out['cv_roc_auc_std']:.4f}")
    return out


# ─── обучение + оценка ────────────────────────────────────────────────────────

def fit_and_evaluate(model, X_train, y_train, X_test, y_test,
                     name: str, recall_target: float, out_dir: str) -> dict:
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    threshold = find_threshold(y_test, y_prob, recall_target)
    y_pred    = (y_prob >= threshold).astype(int)

    auc    = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n{'─'*54}")
    print(f"  {name}   порог={threshold:.3f}")
    print(f"{'─'*54}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"Confusion matrix:\n{np.array(cm)}")
    print(f"ROC-AUC: {auc:.4f}   Recall: {recall:.4f}")

    _save_pr_curve(y_test, y_prob, threshold, name, out_dir)

    return {
        "model_name":       name,
        "threshold":        round(threshold, 4),
        "roc_auc":          round(auc, 4),
        "recall":           round(recall, 4),
        "precision":        round(report["Churn"]["precision"], 4),
        "f1_churn":         round(report["Churn"]["f1-score"], 4),
        "accuracy":         round(report["accuracy"], 4),
        "confusion_matrix": cm,
        "recall_target":    recall_target,
    }


# ─── графики ──────────────────────────────────────────────────────────────────

def _save_pr_curve(y_true, y_prob, threshold, name, out_dir):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    idx = np.argmin(np.abs(thr - threshold)) if len(thr) else 0

    plt.figure(figsize=(6, 4))
    plt.plot(rec, prec, lw=1.5)
    plt.scatter(rec[idx], prec[idx], color="red", zorder=5,
                label=f"порог={threshold:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR-кривая: {name}")
    plt.legend()
    plt.tight_layout()
    slug = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(out_dir, f"pr_{slug}.png"), dpi=120)
    plt.close()


def save_feat_importance(model, feature_names, name, out_dir):
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    plt.figure(figsize=(7, 5))
    imp.tail(15).plot(kind="barh", color="steelblue")
    plt.title(f"Feature importance — {name}")
    plt.tight_layout()
    slug = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(out_dir, f"feat_imp_{slug}.png"), dpi=120)
    plt.close()
    print(f"[train] топ-5 признаков:\n{imp.tail(5).iloc[::-1].to_string()}")


# ─── сохранение модели ────────────────────────────────────────────────────────

def save_model(model, meta: dict, key: str, out_dir: str):
    with open(os.path.join(out_dir, f"{key}_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(out_dir, f"{key}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] сохранено: {key}_model.pkl  {key}_meta.json")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",          default="data/processed/")
    parser.add_argument("--output",        default="results/")
    parser.add_argument("--model",         default="all",
                        choices=["lr", "rf", "gb", "all"])
    parser.add_argument("--recall_target", type=float, default=0.80)
    parser.add_argument("--cv_folds",      type=int,   default=5)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data(args.data)
    feature_names = list(X_train.columns)
    all_models    = get_models()
    keys          = list(all_models) if args.model == "all" else [args.model]

    all_meta = []
    for key in keys:
        model = all_models[key]
        name  = MODEL_NAMES[key]

        print(f"\n{'═'*54}")
        print(f"  {name}")
        print(f"{'═'*54}")

        cv_scores = run_cv(model, X_train, y_train, args.cv_folds, args.seed)

        meta = fit_and_evaluate(
            model, X_train, y_train, X_test, y_test,
            name=name,
            recall_target=args.recall_target,
            out_dir=args.output,
        )
        meta.update(cv_scores)

        save_feat_importance(model, feature_names, name, args.output)
        save_model(model, meta, key, args.output)
        all_meta.append(meta)

    # сводная таблица
    cols = ["model_name", "roc_auc", "recall", "precision",
            "f1_churn", "accuracy", "threshold",
            "cv_recall_mean", "cv_roc_auc_mean"]
    comparison = pd.DataFrame(all_meta)[cols]
    comparison.to_csv(os.path.join(args.output, "comparison.csv"), index=False)
    print(f"\n[train] сравнение моделей:\n{comparison.to_string(index=False)}")

    best = max(all_meta, key=lambda m: m["roc_auc"])
    print(f"\n[train] ✓ лучшая по AUC: {best['model_name']}  "
          f"AUC={best['roc_auc']}  Recall={best['recall']}")


if __name__ == "__main__":
    main()