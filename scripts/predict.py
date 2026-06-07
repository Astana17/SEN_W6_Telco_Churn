"""
predict.py
─────────────────────────────────────────────────────────────────────────────
Запуск:

  # Оценить лучшую модель на тестовом сете
  python scripts/predict.py --test_mode

  # Предсказание на новом сыром CSV
  python scripts/predict.py --input data/new_customers.csv

  # Выбрать конкретную модель
  python scripts/predict.py --test_mode --model_key gb

Сохраняет в results/:
  predictions.csv  — [customerID,] churn_prob, churn_pred
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")

# ─── те же константы что в preprocess.py ─────────────────────────────────────

TARGET       = "Churn"
DROP_COLS    = ["customerID"]
BINARY_COLS  = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
SERVICE_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
ONEHOT_COLS  = ["InternetService", "PaymentMethod"]


# ─── загрузка артефактов ──────────────────────────────────────────────────────

def load_model(key: str, models_dir: str):
    path = os.path.join(models_dir, f"{key}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Модель не найдена: {path}\n"
            f"Запусти: python scripts/train.py --model {key}"
        )
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def load_meta(key: str, models_dir: str) -> dict:
    with open(os.path.join(models_dir, f"{key}_meta.json")) as f:
        meta = json.load(f)
    print(f"[predict] модель: {meta['model_name']}  "
          f"порог={meta['threshold']}  AUC={meta['roc_auc']}  Recall={meta['recall']}")
    return meta


def load_scaler(data_dir: str):
    with open(os.path.join(data_dir, "scaler.pkl"), "rb") as f:
        return pickle.load(f)


def load_feature_names(data_dir: str) -> list:
    with open(os.path.join(data_dir, "feature_names.json")) as f:
        return json.load(f)


# ─── препроцессинг новых данных ───────────────────────────────────────────────

def preprocess(df_raw: pd.DataFrame, feature_names: list, scaler) -> pd.DataFrame:
    """
    Применяет те же трансформации что preprocess.py → encode(),
    затем выравнивает столбцы по feature_names и масштабирует.
    Принимает сырой DataFrame (как из CSV).
    """
    df = df_raw.copy()

    # убираем customerID и Churn если есть
    df.drop(columns=[c for c in DROP_COLS + [TARGET] if c in df.columns], inplace=True)

    # TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # gender
    df["gender"] = (df["gender"] == "Male").astype(int)

    # бинарные
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = (df[col] == "Yes").astype(int)

    # сервисные
    for col in SERVICE_COLS:
        if col in df.columns:
            df[col] = df[col].replace(
                {"No phone service": "No", "No internet service": "No"}
            )
            df[col] = (df[col] == "Yes").astype(int)

    # Contract → ordinal
    if "Contract" in df.columns:
        df["Contract"] = df["Contract"].map(CONTRACT_MAP).astype(int)

    # one-hot
    present = [c for c in ONEHOT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=present, drop_first=True)

    # bool → int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # выравниваем столбцы относительно train
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0          # категория которой не было в новых данных
    df = df[feature_names]       # нужный порядок и только нужные столбцы

    # масштабирование (только transform, не fit!)
    df = pd.DataFrame(scaler.transform(df), columns=feature_names)

    return df


# ─── инференс ─────────────────────────────────────────────────────────────────

def predict(model, X: pd.DataFrame, threshold: float) -> pd.DataFrame:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return pd.DataFrame({"churn_prob": y_prob.round(4), "churn_pred": y_pred})


# ─── оценка на тест-сете ──────────────────────────────────────────────────────

def evaluate(model, data_dir: str, threshold: float, model_name: str):
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\n[predict] {model_name}  порог={threshold:.3f}")
    print("─" * 54)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"ROC-AUC: {auc:.4f}")

    return y_prob, y_pred


# ─── сохранение ───────────────────────────────────────────────────────────────

def save(preds: pd.DataFrame, customer_ids, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    if customer_ids is not None:
        preds.insert(0, "customerID", customer_ids.values)

    path = os.path.join(out_dir, "predictions.csv")
    preds.to_csv(path, index=False)

    n_churn = preds["churn_pred"].sum()
    print(f"\n[predict] сохранено: {path}")
    print(f"[predict] всего клиентов: {len(preds):,}")
    print(f"[predict] предсказан Churn=1: {n_churn:,} ({n_churn/len(preds)*100:.1f}%)")
    print(f"\n[predict] топ-10 по вероятности оттока:")
    print(preds.sort_values("churn_prob", ascending=False).head(10).to_string(index=False))


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=None,
                        help="Сырой CSV с новыми клиентами")
    parser.add_argument("--model_key",  default="gb", choices=["lr", "rf", "gb"])
    parser.add_argument("--models_dir", default="results/")
    parser.add_argument("--data",       default="data/processed/")
    parser.add_argument("--output",     default="results/")
    parser.add_argument("--test_mode",  action="store_true",
                        help="Оценить на X_test.csv (уже обработанный)")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Переопределить порог из meta.json")
    args = parser.parse_args()

    model         = load_model(args.model_key, args.models_dir)
    meta          = load_meta(args.model_key, args.models_dir)
    threshold     = args.threshold if args.threshold is not None else meta["threshold"]

    if args.test_mode:
        # X_test уже обработан preprocess.py — scaler применять не нужно
        y_prob, y_pred = evaluate(model, args.data, threshold, meta["model_name"])
        preds = pd.DataFrame({"churn_prob": y_prob.round(4), "churn_pred": y_pred})
        save(preds, customer_ids=None, out_dir=args.output)

    elif args.input:
        scaler        = load_scaler(args.data)
        feature_names = load_feature_names(args.data)

        df_raw       = pd.read_csv(args.input)
        customer_ids = df_raw["customerID"].copy() if "customerID" in df_raw.columns else None
        print(f"[predict] загружено {len(df_raw):,} новых клиентов")

        X     = preprocess(df_raw, feature_names, scaler)
        preds = predict(model, X, threshold)
        save(preds, customer_ids, out_dir=args.output)

    else:
        parser.error("Укажи --test_mode или --input <csv>")


if __name__ == "__main__":
    main()