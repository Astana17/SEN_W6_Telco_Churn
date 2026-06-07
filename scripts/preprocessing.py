"""
preprocess.py
─────────────────────────────────────────────────────────────────────────────
Запуск:
    python scripts/preprocess.py
    python scripts/preprocess.py --input data/WA_Fn-UseC_-Telco-Customer-Churn.csv
                                  --output data/processed/

Сохраняет в output/:
    X_train.csv, y_train.csv   — после SMOTE, готовы к обучению
    X_test.csv,  y_test.csv    — нетронутый тест
    scaler.pkl                 — StandardScaler (fit только на train)
    feature_names.json         — список столбцов после кодирования
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import pickle

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── константы ────────────────────────────────────────────────────────────────

TARGET     = "Churn"
DROP_COLS  = ["customerID"]

# Yes/No → 1/0
BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]

# Эти три варианта означают одно — услуга не подключена → 0
SERVICE_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# Порядок важен: чем длиннее контракт, тем ниже churn → ordinal
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}

# Нет смыслового порядка → one-hot
ONEHOT_COLS = ["InternetService", "PaymentMethod"]


# ─── шаги обработки ───────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[preprocess] загружено {len(df):,} строк × {df.shape[1]} столбцов")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет ненужные столбцы, чинит типы, заполняет пропуски."""
    df = df.copy()

    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # TotalCharges — object из-за пустых строк у клиентов с tenure=0
    n_bad = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if n_bad:
        print(f"[preprocess] TotalCharges: {n_bad} пустых строк → 0.0")

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Кодирует все категориальные признаки.
    Единственное место в проекте где живёт логика трансформаций.
    """
    df = df.copy()

    # Целевая переменная
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    # gender: Female=0, Male=1
    df["gender"] = (df["gender"] == "Male").astype(int)

    # Бинарные Yes/No → 1/0
    for col in BINARY_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    # Сервисные: "No phone/internet service" → то же что "No" → 0
    for col in SERVICE_COLS:
        df[col] = df[col].replace(
            {"No phone service": "No", "No internet service": "No"}
        )
        df[col] = (df[col] == "Yes").astype(int)

    # Contract → ordinal (0 / 1 / 2)
    unknown = set(df["Contract"].unique()) - set(CONTRACT_MAP)
    if unknown:
        raise ValueError(f"[preprocess] Неизвестные значения Contract: {unknown}")
    df["Contract"] = df["Contract"].map(CONTRACT_MAP).astype(int)

    # InternetService, PaymentMethod → one-hot (drop_first снимает мультиколлинеарность)
    df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=True)

    # Все bool-столбцы от get_dummies → int (совместимость со scaler и joblib)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def split(df: pd.DataFrame, test_size: float, seed: int):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    print(f"[preprocess] train={len(X_train):,}  test={len(X_test):,}  "
          f"churn_train={y_train.mean():.3f}  churn_test={y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


def resample(X_train: pd.DataFrame, y_train: pd.Series, seed: int):
    """SMOTE только на train — test остаётся нетронутым."""
    sm = SMOTE(random_state=seed)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    counts = pd.Series(y_res).value_counts()
    print(f"[preprocess] после SMOTE: {len(X_res):,} строк  "
          f"(0={counts[0]:,}  1={counts[1]:,})")
    return X_res, y_res


def scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """fit — только на train, transform — на обоих."""
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )
    return X_train_sc, X_test_sc, scaler


def save(X_train, X_test, y_train, y_test, scaler, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_test.to_csv( os.path.join(out_dir, "X_test.csv"),  index=False)
    pd.Series(y_train, name=TARGET).reset_index(drop=True).to_csv(
        os.path.join(out_dir, "y_train.csv"), index=False
    )
    pd.Series(y_test, name=TARGET).reset_index(drop=True).to_csv(
        os.path.join(out_dir, "y_test.csv"), index=False
    )

    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    feature_names = list(X_train.columns)
    with open(os.path.join(out_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"[preprocess] сохранено в {out_dir!r}  ({len(feature_names)} признаков)")
    print(f"[preprocess] признаки: {feature_names}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--output",    default="data/processed/")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--no_smote",  action="store_true")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    df = load(args.input)
    df = clean(df)
    df = encode(df)

    X_train, X_test, y_train, y_test = split(df, args.test_size, args.seed)

    if not args.no_smote:
        X_train, y_train = resample(X_train, y_train, args.seed)

    X_train, X_test, scaler = scale(X_train, X_test)

    save(X_train, X_test, y_train, y_test, scaler, args.output)


if __name__ == "__main__":
    main()