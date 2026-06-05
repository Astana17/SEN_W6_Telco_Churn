import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.service_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        self.numeric_cols = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "charges_per_month",
            "n_services",
        ]
        self.categorical_cols = [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "tenure_bucket",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy() if isinstance(X, np.ndarray) else X.copy()
        if "customerID" in X.columns:
            X = X.drop(columns=["customerID"])

        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["charges_per_month"] = X["TotalCharges"] / X["tenure"].replace({0: np.nan})
        X["charges_per_month"] = X["charges_per_month"].replace([np.inf, -np.inf], np.nan)
        X["charges_per_month"] = X["charges_per_month"].fillna(0.0)

        X["n_services"] = (
            X[self.service_cols]
            .apply(lambda row: (row == "Yes").astype(int), axis=1)
            .sum(axis=1)
        )

        X["tenure_bucket"] = pd.cut(
            X["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]
        )
        X["tenure_bucket"] = X["tenure_bucket"].astype(str).fillna("missing")

        return X

    def get_feature_names_out(self, input_features=None):
        return self.numeric_cols + self.categorical_cols


def build_preprocessor():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                [
                    "tenure",
                    "MonthlyCharges",
                    "TotalCharges",
                    "charges_per_month",
                    "n_services",
                ],
            ),
            (
                "categorical",
                categorical_transformer,
                [
                    "gender",
                    "SeniorCitizen",
                    "Partner",
                    "Dependents",
                    "PhoneService",
                    "MultipleLines",
                    "InternetService",
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                    "Contract",
                    "PaperlessBilling",
                    "PaymentMethod",
                    "tenure_bucket",
                ],
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def make_pipeline():
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", build_preprocessor()),
        ]
    )


class ThresholdClassifier(BaseEstimator, TransformerMixin):
    def __init__(self, estimator=None, threshold=0.5):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def set_threshold(self, threshold):
        self.threshold = threshold
        return self
