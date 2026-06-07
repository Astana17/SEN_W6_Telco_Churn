"""Custom transformers and pipeline builders for churn prediction."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INACTIVE_SERVICE_VALUES = {"No", "No internet service", "No phone service"}

SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charges_per_month_of_tenure",
    "n_services",
]

CATEGORICAL_COLS = [
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


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create derived features inside the sklearn pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        if "customerID" in X.columns:
            X = X.drop(columns=["customerID"])

        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

        X["charges_per_month_of_tenure"] = X["TotalCharges"] / X["tenure"].clip(lower=1)
        X["charges_per_month_of_tenure"] = X["charges_per_month_of_tenure"].replace(
            [np.inf, -np.inf], np.nan
        )

        X["n_services"] = X[SERVICE_COLS].apply(
            lambda row: sum(str(value) not in INACTIVE_SERVICE_VALUES for value in row),
            axis=1,
        )

        X["tenure_bucket"] = pd.cut(
            X["tenure"],
            bins=[-1, 12, 24, 48, np.inf],
            labels=["0-12", "13-24", "25-48", "49+"],
        ).astype(str)

        return X


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
            ("numeric", numeric_transformer, NUMERIC_COLS),
            ("categorical", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def make_preprocessing_pipeline():
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", build_preprocessor()),
        ]
    )


def build_model_pipeline(classifier):
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("model", classifier),
        ]
    )


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a fitted pipeline and apply a frozen decision threshold."""

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
