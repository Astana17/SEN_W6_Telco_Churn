"""
Custom sklearn transformers for Telco Churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TotalChargesConverter(BaseEstimator, TransformerMixin):
    """
    Converts TotalCharges column from object dtype to float using
    pd.to_numeric with errors='coerce' (blank strings become NaN).
    Also drops the customerID column if present.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'customerID' in X.columns:
            X = X.drop(columns=['customerID'])
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates derived features from the raw data:

    - tenure_bucket : categorical bin of tenure in months
        bins  : [0, 12, 24, 48, 999]
        labels: ['0-12', '13-24', '25-48', '49+']

    - charges_per_month : TotalCharges / max(tenure, 1)
        Approximates the average monthly spend even for recent customers.

    - n_services : count of services the customer has subscribed to.
        A service is counted as active when its value is NOT 'No' AND NOT
        'No internet service' (avoids counting the placeholder strings that
        appear in columns that depend on internet service).
        Columns examined: PhoneService, MultipleLines, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies.
    """

    SERVICE_COLS = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
    ]

    TENURE_BINS = [0, 12, 24, 48, 999]
    TENURE_LABELS = ['0-12', '13-24', '25-48', '49+']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # tenure_bucket
        if 'tenure' in X.columns:
            X['tenure_bucket'] = pd.cut(
                X['tenure'],
                bins=self.TENURE_BINS,
                labels=self.TENURE_LABELS,
                right=True,
            )

        # charges_per_month
        if 'TotalCharges' in X.columns and 'tenure' in X.columns:
            safe_tenure = X['tenure'].clip(lower=1)
            X['charges_per_month'] = X['TotalCharges'] / safe_tenure

        # n_services
        present_service_cols = [c for c in self.SERVICE_COLS if c in X.columns]
        if present_service_cols:
            def count_active(row):
                count = 0
                for col in present_service_cols:
                    val = row[col]
                    if val != 'No' and val != 'No internet service' and val != 'No phone service':
                        count += 1
                return count

            X['n_services'] = X[present_service_cols].apply(count_active, axis=1)

        return X
