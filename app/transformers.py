import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NumericStringCleaner(BaseEstimator, TransformerMixin):
    """Coerces string-encoded numeric columns (e.g., TotalCharges) to float."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col in self.columns:
            if col in X_copy.columns:
                X_copy[col] = pd.to_numeric(X_copy[col], errors="coerce")
        return X_copy
