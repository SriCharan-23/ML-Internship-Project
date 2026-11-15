import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.transformers import NumericStringCleaner

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
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
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Business Cost Matrix (USD)
COST_FALSE_NEGATIVE = 500.0  # Missed churner (lost customer lifetime value)
COST_FALSE_POSITIVE = 50.0   # Wasted retention incentive/discount


def compute_sha256(df: pd.DataFrame) -> str:
    hashed_rows = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(hashed_rows).hexdigest()


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("cleaner", NumericStringCleaner(columns=["TotalCharges"])),
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(random_state=42)),
        ]
    )


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Finds probability threshold minimizing expected business cost."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best_thresh = 0.5
    min_cost = float("inf")

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        total_cost = (fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)
        if total_cost < min_cost:
            min_cost = total_cost
            best_thresh = float(t)

    return round(best_thresh, 4), round(min_cost / len(y_true), 2)


def compute_reference_distributions(X_train: pd.DataFrame) -> dict:
    """Stores baseline feature histograms/proportions for PSI drift monitoring."""
    cleaned = NumericStringCleaner(columns=["TotalCharges"]).transform(X_train)
    ref = {}
    for col in NUMERIC_COLS:
        series = cleaned[col].dropna()
        counts, bin_edges = np.histogram(series, bins=10)
        proportions = (counts / len(series)).tolist()
        ref[col] = {
            "type": "numeric",
            "bin_edges": bin_edges.tolist(),
            "proportions": proportions,
        }
    for col in CATEGORICAL_COLS:
        vc = cleaned[col].astype(str).value_counts(normalize=True)
        ref[col] = {
            "type": "categorical",
            "proportions": vc.to_dict(),
        }
    return ref


def train_and_export(data_path: str, output_dir: str = "artifacts") -> dict:
    df = pd.read_csv(data_path)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df[FEATURE_COLS]

    data_hash = compute_sha256(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_pipeline()
    param_distributions = {
        "classifier__n_estimators": [100, 150, 200],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__max_depth": [3, 4, 5],
        "classifier__subsample": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_proba = best_model.predict_proba(X_test)[:, 1]
    optimal_threshold, cost_per_customer = find_optimal_threshold(y_test.values, y_proba)
    y_pred = (y_proba >= optimal_threshold).astype(int)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_version = "1.0.0"
    artifact_file = out_path / "model_v1.joblib"
    joblib.dump(best_model, artifact_file)

    metadata = {
        "model_version": model_version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_data_sha256": data_hash,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "optimal_threshold": optimal_threshold,
        "cost_matrix": {
            "false_negative_usd": COST_FALSE_NEGATIVE,
            "false_positive_usd": COST_FALSE_POSITIVE,
            "expected_cost_per_customer_usd": cost_per_customer,
        },
        "metrics": {
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
            "pr_auc": round(float(average_precision_score(y_test, y_proba)), 4),
            "f1_at_threshold": round(float(f1_score(y_test, y_pred)), 4),
        },
        "best_params": search.best_params_,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "reference_distributions": compute_reference_distributions(X_train),
    }

    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    meta = train_and_export(csv_path)
    print(json.dumps({k: v for k, v in meta.items() if k != "reference_distributions"}, indent=2))
