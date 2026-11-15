import numpy as np
import pandas as pd
import joblib

from app.transformers import NumericStringCleaner
from ml.train import find_optimal_threshold


def test_numeric_string_cleaner_coerces_blanks():
    df = pd.DataFrame({"TotalCharges": ["100.50", " ", "", "250.0"]})
    cleaned = NumericStringCleaner(columns=["TotalCharges"]).transform(df)
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])
    assert np.isnan(cleaned["TotalCharges"].iloc[1])
    assert np.isnan(cleaned["TotalCharges"].iloc[2])
    assert cleaned["TotalCharges"].iloc[0] == 100.50


def test_threshold_optimization_favors_recall_under_asymmetric_cost():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_proba = np.array([0.10, 0.20, 0.35, 0.45, 0.25, 0.80])
    thresh, _ = find_optimal_threshold(y_true, y_proba)
    assert thresh < 0.50


def test_artifact_metadata_contains_lineage_and_versions(trained_artifacts):
    meta = trained_artifacts["metadata"]
    assert len(meta["training_data_sha256"]) == 64
    assert "scikit_learn" in meta["environment"]
    assert 0.0 < meta["optimal_threshold"] < 1.0
    assert "roc_auc" in meta["metrics"]


def test_pipeline_handles_unseen_categories_gracefully(trained_artifacts, sample_payload):
    model = joblib.load(trained_artifacts["model_path"])
    mutated = dict(sample_payload)
    mutated["Contract"] = "Three-year-crypto-plan"
    mutated["InternetService"] = "Quantum-Satellite"
    df = pd.DataFrame([mutated])
    proba = model.predict_proba(df)[:, 1]
    assert 0.0 <= proba[0] <= 1.0
