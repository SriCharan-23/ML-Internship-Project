from collections import deque
import json

import joblib
import numpy as np
import pandas as pd

from app.schemas import CustomerFeatures, DriftFeatureReport, PredictionResponse
from app.transformers import NumericStringCleaner
from ml.train import FEATURE_COLS


class ChurnModelService:
    def __init__(self, model_path: str, metadata_path: str, buffer_size: int = 2000):
        self.model = joblib.load(model_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.threshold: float = self.metadata["optimal_threshold"]
        self.model_version: str = self.metadata["model_version"]
        self.data_hash: str = self.metadata["training_data_sha256"]
        self.ref_dist: dict = self.metadata.get("reference_distributions", {})
        self.recent_records: deque[dict] = deque(maxlen=buffer_size)

    def _risk_tier(self, prob: float) -> str:
        if prob >= max(self.threshold + 0.25, 0.70):
            return "HIGH"
        if prob >= self.threshold:
            return "MEDIUM"
        return "LOW"

    def predict_batch(self, customers: list[CustomerFeatures]) -> list[PredictionResponse]:
        records = [c.model_dump() for c in customers]
        for r in records:
            self.recent_records.append(r)

        df = pd.DataFrame(records)[FEATURE_COLS]
        probas = self.model.predict_proba(df)[:, 1]

        results = []
        for cust, prob in zip(customers, probas):
            prob_rounded = round(float(prob), 4)
            results.append(
                PredictionResponse(
                    customerID=cust.customerID,
                    churn_probability=prob_rounded,
                    churn_prediction=bool(prob >= self.threshold),
                    decision_threshold=self.threshold,
                    risk_tier=self._risk_tier(prob),
                    model_version=self.model_version,
                    training_data_sha256=self.data_hash,
                )
            )
        return results

    def compute_drift(self, psi_threshold: float = 0.2) -> dict:
        if len(self.recent_records) < 30:
            return {
                "sample_size": len(self.recent_records),
                "psi_threshold": psi_threshold,
                "drift_detected": False,
                "flagged_features": [],
                "details": [],
            }

        df_recent = pd.DataFrame(list(self.recent_records))[FEATURE_COLS]
        df_recent = NumericStringCleaner(columns=["TotalCharges"]).transform(df_recent)
        eps = 1e-6
        details: list[DriftFeatureReport] = []
        flagged: list[str] = []

        for col, spec in self.ref_dist.items():
            if spec["type"] == "numeric":
                series = df_recent[col].dropna()
                if len(series) == 0:
                    continue
                edges = np.array(spec["bin_edges"], dtype=float)
                edges[0] = -np.inf
                edges[-1] = np.inf
                actual_counts, _ = np.histogram(series, bins=edges)
                actual_props = np.clip(actual_counts / len(series), eps, 1.0)
                expected_props = np.clip(np.array(spec["proportions"], dtype=float), eps, 1.0)
                psi = float(np.sum((actual_props - expected_props) * np.log(actual_props / expected_props)))
            else:
                vc = df_recent[col].astype(str).value_counts(normalize=True).to_dict()
                expected_map: dict = spec["proportions"]
                all_cats = set(expected_map.keys()) | set(vc.keys())
                psi = 0.0
                for cat in all_cats:
                    exp_p = max(float(expected_map.get(cat, 0.0)), eps)
                    act_p = max(float(vc.get(cat, 0.0)), eps)
                    psi += (act_p - exp_p) * np.log(act_p / exp_p)

            psi_rounded = round(float(psi), 4)
            is_drifted = psi_rounded > psi_threshold
            if is_drifted:
                flagged.append(col)
            details.append(DriftFeatureReport(feature=col, psi=psi_rounded, drifted=is_drifted))

        return {
            "sample_size": len(self.recent_records),
            "psi_threshold": psi_threshold,
            "drift_detected": len(flagged) > 0,
            "flagged_features": flagged,
            "details": details,
        }
