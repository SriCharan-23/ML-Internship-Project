import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from ml.train import train_and_export


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory):
    rng = np.random.default_rng(42)
    n = 400
    tenure = rng.integers(0, 72, size=n)
    monthly = np.round(rng.uniform(20.0, 115.0, size=n), 2)
    total = [
        " " if t == 0 else f"{t * m:.2f}" for t, m in zip(tenure, monthly)
    ]
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["Fiber optic", "DSL", "No"], size=n)
    churn_prob = np.where(contract == "Month-to-month", 0.42, 0.08)
    churn = np.where(rng.random(n) < churn_prob, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": [f"CUST-{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], size=n),
            "SeniorCitizen": rng.choice([0, 1], size=n),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Dependents": rng.choice(["Yes", "No"], size=n),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], size=n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=n),
            "InternetService": internet,
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], size=n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size=n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=n),
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                size=n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": churn,
        }
    )

    tmp_dir = tmp_path_factory.mktemp("ml_run")
    csv_path = tmp_dir / "telco_sample.csv"
    df.to_csv(csv_path, index=False)

    art_dir = tmp_dir / "artifacts"
    meta = train_and_export(str(csv_path), str(art_dir))
    return {
        "model_path": str(art_dir / "model_v1.joblib"),
        "metadata_path": str(art_dir / "metadata.json"),
        "metadata": meta,
    }


@pytest.fixture
def client(trained_artifacts, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", trained_artifacts["model_path"])
    monkeypatch.setenv("METADATA_PATH", trained_artifacts["metadata_path"])
    monkeypatch.setenv("CHURN_API_KEY", "test-api-key-123")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_payload():
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85",
    }
