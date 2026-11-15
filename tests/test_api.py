import statistics
import time

AUTH_HEADER = {"X-API-Key": "test-api-key-123"}


def test_health_endpoint_returns_lineage(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_version"] == "1.0.0"
    assert len(body["training_data_sha256"]) == 64


def test_predict_rejects_missing_api_key(client, sample_payload):
    resp = client.post("/v1/predict", json=sample_payload)
    assert resp.status_code == 401


def test_predict_rejects_invalid_api_key(client, sample_payload):
    resp = client.post("/v1/predict", json=sample_payload, headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_predict_single_valid_payload(client, sample_payload):
    resp = client.post("/v1/predict", json=sample_payload, headers=AUTH_HEADER)
    assert resp.status_code == 200
    data = resp.json()
    assert data["customerID"] == "7590-VHVEG"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["churn_prediction"], bool)
    assert data["risk_tier"] in {"LOW", "MEDIUM", "HIGH"}


def test_predict_handles_blank_total_charges_new_customer(client, sample_payload):
    payload = {**sample_payload, "tenure": 0, "TotalCharges": " "}
    resp = client.post("/v1/predict", json=payload, headers=AUTH_HEADER)
    assert resp.status_code == 200


def test_predict_handles_unseen_category_via_api(client, sample_payload):
    payload = {**sample_payload, "PaymentMethod": "Apple Pay Biometric"}
    resp = client.post("/v1/predict", json=payload, headers=AUTH_HEADER)
    assert resp.status_code == 200


def test_predict_rejects_negative_tenure(client, sample_payload):
    payload = {**sample_payload, "tenure": -5}
    resp = client.post("/v1/predict", json=payload, headers=AUTH_HEADER)
    assert resp.status_code == 422
    assert "tenure" in str(resp.json()["detail"])


def test_predict_rejects_invalid_senior_citizen(client, sample_payload):
    payload = {**sample_payload, "SeniorCitizen": 5}
    resp = client.post("/v1/predict", json=payload, headers=AUTH_HEADER)
    assert resp.status_code == 422


def test_predict_rejects_unparseable_total_charges(client, sample_payload):
    payload = {**sample_payload, "TotalCharges": "not_a_number"}
    resp = client.post("/v1/predict", json=payload, headers=AUTH_HEADER)
    assert resp.status_code == 422


def test_predict_rejects_extra_unexpected_fields(client, sample_payload):
    payload = {**sample_payload, "unexpected_crm_column": "foo"}
    resp = client.post("/v1/predict", json=payload, headers=AUTH_HEADER)
    assert resp.status_code == 422


def test_batch_prediction_endpoint(client, sample_payload):
    batch = {
        "customers": [
            {**sample_payload, "customerID": "C-1"},
            {**sample_payload, "customerID": "C-2", "Contract": "Two year", "tenure": 60},
        ]
    }
    resp = client.post("/v1/predict/batch", json=batch, headers=AUTH_HEADER)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_drift_endpoint_flags_shifted_distribution(client, sample_payload):
    drifted_batch = {
        "customers": [
            {**sample_payload, "customerID": f"D-{i}", "MonthlyCharges": 999.99}
            for i in range(40)
        ]
    }
    client.post("/v1/predict/batch", json=drifted_batch, headers=AUTH_HEADER)

    drift_resp = client.get("/v1/drift?psi_threshold=0.2", headers=AUTH_HEADER)
    assert drift_resp.status_code == 200
    report = drift_resp.json()
    assert report["drift_detected"] is True
    assert "MonthlyCharges" in report["flagged_features"]


def test_single_prediction_median_latency_under_100ms(client, sample_payload):
    latencies_ms = []
    for _ in range(30):
        t0 = time.perf_counter()
        resp = client.post("/v1/predict", json=sample_payload, headers=AUTH_HEADER)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        assert resp.status_code == 200

    median_ms = statistics.median(latencies_ms)
    assert median_ms < 100.0, f"Median latency {median_ms:.2f} ms exceeded 100 ms SLA"
