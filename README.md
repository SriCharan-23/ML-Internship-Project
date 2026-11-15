# Telco Customer Churn API

## Local Setup & Training
1. **Install dependencies:**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Train & version the artifact** (place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/`):
   ```bash
   python -m ml.train data/WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```
3. **Run the test suite (17 unit & integration tests):**
   ```bash
   pytest -v
   ```
4. **Start the API locally:**
   ```bash
   export CHURN_API_KEY="super-secret-crm-key"
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Docker Build & Run
```bash
docker build -t telco-churn-api:1.0.0 .
docker images telco-churn-api:1.0.0  # Verify size < 600 MB (~310 MB)

docker run -d --name churn-service \
  -p 8000:8000 \
  -e CHURN_API_KEY="super-secret-crm-key" \
  telco-churn-api:1.0.0
```

## Example Requests

**Health & Lineage Check:**
```bash
curl -s http://localhost:8000/health | jq
```

**Single Customer Prediction:**
```bash
curl -s -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: super-secret-crm-key" \
  -d '{
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": "171.00"
  }' | jq
```

**PSI Drift Check:**
```bash
curl -s "http://localhost:8000/v1/drift?psi_threshold=0.2" \
  -H "X-API-Key: super-secret-crm-key" | jq
```

---

## Safe Retraining & Redeployment Strategy

1. **Automated Triggering:**
   - **Scheduled:** Monthly retraining job using a rolling 12-month window of CRM billing and cancellation outcomes.
   - **Event-Driven:** Triggered automatically when `/v1/drift` reports `PSI > 0.2` across top predictive features (`Contract`, `tenure`, `MonthlyCharges`) for 3 consecutive days, or when realized precision/recall drops below SLA thresholds.
2. **Offline Validation Gate (CI Pipeline):**
   - The candidate pipeline runs `ml/train.py` on the new dataset snapshot, recording a new `training_data_sha256` and `model_version`.
   - Automated promotion checks verify: (a) `PR-AUC` and expected cost per customer improve over or match the production baseline, (b) `pytest` passes 100%, and (c) p50 latency remains under 100 ms.
3. **Shadow Deployment:**
   - Deploy `v2` alongside `v1` in Kubernetes. Mirror 100% of live CRM traffic asynchronously to `v2` without returning `v2` predictions to the CRM. Compare prediction score distributions, latency, and error rates over 48 hours.
4. **Canary Rollout & Instant Rollback:**
   - Shift live traffic progressively (`10% -> 50% -> 100%`) using weighted ingress routing while keeping `v1` warm. If HTTP 5xx rates exceed 0.5% or p95 latency spikes above 150 ms, traffic automatically reverts to `v1`.
