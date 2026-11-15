from contextlib import asynccontextmanager
import json
import logging
import os
import secrets
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.predictor import ChurnModelService
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    DriftCheckResponse,
    PredictionResponse,
)

logger = logging.getLogger("churn_api")
logging.basicConfig(level=logging.INFO, format="%(message)s")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_expected_api_key() -> str:
    return os.getenv("CHURN_API_KEY", "dev-secret-key-change-in-prod")


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    expected = get_expected_api_key()
    if not api_key or not secrets.compare_digest(api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", "artifacts/model_v1.joblib")
    metadata_path = os.getenv("METADATA_PATH", "artifacts/metadata.json")
    app.state.service = ChurnModelService(model_path=model_path, metadata_path=metadata_path)
    yield


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Real-time & batch churn scoring service with lineage tracking and PSI drift detection.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def structured_request_logger(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(latency_ms)

    log_entry = {
        "event": "http_request",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": latency_ms,
    }
    logger.info(json.dumps(log_entry))
    return response


def get_service(request: Request) -> ChurnModelService:
    return request.app.state.service


@app.get("/health", tags=["Operations"])
def health_check(service: ChurnModelService = Depends(get_service)):
    return {
        "status": "healthy",
        "model_version": service.model_version,
        "training_data_sha256": service.data_hash,
        "decision_threshold": service.threshold,
        "metrics": service.metadata.get("metrics", {}),
    }


@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Inference"],
)
def predict_single(
    customer: CustomerFeatures,
    service: ChurnModelService = Depends(get_service),
):
    return service.predict_batch([customer])[0]


@app.post(
    "/v1/predict/batch",
    response_model=BatchPredictionResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Inference"],
)
def predict_batch(
    payload: BatchPredictionRequest,
    service: ChurnModelService = Depends(get_service),
):
    preds = service.predict_batch(payload.customers)
    return BatchPredictionResponse(predictions=preds, count=len(preds))


@app.get(
    "/v1/drift",
    response_model=DriftCheckResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Monitoring"],
)
def check_drift(
    psi_threshold: float = 0.2,
    service: ChurnModelService = Depends(get_service),
):
    return service.compute_drift(psi_threshold=psi_threshold)
