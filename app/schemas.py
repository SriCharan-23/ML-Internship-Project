from typing import Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customerID: str = Field(..., min_length=1, max_length=64, examples=["7590-VHVEG"])
    gender: str = Field(..., min_length=1, examples=["Female"])
    SeniorCitizen: Union[int, str] = Field(..., examples=[0])
    Partner: str = Field(..., min_length=1, examples=["Yes"])
    Dependents: str = Field(..., min_length=1, examples=["No"])
    tenure: int = Field(..., ge=0, le=120, examples=[12])
    PhoneService: str = Field(..., min_length=1, examples=["Yes"])
    MultipleLines: str = Field(..., min_length=1, examples=["No"])
    InternetService: str = Field(..., min_length=1, examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., min_length=1, examples=["No"])
    OnlineBackup: str = Field(..., min_length=1, examples=["Yes"])
    DeviceProtection: str = Field(..., min_length=1, examples=["No"])
    TechSupport: str = Field(..., min_length=1, examples=["No"])
    StreamingTV: str = Field(..., min_length=1, examples=["No"])
    StreamingMovies: str = Field(..., min_length=1, examples=["No"])
    Contract: str = Field(..., min_length=1, examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., min_length=1, examples=["Yes"])
    PaymentMethod: str = Field(..., min_length=1, examples=["Electronic check"])
    MonthlyCharges: float = Field(..., gt=0, le=10000, examples=[70.35])
    TotalCharges: Union[float, int, str] = Field(..., examples=["844.20"])

    @field_validator("SeniorCitizen")
    @classmethod
    def validate_senior(cls, v: Union[int, str]) -> str:
        str_val = str(v).strip()
        if str_val not in {"0", "1"}:
            raise ValueError("SeniorCitizen must be 0 or 1")
        return str_val

    @field_validator("TotalCharges")
    @classmethod
    def validate_total_charges(cls, v: Union[float, int, str]) -> Union[float, str]:
        if isinstance(v, (int, float)):
            if v < 0:
                raise ValueError("TotalCharges cannot be negative")
            return float(v)
        stripped = v.strip()
        if stripped == "":
            return stripped  # Handled by NumericStringCleaner + median imputer
        try:
            num = float(stripped)
            if num < 0:
                raise ValueError("TotalCharges cannot be negative")
            return stripped
        except ValueError as exc:
            raise ValueError("TotalCharges must be a non-negative number or blank string") from exc


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    customerID: str
    churn_probability: float
    churn_prediction: bool
    decision_threshold: float
    risk_tier: str
    model_version: str
    training_data_sha256: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class DriftFeatureReport(BaseModel):
    feature: str
    psi: float
    drifted: bool


class DriftCheckResponse(BaseModel):
    sample_size: int
    psi_threshold: float
    drift_detected: bool
    flagged_features: list[str]
    details: list[DriftFeatureReport]
