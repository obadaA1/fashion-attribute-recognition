from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class TopPrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class AttributePrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: list[TopPrediction] = Field(default_factory=list)


class FashionPredictionResponse(BaseModel):
    model_version: str
    predictions: dict[str, AttributePrediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    message: str | None = None


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    version: str | None = None
    architecture: str | None = None
    training_date: str | None = None
    metrics: dict[str, float | str] = Field(default_factory=dict)

