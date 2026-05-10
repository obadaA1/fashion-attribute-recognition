import logging
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from fashion_api.api.errors import http_exception_handler, unhandled_exception_handler
from fashion_api.api.middleware import request_id_middleware
from fashion_api.api.schemas import FashionPredictionResponse, HealthResponse, ModelInfoResponse
from fashion_api.api.validation import read_valid_image
from fashion_api.core.artifacts import validate_artifacts
from fashion_api.core.logging import configure_logging
from fashion_api.core.settings import get_settings
from fashion_api.ml.model import build_model, load_checkpoint, predict_image

logger = logging.getLogger("fashion_api")
settings = get_settings()


class FashionModelService:
    def __init__(self) -> None:
        self.status = validate_artifacts(settings.model_root)
        self._model = None

    @property
    def model_loaded(self) -> bool:
        return self.status.ready and self._model is not None

    @property
    def ready(self) -> bool:
        return self.status.ready

    @property
    def version(self) -> str | None:
        return self.status.version

    def load(self) -> None:
        if not self.status.ready:
            return
        if self._model is not None:
            return
        metadata = self.status.metadata
        model = build_model(
            device=str(metadata.get("device", "cpu")),
            backbone_repo=str(metadata.get("backbone_repo", settings.model_root / "dinov2")),
            backbone_source=str(metadata.get("backbone_source", "local")),
        )
        load_checkpoint(
            model,
            settings.model_root / str(metadata.get("checkpoint", "model.pt")),
            str(metadata.get("device", "cpu")),
        )
        self._model = model
        logger.info("fashion model loaded")

    def info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_loaded=self.model_loaded,
            version=self.status.version,
            architecture=self.status.architecture,
            training_date=self.status.metadata.get("training_date"),
            metrics=self.status.metadata.get("metrics", {}),
        )

    def predict(self, image_bytes: bytes) -> FashionPredictionResponse:
        if not self.ready:
            raise RuntimeError(self.status.message)
        self.load()
        if self._model is None:
            raise RuntimeError("Model failed to load.")
        predictions = {
            attr: {
                "label": prediction.label,
                "confidence": prediction.confidence,
                "top_k": prediction.top_k,
            }
            for attr, prediction in predict_image(
                self._model,
                image_bytes,
                str(self.status.metadata.get("device", "cpu")),
            ).items()
        }
        return FashionPredictionResponse(model_version=self.version or "unknown", predictions=predictions)


@lru_cache
def get_model_service() -> FashionModelService:
    return FashionModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    service = get_model_service()
    if service.ready:
        try:
            service.load()
        except Exception:
            logger.exception("fashion model startup load failed")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Fashion Attribute Recognition API",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    limiter = Limiter(key_func=get_remote_address, default_limits=[])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.middleware("http")(request_id_middleware)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "x-request-id"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        service = get_model_service()
        status_value = "ok" if service.model_loaded else "warming"
        return HealthResponse(
            status=status_value,
            model_loaded=service.model_loaded,
            model_version=service.version,
            message=service.status.message,
        )

    @app.get("/health/live", response_model=HealthResponse)
    def live() -> HealthResponse:
        service = get_model_service()
        return HealthResponse(status="ok", model_loaded=service.model_loaded, model_version=service.version)

    @app.get("/health/ready", response_model=HealthResponse)
    def ready() -> HealthResponse:
        service = get_model_service()
        if not service.ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=service.status.message)
        service.load()
        return HealthResponse(status="ok", model_loaded=service.model_loaded, model_version=service.version)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        return get_model_service().info()

    @app.post("/predict", response_model=FashionPredictionResponse)
    @limiter.limit("10/minute")
    async def predict(request: Request, file: UploadFile) -> FashionPredictionResponse:
        image_bytes = await read_valid_image(file, settings.max_image_bytes)
        service = get_model_service()
        if not service.ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=service.status.message)
        return service.predict(image_bytes)

    return app


app = create_app()
