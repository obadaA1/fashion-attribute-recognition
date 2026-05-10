import json
from pathlib import Path

from src.models.fashion_multihead import build_model, load_checkpoint, predict_image

from .schemas import AttributePrediction, FashionPredictionResponse, ModelInfoResponse, TopPrediction

SAMPLE_PREDICTIONS = {
    "color": AttributePrediction(
        label="navy",
        confidence=0.91,
        top_k=[TopPrediction(label="navy", confidence=0.91), TopPrediction(label="black", confidence=0.06)],
    ),
    "pattern": AttributePrediction(
        label="solid",
        confidence=0.88,
        top_k=[TopPrediction(label="solid", confidence=0.88), TopPrediction(label="geometric", confidence=0.07)],
    ),
    "material": AttributePrediction(
        label="cotton",
        confidence=0.73,
        top_k=[TopPrediction(label="cotton", confidence=0.73), TopPrediction(label="synthetic", confidence=0.18)],
    ),
    "fabric_texture": AttributePrediction(
        label="smooth",
        confidence=0.69,
        top_k=[
            TopPrediction(label="smooth", confidence=0.69),
            TopPrediction(label="ribbed/structured", confidence=0.19),
        ],
    ),
}


class FashionModelService:
    def __init__(self, model_root: Path) -> None:
        self.model_root = model_root
        self.metadata = self._load_metadata()
        self.weights_path = model_root / str(self.metadata.get("checkpoint", "model.pt"))
        self.device = str(self.metadata.get("device", "cpu"))
        self.backbone_repo = str(self.metadata.get("backbone_repo", model_root / "dinov2"))
        self.backbone_source = str(
            self.metadata.get("backbone_source", "local" if Path(self.backbone_repo).exists() else "github")
        )
        self._model = None

    def _load_metadata(self) -> dict:
        metadata_path = self.model_root / "metadata.json"
        if not metadata_path.exists():
            return {}
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    @property
    def model_loaded(self) -> bool:
        return bool(self.metadata) and self.weights_path.exists()

    @property
    def version(self) -> str | None:
        value = self.metadata.get("version")
        return str(value) if value else None

    def info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_loaded=self.model_loaded,
            version=self.version,
            architecture=(
                self.metadata.get("architecture", "DINOv2 ViT-B/14 multi-head classifier") if self.metadata else None
            ),
            training_date=self.metadata.get("training_date"),
            metrics=self.metadata.get("metrics", {}),
        )

    def predict(self, image_bytes: bytes) -> FashionPredictionResponse:
        if not self.model_loaded:
            raise RuntimeError("Model artifacts are not mounted or loaded.")

        if self._model is None:
            self._model = build_model(
                device=self.device,
                backbone_repo=self.backbone_repo,
                backbone_source=self.backbone_source,
            )
            load_checkpoint(self._model, self.weights_path, self.device)

        predictions = {
            attr: AttributePrediction(
                label=prediction.label,
                confidence=prediction.confidence,
                top_k=[
                    TopPrediction(label=str(item["label"]), confidence=float(item["confidence"]))
                    for item in prediction.top_k
                ],
            )
            for attr, prediction in predict_image(self._model, image_bytes, self.device).items()
        }

        return FashionPredictionResponse(
            model_version=self.version or "unknown",
            predictions=predictions,
        )
