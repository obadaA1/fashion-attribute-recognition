from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fashion_api.ml.labels import LABEL_MAPS, PUBLIC_ATTRIBUTE_NAMES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EMBED_DIM = 768


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    top_k: list[dict[str, float | str]]


def _import_torch_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("Fashion inference dependencies are missing.") from exc
    return torch, nn, Image, transforms


def build_eval_transform():
    _, _, _, transforms = _import_torch_stack()
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_model(device: str = "cpu", backbone_repo: str = "facebookresearch/dinov2", backbone_source: str = "github"):
    torch, nn, _, _ = _import_torch_stack()

    class FashionMultiHeadModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.hub.load(backbone_repo, "dinov2_vitb14", source=backbone_source)
            self.heads = nn.ModuleDict(
                {attr: nn.Linear(EMBED_DIM, len(labels)) for attr, labels in LABEL_MAPS.items()}
            )

        def forward(self, x):
            feats = self.backbone(x)
            return {attr: head(feats) for attr, head in self.heads.items()}

    model = FashionMultiHeadModel().to(device)
    model.eval()
    return model


def load_checkpoint(model, checkpoint_path: Path, device: str):
    torch, _, _, _ = _import_torch_stack()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return checkpoint


def predict_image(model, image_bytes: bytes, device: str = "cpu", top_k: int = 2) -> dict[str, Prediction]:
    torch, _, Image, _ = _import_torch_stack()
    transform = build_eval_transform()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)

    predictions: dict[str, Prediction] = {}
    for attr, values in logits.items():
        probs = torch.softmax(values[0], dim=0).detach().cpu()
        k = min(top_k, len(LABEL_MAPS[attr]))
        confs, indices = torch.topk(probs, k=k)
        top = [
            {"label": LABEL_MAPS[attr][idx.item()], "confidence": float(conf.item())}
            for conf, idx in zip(confs, indices, strict=True)
        ]
        predictions[PUBLIC_ATTRIBUTE_NAMES[attr]] = Prediction(
            label=str(top[0]["label"]),
            confidence=float(top[0]["confidence"]),
            top_k=top,
        )
    return predictions
