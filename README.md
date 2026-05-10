# Multi-Head Fashion Attribute Recognition

Self-supervised backbones (DINOv2) + VLM-audited silver labels for joint per-image classification of **color, pattern, material, and texture** on a 13,355-image TextileNet subset.

**Mean test macro-F1: 0.765** (DINOv2 ViT-B/14) vs **0.649** (ResNet-50, identical recipe) — **+11.6 pp** absolute. DINOv2 wins on every head and on all 29 classes.

📓 [Notebook](./research/fashion_attributes.ipynb) · 🔬 [Provenance manifest](./research/data_manifests/manifest.json)

Large research exports such as the PDF report and generated figures are intentionally kept out of normal git history. Publish them through Git LFS, GitHub Releases, or external artifact storage when needed.

---

## TL;DR

Public fashion datasets each cover at most two of {color, pattern, material, texture}; none label all four jointly per image. I built a 13,355-image TextileNet subset and hybrid-labeled it: rule-based mapping for material (perfect-precision by construction), Marqo-FashionSigLIP zero-shot with 5-template prompt ensembling and confidence-threshold abstain for color / pattern / texture. A 210-image stratified human audit anchors silver-label trust at 0.971 / 0.803 / 0.681 macro-F1. A DINOv2 ViT-B/14 backbone with four parallel linear heads reaches mean test macro-F1 0.765 — uniformly +11.6 pp over an identically-recipe-trained ResNet-50. Cramér's V on material×texture (V_gt=0.318, V_pred=0.353, Δ=+0.035) confirms the multi-head design preserves rather than collapses the joint distribution.

## Method

- **Backbone:** DINOv2 ViT-B/14 (86.6M params, self-supervised on LVD-142M)
- **Heads:** 4 parallel linear (768 → {6, 12, 6, 5}; 22.3k params)
- **Loss:** sum of 4 class-weighted CE, abstained labels masked via `ignore_index=-100`
- **Schedule:** 20 epochs, AdamW, discriminative LRs (1e-5 backbone / 1e-4 heads), 2-epoch warmup + 18-epoch cosine, AMP fp16, batch 64
- **Hardware:** single NVIDIA T4

## Results

| Head     | n eval | Acc.  | DINOv2 F1 | ResNet-50 F1 | Δ      |
|----------|-------:|------:|----------:|-------------:|-------:|
| material |  3,760 | 0.747 |     0.744 |        0.612 | +13.2  |
| color    |  3,743 | 0.842 |     0.827 |        0.738 |  +8.9  |
| pattern  |  3,527 | 0.822 |     0.737 |        0.616 | +12.1  |
| texture  |  3,487 | 0.742 |     0.753 |        0.630 | +12.3  |
| **mean** |    —   | **0.788** | **0.765** | **0.649** | **+11.6** |

Same data, splits, augmentation, optimizer, schedule, class weights. Only the backbone changes.

## Reproduction
git clone https://github.com/obadaA1/fashion-attribute-recognition
cd fashion-attribute-recognition
pip install -r requirements.txt

## Production inference

The original model was trained in `fashion_attributes.ipynb` on Google Colab.
Production serving now lives in importable modules under `src/fashion_api`:

- `src/fashion_api/ml/model.py` defines the locked label maps, DINOv2
  multi-head architecture, ImageNet preprocessing, checkpoint loading, and
  top-k prediction conversion.
- `src/fashion_api/api` wraps that model with FastAPI, upload validation,
  health checks, and model metadata endpoints.
- `src/fashion_api/core` owns settings, structured logging, and model artifact
  validation.

Runtime checkpoints stay outside git and are mounted on the UN1290. See
`docs/deployment.md` for the expected `/models/fashion/current` layout,
including the local DINOv2 hub repository used for network-free production
startup.

## API contract

- `GET /health` - reports service and model-artifact status.
- `GET /health/live` - process liveness check.
- `GET /health/ready` - readiness check; returns 503 until artifacts validate and the model is loadable.
- `GET /model-info` - reports active model metadata without exposing filesystem paths or secrets.
- `POST /predict` - accepts one JPEG/PNG image up to 10 MB.

Structured errors use one shape:

```json
{
  "error": {
    "code": "request_error",
    "message": "Uploaded file does not look like a valid JPEG or PNG image.",
    "request_id": "2f8c8a2a-5c5d-47f1-8f9a-7ec5f4e2d6d9"
  }
}
```

Example request:

```bash
curl -fsS http://127.0.0.1:8011/health/live
curl -fsS http://127.0.0.1:8011/model-info
curl -fsS -X POST http://127.0.0.1:8011/predict \
  -F "file=@sample-garment.jpg;type=image/jpeg"
```

Example response:

```json
{
  "model_version": "v1",
  "predictions": {
    "color": {"label": "navy", "confidence": 0.91, "top_k": []},
    "pattern": {"label": "solid", "confidence": 0.88, "top_k": []},
    "material": {"label": "cotton", "confidence": 0.73, "top_k": []},
    "texture": {"label": "smooth", "confidence": 0.69, "top_k": []}
  }
}
```

## Operations

CI/CD is defined in `.github/workflows/ci.yml`:

- Ruff linting.
- Mypy type checking.
- Pytest with coverage threshold.
- Dependency audit with `pip-audit`.
- Secret scanning with Gitleaks.
- Docker build on PR/push.
- GHCR publish and Trivy image scan on `main`/version tags.

Tests live in `tests/` and cover health/readiness behavior, artifact validation, upload validation, structured error responses, and mocked prediction contracts.

Docker Compose is available in `docker-compose.example.yml`. It binds only to `127.0.0.1`, runs as a non-root user, drops Linux capabilities, uses a read-only root filesystem, and mounts model artifacts read-only.

CORS is configured through `FASHION_CORS_ORIGINS` and should be restricted in production to:

```text
https://obadaalsehli.com
```
