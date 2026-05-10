# Fashion API Deployment

The API is designed to run on the UN1290 behind Cloudflare Tunnel. Bind the container to localhost only and let Cloudflare provide public HTTPS.

## Artifact layout

Model artifacts are mounted read-only and are not committed to git:

```text
/models/fashion/v1/metadata.json
/models/fashion/v1/manifest.json
/models/fashion/v1/model.pt
/models/fashion/v1/dinov2/
/models/fashion/current -> /models/fashion/v1
```

Example `metadata.json`:

```json
{
  "version": "v1",
  "architecture": "DINOv2 ViT-B/14 multi-head classifier",
  "checkpoint": "model.pt",
  "device": "cpu",
  "backbone_repo": "/models/fashion/current/dinov2",
  "backbone_source": "local",
  "training_date": "2026-04-01",
  "metrics": {
    "mean_macro_f1": 0.765
  }
}
```

Example `manifest.json`:

```json
{
  "model_version": "v1",
  "architecture": "DINOv2 ViT-B/14 multi-head classifier",
  "created_at": "2026-04-01T00:00:00Z",
  "labels": {
    "color": ["black", "blue", "brown", "green", "red", "white"],
    "pattern": ["solid", "striped", "plaid", "floral", "graphic"],
    "material": ["cotton", "denim", "leather", "polyester", "silk", "wool"],
    "texture": ["smooth", "knit", "woven", "ribbed", "fuzzy"]
  },
  "artifacts": [
    {"path": "model.pt", "sha256": "<sha256>"},
    {"path": "dinov2/hubconf.py", "sha256": "<sha256>"}
  ]
}
```

The notebook checkpoint is expected to contain either a raw state dict or the
Colab training dictionary with `model_state`. Export the best checkpoint from
Drive to `/models/fashion/v1/model.pt`; do not commit it. For production,
mirror the DINOv2 torch hub repository into `/models/fashion/v1/dinov2` and
set `backbone_source` to `local` so inference does not depend on runtime
network access.

## Local run

```bash
pip install -r requirements.txt
FASHION_MODEL_ROOT=/models/fashion/current uvicorn fashion_api.api.app:app --host 127.0.0.1 --port 8011
```

## Docker run

```bash
docker compose -f docker-compose.example.yml up -d --build
```

The compose example follows the production security baseline:

- binds only to `127.0.0.1`
- runs as the non-root `app` user from the image
- drops Linux capabilities
- enables `no-new-privileges`
- mounts model artifacts read-only
- uses a read-only root filesystem plus a small `/tmp` tmpfs
- exposes healthchecks for Docker and Cloudflare

Cloudflare Tunnel route:

```text
fashion-api.obadaalsehli.com -> http://127.0.0.1:8011
```

Recommended Cloudflare rule: 10 requests/minute per IP for this hostname.
