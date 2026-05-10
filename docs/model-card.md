# Fashion Attribute Recognition Model Card

## Intended Use

Classify one fashion image into color, pattern, material, and fabric texture attributes for a public portfolio demo.

## Inputs and Outputs

- Input: one JPEG or PNG image, maximum 10 MB.
- Output: top prediction and top-k confidence values for each supported attribute head.

## Runtime Artifacts

Artifacts are mounted read-only at `/models/fashion/current`:

- `metadata.json`
- `manifest.json`
- `model.pt`
- `dinov2/`

`manifest.json` records artifact names, SHA256 checksums, model type, version, architecture, and label maps.

## Limitations

The model is trained on a TextileNet-derived subset and may perform poorly on non-fashion images, unusual lighting, occlusion, layered garments, or attributes outside the fixed label maps.

## Operational Notes

The API refuses readiness until model metadata, manifest, checkpoint, local DINOv2 repository, and label maps validate successfully.

