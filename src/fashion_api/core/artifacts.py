from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from fashion_api.ml.labels import LABEL_MAPS


def _status(
    ready: bool,
    version: str | None,
    architecture: str | None,
    message: str,
    metadata: dict,
    manifest: dict,
) -> ArtifactStatus:
    return ArtifactStatus(ready, version, architecture, message, metadata, manifest)


@dataclass(frozen=True)
class ArtifactStatus:
    ready: bool
    version: str | None
    architecture: str | None
    message: str
    metadata: dict
    manifest: dict


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts(model_root: Path) -> ArtifactStatus:
    metadata = _read_json(model_root / "metadata.json")
    manifest = _read_json(model_root / "manifest.json")
    version = str(metadata.get("version")) if metadata.get("version") else None
    architecture = metadata.get("architecture")

    if not metadata:
        return _status(False, None, None, "metadata.json is missing or invalid.", metadata, manifest)
    if not manifest:
        return _status(False, version, architecture, "manifest.json is missing or invalid.", metadata, manifest)

    checkpoint = model_root / str(metadata.get("checkpoint", "model.pt"))
    if not checkpoint.exists():
        return _status(False, version, architecture, "model checkpoint is missing.", metadata, manifest)

    backbone_repo = Path(str(metadata.get("backbone_repo", model_root / "dinov2")))
    if str(metadata.get("backbone_source", "local")) == "local" and not backbone_repo.exists():
        return _status(False, version, architecture, "local DINOv2 repository is missing.", metadata, manifest)

    manifest_labels = manifest.get("labels", {})
    if manifest_labels and manifest_labels != LABEL_MAPS:
        return _status(
            False,
            version,
            architecture,
            "manifest label maps do not match service label maps.",
            metadata,
            manifest,
        )

    for artifact in manifest.get("artifacts", []):
        relative_path = artifact.get("path")
        expected_sha = artifact.get("sha256")
        if not relative_path:
            return _status(False, version, architecture, "manifest artifact entry is missing path.", metadata, manifest)
        artifact_path = model_root / str(relative_path)
        if artifact.get("type") == "directory":
            if artifact.get("required", True) and not artifact_path.is_dir():
                return _status(
                    False,
                    version,
                    architecture,
                    f"artifact directory is missing: {relative_path}",
                    metadata,
                    manifest,
                )
            continue
        if artifact.get("required", True) and not artifact_path.is_file():
            return _status(
                False,
                version,
                architecture,
                f"artifact file is missing: {relative_path}",
                metadata,
                manifest,
            )
        if expected_sha and artifact_path.is_file() and _sha256(artifact_path) != expected_sha:
            return _status(
                False,
                version,
                architecture,
                f"artifact checksum mismatch: {relative_path}",
                metadata,
                manifest,
            )

    return _status(True, version, architecture, "Artifacts validated.", metadata, manifest)
