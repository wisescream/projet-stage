"""Local immutable model registry with explicit promotion and rollback."""

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_model_manifest(model_path, output_dir, *, metrics, feature_names, training_period,
                         calibration_method):
    model_path, output_dir = Path(model_path), Path(output_dir)
    digest = sha256_file(model_path)
    manifest = {
        "schema_version": 1,
        "model_version": f"hgb-{digest[:16]}",
        "artifact": model_path.name,
        "sha256": digest,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_period": training_period,
        "feature_count": len(feature_names),
        "feature_schema": list(feature_names),
        "validation_metrics": metrics,
        "calibration_method": calibration_method,
        "status": "trained_not_promoted",
    }
    path = output_dir / "model-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def promote_model(model_path, manifest_path, registry_dir):
    model_path, manifest_path, registry_dir = Path(model_path), Path(manifest_path), Path(registry_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256_file(model_path)
    if digest != manifest.get("sha256"):
        raise ValueError("Model hash does not match its manifest")
    version = manifest.get("model_version")
    if not isinstance(version, str) or not version or Path(version).name != version:
        raise ValueError("Manifest has an invalid model_version")
    destination = registry_dir / "versions" / version
    if destination.exists():
        raise FileExistsError(f"Model version already exists: {version}")
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(model_path, destination / "model.joblib")
    promoted_manifest = {**manifest, "artifact": "model.joblib", "status": "promoted",
                         "promoted_at": datetime.now(timezone.utc).isoformat()}
    (destination / "model-manifest.json").write_text(
        json.dumps(promoted_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    registry_path = registry_dir / "registry.json"
    current = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {
        "schema_version": 1, "active_version": None, "previous_version": None, "promotions": []}
    current["previous_version"] = current.get("active_version")
    current["active_version"] = version
    current.setdefault("promotions", []).append({"version": version, "promoted_at": promoted_manifest["promoted_at"]})
    _write_json_atomic(registry_path, current)
    return current


def rollback_model(registry_dir):
    registry_dir = Path(registry_dir)
    registry_path = registry_dir / "registry.json"
    if not registry_path.exists():
        raise FileNotFoundError("Model registry has no promoted version")
    current = json.loads(registry_path.read_text(encoding="utf-8"))
    active, previous = current.get("active_version"), current.get("previous_version")
    if not previous:
        raise ValueError("No previous model version is available for rollback")
    model = registry_dir / "versions" / previous / "model.joblib"
    if not model.is_file():
        raise FileNotFoundError(f"Previous model artifact is missing: {model}")
    current["active_version"], current["previous_version"] = previous, active
    current.setdefault("rollbacks", []).append({
        "from_version": active,
        "to_version": previous,
        "rolled_back_at": datetime.now(timezone.utc).isoformat(),
    })
    _write_json_atomic(registry_path, current)
    return current


def active_model_path(registry_dir):
    registry_dir = Path(registry_dir)
    registry = json.loads((registry_dir / "registry.json").read_text(encoding="utf-8"))
    version = registry.get("active_version")
    if not version:
        raise ValueError("No active model has been promoted")
    path = registry_dir / "versions" / version / "model.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Active model artifact is missing: {path}")
    return path