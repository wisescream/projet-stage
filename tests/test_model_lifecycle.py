import json

import pytest

from fraud_detection.model_lifecycle import (
    active_model_path,
    promote_model,
    rollback_model,
    write_model_manifest,
)


def train_artifact(tmp_path, name, contents):
    model = tmp_path / name
    model.write_bytes(contents)
    output = tmp_path / f"{name}-artifacts"
    output.mkdir()
    manifest = write_model_manifest(
        model,
        output,
        metrics={"average_precision": 0.4, "brier_score_calibrated": 0.03},
        feature_names=["TransactionAmt", "ProductCD"],
        training_period={"start": 1, "end": 10, "rows": 10},
        calibration_method="identity_insufficient_calibration_data",
    )
    return model, output / "model-manifest.json", manifest


def test_promote_and_rollback_keep_immutable_versioned_artifacts(tmp_path):
    registry = tmp_path / "registry"
    first, first_manifest_path, first_manifest = train_artifact(tmp_path, "first.joblib", b"model-one")
    second, second_manifest_path, second_manifest = train_artifact(tmp_path, "second.joblib", b"model-two")

    first_registry = promote_model(first, first_manifest_path, registry)
    assert first_registry["active_version"] == first_manifest["model_version"]
    assert active_model_path(registry).read_bytes() == b"model-one"

    second_registry = promote_model(second, second_manifest_path, registry)
    assert second_registry["active_version"] == second_manifest["model_version"]
    assert second_registry["previous_version"] == first_manifest["model_version"]
    rolled_back = rollback_model(registry)
    assert rolled_back["active_version"] == first_manifest["model_version"]
    assert active_model_path(registry).read_bytes() == b"model-one"
    assert len(json.loads((registry / "registry.json").read_text())["rollbacks"]) == 1


def test_promote_rejects_a_manifest_hash_mismatch(tmp_path):
    model, manifest_path, _ = train_artifact(tmp_path, "tampered.joblib", b"trusted")
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        promote_model(model, manifest_path, tmp_path / "registry")
