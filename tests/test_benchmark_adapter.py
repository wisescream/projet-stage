import hashlib
import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from fraud_detection.pipeline import build_model
from fraud_detection.simulation.decision import BenchmarkModel
from fraud_detection.simulation.replay import build_synthetic_bundle, timeline, write_bundle
from fraud_detection.simulation.schemas import identifier


def test_real_model_reference_binding_schema_and_hash(tmp_path):
    synthetic = tmp_path / "synthetic"
    build_synthetic_bundle(synthetic, 1)
    event = next(timeline(synthetic))
    event = event.model_copy(update={"source": "ieee_replay", "feature_ref": identifier("ref", event.transaction_id)})
    frame = pd.DataFrame({"TransactionAmt": np.tile([10, 100], 40).astype("float32"),
                          "ProductCD": ["C", "W"] * 40, "D1": np.nan})
    model = build_model(frame.columns, max_iter=5)
    with threadpool_limits(limits=1):
        model.fit(frame, np.tile([0, 1], 40))
        expected = float(model.predict_proba(frame.iloc[:1])[0, 1])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    data = {"TransactionAmt": 10.0, "ProductCD": "C", "D1": None}
    record = {"features": data, "authorization": event.model_dump(mode="json")}
    bundle = tmp_path / "replay"
    write_bundle(bundle, [event], [], [(event.feature_ref, json.dumps(record))],
                 {"model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    adapter = BenchmarkModel(path, bundle / "features.sqlite")
    assert adapter.score(event) == expected
    explanation = adapter.explain_features(event)
    assert explanation["method"] == "single_feature_missing_value_perturbation"
    assert explanation["effects"]
    assert all("probability_delta" in effect for effect in explanation["effects"])
    with pytest.raises(ValueError, match="differs"):
        adapter.score(event.model_copy(update={"country": "US"}))
    record["features"]["isFraud"] = 1
    with sqlite3.connect(bundle / "features.sqlite") as db:
        db.execute("UPDATE features SET payload=?", (json.dumps(record),))
    with pytest.raises(ValueError, match="schema"):
        adapter.score(event)
    with sqlite3.connect(bundle / "features.sqlite") as db:
        db.execute("UPDATE metadata SET payload=?", (json.dumps({"model_sha256": "wrong-model"}),))
    assert BenchmarkModel(path, bundle / "features.sqlite").version == "unavailable"
