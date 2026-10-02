"""Build a pseudonymized replay bundle with strictly separate features and labels."""
import hashlib
import heapq
import hmac
import json
import secrets
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from fraud_detection.pipeline import ID, TARGET, TIME, features, load_identity, merge_identity, read_table
from fraud_detection.simulation.schemas import Authorization, FraudLabel, identifier

AMOUNT_BINS = [0, 50, 100, 250, 1000, 5000, float("inf")]


def token(key, prefix, *values):
    # A fresh, non-persisted HMAC key per bundle prevents dictionary reversal of low-cardinality codes.
    digest = hmac.new(key, json.dumps(values, default=str).encode(), hashlib.sha256).hexdigest()[:24]
    return f"{prefix}_{digest}"


def label_for(event, label, days):
    when = event.event_time + timedelta(days=days)
    return FraudLabel(event_id=identifier("evt", event.event_id + ":label"),
                      transaction_id=event.transaction_id, trace_id=event.trace_id,
                      event_time=when, label_time=when, authorization_time=event.event_time,
                      label=int(label), label_type="chargeback" if label else "historical_confirmation")


def write_bundle(output, events, labels, feature_rows, metadata):
    output = Path(output)
    # Never mix a new bundle with old feature references or overwrite an active replay.
    output.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(output / "features.sqlite") as db:
        db.execute("CREATE TABLE features (reference TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        db.executemany("INSERT INTO features VALUES (?, ?)", feature_rows)
        db.execute("CREATE TABLE metadata (payload TEXT NOT NULL)")
        db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata, allow_nan=False),))
    for name, values in [("events.jsonl", events), ("labels.jsonl", labels)]:
        with (output / name).open("w", encoding="utf-8") as stream:
            for event in sorted(values, key=lambda e: (e.event_time, e.event_id)):
                stream.write(event.model_dump_json() + "\n")
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_ieee_bundle(data_dir, artifacts, output, limit=1000):
    if limit < 1:
        raise ValueError("limit must be positive")
    data_dir, artifacts = Path(data_dir), Path(artifacts)
    metrics = json.loads((artifacts / "metrics.json").read_text())
    model_path = artifacts / "model-validation.joblib"
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    wanted = set(pd.read_csv(artifacts / "validation_predictions.csv", nrows=limit)[ID])
    parts = []
    for chunk in read_table(data_dir / "train_transaction.csv", chunksize=20000):
        selected = chunk.loc[chunk[ID].isin(wanted)]
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise ValueError("No held-out transactions found")
    frame = merge_identity(pd.concat(parts), load_identity(data_dir / "train_identity.csv")).sort_values(TIME, kind="stable")
    if set(frame[ID]) != wanted or not (frame[TIME] > metrics["validation"]["train_max_time"]).all():
        raise ValueError("Replay must contain only held-out rows strictly after model training")
    # Reference distribution comes from training history, not the replay's future.
    amounts = pd.read_csv(data_dir / "train_transaction.csv", usecols=[TIME, "TransactionAmt"])
    reference = np.histogram(amounts.loc[amounts[TIME] <= metrics["validation"]["train_max_time"], "TransactionAmt"], bins=AMOUNT_BINS)[0]
    key, run = secrets.token_bytes(32), uuid4().hex
    base = datetime(2026, 9, 11, tzinfo=timezone.utc)
    first = float(frame[TIME].min())
    events, labels, rows = [], [], []
    for i, (_, row) in enumerate(frame.iterrows()):
        transaction = identifier("txn", f"{run}:{int(row[ID])}")
        reference_id = identifier("ref", transaction)
        country = ["FR", "DE", "GB", "US", "ES"][i % 5]
        event = Authorization(
            event_id=identifier("evt", transaction + ":authorize"), transaction_id=transaction,
            trace_id=identifier("trace", transaction), event_time=base + timedelta(seconds=float(row[TIME]) - first),
            source="ieee_replay", customer_token=token(key, "cus", row.get("card1"), row.get("card2")),
            card_token=token(key, "card", row.get("card1"), row.get("card2")),
            merchant_id=token(key, "merchant", row.get("ProductCD"), i % 17),
            amount=f"{float(row['TransactionAmt']):.2f}", currency="EUR", country=country,
            channel=["ecommerce", "pos", "mobile"][i % 3],
            device_id=token(key, "device", row.get("DeviceInfo"), row.get("card1")),
            ip_hash=token(key, "ip", row.get("addr1"), row.get("addr2"), i % 7),
            merchant_category=str(row["ProductCD"]), three_ds={"authenticated": i % 7 != 0},
            identity_data_missing=bool(pd.isna(row.get("id_01"))), feature_ref=reference_id,
        )
        payload = {k: None if pd.isna(v) else v for k, v in features(row.to_frame().T).iloc[0].items()}
        # Identity/email fields are benchmark data, kept outside the banking event bus.
        rows.append((reference_id, json.dumps({"features": payload, "authorization": event.model_dump(mode="json")}, allow_nan=False)))
        events.append(event)
        labels.append(label_for(event, row[TARGET], 7 if row[TARGET] else 1))
    return write_bundle(output, events, labels, rows, {
        "simulation": True, "source": "ieee_replay", "run_id": run, "count": len(events),
        "model_sha256": model_hash, "model_train_max_time": metrics["validation"]["train_max_time"],
        "amount_reference_counts": reference.tolist(),
        "warning": "Synthetic banking context; IEEE fraud labels simulate confirmations, not observed chargebacks.",
    })


def build_synthetic_bundle(output, count=40):
    if count < 1:
        raise ValueError("count must be positive")
    key, run = secrets.token_bytes(32), uuid4().hex
    events, labels = [], []
    for i in range(count):
        suspicious = i % 10 >= 5
        transaction = identifier("txn", f"{run}:{i}")
        event = Authorization(
            event_id=identifier("evt", transaction), transaction_id=transaction,
            trace_id=identifier("trace", transaction), event_time=datetime(2026, 9, 11, tzinfo=timezone.utc) + timedelta(seconds=i * 2),
            source="synthetic", customer_token=token(key, "cus", i // 10), card_token=token(key, "card", i // 10),
            merchant_id=token(key, "merchant", i % 4), amount="9000.00" if i % 10 == 9 else "49.90",
            currency="EUR", country="US" if suspicious else "FR", channel="ecommerce",
            device_id=token(key, "device", i if suspicious else i // 10), ip_hash=token(key, "ip", i % 3),
            merchant_category="retail", three_ds={"authenticated": not suspicious},
        )
        events.append(event)
        labels.append(label_for(event, suspicious, 7 if suspicious else 1))
    return write_bundle(output, events, labels, [], {
        "simulation": True, "source": "synthetic", "run_id": run, "count": count,
        "model_sha256": None, "amount_reference_counts": [1] * 6,
        "warning": "Scenario labels are fabricated; not evidence of ML accuracy. No compatible IEEE features: rules-only safe review.",
    })


def timeline(bundle):
    def read(name, contract):
        previous = None
        with (Path(bundle) / name).open(encoding="utf-8") as stream:
            for line in stream:
                event = contract.model_validate_json(line)
                if previous is not None and event.event_time < previous:
                    raise ValueError("Replay files must be chronological")
                previous = event.event_time
                yield event
    return heapq.merge(read("events.jsonl", Authorization), read("labels.jsonl", FraudLabel), key=lambda e: e.event_time)


def replay(bundle, publish, *, speed=86400, wait=True, sleep=time.sleep):
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be positive and finite")
    previous = None
    count = 0
    for event in timeline(bundle):
        if wait and previous is not None:
            sleep(max(0, (event.event_time - previous).total_seconds()) / speed)
        topic = "transactions-incoming" if isinstance(event, Authorization) else "fraud-labels"
        publish(topic, event.model_dump(mode="json"))
        previous = event.event_time
        count += 1
    return count
