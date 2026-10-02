"""Hybrid decisions. Recent context drives rules, not untrained model features."""
import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from threadpoolctl import threadpool_limits

from fraud_detection.pipeline import CATEGORICAL
from fraud_detection.simulation.schemas import Decision, Thresholds, identifier, utcnow

LOG = logging.getLogger(__name__)


class BenchmarkModel:
    """Trusted local IEEE holdout model + read-only feature references, never label lookup."""
    def __init__(self, model_path, feature_db):
        self.model, self.version = None, "unavailable"
        self.feature_db = Path(feature_db).resolve()
        self.lock = threading.Lock()
        self.anomaly_model = None
        self.anomaly_min, self.anomaly_max = 0.0, 1.0
        try:
            digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
            with self.connect() as db:
                metadata = json.loads(db.execute("SELECT payload FROM metadata").fetchone()[0])
            if metadata["model_sha256"] != digest:
                raise ValueError("Model must match the bundle's unseen-holdout model")
            self.model = joblib.load(model_path)  # Trusted local files only.
            self.version = "ieee-hgb-" + digest[:12]
            self._fit_anomaly_model()
        except (OSError, ValueError, sqlite3.Error, KeyError):
            LOG.warning("Benchmark model unavailable; safe rules-only fallback enabled")

    def connect(self):
        return sqlite3.connect(self.feature_db.as_uri() + "?mode=ro", uri=True)

    def _frame(self, record):
        values = {column: np.asarray([np.nan if value is None else value],
                                     dtype=object if column in CATEGORICAL else np.float32)
                  for column, value in record["features"].items()}
        return pd.DataFrame(values, columns=self.model.feature_names_in_)

    def _fit_anomaly_model(self):
        """Fit an unlabeled detector on the private replay feature reference set."""
        with self.connect() as db:
            records = [json.loads(row[0]) for row in db.execute("SELECT payload FROM features")]
        if len(records) < 10:
            return
        frames = pd.concat([self._frame(record) for record in records], ignore_index=True)
        matrix = self.model.named_steps["preprocess"].transform(frames)
        self.anomaly_model = IsolationForest(n_estimators=100, random_state=42, contamination="auto")
        self.anomaly_model.fit(matrix)
        values = self.anomaly_model.score_samples(matrix)
        self.anomaly_min, self.anomaly_max = float(values.min()), float(values.max())

    def score(self, event):
        if self.model is None or event.feature_ref is None:
            raise ValueError("No compatible benchmark model/features")
        with self.connect() as db:
            row = db.execute("SELECT payload FROM features WHERE reference=?", (event.feature_ref,)).fetchone()
        if not row:
            raise ValueError("Unknown benchmark feature reference")
        record = json.loads(row[0])
        if record["authorization"] != event.model_dump(mode="json"):
            raise ValueError("Authorization differs from its bound benchmark feature record")
        if set(record["features"]) != set(self.model.feature_names_in_):
            raise ValueError("Benchmark feature schema differs from the trained model")
        frame = self._frame(record)
        with self.lock, threadpool_limits(limits=1):
            return float(self.model.predict_proba(frame)[0, 1])

    def explain_features(self, event, *, limit=8):
        if self.model is None or event.feature_ref is None:
            return {"method": "unavailable", "effects": []}
        with self.connect() as db:
            row = db.execute("SELECT payload FROM features WHERE reference=?", (event.feature_ref,)).fetchone()
        if not row:
            return {"method": "unavailable", "effects": []}
        record = json.loads(row[0])
        if record["authorization"] != event.model_dump(mode="json"):
            return {"method": "unavailable", "effects": []}

        preprocessing = self.model.named_steps["preprocess"]
        classifier = self.model.named_steps["classifier"]
        original_names = list(self.model.feature_names_in_)
        numeric_names = [name for name in original_names if name not in CATEGORICAL]
        categorical_names = [name for name in original_names if name in CATEGORICAL]
        transformed_names = numeric_names + categorical_names
        gains = np.zeros(len(transformed_names), dtype=float)
        try:
            for iteration in classifier._predictors:
                for predictor in iteration:
                    nodes = predictor.nodes
                    for node in nodes:
                        feature_index = int(node["feature_idx"])
                        gain = float(node["gain"])
                        if feature_index >= 0 and gain > 0 and not bool(node["is_leaf"]):
                            gains[feature_index] += gain
        except (AttributeError, KeyError, TypeError, ValueError):
            return {"method": "unavailable", "effects": []}

        raw_features = record["features"]
        frame = self._frame(record)
        ranked = np.argsort(-gains, kind="stable")
        selected = []
        for index in ranked:
            name = transformed_names[index]
            if gains[index] > 0 and name in raw_features and name not in selected:
                selected.append(name)
            if len(selected) >= limit:
                break
        if not selected:
            return {"method": "unavailable", "effects": []}

        with self.lock, threadpool_limits(limits=1):
            current_score = float(self.model.predict_proba(frame)[0, 1])
            effects = []
            for name in selected:
                changed = frame.copy()
                changed.loc[:, name] = np.nan
                missing_score = float(self.model.predict_proba(changed)[0, 1])
                effects.append({"feature": name, "probability_delta": current_score - missing_score,
                               "score_without_feature": missing_score})
        effects.sort(key=lambda item: abs(item["probability_delta"]), reverse=True)
        return {"method": "single_feature_missing_value_perturbation", "effects": effects,
                "note": "Local model sensitivity only; not causal, additive, or used to change the decision."}

    def anomaly_score(self, event):
        if self.anomaly_model is None or event.feature_ref is None:
            return None
        with self.connect() as db:
            row = db.execute("SELECT payload FROM features WHERE reference=?", (event.feature_ref,)).fetchone()
        if not row:
            return None
        record = json.loads(row[0])
        if record["authorization"] != event.model_dump(mode="json"):
            return None
        matrix = self.model.named_steps["preprocess"].transform(self._frame(record))
        raw = float(self.anomaly_model.score_samples(matrix)[0])
        span = max(self.anomaly_max - self.anomaly_min, 1e-9)
        return float(np.clip((self.anomaly_max - raw) / span, 0, 1))


class DecisionEngine:
    def __init__(self, model, thresholds=None, blocked_cards=()):
        self.model = model
        self.thresholds = thresholds or Thresholds()
        self.blocked_cards = set(blocked_cards)

    def decide(self, event, context):
        start, received = time.perf_counter(), utcnow()
        reasons, hard_block, review = [], False, False
        if event.card_token in self.blocked_cards:
            reasons.append("blocked_card")
            hard_block = True
        if event.amount >= 5000:
            reasons.append("unusual_amount")
            review = True
        if context.transactions_5m >= 4 or context.failures_5m >= 3:
            reasons.append("high_velocity")
            review = True
        if context.last_device and context.last_device != event.device_id:
            reasons.append("new_device")
            review = True
        if context.last_country and context.last_country != event.country and event.event_time.timestamp() - context.last_time <= 300:
            reasons.append("country_change")
            review = True
        if context.cards_on_device_24h >= 3:
            reasons.append("multiple_cards_on_device")
            review = True
        if not event.three_ds.authenticated and event.amount >= 1000:
            reasons.append("three_ds_missing")
            review = True
        if event.identity_data_missing:
            reasons.append("identity_data_missing")  # Informational; not alone a blocking rule.
        score, anomaly_score, version = None, None, getattr(self.model, "version", "unavailable")
        try:
            score = self.model.score(event)
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid model probability")
        except Exception:
            # A supervised scoring failure is fail-safe: require manual review.
            score, version, review = None, "unavailable", True
            reasons.append("model_unavailable")
        if score is not None and hasattr(self.model, "anomaly_score"):
            try:
                anomaly_score = self.model.anomaly_score(event)
                if anomaly_score is not None and not np.isfinite(anomaly_score):
                    raise ValueError("Invalid anomaly score")
            except Exception:
                anomaly_score = None
                reasons.append("anomaly_unavailable")
        if score is not None and score >= self.thresholds.block:
            reasons.append("model_block_threshold")
            hard_block = True
        elif score is not None and score >= self.thresholds.review:
            reasons.append("model_review_threshold")
            review = True
        if anomaly_score is not None and anomaly_score >= self.thresholds.anomaly_review:
            reasons.append("anomaly_review_threshold")
            review = True
        outcome = "decline" if hard_block else "manual_review" if review else "approve"
        if not reasons:
            reasons.append("below_risk_thresholds")
        return Decision(event_id=identifier("evt", event.event_id + ":decision"), request_event_id=event.event_id,
                        transaction_id=event.transaction_id, trace_id=event.trace_id, event_time=event.event_time,
                        decision=outcome, risk_score=score, anomaly_score=anomaly_score, thresholds=self.thresholds, reasons=reasons,
                        model_version=version, received_at=received, decided_at=utcnow(),
                        feature_timestamp=event.event_time, decision_time_ms=(time.perf_counter() - start) * 1000,
                        context=context, amount=float(event.amount), source=event.source)
