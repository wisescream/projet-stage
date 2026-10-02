"""Deduplicated ClickHouse history and delayed-label feedback (not training input)."""
import json
import time
from pathlib import Path

import clickhouse_connect
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detection.simulation import monitoring as metrics
from fraud_detection.simulation.replay import AMOUNT_BINS
from fraud_detection.simulation.schemas import Decision, FraudLabel, PaymentEvent, identifier
from fraud_detection.simulation.transport import KafkaPublisher, consume_loop, consumer

DDL = """CREATE TABLE IF NOT EXISTS banking_events (
 event_id String, transaction_id String, event_type LowCardinality(String),
 event_time DateTime64(3, 'UTC'), decision LowCardinality(String),
 risk_score Nullable(Float64), label Int8, amount Float64, source LowCardinality(String), payload String
) ENGINE = ReplacingMergeTree ORDER BY event_id"""


class History:
    def __init__(self, host="localhost", port=8123):
        self.client = clickhouse_connect.get_client(host=host, port=port)
        self.client.command(DDL)

    def insert(self, event):
        data = event.model_dump(mode="json")
        self.client.insert("banking_events", [[event.event_id, event.transaction_id, event.event_type,
                           event.event_time, data.get("decision", ""), data.get("risk_score"),
                           data.get("label", -1), data.get("amount", 0), data.get("source", ""),
                           json.dumps(data, allow_nan=False)]],
                           column_names=["event_id", "transaction_id", "event_type", "event_time", "decision",
                                         "risk_score", "label", "amount", "source", "payload"])

    def report(self):
        decisions = self.client.query("SELECT transaction_id, decision, risk_score, amount, source FROM banking_events FINAL WHERE event_type='fraud_scored'").result_rows
        labels = dict(self.client.query("SELECT transaction_id, argMax(label,event_time) FROM banking_events FINAL WHERE event_type='fraud_label_received' GROUP BY transaction_id").result_rows)
        totals = {name: sum(row[1] == name for row in decisions) for name in ("approve", "manual_review", "decline")}
        by_source = {}
        for source in ("ieee_replay", "synthetic"):
            joined = [row for row in decisions if row[0] in labels and row[4] == source]
            counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
            for transaction, decision, score, amount, _ in joined:
                flagged, label = decision != "approve", labels[transaction]
                counts[("t" if flagged == bool(label) else "f") + ("p" if flagged else "n")] += 1
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision = tp / (tp + fp) if tp + fp else 0
            recall = tp / (tp + fn) if tp + fn else 0
            stats = {"joined_labels": len(joined), "confusion": counts, "precision": precision, "recall": recall,
                     "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
                     "definition": "Positive operational prediction = review or decline; scenario labels are not bank ground truth"}
            scored = [row for row in joined if row[2] is not None]
            y = [labels[row[0]] for row in scored]
            if source == "ieee_replay" and len(set(y)) == 2:
                stats.update(roc_auc=float(roc_auc_score(y, [r[2] for r in scored])),
                             average_precision=float(average_precision_score(y, [r[2] for r in scored])))
            by_source[source] = stats
        return {"simulation": True, "decisions": totals, "unique_decisions": len(decisions),
                "labels": len(labels), "unmatched_labels": len(set(labels) - {r[0] for r in decisions}),
                "feedback": by_source}

    def refresh_metrics(self, reference):
        report = self.report()
        for decision, count in report["decisions"].items():
            metrics.HISTORY.labels(decision).set(count)
        for outcome in ("tp", "fp", "tn", "fn"):
            metrics.CONFIRMED.labels(outcome).set(report["feedback"]["ieee_replay"]["confusion"][outcome])
        metrics.UNMATCHED.set(report["unmatched_labels"])
        amounts = self.client.query("SELECT amount FROM banking_events FINAL WHERE event_type='fraud_scored' AND source='ieee_replay'").result_rows
        if amounts:
            observed = np.histogram([r[0] for r in amounts], bins=AMOUNT_BINS)[0] + .5
            expected = np.asarray(reference, dtype=float) + .5
            observed, expected = observed / observed.sum(), expected / expected.sum()
            metrics.DRIFT.set(float(np.sum((observed - expected) * np.log(observed / expected))))


def run_sink(bootstrap, host, port, bundle, group="fraud-history-v1"):
    history, publisher = History(host, port), KafkaPublisher(bootstrap)
    reference = json.loads((Path(bundle) / "manifest.json").read_text())["amount_reference_counts"]
    last_refresh = 0

    def refresh():
        nonlocal last_refresh
        if time.monotonic() - last_refresh > 5:
            history.refresh_metrics(reference)
            last_refresh = time.monotonic()

    def handle(topic, raw):
        contract = {"fraud-decisions": Decision, "fraud-labels": FraudLabel, "payment-events": PaymentEvent}[topic]
        event = contract.model_validate_json(raw)
        history.insert(event)
        if isinstance(event, FraudLabel):
            metrics.LABEL_DELAY.observe((event.label_time - event.authorization_time).total_seconds() / 86400)
            if event.label:
                chargeback = PaymentEvent(event_id=identifier("evt", event.event_id + ":chargeback"),
                                          transaction_id=event.transaction_id, trace_id=event.trace_id,
                                          event_time=event.event_time, event_type="chargeback_received", parent_event_id=event.event_id)
                publisher.send("payment-events", chargeback.model_dump(mode="json"))

    consume_loop(consumer(bootstrap, group, ["fraud-decisions", "fraud-labels", "payment-events"]),
                 publisher, handle, after_poll=refresh, group=group)
