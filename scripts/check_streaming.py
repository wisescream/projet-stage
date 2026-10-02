"""Destructive only to this demo's service processes: run on the isolated local Compose stack.

Requires real Kafka, Redis, ClickHouse, Prometheus and Grafana. Starts/stops native
Python services, briefly restarts the demo Redis container, and publishes replay data.
No mocks or synthetic substitutes for infrastructure. Does not delete volumes.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
import redis
from confluent_kafka import Consumer, TopicPartition

from fraud_detection.simulation.analytics import History
from fraud_detection.simulation.replay import timeline
from fraud_detection.simulation.schemas import Authorization
from fraud_detection.simulation.transport import KafkaPublisher, ensure_topics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts-stream-check"
OUTPUT.mkdir(exist_ok=True)


def eventually(check, timeout=90):
    """Bounded synchronization assertions for asynchronous integration tests."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                return result
        except (httpx.HTTPError, redis.RedisError) as error:
            last = type(error).__name__
        time.sleep(.25)
    raise AssertionError(f"Integration condition did not complete within {timeout}s ({last})")


def main():
    processes, logs = [], []
    run = uuid4().hex
    environment = {**os.environ, "MODEL_PATH": str(ROOT / "artifacts/model-validation.joblib"),
                   "FEATURE_DB": str(ROOT / "artifacts-replay/features.sqlite"),
                   "STATE_NAMESPACE": "stream-check-" + run, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}

    def start(name, command, env=environment):
        log = (OUTPUT / (name + ".log")).open("w", encoding="utf-8")
        logs.append(log)
        process = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        return process

    def start_api(name="api", port=8000, env=environment):
        process = start(name, ["-m", "uvicorn", "fraud_detection.simulation.api:app", "--host", "127.0.0.1", "--port", str(port)], env)
        def ready():
            assert process.poll() is None, f"{name} exited; inspect {OUTPUT / (name + '.log')}"
            response = httpx.get(f"http://127.0.0.1:{port}/health/ready", timeout=2, trust_env=False)
            return response.status_code == 200 and not response.json()["rules_fallback"]
        eventually(ready, 30)
        return process

    def stop(process):
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)

    ensure_topics("127.0.0.1:19092")
    events = [e for e in timeline(ROOT / "artifacts-replay") if isinstance(e, Authorization)]
    labels = [e for e in timeline(ROOT / "artifacts-replay") if not isinstance(e, Authorization)]
    history = History()
    publisher = KafkaPublisher("127.0.0.1:19092")
    inspectors = {group: Consumer({"bootstrap.servers": "127.0.0.1:19092", "group.id": group, "enable.auto.commit": False})
                  for group in ("fraud-scorer-v1", "fraud-history-v1")}

    def drained(group, topics):
        inspector = inspectors[group]
        for topic in topics:
            tp = TopicPartition(topic, 0)
            high = inspector.get_watermark_offsets(tp, timeout=10)[1]
            committed = inspector.committed([tp], timeout=10)[0].offset
            if committed != high and not (high == 0 and committed < 0):
                return False
        return True

    try:
        # Isolated test stack only: replay the complete retained input from offset zero.
        # A fresh state namespace must not resume half-way through an older run.
        inspectors["fraud-scorer-v1"].commit(offsets=[TopicPartition("transactions-incoming", 0, 0)], asynchronous=False)
        api = start_api()
        worker = start("worker", ["-m", "fraud_detection.simulation", "worker"])
        sink = start("sink", ["-m", "fraud_detection.simulation", "sink"])
        # Primary pass is uncached on this run's Redis namespace, with actual model scores.
        for event in events:
            publisher.send("transactions-incoming", event.model_dump(mode="json"))
        eventually(lambda: drained("fraud-scorer-v1", ["transactions-incoming"]), 180)
        eventually(lambda: drained("fraud-history-v1", ["fraud-decisions", "payment-events", "fraud-labels"]), 180)
        ids = {e.transaction_id for e in events}
        rows = history.client.query("SELECT transaction_id, payload FROM banking_events FINAL WHERE event_type='fraud_scored'").result_rows
        decisions = {transaction: json.loads(payload) for transaction, payload in rows if transaction in ids}
        assert set(decisions) == ids
        assert all(d["risk_score"] is not None for d in decisions.values())
        assert len({d["request_event_id"] for d in decisions.values()}) == len(events)
        # Labels are released only after this authorization pass, at their later simulation timestamps.
        for label in labels:
            publisher.send("fraud-labels", label.model_dump(mode="json"))
        eventually(lambda: drained("fraud-history-v1", ["fraud-decisions", "payment-events", "fraud-labels"]), 180)
        report = history.report()
        assert report["feedback"]["ieee_replay"]["joined_labels"] >= len(events)
        assert report["unmatched_labels"] == 0
        with httpx.Client(timeout=10, trust_env=False) as http:
            first = http.post("http://localhost:8000/score", json=events[0].model_dump(mode="json"))
            assert first.status_code == 200 and first.json() == decisions[events[0].transaction_id]
            explanation = http.get(f"http://localhost:8000/decisions/{events[0].event_id}/explanation")
            assert explanation.status_code == 200 and explanation.json()["analyst_only"]
            # A state outage must not create an unpersisted or duplicate decision.
            subprocess.run(["docker", "compose", "-f", "compose.host.yaml", "stop", "redis"], check=True, capture_output=True)
            try:
                assert http.post("http://localhost:8000/score", json=events[0].model_dump(mode="json")).status_code == 503
            finally:
                subprocess.run(["docker", "compose", "-f", "compose.host.yaml", "up", "-d", "--wait", "redis"], check=True, capture_output=True)
            stop(api)
            api = start_api("api-restarted")
            assert http.post("http://localhost:8000/score", json=events[0].model_dump(mode="json")).json() == first.json()
        # Stop the scorer, append duplicates, restart: pending offsets are resumed.
        stop(worker)
        for event in events:
            publisher.send("transactions-incoming", event.model_dump(mode="json"))
        worker = start("worker-restarted", ["-m", "fraud_detection.simulation", "worker"])
        eventually(lambda: drained("fraud-scorer-v1", ["transactions-incoming"]), 180)
        eventually(lambda: drained("fraud-history-v1", ["fraud-decisions", "payment-events", "fraud-labels"]), 180)
        repeated = history.client.query("SELECT transaction_id, payload FROM banking_events FINAL WHERE event_type='fraud_scored'").result_rows
        assert {t: json.loads(p) for t, p in repeated if t in ids} == decisions
        # Invalid messages produce a DLQ pointer without copying their payload.
        dlq = Consumer({"bootstrap.servers": "127.0.0.1:19092", "group.id": "check-dlq-" + run, "enable.auto.commit": False})
        end = dlq.get_watermark_offsets(TopicPartition("fraud-dead-letter", 0), timeout=10)[1]
        dlq.assign([TopicPartition("fraud-dead-letter", 0, end)])
        publisher.send("transactions-incoming", {"unexpected": "DO_NOT_COPY_INVALID_PAYLOAD"})
        message = eventually(lambda: dlq.poll(.2), 30)
        assert not message.error()
        pointer = json.loads(message.value())
        assert pointer["reason"] == "invalid_contract" and "DO_NOT_COPY" not in message.value().decode()
        dlq.close()
        eventually(lambda: drained("fraud-scorer-v1", ["transactions-incoming"]))
        # Independent real-HTTP, uncached sequential benchmark using another Redis namespace.
        benchmark = start_api("benchmark", 8001, {**environment, "STATE_NAMESPACE": "latency-check-" + run})
        latencies = []
        with httpx.Client(timeout=10, trust_env=False) as http:
            for event in events[:100]:
                start_time = time.perf_counter()
                response = http.post("http://localhost:8001/score", json=event.model_dump(mode="json"))
                latencies.append((time.perf_counter() - start_time) * 1000)
                assert response.status_code == 200 and response.json()["risk_score"] is not None
        stop(benchmark)
        quantiles = dict(zip(["p50_ms", "p95_ms", "p99_ms"], np.percentile(latencies, [50, 95, 99]).tolist()))
        quantiles.update(count=len(latencies), requests_per_second=1000 * len(latencies) / sum(latencies),
                         errors=0, mode="sequential uncached HTTP including Redis and model, cold first request included")
        def all_targets_up():
            targets = httpx.get("http://127.0.0.1:9090/api/v1/targets", timeout=5, trust_env=False).json()["data"]["activeTargets"]
            return len(targets) == 3 and all(t["health"] == "up" for t in targets)
        eventually(all_targets_up, 45)
        dashboard = httpx.get("http://127.0.0.1:3000/api/dashboards/uid/banking-simulation", timeout=5, trust_env=False)
        dashboard.raise_for_status()
        for panel in dashboard.json()["dashboard"]["panels"]:
            for target in panel["targets"]:
                result = httpx.get("http://127.0.0.1:9090/api/v1/query", params={"query": target["expr"]}, timeout=5, trust_env=False).json()
                assert result["status"] == "success", (panel["title"], result)
        result = {"simulation": True, "checked_transactions": len(events), "coverage": "all transactions in this replay bundle, not the full IEEE holdout",
                  "checks": ["real Kafka/Redis/FastAPI/ClickHouse flow", "deferred labels joined", "Redis outage returns 503", "Redis + API restart retains exact decision",
                             "consumer restart resumes duplicate backlog", "one logical decision per event", "invalid message DLQ without raw payload",
                             "3 Prometheus targets up", "Grafana dashboard loaded and all PromQL queries accepted"],
                  "http_latency": quantiles, "prototype_targets": {"p95_under_100ms": quantiles["p95_ms"] < 100,
                  "p99_under_250ms": quantiles["p99_ms"] < 250, "guarantee": False}, "feedback": history.report()}
        (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        for process in reversed(processes):
            stop(process)
        for log in logs:
            log.close()
        for inspector in inspectors.values():
            inspector.close()


if __name__ == "__main__":
    main()
