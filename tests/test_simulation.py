import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import httpx
import pytest
import redis
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fraud_detection.simulation.api import create_app
from fraud_detection.simulation.decision import BenchmarkModel, DecisionEngine
from fraud_detection.simulation.explanations import explain
from fraud_detection.simulation.replay import build_synthetic_bundle, label_for, replay, timeline
from fraud_detection.simulation.schemas import Authorization, Context, FraudLabel, Thresholds, identifier
from fraud_detection.simulation.state import Conflict, MemoryState, snapshot
from fraud_detection.simulation.transport import PermanentMessageError, handle_authorization, payment_events, retry


class FixedModel:
    version = "test-model"

    def __init__(self, value=.1):
        self.value = value

    def score(self, event):
        if self.value is None:
            raise RuntimeError("Model down")
        return self.value


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    build_synthetic_bundle(root, 10)
    return root


@pytest.fixture
def event(bundle):
    return next(e for e in timeline(bundle) if isinstance(e, Authorization))


def test_strict_separation_and_delayed_replay(bundle):
    events = list(timeline(bundle))
    assert len(events) == 20
    assert [e.event_time for e in events] == sorted(e.event_time for e in events)
    authorizations = {e.transaction_id: e for e in events if isinstance(e, Authorization)}
    for e in events:
        if isinstance(e, Authorization):
            assert "label" not in e.model_dump() and "isFraud" not in e.model_dump()
            assert "DeviceInfo" not in e.model_dump()
            assert e.simulation is True and e.card_token.startswith("card_")
        else:
            assert e.label_time > authorizations[e.transaction_id].event_time
    sent, waits = [], []
    count = replay(bundle, lambda topic, value: sent.append((topic, value)), speed=86400, sleep=waits.append)
    assert count == 20 and len(waits) == 19 and max(waits) > 5
    assert sum(topic == "fraud-labels" for topic, value in sent) == 10
    assert all(value["event_type"] == "authorization_requested" for topic, value in sent if topic == "transactions-incoming")
    with sqlite3.connect(bundle / "features.sqlite") as db:
        assert db.execute("SELECT count(*) FROM features").fetchone()[0] == 0
    with pytest.raises(FileExistsError):
        build_synthetic_bundle(bundle)


@pytest.mark.parametrize("extra", [{"isFraud": 1}, {"label": 1}, {"card_number": "forbidden"}, {"cvv": "forbidden"}, {"schema_version": 2}, {"simulation": False}, {"amount": "NaN"}, {"amount": "0"}, {"event_time": "2026-09-11T12:00:00"}])
def test_contract_rejects_sensitive_unknown_or_invalid_fields(event, extra):
    with pytest.raises(ValidationError):
        Authorization.model_validate({**event.model_dump(mode="json"), **extra})


def test_labels_cannot_arrive_at_authorization_time(event):
    label = label_for(event, 1, 7).model_dump(mode="json")
    label.update(label_time=event.event_time.isoformat(), event_time=event.event_time.isoformat())
    with pytest.raises(ValidationError):
        FraudLabel.model_validate(label)


@pytest.mark.parametrize("value,expected", [(.1, "approve"), (.55, "manual_review"), (.98, "decline"), (None, "manual_review"), (float("nan"), "manual_review")])
def test_thresholds_and_fail_safe(event, value, expected):
    decision = DecisionEngine(FixedModel(value)).decide(event, Context())
    assert decision.decision == expected
    assert decision.request_event_id == event.event_id
    assert decision.trace_id == event.trace_id
    assert decision.feature_timestamp == event.event_time
    assert decision.decided_at >= decision.received_at
    if value is None or value != value:
        assert decision.risk_score is None and "model_unavailable" in decision.reasons


def test_strict_rules_override_model_and_outage(event):
    for value in (.01, None):
        decision = DecisionEngine(FixedModel(value), blocked_cards=[event.card_token]).decide(event, Context())
        assert decision.decision == "decline" and "blocked_card" in decision.reasons
        high = DecisionEngine(FixedModel(value)).decide(event, Context(transactions_5m=4))
        assert high.decision == "manual_review" and "high_velocity" in high.reasons
    amount = DecisionEngine(FixedModel(.01)).decide(event.model_copy(update={"amount": 5000}), Context())
    assert amount.decision == "manual_review" and "unusual_amount" in amount.reasons
    with pytest.raises(ValidationError):
        Thresholds(review=.95, block=.9)


def test_anomaly_failure_preserves_supervised_score(event):
    class AnomalyFailure(FixedModel):
        def anomaly_score(self, event):
            raise RuntimeError("anomaly detector unavailable")

    decision = DecisionEngine(AnomalyFailure(.7)).decide(event, Context())
    assert decision.risk_score == .7
    assert decision.anomaly_score is None
    assert decision.decision == "manual_review"
    assert "anomaly_unavailable" in decision.reasons
    assert "model_unavailable" not in decision.reasons


def test_previous_only_context_with_equal_times_and_windows(event):
    now = event.event_time.timestamp()
    row = {"time": now - 1, "cents": 2000, "country": "FR", "device": "device_abc", "card": "card_abc", "decision": "decline"}
    history = [dict(row, time=now - 3700), row, dict(row, time=now), dict(row, time=now + 1)]
    result = snapshot(event, history, history)
    assert result.transactions_5m == 1 and result.amount_1h == 20
    assert result.failures_5m == 1 and result.last_time == now - 1
    assert result.cards_on_device_24h == 1


def test_duplicate_concurrency_conflict_and_ordering(event):
    store, engine = MemoryState(), DecisionEngine(FixedModel())
    def process(_):
        return store.process(event, lambda context: engine.decide(event, context))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(process, range(16)))
    assert sum(not duplicate for decision, duplicate in results) == 1
    assert all(decision == results[0][0] for decision, duplicate in results)
    changed = event.model_copy(update={"country": "US"})
    with pytest.raises(Conflict, match="different payload"):
        store.process(changed, lambda context: engine.decide(changed, context))
    other = event.model_copy(update={"event_id": identifier("evt", "other")})
    with pytest.raises(Conflict, match="already"):
        store.process(other, lambda context: engine.decide(other, context))
    past = event.model_copy(update={"event_id": identifier("evt", "past"), "transaction_id": identifier("txn", "past"), "event_time": event.event_time - timedelta(seconds=1)})
    with pytest.raises(Conflict, match="Out-of-order"):
        store.process(past, lambda context: engine.decide(past, context))


def test_api_validation_idempotence_and_explanation(event):
    with TestClient(create_app(MemoryState(), DecisionEngine(FixedModel()))) as client:
        assert client.get("/health/ready").status_code == 200
        first = client.post("/score", json=event.model_dump(mode="json"))
        assert first.status_code == 200
        assert client.post("/score", json=event.model_dump(mode="json")).json() == first.json()
        assert client.post("/score", json={**event.model_dump(mode="json"), "country": "US"}).status_code == 409
        assert client.post("/score", json={**event.model_dump(mode="json"), "isFraud": 1}).status_code == 422
        text = client.get(f"/decisions/{event.event_id}/explanation").json()
        assert text["analyst_only"] and text["generated_by"] == "template"
        assert text["facts"]["decision"] == first.json()["decision"]
        assert "fraud_api_seconds" in client.get("/metrics").text


def test_redis_failure_returns_retryable_error(event):
    class DownState(MemoryState):
        def process(self, event, decide):
            raise redis.ConnectionError("unavailable")
    with TestClient(create_app(DownState(), DecisionEngine(FixedModel()))) as client:
        response = client.post("/score", json=event.model_dump(mode="json"))
        assert response.status_code == 503
        assert "decision" not in response.json()


def test_backoff_and_permanent_rejection():
    calls, waits = [], []
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("transient")
        return "ok"
    assert retry(flaky, sleep=waits.append) == "ok"
    assert waits == [.1, .2]
    with pytest.raises(PermanentMessageError):
        retry(lambda: (_ for _ in ()).throw(PermanentMessageError()), sleep=waits.append)
    assert waits == [.1, .2]


def test_publication_failure_reuses_exact_decision(event):
    class Publisher:
        sent = []
        fail = True
        def send(self, topic, payload):
            self.sent.append((topic, payload))
            if self.fail:
                self.fail = False
                raise RuntimeError("crash after broker accepted decision")
    publisher = Publisher()
    with TestClient(create_app(MemoryState(), DecisionEngine(FixedModel()))) as client:
        retry(lambda: handle_authorization(event.model_dump_json(), client, publisher, "http://testserver"), sleep=lambda _: None)
    decisions = [value for topic, value in publisher.sent if topic == "fraud-decisions"]
    assert len(decisions) == 2 and decisions[0] == decisions[1]
    types = {value["event_type"] for topic, value in publisher.sent}
    assert {"fraud_scored", "authorization_approved", "payment_captured"} <= types


def test_lifecycle_and_llm_do_not_change_decision(event):
    decision = DecisionEngine(FixedModel(.7)).decide(event, Context())
    before = decision.model_dump_json()
    assert [e.event_type for e in payment_events(decision)] == ["manual_review_requested"]
    explanation = explain(decision)
    assert explanation["retrieved_context"]
    with pytest.raises(ValueError, match="local"):
        explain(decision, llm_url="https://example.com")
    assert decision.model_dump_json() == before


def test_unavailable_benchmark_model(tmp_path, event):
    model = BenchmarkModel(tmp_path / "missing.joblib", tmp_path / "missing.sqlite")
    assert model.version == "unavailable"
    assert DecisionEngine(model).decide(event, Context()).decision == "manual_review"


def test_investigation_feedback_and_api_key(monkeypatch, event):
    monkeypatch.setenv("FRAUD_API_KEY", "test-key")
    with TestClient(create_app(MemoryState(), DecisionEngine(FixedModel(.7))),
                    raise_server_exceptions=False) as client:
        payload = event.model_dump(mode="json")
        assert client.post("/score", json=payload).status_code == 401
        response = client.post("/score", json=payload, headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        case_id = response.json()["request_event_id"]
        assert client.get("/investigations").status_code == 401
        cases = client.get("/investigations", headers={"X-API-Key": "test-key"}).json()
        assert len(cases) == 1 and cases[0]["status"] == "open"
        assert client.get(f"/decisions/{case_id}/explanation").status_code == 401
        assert client.get(f"/decisions/{case_id}/explanation",
                  headers={"X-API-Key": "test-key"}).status_code == 200
        feedback = client.post(f"/investigations/{case_id}/feedback",
                               headers={"X-API-Key": "test-key"},
                               json={"verdict": "legitimate", "analyst": "alice", "note": "verified"})
        assert feedback.status_code == 200
        assert feedback.json()["status"] == "confirmed"


def test_investigation_filters_and_delayed_label_outbox_retry(event):
    class Publisher:
        def __init__(self):
            self.events = []
            self.fail_once = True

        def send(self, topic, payload):
            self.events.append((topic, json.loads(json.dumps(payload))))
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("broker acknowledgement lost")

    publisher = Publisher()
    with TestClient(create_app(MemoryState(), DecisionEngine(FixedModel(.7)), publisher),
                    raise_server_exceptions=False) as client:
        response = client.post("/score", json=event.model_dump(mode="json"))
        case_id = response.json()["request_event_id"]
        filtered = client.get("/investigations", params={"decision": "manual_review", "min_score": 0.6,
                                  "status": "open", "search": "test-model"})
        assert filtered.status_code == 200 and [case["case_id"] for case in filtered.json()] == [case_id]
        assert client.get("/investigations", params={"min_score": 0.8}).json() == []

        feedback = {"verdict": "confirmed_fraud", "analyst": "alice", "note": "confirmed"}
        endpoint = f"/investigations/{case_id}/feedback"
        assert client.post(endpoint, json=feedback).status_code == 503
        retried = client.post(endpoint, json=feedback)
        assert retried.status_code == 200
        assert retried.json()["label_event_published"] is True
        assert publisher.events[0] == publisher.events[1]
        assert publisher.events[0][0] == "fraud-labels"
        label = FraudLabel.model_validate(publisher.events[1][1])
        assert label.label == 1 and label.label_time > label.authorization_time


def test_retry_repairs_case_after_case_creation_failure(monkeypatch, event):
    import fraud_detection.simulation.api as api_module

    create_case = api_module.create_case
    calls = []

    def fail_once(store, decision):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("simulated failure after decision persistence")
        return create_case(store, decision)

    monkeypatch.setattr(api_module, "create_case", fail_once)
    with TestClient(create_app(MemoryState(), DecisionEngine(FixedModel(.7))),
                    raise_server_exceptions=False) as client:
        payload = event.model_dump(mode="json")
        assert client.post("/score", json=payload).status_code == 500
        retry = client.post("/score", json=payload)
        assert retry.status_code == 200
        assert len(client.get("/investigations").json()) == 1
