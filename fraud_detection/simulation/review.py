"""Durable-enough review case helpers for the local simulation store."""
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class ReviewFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: str = Field(pattern="^(confirmed_fraud|legitimate|needs_more_evidence)$")
    analyst: str = Field(min_length=1, max_length=80)
    note: str = Field(default="", max_length=2000)


def _now():
    return datetime.now(timezone.utc).isoformat()


def create_case(store, decision):
    if decision.decision == "approve":
        return None
    key = "review:" + decision.request_event_id
    if hasattr(store, "values"):
        existing = store.values.get(key)
        if existing:
            return existing
        case = {"case_id": decision.request_event_id, "status": "open",
                "created_at": _now(), "decision": decision.model_dump(mode="json"), "feedback": []}
        store.values[key] = case
        return case
    client_key = store.prefix + key
    existing = store.client.get(client_key)
    if existing:
        import json
        return json.loads(existing)
    import json
    case = {"case_id": decision.request_event_id, "status": "open",
            "created_at": _now(), "decision": decision.model_dump(mode="json"), "feedback": []}
    store.client.set(client_key, json.dumps(case))
    return case


def get_case(store, case_id):
    key = "review:" + case_id
    if hasattr(store, "values"):
        return store.values.get(key)
    import json
    raw = store.client.get(store.prefix + key)
    return json.loads(raw) if raw else None


def save_case(store, case):
    key = "review:" + case["case_id"]
    if hasattr(store, "values"):
        store.values[key] = case
    else:
        import json
        store.client.set(store.prefix + key, json.dumps(case))


def list_cases(store, *, decision=None, min_score=None, max_score=None, status=None,
               search=None, from_time=None, to_time=None, offset=0, limit=100):
    if hasattr(store, "values"):
        cases = [value for key, value in store.values.items() if key.startswith("review:")]
    else:
        import json
        keys = list(store.client.scan_iter(store.prefix + "review:*"))
        cases = [json.loads(raw) for raw in store.client.mget(keys) if raw]
    query = (search or "").casefold()
    filtered = []
    for case in cases:
        decision_data = case["decision"]
        score = decision_data.get("risk_score")
        created = case.get("created_at", "")
        if decision and decision_data.get("decision") != decision:
            continue
        if status and case.get("status") != status:
            continue
        if min_score is not None and (score is None or score < min_score):
            continue
        if max_score is not None and (score is None or score > max_score):
            continue
        if from_time and created < from_time:
            continue
        if to_time and created > to_time:
            continue
        searchable = " ".join([case.get("case_id", ""), decision_data.get("model_version", ""),
                               *decision_data.get("reasons", [])]).casefold()
        if query and query not in searchable:
            continue
        filtered.append(case)
    filtered.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return filtered[offset:offset + limit]


def add_feedback(store, case_id, feedback):
    case = get_case(store, case_id)
    if case is None:
        return None
    item = {**feedback.model_dump(), "submitted_at": _now()}
    case["feedback"].append(item)
    case["status"] = "confirmed" if feedback.verdict in {"confirmed_fraud", "legitimate"} else "open"
    key = "review:" + case_id
    save_case(store, case)
    return case