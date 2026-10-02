"""Atomic decision/context persistence; single chronological prototype stream."""
import hashlib
import json
import threading
import time

from redis.exceptions import WatchError

from fraud_detection.simulation import monitoring as metrics
from fraud_detection.simulation.schemas import Context, Decision


class Conflict(ValueError):
    pass


def fingerprint(event):
    return hashlib.sha256(json.dumps(event.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


def cached_result(raw, event):
    cached = json.loads(raw)
    if cached["fingerprint"] != fingerprint(event):
        raise Conflict("event_id reused with a different payload")
    return Decision.model_validate(cached["decision"]), True


def snapshot(event, customer, device):
    now = event.event_time.timestamp()
    # Equal timestamps are excluded too: statistics use strictly preceding events.
    recent = [r for r in customer if now - 3600 <= r["time"] < now]
    five = [r for r in recent if r["time"] >= now - 300]
    devices = [r for r in device if now - 86400 <= r["time"] < now]
    last = recent[-1] if recent else None
    return Context(transactions_5m=len(five), amount_1h=sum(r["cents"] for r in recent) / 100,
                   failures_5m=sum(r["decision"] == "decline" for r in five),
                   cards_on_device_24h=len({r["card"] for r in devices}),
                   last_country=last["country"] if last else None, last_device=last["device"] if last else None,
                   last_time=last["time"] if last else None,
                   device_first_seen=devices[0]["time"] if devices else None)


def advance(event, decision, customer, device):
    now = event.event_time.timestamp()
    record = {"time": now, "cents": int(event.amount * 100), "country": event.country,
              "device": event.device_id, "card": event.card_token, "decision": decision.decision}
    return ([r for r in customer if r["time"] >= now - 3600] + [record],
            [r for r in device if r["time"] >= now - 86400] + [record])


class MemoryState:
    """Test/local benchmark only. No restart durability; deployment uses RedisState."""
    def __init__(self):
        self.values = {}
        self.lock = threading.Lock()

    def ping(self):
        return True

    def process(self, event, decide):
        with self.lock:
            key = "event:" + event.event_id
            if key in self.values:
                return cached_result(self.values[key], event)
            if event.transaction_id in self.values:
                raise Conflict("transaction_id already has an authorization")
            if event.event_time.timestamp() < self.values.get("watermark", -1):
                raise Conflict("Out-of-order event: use a new isolated replay namespace")
            customer = self.values.get(event.customer_token, [])
            device = self.values.get(event.device_id, [])
            decision = decide(snapshot(event, customer, device))
            customer, device = advance(event, decision, customer, device)
            self.values.update({key: json.dumps({"fingerprint": fingerprint(event), "event": event.model_dump(mode="json"),
                                                 "decision": decision.model_dump(mode="json")}),
                                event.transaction_id: event.event_id, event.customer_token: customer,
                                event.device_id: device, "watermark": event.event_time.timestamp()})
            return decision, False


class RedisState:
    def __init__(self, client, namespace="bank-demo"):
        self.client, self.prefix = client, namespace + ":"

    def ping(self):
        return self.client.ping()

    def process(self, event, decide):
        keys = [self.prefix + k for k in ["event:" + event.event_id, "transaction:" + event.transaction_id,
                "customer:" + event.customer_token, "device:" + event.device_id, "watermark"]]
        for attempt in range(8):
            with self.client.pipeline() as pipe:
                try:
                    with metrics.REDIS.time():
                        pipe.watch(*keys)
                        cached, transaction, customer, device, watermark = pipe.mget(keys)
                    if cached:
                        return cached_result(cached, event)
                    if transaction:
                        raise Conflict("transaction_id already has an authorization")
                    if watermark and event.event_time.timestamp() < float(watermark):
                        raise Conflict("Out-of-order event: use a new isolated replay namespace")
                    customer, device = json.loads(customer or "[]"), json.loads(device or "[]")
                    decision = decide(snapshot(event, customer, device))
                    customer, device = advance(event, decision, customer, device)
                    pipe.multi()
                    # Cache/transaction guard/watermark have no TTL: retry dedup survives restarts.
                    pipe.set(keys[0], json.dumps({"fingerprint": fingerprint(event),
                                                 "event": event.model_dump(mode="json"),
                                                 "decision": decision.model_dump(mode="json")}))
                    pipe.set(keys[1], event.event_id)
                    pipe.set(keys[2], json.dumps(customer), ex=86400)
                    pipe.set(keys[3], json.dumps(device), ex=86400)
                    pipe.set(keys[4], event.event_time.timestamp())
                    with metrics.REDIS.time():
                        pipe.execute()
                    return decision, False
                except WatchError:
                    time.sleep(0.005 * 2 ** attempt)
        raise RuntimeError("Concurrent state updates exhausted retries; request can be retried")
