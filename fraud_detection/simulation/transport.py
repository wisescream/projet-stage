"""At-least-once Kafka delivery; stable event IDs provide idempotent effects."""
import hashlib
import json
import logging
import time
from datetime import timedelta

import httpx
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic
from pydantic import ValidationError

from fraud_detection.simulation import monitoring as metrics
from fraud_detection.simulation.schemas import Authorization, Decision, PaymentEvent, identifier, utcnow

TOPICS = ["transactions-incoming", "fraud-decisions", "payment-events", "fraud-labels", "fraud-dead-letter"]
LOG = logging.getLogger(__name__)


class PermanentMessageError(ValueError):
    pass


def retry(operation, attempts=5, sleep=time.sleep):
    for attempt in range(attempts):
        try:
            return operation()
        except (PermanentMessageError, ValidationError):
            raise
        except Exception:
            if attempt == attempts - 1:
                raise
            metrics.ERRORS.labels("retry").inc()
            sleep(.1 * 2 ** attempt)


class KafkaPublisher:
    def __init__(self, bootstrap):
        self.producer = Producer({"bootstrap.servers": bootstrap, "enable.idempotence": True,
                                  "acks": "all", "delivery.timeout.ms": 10000})

    def send(self, topic, payload):
        errors = []
        self.producer.produce(topic, key=b"chronological-demo", value=json.dumps(payload, allow_nan=False).encode(),
                              on_delivery=lambda error, message: errors.append(error) if error else None)
        if self.producer.flush(12) or errors:
            raise RuntimeError("Kafka publish not acknowledged")
        metrics.KAFKA_MESSAGES.labels(topic, "published").inc()


def ensure_topics(bootstrap):
    admin = AdminClient({"bootstrap.servers": bootstrap})
    futures = admin.create_topics(
        [NewTopic(topic, num_partitions=1, replication_factor=1, config={"retention.ms": "604800000"}) for topic in TOPICS])
    for future in futures.values():
        try:
            future.result(timeout=20)
        except KafkaException as error:
            if error.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise


def consumer(bootstrap, group, topics):
    client = Consumer({"bootstrap.servers": bootstrap, "group.id": group,
                       "enable.auto.commit": False, "auto.offset.reset": "earliest",
                       "max.poll.interval.ms": 300000})
    client.subscribe(topics)
    return client


def payment_events(decision):
    types = {"approve": ["authorization_approved", "payment_captured"],
             "decline": ["authorization_declined"], "manual_review": ["manual_review_requested"]}[decision.decision]
    if decision.decision == "approve" and int(hashlib.sha256(decision.transaction_id.encode()).hexdigest()[:4], 16) % 11 == 0:
        types = types + ["payment_refunded"]
    for i, kind in enumerate(types):
        yield PaymentEvent(event_id=identifier("evt", decision.event_id + ":" + kind),
                           transaction_id=decision.transaction_id, trace_id=decision.trace_id,
                           event_time=decision.event_time + timedelta(seconds=i),
                           event_type=kind, parent_event_id=decision.event_id)


def handle_authorization(raw, http, publisher, api_url):
    event = Authorization.model_validate_json(raw)
    response = http.post(api_url.rstrip("/") + "/score", json=event.model_dump(mode="json"))
    if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
        raise PermanentMessageError("Authorization rejected by schema/idempotence/ordering contract")
    response.raise_for_status()
    decision = Decision.model_validate(response.json())
    if decision.request_event_id != event.event_id or decision.transaction_id != event.transaction_id:
        raise PermanentMessageError("Scorer response does not match authorization")
    publisher.send("fraud-decisions", decision.model_dump(mode="json"))
    for payment in payment_events(decision):
        publisher.send("payment-events", payment.model_dump(mode="json"))


def _record_consumer_lag(client, group):
    partitions = client.assignment()
    if not partitions:
        return
    committed = client.committed(partitions, timeout=2)
    for partition, offset in zip(partitions, committed):
        try:
            _, high = client.get_watermark_offsets(partition, timeout=2)
            current = max(offset.offset, 0)
            metrics.KAFKA_LAG.labels(group, partition.topic, str(partition.partition)).set(max(high - current, 0))
        except KafkaException:
            metrics.ERRORS.labels("lag_probe").inc()


def consume_loop(client, publisher, handler, *, after_poll=None, group="unknown"):
    try:
        while True:
            message = client.poll(1)
            if after_poll:
                after_poll()
            if message is None:
                continue
            if message.error():
                raise KafkaException(message.error())
            try:
                with metrics.KAFKA.time():
                    retry(lambda: handler(message.topic(), message.value()))
            except (ValidationError, PermanentMessageError):
                # Do not copy invalid payloads (possibly raw PII) into the DLQ/logs.
                pointer = {"schema_version": 1, "simulation": True, "event_type": "dead_letter",
                           "event_id": identifier("evt", f"{message.topic()}:{message.partition()}:{message.offset()}"),
                           "source_topic": message.topic(), "source_partition": message.partition(),
                           "source_offset": message.offset(), "reason": "invalid_contract",
                           "payload_sha256": hashlib.sha256(message.value() or b"").hexdigest(),
                           "received_at": utcnow().isoformat()}
                retry(lambda: publisher.send("fraud-dead-letter", pointer))
                metrics.ERRORS.labels("dead_letter").inc()
            # Transient exhaustion exits before committing. Restart resumes the same offset.
            client.commit(message=message, asynchronous=False)
            metrics.KAFKA_MESSAGES.labels(message.topic(), "consumed").inc()
            _record_consumer_lag(client, group)
    finally:
        client.close()


def run_worker(bootstrap, api_url, group="fraud-scorer-v1"):
    publisher = KafkaPublisher(bootstrap)
    with httpx.Client(timeout=10, trust_env=False) as http:
        consume_loop(consumer(bootstrap, group, ["transactions-incoming"]), publisher,
                     lambda topic, raw: handle_authorization(raw, http, publisher, api_url), group=group)
