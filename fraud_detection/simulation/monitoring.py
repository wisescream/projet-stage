"""Low-cardinality metrics only: never use customer/event tokens as labels."""
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()
REQUESTS = Counter("fraud_api_requests_total", "API responses", ["status"], registry=REGISTRY)
DECISIONS = Counter("fraud_decisions_total", "New persisted decisions", ["decision"], registry=REGISTRY)
DUPLICATES = Counter("fraud_duplicate_requests_total", "Cached authorizations", registry=REGISTRY)
LATENCY = Histogram("fraud_api_seconds", "End-to-end score request latency", buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 5), registry=REGISTRY)
REDIS = Histogram("fraud_redis_seconds", "Redis read/commit latency excluding model inference", registry=REGISTRY)
KAFKA = Histogram("fraud_kafka_processing_seconds", "Consumer processing including scoring and publishing", registry=REGISTRY)
KAFKA_MESSAGES = Counter("fraud_kafka_messages_total", "Kafka messages published or successfully consumed", ["topic", "direction"], registry=REGISTRY)
KAFKA_LAG = Gauge("fraud_kafka_consumer_lag", "Unprocessed Kafka offsets by consumer group and topic partition", ["group", "topic", "partition"], registry=REGISTRY)
ERRORS = Counter("fraud_stream_errors_total", "Stream failures", ["kind"], registry=REGISTRY)
LABEL_DELAY = Histogram("fraud_label_delay_days", "Simulated confirmation delay", buckets=(1, 2, 7, 14, 30), registry=REGISTRY)
HISTORY = Gauge("fraud_history_decisions", "Deduplicated persisted decisions", ["decision"], registry=REGISTRY)
CONFIRMED = Gauge("fraud_feedback_count", "Deduplicated joined historical labels; positive prediction = review or decline", ["outcome"], registry=REGISTRY)
DRIFT = Gauge("fraud_amount_psi", "Population stability index: replay amounts versus training reference", registry=REGISTRY)
UNMATCHED = Gauge("fraud_unmatched_labels", "Labels waiting for a decision", registry=REGISTRY)
