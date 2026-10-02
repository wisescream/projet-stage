"""Run: python -m fraud_detection.simulation --help."""
import argparse
import json
import os

from prometheus_client import start_http_server

from fraud_detection.cli import positive_int
from fraud_detection.simulation import monitoring
from fraud_detection.simulation.analytics import History, run_sink
from fraud_detection.simulation.replay import build_ieee_bundle, build_synthetic_bundle, replay
from fraud_detection.simulation.transport import KafkaPublisher, ensure_topics, retry, run_worker


def main():
    parser = argparse.ArgumentParser(description="Local simulated banking flow, not a production service")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-replay")
    build.add_argument("--data-dir", default="ieee-fraud-detection")
    build.add_argument("--artifacts", default="artifacts")
    build.add_argument("--output", default="artifacts-replay")
    build.add_argument("--limit", type=positive_int, default=1000)
    synthetic = commands.add_parser("build-synthetic")
    synthetic.add_argument("--output", default="artifacts-synthetic")
    synthetic.add_argument("--count", type=positive_int, default=40)
    produce = commands.add_parser("replay")
    produce.add_argument("--bundle", default="artifacts-replay")
    produce.add_argument("--speed", type=float, default=86400)
    produce.add_argument("--no-wait", action="store_true", help="Skip wall-clock waits, preserve simulated ordering")
    worker = commands.add_parser("worker")
    worker.add_argument("--api-url", default=os.getenv("FRAUD_API_URL", "http://localhost:8000"))
    sink = commands.add_parser("sink")
    sink.add_argument("--bundle", default="artifacts-replay")
    report = commands.add_parser("report")
    topics = commands.add_parser("create-topics")
    for command in (produce, worker, sink, topics):
        command.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP", "127.0.0.1:19092"))
    for command in (sink, report):
        command.add_argument("--clickhouse-host", default=os.getenv("CLICKHOUSE_HOST", "localhost"))
        command.add_argument("--clickhouse-port", type=positive_int, default=8123)
    for command in (worker, sink):
        command.add_argument("--metrics-port", type=positive_int, default=9101 if command is worker else 9102)
    args = parser.parse_args()
    if args.command == "build-replay":
        print(json.dumps(build_ieee_bundle(args.data_dir, args.artifacts, args.output, args.limit), indent=2))
    elif args.command == "build-synthetic":
        print(json.dumps(build_synthetic_bundle(args.output, args.count), indent=2))
    elif args.command == "create-topics":
        ensure_topics(args.bootstrap)
    elif args.command == "replay":
        publisher = KafkaPublisher(args.bootstrap)
        count = replay(args.bundle, lambda topic, event: retry(lambda: publisher.send(topic, event)), speed=args.speed, wait=not args.no_wait)
        print(json.dumps({"published": count, "simulation": True}))
    elif args.command == "report":
        print(json.dumps(History(args.clickhouse_host, args.clickhouse_port).report(), indent=2))
    else:
        start_http_server(args.metrics_port, registry=monitoring.REGISTRY)
        if args.command == "worker":
            run_worker(args.bootstrap, args.api_url)
        else:
            run_sink(args.bootstrap, args.clickhouse_host, args.clickhouse_port, args.bundle)


if __name__ == "__main__":
    main()
