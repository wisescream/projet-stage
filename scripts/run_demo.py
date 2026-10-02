"""Run and verify the local Kafka -> FastAPI/Redis -> ClickHouse demo."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "artifacts-replay"
REQUIRED_BUNDLE_FILES = ("manifest.json", "features.sqlite", "events.jsonl", "labels.jsonl")


def docker_command() -> list[str]:
    executable = shutil.which("docker")
    if executable:
        return [executable]
    local_app_data = os.environ.get("LOCALAPPDATA")
    candidates = []
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs/DockerDesktop/resources/bin/docker.exe")
    candidates.append(Path("C:/Program Files/Docker/Docker/resources/bin/docker.exe"))
    for candidate in candidates:
        if candidate.is_file():
            return [str(candidate)]
    raise RuntimeError("Docker CLI introuvable. Installez Docker Desktop et verifiez son PATH.")


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def progress(message: str) -> None:
    print(f"[{datetime.now().astimezone().strftime('%H:%M:%S')}] {message}", flush=True)


def request_json(url: str, timeout: float = 3) -> tuple[int, object]:
    with urlopen(url, timeout=timeout) as response:
        body = response.read()
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = body.decode("utf-8", errors="replace")
        return response.status, payload


def wait_for(check, label: str, timeout: int) -> object:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                return result
        except Exception as error:
            last_error = str(error)
        time.sleep(1)
    raise RuntimeError(f"Delai depasse en attendant {label}. Derniere erreur: {last_error}")


def ensure_bundle(bundle: Path, data_dir: Path, artifacts: Path, limit: int) -> None:
    if not bundle.exists():
        run([
            sys.executable, "-m", "fraud_detection.simulation", "build-replay",
            "--data-dir", str(data_dir), "--artifacts", str(artifacts),
            "--output", str(bundle), "--limit", str(limit),
        ])
    missing = [name for name in REQUIRED_BUNDLE_FILES if not (bundle / name).is_file()]
    if missing:
        raise RuntimeError(f"Bundle incomplet dans {bundle}: fichiers manquants {missing}")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source") != "ieee_replay" or not manifest.get("simulation"):
        raise RuntimeError("Le bundle doit etre une simulation IEEE-CIS valide.")


def load_expected_events(bundle: Path, limit: int) -> tuple[list[dict], set[str], set[str]]:
    events = []
    request_ids, transaction_ids = set(), set()
    if limit < 1:
        raise ValueError("--events doit etre superieur a zero")
    with (bundle / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("event_type") != "authorization_requested":
                continue
            events.append(event)
            request_ids.add(event["event_id"])
            transaction_ids.add(event["transaction_id"])
            if len(events) == limit:
                break
    if not events:
        raise RuntimeError("Le bundle ne contient aucune autorisation a rejouer.")
    return events, request_ids, transaction_ids


def write_demo_slice(bundle: Path, output: Path, transaction_ids: set[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    selected = []
    with (bundle / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("transaction_id") in transaction_ids:
                selected.append(event)
    labels = []
    with (bundle / "labels.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            label = json.loads(line)
            if label.get("transaction_id") in transaction_ids:
                labels.append(label)
    for name, values in (("events.jsonl", selected), ("labels.jsonl", labels)):
        with (output / name).open("w", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Demonstration E2E Kafka, Redis, FastAPI et ClickHouse")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "ieee-fraud-detection")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--limit", type=int, default=100, help="Transactions pour creer le bundle s'il n'existe pas")
    parser.add_argument("--events", type=int, default=5, help="Autorisations a rejouer pour cette demonstration")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-build", action="store_true", help="Ne pas reconstruire les images Docker")
    parser.add_argument("--skip-compose-up", action="store_true", help="Reutiliser la stack deja demarree")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    data_dir = args.data_dir.resolve()
    artifacts = args.artifacts.resolve()

    try:
        from confluent_kafka import Consumer, KafkaError
        import clickhouse_connect
    except ImportError as error:
        raise RuntimeError('Installez les dependances streaming: python -m pip install -e ".[streaming]"') from error

    ensure_bundle(bundle, data_dir, artifacts, args.limit)
    events, request_ids, transaction_ids = load_expected_events(bundle, args.events)
    progress(f"Bundle pret: {len(events)} transactions IEEE-CIS.")
    docker = docker_command()
    compose = [*docker, "compose", "-f", str(ROOT / "compose.yaml")]
    if not args.skip_compose_up:
        command = [*compose, "up", "-d"]
        if not args.no_build:
            command.append("--build")
        progress("Demarrage de la stack Docker Compose.")
        run(command)

    progress("Attente de FastAPI/Redis, Prometheus et Grafana.")
    wait_for(lambda: request_json("http://127.0.0.1:8000/health/ready")[0] == 200,
             "FastAPI et Redis", args.timeout)
    wait_for(lambda: request_json("http://127.0.0.1:9090/-/healthy")[0] == 200,
             "Prometheus", args.timeout)
    wait_for(lambda: request_json("http://127.0.0.1:3000/api/health")[0] == 200,
             "Grafana", args.timeout)

    consumer = Consumer({
        "bootstrap.servers": "127.0.0.1:19092",
        "group.id": "demo-check-" + uuid4().hex,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe(["fraud-decisions"])
    consumer.poll(1)
    environment = {**os.environ, "KAFKA_BOOTSTRAP": "127.0.0.1:19092"}
    try:
        with tempfile.TemporaryDirectory(prefix="fraud-demo-") as temp_path:
            demo_bundle = Path(temp_path)
            write_demo_slice(bundle, demo_bundle, transaction_ids)
            progress("Publication du replay court dans Kafka.")
            run([
                sys.executable, "-m", "fraud_detection.simulation", "replay",
                "--bundle", str(demo_bundle), "--no-wait", "--bootstrap", "127.0.0.1:19092",
            ], env=environment)

            decisions: dict[str, dict] = {}
            deadline = time.monotonic() + args.timeout
            progress("Attente des decisions sur fraud-decisions.")
            while request_ids - decisions.keys() and time.monotonic() < deadline:
                message = consumer.poll(1)
                if message is None:
                    continue
                if message.error():
                    if message.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise RuntimeError(f"Erreur Kafka: {message.error()}")
                decision = json.loads(message.value())
                request_id = decision.get("request_event_id")
                if request_id in request_ids:
                    decisions[request_id] = decision
            missing_requests = sorted(request_ids - decisions.keys())
            if missing_requests:
                raise RuntimeError(f"Decisions Kafka manquantes: {len(missing_requests)}/{len(request_ids)}")

            progress("Verification de la persistance des decisions dans ClickHouse.")
            client = clickhouse_connect.get_client(host="127.0.0.1", port=8123, connect_timeout=5)
            try:
                def persisted_event_counts():
                    result = client.query(
                        "SELECT event_type, countDistinct(event_id) FROM banking_events FINAL "
                        "WHERE transaction_id IN {ids:Array(String)} GROUP BY event_type",
                        parameters={"ids": sorted(transaction_ids)},
                    )
                    return {name: int(count) for name, count in result.result_rows}

                def all_events_persisted():
                    counts = persisted_event_counts()
                    expected = len(transaction_ids)
                    complete = (counts.get("fraud_scored", 0) >= expected
                                and counts.get("fraud_label_received", 0) >= expected
                                and sum(count for name, count in counts.items() if name in {
                                    "authorization_approved", "authorization_declined",
                                    "manual_review_requested", "payment_captured", "payment_refunded",
                                    "chargeback_received",
                                }) >= expected)
                    return counts if complete else None

                persisted_events = wait_for(all_events_persisted,
                                            "decisions, labels et evenements de paiement dans ClickHouse", args.timeout)
            finally:
                client.close()
    finally:
        consumer.close()

    counts = {name: sum(decision["decision"] == name for decision in decisions.values())
              for name in ("approve", "manual_review", "decline")}
    report = {
        "simulation": True,
        "bundle": str(bundle.relative_to(ROOT)) if bundle.is_relative_to(ROOT) else str(bundle),
        "transactions_published": len(events),
        "decisions_consumed_from_kafka": len(decisions),
        "clickhouse_event_counts": persisted_events,
        "decision_counts": counts,
        "services_checked": ["Kafka", "Redis via FastAPI readiness", "FastAPI", "ClickHouse", "Prometheus", "Grafana"],
        "note": "Flux simule a partir des donnees IEEE-CIS, pas un flux bancaire de production.",
    }
    output_dir = ROOT / "artifacts-demo"
    output_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = output_dir / f"e2e-report-{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report": str(report_path.relative_to(ROOT))}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ECHEC E2E: {error}", file=sys.stderr)
        raise SystemExit(1) from error