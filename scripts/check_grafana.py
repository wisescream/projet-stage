"""Validate the provisioned dashboard and its live Prometheus queries."""
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
PROMETHEUS = "http://localhost:9090"
GRAFANA = "http://localhost:3000"
REQUIRED_TARGETS = {"fraud-api", "fraud-worker", "fraud-history"}


def get_json(url):
    with urlopen(url, timeout=5) as response:
        return json.load(response)


def main():
    dashboard = json.loads((ROOT / "deploy/grafana/dashboards/banking.json").read_text(encoding="utf-8"))
    provisioned = get_json(f"{GRAFANA}/api/dashboards/uid/{dashboard['uid']}")
    if provisioned["dashboard"]["uid"] != dashboard["uid"]:
        raise AssertionError("Grafana returned a different dashboard")

    targets = get_json(f"{PROMETHEUS}/api/v1/targets")["data"]["activeTargets"]
    healthy = {target["labels"].get("job") for target in targets if target["health"] == "up"}
    missing = REQUIRED_TARGETS - healthy
    if missing:
        raise AssertionError(f"Prometheus targets down or absent: {sorted(missing)}")

    empty = []
    for panel in provisioned["dashboard"]["panels"]:
        for target in panel.get("targets", []):
            result = get_json(f"{PROMETHEUS}/api/v1/query?{urlencode({'query': target['expr']})}")
            if result["status"] != "success":
                raise AssertionError(f"{panel['title']} ({target['refId']}): {result}")
            count = len(result["data"]["result"])
            print(f"{panel['title']} [{target['refId']}]: {count} series")
            if count == 0:
                empty.append(f"{panel['title']}[{target['refId']}]")

    if empty:
        print("Queries valid but currently empty: " + ", ".join(empty))
    print(f"Dashboard {dashboard['uid']} provisioned; targets up: {sorted(healthy)}")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, HTTPError, URLError, TimeoutError) as error:
        print(f"Grafana validation failed: {error}", file=sys.stderr)
        raise SystemExit(1)