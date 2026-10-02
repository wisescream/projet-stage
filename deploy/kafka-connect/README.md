# Optional Kafka Connect sink

The default local profile uses the Python sink because it is self-contained and
does not require connector plugins. The `connect` profile is an infrastructure
option for deployments that require Kafka Connect:

```shell
docker compose --profile connect up -d kafka-connect
curl -X POST http://localhost:8083/connectors \
  -H 'Content-Type: application/json' \
  --data @deploy/kafka-connect/fraud-decisions-clickhouse.json
```

The Connect image must contain `com.clickhouse.kafka.connect.ClickHouseSinkConnector`
and must receive `/etc/kafka-connect/secrets.properties` from a secret manager.
The checked-in configuration intentionally contains no password. Do not expose
port 8083 outside a private network.