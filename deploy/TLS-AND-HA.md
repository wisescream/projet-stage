# TLS, secrets, and high availability

The Compose files are local simulation environments. For a real deployment:

- terminate TLS at a private ingress or service mesh and forward only to the internal API network;
- inject `FRAUD_API_KEY`, Redis credentials, ClickHouse credentials, and model paths from a secret manager;
- run at least three Kafka brokers with replication factor 3 and separate controller quorum;
- use Redis Sentinel or Redis Cluster with persistent encrypted storage;
- use a ClickHouse replicated database and distributed table;
- run multiple API and worker replicas behind health-checked load balancing;
- configure Prometheus and Grafana authentication and restrict their network exposure.

No local certificate or fake replica is included: those would create a false sense
of security and are not interchangeable with deployment-managed trust roots and
failover testing.