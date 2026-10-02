# Infrastructure et deploiement

## Compose local

`compose.yaml` demarre :

| Service | Role |
| --- | --- |
| Redpanda | Kafka compatible pour les topics |
| Redis | etat recent et idempotence |
| ClickHouse | historique analytique |
| API | FastAPI et scoring |
| worker | consommation des autorisations |
| sink | ingestion Python vers ClickHouse |
| Prometheus | collecte metriques |
| Grafana | dashboards |

Les ports sont limites a loopback et le fichier indique explicitement que ce
n'est pas une configuration production.

## Variante Windows

`compose.host.yaml` conserve Kafka, Redis et ClickHouse dans Compose, mais lance
les services Python sur l'hote. Cette variante evite les problemes de partage de
repertoires Docker sous Windows.

## Kafka Connect

Le service `kafka-connect` est dans le profil optionnel `connect` :

```shell
docker compose --profile connect up -d kafka-connect
```

La configuration du connecteur ClickHouse se trouve dans
`deploy/kafka-connect/fraud-decisions-clickhouse.json`.

Il faut fournir :

- une image contenant le plugin ClickHouse Connect;
- un secret `secrets.properties` hors Git;
- une configuration de replication adaptee au cluster;
- un reseau prive pour le port REST Connect.

Le sink Python reste le choix local autonome.

## Modele distant

L'API accepte `REMOTE_LLM_URL`, `REMOTE_LLM_MODEL` et
`REMOTE_LLM_API_KEY`. Le modele distant est une dependance d'explication et ne
bloque pas le scoring si le service est indisponible.

## Production non couverte par Compose

Pour une vraie production, deployer :

- Kafka avec plusieurs brokers et replication factor 3;
- Redis Sentinel ou Cluster;
- ClickHouse replique et tables distribuees;
- plusieurs replicas API et workers;
- ingress TLS et authentification;
- secrets depuis Vault, Kubernetes Secrets ou service cloud;
- sauvegardes et procedures de restauration.

Voir `deploy/TLS-AND-HA.md` pour les exigences detaillees.
