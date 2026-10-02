# Tests, observabilite et exploitation

## Tests unitaires

```shell
python -m pytest -q
```

`tests/test_pipeline.py` couvre :

- lecture et types CSV;
- jointures identity;
- validation des identifiants;
- split temporel;
- absence de fuite de categories;
- generation de soumission;
- reproductibilite;
- CLI train/predict;
- schema des fichiers test.

`tests/test_simulation.py` couvre :

- contrats Pydantic;
- labels differes;
- seuils et fail-safe;
- etat precedent et fenetres;
- concurrence et idempotence;
- API et explication;
- panne Redis;
- retry et publication Kafka;
- dossiers analyste et feedback;
- authentification API.

## Test d'integration streaming

`scripts/check_streaming.py` verifie sur la stack Compose :

- Kafka, Redis, FastAPI, ClickHouse;
- labels differes et rapprochement;
- redemarrage Redis et API;
- reprise worker;
- absence de doublons;
- dead-letter sans payload brut;
- Prometheus et Grafana;
- latence HTTP reelle.

Ce script consomme et publie dans l'environnement de demo. Il ne doit pas etre
lance sur des topics de production.

## Alimenter Grafana avec le replay

Les dashboards sont silencieux tant qu'aucun evenement n'est produit. Demarrer
la stack puis lancer le producteur one-shot depuis l'image Python :

```shell
docker compose up -d --wait
docker compose run --rm topics python -m fraud_detection.simulation replay --bundle /bundle --no-wait --bootstrap kafka:9092
```

Le producteur publie autorisations et labels. Le worker calcule les decisions,
le sink les ecrit dans ClickHouse et Prometheus collecte les metriques. Ouvrir
`http://localhost:3000` apres quelques secondes.

Les volumes locaux persistent. Un replay identique est deduplique par les
identifiants stables et ne doit pas etre interprete comme une nouvelle
evaluation du modele.

## Metriques Prometheus

`simulation/monitoring.py` expose notamment :

- requetes API par statut;
- decisions par type;
- doublons;
- latence API, Redis et Kafka;
- erreurs et dead letters;
- delai de labels;
- decisions rapprochees TP/FP/TN/FN;
- derive des montants;
- labels non rapproches.

La couche de transport expose aussi le nombre de messages publies/consommes et
le lag de l'offset par consumer group, topic et partition. Les dimensions restent
des valeurs techniques a faible cardinalite, sans identifiants client.

Le dashboard Grafana est dans `deploy/grafana/dashboards/banking.json`.
Les regles prototype sont dans `deploy/alerts.yml` et couvrent disponibilite
API, lag Kafka, taux d'erreur, PSI des montants et labels non rapproches.

## Alertes recommandees

Ajouter en production des alertes sur :

- API indisponible ou taux 5xx;
- p95/p99 au-dessus du budget;
- backlog Kafka;
- dead letters;
- Redis ou ClickHouse indisponible;
- score ou taux de revue anormal;
- derive PSI;
- labels non rapproches;
- quota ou erreurs du LLM distant.

## Incident et reprise

La reprise doit verifier :

1. offsets Kafka non commites avant traitement termine;
2. idempotence par event et transaction;
3. decisions deja persistees retournees a l'identique;
4. absence d'approbation automatique si le feature store est indisponible;
5. restauration Redis/ClickHouse depuis sauvegarde;
6. rotation des secrets apres incident.

## Environnements

| Environnement | Usage |
| --- | --- |
| tests | pytest, pas d'infrastructure externe |
| smoke | entrainement reduit, pas une mesure de performance |
| replay | 300 transactions holdout et labels simules |
| local streaming | Compose complet sur loopback |
| production | non fournie, doit etre construite et testee |
