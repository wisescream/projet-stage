# Demonstration E2E

Cette demonstration relance le flux complet IEEE-CIS simule : publication dans
Kafka, scoring par le worker via FastAPI, enrichissement Redis, decision Kafka,
puis persistance dans ClickHouse. Le script verifie aussi la disponibilite de
Prometheus et Grafana. Il ne supprime ni les volumes Docker ni les bundles
existants.

## Prerequis

- Docker Desktop avec Docker Compose demarre ;
- Python 3.10 ou plus recent ;
- dependances Python streaming installees :

```powershell
python -m pip install -e ".[streaming]"
```

- fichiers de donnees IEEE-CIS sous `ieee-fraud-detection/` ;
- modele et metriques batch sous `artifacts/` (`model-validation.joblib`,
  `metrics.json`, `validation_predictions.csv`).

Si les artefacts batch manquent, les generer une fois :

```powershell
python -m fraud_detection train --data-dir ieee-fraud-detection --output-dir artifacts
```

Le bundle de replay `artifacts-replay/` est reutilise s'il existe. S'il manque,
le script le cree avec 100 transactions par defaut avant la construction Docker.
Il ne remplace jamais un dossier de bundle existant.

## Lancer la demonstration

Depuis la racine du depot :

```powershell
python scripts/run_demo.py
```

La premiere execution construit et demarre la stack. Le script attend que les
services soient accessibles, s'abonne au topic de decisions, rejoue le bundle
sans delai (5 autorisations par defaut, avec leurs labels associes), puis attend
les decisions et leur persistance ClickHouse. Le bundle original n'est pas
modifie ; seul un petit replay temporaire est prepare.

Pour reutiliser une stack deja demarree sans reconstruire les images :

```powershell
python scripts/run_demo.py --skip-compose-up
```

Pour redemarrer Compose sans reconstruire les images :

```powershell
python scripts/run_demo.py --no-build
```

Options disponibles :

- `--bundle chemin` : bundle IEEE-CIS a rejouer ;
- `--data-dir chemin` : emplacement des CSV, utilise seulement si le bundle doit etre cree ;
- `--artifacts chemin` : artefacts batch, utilises seulement si le bundle doit etre cree ;
- `--limit N` : nombre de transactions du nouveau bundle, 100 par defaut ;
- `--events N` : autorisations rejouees pour cette execution, 5 par defaut ;
- `--timeout secondes` : delai d'attente des composants et resultats, 120 par defaut.

Le script echoue avec un code non nul si un service n'est pas disponible, si une
decision attendue manque du topic Kafka ou si ClickHouse n'a pas persiste toutes
les decisions. Le rapport JSON horodate est ecrit dans `artifacts-demo/`.

## Composants controles

| Composant | Controle effectue |
| --- | --- |
| Kafka | Lecture du topic `fraud-decisions` et correspondance par `request_event_id` |
| FastAPI | Endpoint `/health/ready` ; la readiness verifie aussi Redis et le moteur |
| Redis | Ping par la readiness FastAPI et etat/idempotence utilises pendant le scoring |
| ClickHouse | Decisions, labels differes et evenements de paiement persistes pour les transactions rejouees |
| Prometheus | Endpoint de sante `/-/healthy` |
| Grafana | Endpoint de sante `/api/health` |

Le flux metier est :

```text
IEEE replay -> Kafka transactions-incoming -> worker -> FastAPI -> Redis
                                              -> modele + regles
           <- Kafka fraud-decisions <- worker <- decision
                    -> sink -> ClickHouse -> Prometheus/Grafana
```

Les messages sont des evenements pseudonymises construits depuis IEEE-CIS. Les
labels sont simules et ne constituent pas des chargebacks bancaires observes.
Cette demo verifie l'integration technique ; elle ne certifie ni un systeme de
production bancaire, ni la performance reelle du modele.

## Organisation du depot

```text
fraud_detection/              pipeline batch et application de scoring
  simulation/                 API, contrats, decision, Kafka/Redis/ClickHouse
scripts/                      verification et lancement de demos
tests/                        tests unitaires et contractuels
docs/                         architecture, ML, operations et guides
deploy/                       Compose-adjacent, Grafana, Prometheus, Connect
ieee-fraud-detection/         donnees locales non versionnees
artifacts*/                   modeles, bundles et resultats locaux non versionnes
```

Les CSV, modeles, bundles, rapports d'execution et sorties de build ne sont pas
des sources de code et restent hors de Git.