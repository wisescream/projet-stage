# Detection de fraude IEEE-CIS

Ce depot contient deux parcours complementaires : un pipeline batch supervise
sur le dataset IEEE-CIS et une simulation locale de scoring en flux. La simulation
relie Kafka, un worker Python, FastAPI, Redis, les regles et le modele, puis
ClickHouse, Prometheus et Grafana.

> Le flux est une simulation technique basee sur IEEE-CIS, pas un flux bancaire
> de production. Les labels de replay simulent des confirmations de fraude.

## Demarrage rapide

Prerequis : Python 3.10+, Docker Desktop/Compose, CSV IEEE-CIS locaux et artefacts
modele pour le scoring. Depuis la racine du depot :

```powershell
python -m pip install -e ".[streaming]"
python scripts/run_demo.py
```

Le lanceur demarre Compose si necessaire, rejoue cinq transactions par defaut,
verifie les decisions lues dans Kafka et leur persistance ClickHouse, puis
controle FastAPI/Redis, Prometheus et Grafana. Le rapport JSON est enregistre
sous `artifacts-demo/`.

Pour relancer sur une stack deja demarree :

```powershell
python scripts/run_demo.py --skip-compose-up
```

Les pre-requis, options, controles et limites sont decrits dans le
[guide de demonstration E2E](docs/DEMO.md).

## Pipeline batch

Entrainement complet avec une validation chronologique :

```powershell
python -m fraud_detection train --data-dir ieee-fraud-detection --output-dir artifacts
```

Prediction a partir d'un modele enregistre :

```powershell
python -m fraud_detection predict --model artifacts/model.joblib --data-dir ieee-fraud-detection --output artifacts/submission-reloaded.csv
```

Le detail des donnees, features, evaluations et limites est dans le
guide [Pipeline ML](docs/ML-PIPELINE.md).

Pour comparer plusieurs fenetres temporelles avec calibration, capacite de
revue et scenarios de cout :

```powershell
python -m fraud_detection evaluate-rolling --data-dir ieee-fraud-detection --output-dir artifacts/rolling
```

Un entrainement cree `model-manifest.json`, mais ne promeut pas le modele.
Promotion et rollback sont explicites :

```powershell
python -m fraud_detection model-promote
python -m fraud_detection model-active
python -m fraud_detection model-rollback
```

## Structure du depot

```text
deploy/                configuration Prometheus, Grafana et Kafka Connect
fraud_detection/       code Python du produit
	pipeline.py          entrainement, evaluation et prediction batch
	model_*.py           comparaison, evaluation et cycle de vie des modeles
	simulation/          API, scoring, replay, regles et analytics
scripts/               lanceurs et controles operationnels
tests/                 tests du pipeline et de la simulation
docs/                  guides utilisateur, architecture et exploitation
deploy/                Compose, Prometheus, Grafana et Kafka Connect
artifacts*/            donnees generees localement, ignorees par Git
ieee-fraud-detection/  donnees source locales, ignorees par Git
compose*.yaml          variantes de la stack Docker Compose
Dockerfile             image de l'application
pyproject.toml         dependances, CLI et configuration de tests
```

## Documentation

- [Index de la documentation](docs/README.md)
- [Guide E2E](docs/DEMO.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Pipeline ML](docs/ML-PIPELINE.md)
- [Scoring temps reel](docs/RUNTIME-SCORING.md)
- [Explicabilite et RAG](docs/EXPLAINABILITY.md)
- [Workflow analyste](docs/ANALYST-WORKFLOW.md)
- [Infrastructure](docs/INFRASTRUCTURE.md)
- [Operations et observabilite](docs/OPERATIONS.md)
- [Feuille de route](docs/ROADMAP.md)

## Tests de developpement

```powershell
python -m pytest -q
```

Les tests automatises valident les contrats et fonctions. Pour verifier les
composants reels de Compose et le flux Kafka jusqu'a ClickHouse, lancer
`python scripts/run_demo.py`.
