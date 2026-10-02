# Documentation du projet

Cette documentation couvre le code present dans le depot, son fonctionnement et
ses limites. Le projet contient un pipeline ML batch et une simulation de systeme
fraude temps reel.

## Parcours recommande

1. [Demonstration E2E](DEMO.md)
2. [Vue d'ensemble](ARCHITECTURE.md)
3. [Pipeline ML](ML-PIPELINE.md)
4. [Scoring temps reel](RUNTIME-SCORING.md)
5. [Explicabilite, RAG et LLM](EXPLAINABILITY.md)
6. [Investigation et feedback analyste](ANALYST-WORKFLOW.md)
7. [Infrastructure et deploiement](INFRASTRUCTURE.md)
8. [Tests, observabilite et exploitation](OPERATIONS.md)
9. [Modele distant sans GPU](REMOTE-LLM.md)
10. [Feuille de route](ROADMAP.md)

## Statut des composants

| Composant | Statut |
| --- | --- |
| HGB sklearn supervise | Implemente et valide |
| Isolation Forest | Implemente pour le replay IEEE |
| XGBoost / LightGBM | Comparaison optionnelle, dependances non incluses |
| Kafka / Redis / FastAPI / ClickHouse | Simulation Compose implementee |
| Prometheus / Grafana | Dashboards et metriques implementees |
| Kafka Connect | Profil optionnel, plugin a installer |
| RAG controle | Recherche locale par mots-cles implementee |
| LLM distant | API HTTPS compatible OpenAI, optionnelle |
| SHAP | Non implemente ; raisons deterministes disponibles |
| TLS / HA production | Documentation et profil de deploiement, validation externe requise |

## Limite importante

Les donnees IEEE-CIS et les labels du replay ne constituent pas des donnees
bancaires de production. Les resultats de simulation servent a valider le flux
technique, pas a prouver une performance metier reelle.
