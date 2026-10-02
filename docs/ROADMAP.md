# Feuille de route apres la partie applicative

## 1. Environnement distant de modele

- Choisir un fournisseur HTTPS compatible OpenAI avec quota gratuit ou budget
  controle.
- Configurer `REMOTE_LLM_*` uniquement par secrets.
- Garder Ollama comme integration locale de secours.
- Mesurer le cout, la latence, les quotas et la disponibilite.

## 2. Infrastructure

- Installer Kafka Connect et le plugin ClickHouse.
- Tester le connecteur sur `fraud-decisions`, `payment-events` et `fraud-labels`.
- Passer Kafka a plusieurs brokers avec replication factor 3.
- Utiliser Redis Sentinel ou Redis Cluster.
- Utiliser ClickHouse replique et tables distribuees.

Le profil local utilise encore le sink Python. Le profil Connect est optionnel
afin de conserver un demarrage simple pour le developpement.

## 3. Securite

- TLS a l'ingress et entre services sensibles.
- Secrets dans Vault, Docker secrets ou Kubernetes Secrets.
- Authentification API et roles analystes.
- Restriction reseau de Kafka Connect, Prometheus et Grafana.
- Rotation des cles et audit des acces.

Voir `deploy/TLS-AND-HA.md`.

## 4. Qualite modele

- [x] Calibration Platt chronologique et diagnostics Brier / bins.
- [x] Validation expanding-window multi-periodes et rapport PR-AUC par periode.
- [x] Capacite de revue/refus, precision/rappel et scenarios de cout explicites.
- [ ] Comparer HGB, XGBoost et LightGBM sur les memes fenetres walk-forward.
- [ ] Valider les hypotheses de cout avec les operations fraude de la banque.
- [ ] Evaluer Isolation Forest sans fuite de labels sur des periodes futures.
- [ ] Ajouter SHAP approuve pour le modele et le mapping de features retenus.

## 5. Performance et reprise

- Test de charge API et Kafka.
- Mesure p50, p95 et p99.
- Arret/redemarrage Redis, Kafka et ClickHouse.
- Verification de l'idempotence, des offsets et de l'absence de perte.
- Test de plusieurs replicas API et workers.

## 6. MLOps et exploitation

- [x] Manifeste modele avec SHA-256, schema features, periode et metriques.
- [x] Registry local versionne avec promotion/rollback explicites.
- [ ] Registry centralise, revue a deux personnes et rollback orchestre en production.
- Pipeline train, validation, promotion et rollback.
- Monitoring de derive, qualite, latence et volumes.
- Conservation immuable des decisions et feedbacks analystes.
- Documentation des alertes et procedures d'astreinte.

## Critere de passage en production

Le projet ne doit etre declare production-ready qu'apres validation securite,
charge, panne, reprise, confidentialite et cout du fournisseur distant.

Les fonctionnalites cochees sont des mecanismes de prototype. Elles ne valent
pas validation statistique/metier ou certification de production.
