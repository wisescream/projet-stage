# Scoring temps reel

## Entree

Une autorisation arrive sur le topic Kafka `transactions-incoming` et respecte
le contrat Pydantic `Authorization` dans `simulation/schemas.py`.

Le contrat contient des tokens pseudonymises, le montant, pays, canal, appareil,
3-D Secure, indication identity manquante et une reference privee de features
pour les evenements IEEE replay.

Les champs inconnus, labels precoces, PAN, CVV et dates invalides sont refuses.

## Etat temps reel

`simulation/state.py` fournit :

- `MemoryState` pour les tests locaux;
- `RedisState` pour le service Compose.

L'etat conserve l'historique client et appareil necessaire aux fenetres :

- transactions des cinq dernieres minutes;
- echecs des cinq dernieres minutes;
- montant recent;
- cartes vues sur appareil pendant 24 heures;
- dernier pays et dernier appareil;
- watermark temporel.

Les identifiants d'evenement et de transaction assurent l'idempotence et
protegent contre les doublons et les evenements hors ordre.

## Inference

`simulation/decision.py` combine trois familles de signaux :

1. score supervise `risk_score` du modele IEEE holdout;
2. score atypique `anomaly_score` produit par Isolation Forest;
3. regles de contexte et regles de securite.

Le modele benchmark verifie le hash du modele et la correspondance exacte entre
l'autorisation et la feature reference SQLite avant l'inference.

## Politique de decision

Seuils par defaut :

```text
review = 0.55
block = 0.98
anomaly_review = 0.80
```

- `approve` : aucun signal de revue ou de refus;
- `manual_review` : score, anomalie ou contexte exige une verification;
- `decline` : carte bloquee ou score modele au seuil de blocage.

Montant inhabituel, velocite, changement d'appareil, changement de pays,
plusieurs cartes et 3-D Secure absent declenchent une revue. L'anomalie ne peut
pas declencher seule un refus.

En cas d'indisponibilite du modele, le systeme refuse l'autorisation automatique
et demande une revue. Il ne revient jamais a un approve silencieux.

## Sorties Kafka

Le worker publie :

- `fraud-decisions` avec la decision complete;
- `payment-events` avec autorisation approuvee, refus ou demande de revue;
- `fraud-dead-letter` pour les messages invalides, sans recopier le payload.

Le producteur active l'idempotence Kafka et le consommateur valide les contrats
avant commit d'offset.

## API

| Endpoint | Fonction |
| --- | --- |
| `GET /health/live` | processus vivant |
| `GET /health/ready` | Redis disponible et modele charge |
| `POST /score` | scorer une autorisation |
| `GET /decisions/{event_id}/explanation` | explication analyste |
| `GET /investigations` | dossiers a risque |
| `POST /investigations/{case_id}/feedback` | verdict humain |
| `GET /analyst` | console analyste minimale |
| `GET /metrics` | metriques Prometheus |

`FRAUD_API_KEY` active une authentification simple par header `X-API-Key`. Pour
la production, elle doit etre remplacee ou completee par OAuth2/OIDC et des
roles.

## Limite de performance connue

Le test de streaming precedent mesurait environ 529 ms en p95 et 614 ms en p99
sur un traitement HTTP sequentiel. Ces valeurs sont un signal de prototype et
necessitent une vraie campagne de charge avant de fixer un SLA.
