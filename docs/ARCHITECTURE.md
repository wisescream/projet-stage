# Architecture du projet

## Flux actuel

```text
transactions-incoming (Kafka)
        |
        v
FastAPI + Redis + modele supervise + Isolation Forest + regles
        |
        +--> fraud-decisions (Kafka)
        +--> payment-events (Kafka)
        +--> dossiers analyste + feedback humain
        |
        v
ClickHouse <- sink Python local ou Kafka Connect optionnel
        |
        v
Prometheus -> Grafana
```

## Conformite avec le schema fourni

Le projet suit bien le meme principe que sur le schema visuel :

- `transactions-incoming` correspond au flux principal de transactions Kafka.
- Le bloc `Feature Store / Redis` correspond a la memoire de contexte recente et aux
  donnees de contexte client, montant, historique court, appareil, adresse et
  dernier comportement.
- Le bloc `Microservice Inference` correspond au service FastAPI qui lit le message,
  enrichit les features, applique les regles et appelle le modele.
- Le bloc `Modele hybride` correspond a l'assemblage des sorties : modele supervise
  IEEE, Isolation Forest et regles de decision.
- Le bloc `Decision de fraude` correspond a la sortie de decision (approve,
  manual_review, decline) et au score associe.
- Le bloc `Kafka` correspond a la publication sur `fraud-decisions` et aux evenements
  de paiement / labels.
- Le bloc `ClickHouse + Grafana` correspond au stockage analytique et au dashboard de
  supervision.
- Le bloc `LLM local / expliqueur` correspond a une couche d'explication uniquement.
  Le LLM ne prend aucune decision et ne modifie pas la note de risque.

Le diagramme de l'image est donc conforme a la philosophie de l'implementation :
architecture de scoring hybrid, publication de decisions, observabilite et couche
explicative dediee a l'analyste. L'element important est que le score reste un
produit de l'IA et des regles, tandis que l'IA de langage sert seulement a rendre
la decision comprensible.

Le modele LLM est uniquement utilise pour reformuler une explication a destination
analyste. Il ne prend aucune decision et ne modifie jamais le score.

## Couplage detaille Kafka / Redis / FastAPI

Le flux concret de l'application correspond aux fichiers suivants :

- [fraud_detection/simulation/transport.py](../fraud_detection/simulation/transport.py) : publication sur Kafka, creation des topics, gestion de la DLQ, consommation des messages.
- [fraud_detection/simulation/api.py](../fraud_detection/simulation/api.py) : service FastAPI, endpoints `/score`, `/health/ready`, `/investigations`, `/metrics`.
- [fraud_detection/simulation/state.py](../fraud_detection/simulation/state.py) : couche Redis pour l'idempotence, le contexte transactionnel et la concurrence.
- [fraud_detection/simulation/decision.py](../fraud_detection/simulation/decision.py) : moteur de decision et combinaison regles + score + anomalies.

### Sequencement fonctionnel

1. Le producteur lit un bundle de transactions et publie sur `transactions-incoming`.
2. Le worker Kafka consomme le message et l'envoie a l'API FastAPI via `POST /score`.
3. L'API valide la structure du message, exige la cle API si elle est activee, puis
   appelle la couche de decision.
4. La couche de decision combine le score IEEE, les regles fortes et l'anomalie.
5. L'API appelle le store Redis pour verrouiller l'idempotence et enregistrer le
   resultat associe a `event_id`.
6. Si la decision est valide, le worker publie sur `fraud-decisions` puis sur
   `payment-events`.
7. Le sink Kafka enregistre dans ClickHouse et alimente la surveillance Grafana.
8. Les dossiers analyste sont alimentes par l'API et la base Redis / ClickHouse.
9. Le LLM explique la decision a partir d'un contexte structure, sans modifier la
   note de risque ni la decision.

### Exemple de flux complet

```text
Kafka producer / replay
    -> transactions-incoming
        -> worker Kafka
            -> FastAPI /score
                -> RedisState.process()
                -> DecisionEngine.decide()
                -> return Decision
            -> Kafka fraud-decisions
            -> Kafka payment-events
        -> sink / ClickHouse
            -> Prometheus / Grafana
```

### Contraintes de robustesse presentes dans le code

- `enable.idempotence` et `acks=all` sur le producer Kafka.
- gestion explicite des messages invalides vers `fraud-dead-letter`.
- `Conflict` et `409` sur doublons ou contestations de contexte.
- `503` si Redis ou le service de contexte est indisponible.
- `event_id` comme identifiant logique stable pour la dedupe.
- `payment-events` publie une decision de paiement departagee selon `approve`, `manual_review` ou `decline`.

Cette structure correspond bien a un flux bancaire simule, avec l'organisation demandee par le schema : ingestion, enrichissement, inference, decision, publication, stockage analytique et explication analytique.

## Scoring

- Le modele IEEE supervise produit `risk_score`.
- Isolation Forest produit `anomaly_score` sans utiliser les labels.
- Les regles fortes et les seuils produisent `approve`, `manual_review` ou `decline`.
- Une anomalie ne peut declencher qu'une revue.
- Une indisponibilite du modele active le comportement fail-safe de revue.

## Explications et RAG

Les raisons deterministes sont toujours produites en premier. Un petit corpus
controle dans `fraud_detection/simulation/knowledge.py` est recherche par mots
cles et retourne les documents, versions et correspondances utilises.

Le LLM est donc une couche d'explication, pas une source de verite. Le contexte
RAG et les faits envoyes doivent rester depourvus de donnees personnelles.

## Deux modes LLM

### Distant, recommande sans GPU

Le service utilise une API HTTPS compatible OpenAI :

- `REMOTE_LLM_URL`
- `REMOTE_LLM_MODEL`
- `REMOTE_LLM_API_KEY`

La cle doit venir d'un secret d'environnement ou d'un gestionnaire de secrets.
Aucune cle ne doit etre commitee. Les offres gratuites et leurs quotas changent;
le fournisseur doit donc etre choisi au moment du deploiement.

### Local, conserve pour l'integration

Ollama reste disponible avec :

- `LOCAL_LLM_URL`
- `LOCAL_LLM_MODEL`

Le mode local est utile pour les tests, mais il n'est pas necessaire pour faire
fonctionner le scoring ou la console analyste.

## Limites actuelles

Le projet reste une simulation IEEE-CIS. Les labels du replay sont simules, les
secrets ne sont pas geres par un coffre dans Compose, et la haute disponibilite
necessite un deploiement externe.
