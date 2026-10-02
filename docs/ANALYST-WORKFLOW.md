# Investigation et feedback humain

## Creation d'un dossier

Une decision `manual_review` ou `decline` cree un dossier via
`simulation/review.py`. Une decision `approve` ne cree pas de dossier.

Le dossier conserve :

- identifiant de cas;
- date de creation;
- decision originale;
- score et score d'anomalie;
- raisons et contexte;
- liste des feedbacks.

En local, `MemoryState` stocke les dossiers en memoire. Avec Redis, ils sont
stockes sous le namespace de l'application.

## Verdicts

Le schema `ReviewFeedback` accepte :

- `confirmed_fraud`;
- `legitimate`;
- `needs_more_evidence`.

Chaque feedback contient analyste, note et timestamp. Les deux premiers ferment
le dossier en statut `confirmed`; le dernier le laisse ouvert.

## Console

`GET /analyst` retourne une interface HTML minimale permettant de :

- lister les dossiers;
- voir decision, score et raisons;
- confirmer fraude;
- confirmer transaction legitime.

Cette interface est un prototype et ne fournit pas encore de gestion de roles,
de journal d'audit immuable ou de SSO. Elle fournit maintenant des filtres
serveur par decision, score minimum/maximum, statut, intervalle ISO-8601 et
recherche textuelle, avec offset/limit via l'API. La console presente la version
du modele et peut demander une explication locale de sensibilite par feature.

Cette explication remplace une feature par une valeur manquante et mesure le
changement local du score. Elle n'est ni causale, ni additive comme SHAP, et ne
modifie jamais la decision.

## Label differe depuis le verdict

Quand `KAFKA_BOOTSTRAP` est configure, les verdicts finaux `confirmed_fraud` et
`legitimate` produisent un `fraud-labels` avec un horodatage strictement posterieur
a l'autorisation. Le payload est conserve dans l'outbox du dossier avant la
publication afin qu'un retry reutilise exactement le meme event. Les labels
restent separes des features et ne sont jamais reinjectes automatiquement dans
l'entrainement.

## Utilisation API

```shell
curl http://localhost:8000/investigations \
  -H "X-API-Key: $FRAUD_API_KEY"

curl -X POST http://localhost:8000/investigations/evt_xxx/feedback \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $FRAUD_API_KEY" \
  -d '{"verdict":"confirmed_fraud","analyst":"alice","note":"chargeback confirme"}'
```

## Reutilisation des labels

Les verdicts humains ne sont pas automatiquement injectes dans l'entrainement.
Avant reutilisation, il faut :

1. valider le schema et la qualite des annotations;
2. separer les dates d'autorisation et de confirmation;
3. dedupliquer par transaction;
4. versionner le dataset de feedback;
5. refaire une validation temporelle;
6. obtenir une approbation avant promotion du modele.

Cette separation evite qu'un feedback tardif ou errone cree une fuite de cible.
