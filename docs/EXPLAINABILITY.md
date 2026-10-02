# Explicabilite, RAG et LLM

## Principes

L'explication ne participe jamais au calcul de la decision. La decision est

- carte bloquee;
- montant inhabituel;
- velocite;
- nouvel appareil;
- changement de pays;
- plusieurs cartes sur appareil;
- 3-D Secure absent;
- identity absente;
- seuil modele;
- seuil anomalie;
- modele indisponible.

Ces raisons sont la source principale pour l'analyste. Elles sont preferables a
une justification generative quand une trace exacte est necessaire.

## RAG local controle

`simulation/knowledge.py` contient un petit corpus versionne de politiques. La
fonction `retrieve_context()` :

1. tokenise la requete construite a partir des raisons;
2. compte les mots communs avec les documents;
3. trie les documents par correspondance puis identifiant;
4. retourne identifiant, titre, version, texte et nombre de correspondances.

Ce RAG est volontairement simple et deterministe. Il ne remplace pas encore une
base vectorielle avec embeddings, filtrage par version et controle editorial.

## LLM local

Ollama est supporte par :

```text
LOCAL_LLM_URL=http://localhost:11434
LOCAL_LLM_MODEL=qwen2.5:0.5b
```

Le prompt impose une reformulation des faits sans ajout ni decision.

## LLM distant

Une API HTTPS compatible OpenAI est supportee par :

```text
REMOTE_LLM_URL=https://provider.example/v1
REMOTE_LLM_MODEL=nom-du-modele
REMOTE_LLM_API_KEY=secret
```

La cle doit etre injectee par secret manager ou variable d'environnement. Le
mode distant est utile sans GPU local et reste optionnel.

## Protection des donnees

Le prompt transmet uniquement :

- decision;
- risk score;
- anomaly score;
- raisons;
- version modele.

Les tokens client, carte, appareil et IP ne sont pas transmis au LLM.

## Fallback

Si le LLM echoue, la reponse conserve l'explication template et ajoute un statut
d'indisponibilite. Une explication generee porte `requires_human_validation`.

## Effets locaux des features

Pour les references IEEE privees, l'API calcule une sensibilite locale en
remplacant une feature importante par une valeur manquante et en observant le
changement du score. La reponse expose la methode
`single_feature_missing_value_perturbation` et le delta mesure.

Cette methode n'est ni causale, ni additive comme SHAP, ni utilisee par le moteur
de decision. Les signaux deterministes des regles restent separes des effets du
modele. SHAP pourra etre ajoute ensuite avec un explainer et un mapping de
features explicitement valides pour le modele retenu.
