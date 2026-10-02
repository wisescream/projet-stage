# Utiliser un modele gratuit distant

Le scoring fraude fonctionne sans LLM. Le modele distant sert uniquement a
produire une explication lisible pour l'analyste.

## Contrat fournisseur

Le fournisseur doit proposer une API HTTPS de type OpenAI Chat Completions avec
une reponse de la forme `choices[0].message.content`.

Les offres gratuites ou free-tier varient selon le pays, la date et les quotas.
Le code ne depend donc pas d'un fournisseur unique. Un endpoint compatible peut
etre fourni par une plateforme de modeles heberges, un endpoint de laboratoire
ou un proxy d'equipe.

## Configuration

Dans l'environnement du service API :

```text
REMOTE_LLM_URL=https://provider.example/v1
REMOTE_LLM_MODEL=nom-du-modele-instruct
REMOTE_LLM_API_KEY=cle-injectee-par-secret-manager
```

Ne jamais placer `REMOTE_LLM_API_KEY` dans le depot, les fichiers Compose ou les
logs. En production, utiliser Vault, Docker secrets, Kubernetes Secrets ou le
secret manager du cloud.

## Exemple de lancement

```shell
$env:REMOTE_LLM_URL = "https://provider.example/v1"
$env:REMOTE_LLM_MODEL = "nom-du-modele-instruct"
$env:REMOTE_LLM_API_KEY = "a-definir-dans-le-terminal"
python -m uvicorn fraud_detection.simulation.api:app --host 127.0.0.1 --port 8000
```

Les faits envoyes au modele sont limites a la decision, aux scores, aux raisons
et a la version du modele. Les tokens client, carte, appareil et IP ne sont pas
ajoutes au prompt.

## Comportement de secours

- Si la variable distante est absente, l'explication template fonctionne.
- Si l'endpoint echoue ou depasse le delai, le template est conserve.
- Si l'URL n'est pas HTTPS ou si la cle manque, la configuration est rejetee.
- Le LLM ne peut jamais changer `approve`, `manual_review` ou `decline`.

## Verification

Tester d'abord avec un endpoint de test et un quota faible. Verifier ensuite :

1. absence de secret dans Git et les logs;
2. retour template en cas de panne distante;
3. conservation de la decision originale;
4. validation humaine obligatoire pour une explication generee par LLM;
5. respect des conditions de confidentialite du fournisseur.
