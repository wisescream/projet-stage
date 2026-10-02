# Pipeline ML batch

## Objectif

Le pipeline lit les fichiers IEEE-CIS, construit une table transaction + identity,
entraine un classifieur de fraude et produit des probabilites pour le fichier de
soumission.

## Entrees

Le repertoire de donnees doit contenir :

```text
train_transaction.csv
train_identity.csv
test_transaction.csv
test_identity.csv
sample_submission.csv
```

Les transactions sont identifiees par `TransactionID`, le temps par
`TransactionDT` et la cible par `isFraud`.

## Chargement et controle

`fraud_detection/pipeline.py` applique les controles suivants :

- types explicites pour eviter les differences entre chunks CSV;
- normalisation de `id-XX` vers `id_XX`;
- identifiants uniques et non nuls;
- jointure identity en `left join`;
- rejet des collisions de colonnes;
- rejet d'une cible dans les donnees identity;
- controle des valeurs temporelles et labels 0/1.

Les colonnes identite manquantes ne suppriment jamais une transaction.

## Features

`features()` retire `TransactionID` et `isFraud`. Les colonnes connues comme
categorical sont converties en chaines puis encodees par `OrdinalEncoder`.

Les categories inconnues et manquantes recoivent `-1`. Les colonnes numeriques
restent numeriques et les valeurs manquantes sont gerees par le classifieur.

Cette representation est un baseline : les codes ordinaux sont traites comme
des nombres, ce qui ne remplace pas un modele natif categorical.

## Split temporel

`chronological_split()` trie par `TransactionDT` et reserve les transactions
les plus recentes pour la validation. Les timestamps egaux restent dans la meme
partition et aucun timestamp d'entrainement n'atteint la validation.

Ce choix limite les fuites temporelles, mais une seule fenetre ne prouve pas la
robustesse future. Une validation glissante est recommandee avant production.

## Modele actuel

Le modele par defaut est `HistGradientBoostingClassifier` dans une pipeline
scikit-learn :

```text
ColumnTransformer
  numeric: passthrough
  categorical: string conversion + OrdinalEncoder
HistGradientBoostingClassifier
```

L'early stopping est desactive afin de garder une evaluation temporelle
reproductible.

`fraud_detection/model_comparison.py` detecte optionnellement XGBoost et
LightGBM. Ces bibliotheques ne sont pas des dependances obligatoires du projet.

## Evaluation

Les metriques produites dans `metrics.json` sont :

- ROC-AUC : qualite du classement des transactions;
- average precision : qualite du classement en classe rare;
- log loss : qualite probabiliste;
- taux de fraude;
- tailles et limites temporelles des partitions.

L'accuracy n'est pas utilisee comme metrique principale car la fraude est
fortement desequilibree.

### Validation temporelle glissante

L'entrainement produit une calibration Platt chronologique sur la fin de la
periode d'entrainement, puis evalue la periode holdout. Pour comparer plusieurs
periodes successives et publier precision/rappel aux capacites de revue, couts
estimes et calibration par periode :

```shell
python -m fraud_detection evaluate-rolling --data-dir ieee-fraud-detection --output-dir artifacts/rolling
```

Les sorties `rolling-evaluation.json` et `rolling-periods.csv` contiennent les
metriques par fold. Chaque fenetre utilise un entrainement expanding-window et
une tranche chronologique interne pour la calibration. Les petits jeux de
developpement n'apprennent pas un calibrateur instable ; le rapport l'indique.

L'analyse de cout compare trois hypotheses explicites (low/medium/high) pour
fraude manquee, revue manuelle et faux refus. Les taux de revue/refus sont des
capacites de classement par score, pas des seuils choisis sur les labels. Les
montants sont des hypotheses a remplacer par des couts metier valides.

## Artefacts

| Artefact | Role |
| --- | --- |
| `model-validation.joblib` | modele entraine uniquement sur la partition train |
| `model.joblib` | modele refit sur toutes les lignes chargees |
| `validation_predictions.csv` | labels et probabilites de validation |
| `submission.csv` | probabilites test alignees sur le sample |
| `metrics.json` | metriques, versions et parametres |
| `model-manifest.json` | hash, version, schema des features, periode et metriques |

Le fichier validation est celui utilise pour construire le replay IEEE. Il ne
faut pas utiliser `model.joblib` pour evaluer ce meme holdout.

## Commandes

```shell
python -m fraud_detection train --data-dir ieee-fraud-detection --output-dir artifacts
python -m fraud_detection train --train-rows 50000 --max-iter 30 --output-dir artifacts-smoke
python -m fraud_detection predict --model artifacts/model.joblib --output artifacts/submission-reloaded.csv
```

`--train-rows` prend les premieres lignes, ce n'est pas un echantillonnage
aleatoire. La soumission test conserve toutefois toutes les lignes test.

## Risques connus

- calibration des probabilites non demontree;
- absence de validation glissante;
- baseline ordinale plutot que categorical native;
- chargement joblib reserve a des fichiers de confiance;
- entrainement complet consommateur de plusieurs Go de RAM.

## Promotion et rollback explicites

Un entrainement n'est pas automatiquement promu. Apres revue du manifeste et
des metriques, les operations suivantes conservent chaque modele dans une
version immuable :

```shell
python -m fraud_detection model-promote --model artifacts/model.joblib --manifest artifacts/model-manifest.json --registry-dir artifacts/model-registry
python -m fraud_detection model-active --registry-dir artifacts/model-registry
python -m fraud_detection model-rollback --registry-dir artifacts/model-registry
```

La promotion verifie le SHA-256 du manifeste avant copie. Le rollback ne fait
que basculer le pointeur actif vers la version precedente ; il ne supprime
aucun artefact. Le chargement du modele actif par les services doit etre
configure explicitement au deploiement.
