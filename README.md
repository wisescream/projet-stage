# IEEE-CIS fraud detection pipeline

A reproducible CPU baseline for the local IEEE-CIS transaction and identity CSVs. It trains a scikit-learn histogram gradient-boosted classifier, evaluates on later transactions, refits on all loaded training rows, and generates a complete submission of fraud probabilities.

## Setup

Python 3.10+ is required. From this project directory:

```shell
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows Command Prompt, or `source .venv/bin/activate` on Linux/macOS, then install:

```shell
python -m pip install -e ".[test]"
```

Keep the data locally in `ieee-fraud-detection/` (or pass `--data-dir`):

```text
ieee-fraud-detection/
  train_transaction.csv
  train_identity.csv
  test_transaction.csv
  test_identity.csv
  sample_submission.csv
```

The datasets and generated model/prediction artifacts are excluded from Git. Do not upload the competition data; obtain it separately under its applicable terms.

## Train, evaluate, and submit

Full training, with an 80/20 chronological holdout and 100 boosting iterations:

```shell
python -m fraud_detection train --data-dir ieee-fraud-detection --output-dir artifacts
```

For a faster first run:

```shell
python -m fraud_detection train --train-rows 50000 --max-iter 30 --output-dir artifacts-smoke
```

`--train-rows` selects the **first N training transactions**, not a random sample. This is a development smoke run, not a full-data performance estimate. All test transactions are still predicted, so its submission has complete coverage. Omit the flag for full training. The CLI logs progress and prints a JSON report.

Options: `--validation-fraction` (default `0.2`), `--max-iter` (`100`), `--seed` (`42`), `--threads` (`4`), `--chunk-size` (`20000`). Training is in-memory; the full dataset and intermediate transforms can require several GB of RAM. Test transactions are predicted in chunks; reduce `--chunk-size` for lower prediction memory use. Identity data is loaded in full. Both holdout partitions must contain fraud and non-fraud examples; very small row limits may fail this check.

### Outputs

| File | Contents |
| --- | --- |
| `metrics.json` | Holdout ROC-AUC, average precision, log loss, class prevalence, constant baselines, split boundaries, row counts, settings, library versions |
| `validation_predictions.csv` | Holdout `TransactionID`, observed `isFraud`, predicted `fraud_probability` |
| `model.joblib` | Final fitted preprocessing + classifier, refitted on all loaded training rows |
| `submission.csv` | Exactly `TransactionID,isFraud`, with probabilities aligned to sample-submission ID order |

The pipeline rejects duplicate or null IDs, invalid labels, schema mismatches, incomplete prediction coverage, and non-finite/out-of-range probabilities. Existing outputs at the chosen paths are overwritten on a successful rerun; use separate output directories for experiments.

## Predict with a saved model

```shell
python -m fraud_detection predict --model artifacts/model.joblib --data-dir ieee-fraud-detection --output artifacts/submission-reloaded.csv
```

**Only load model files you trust.** Joblib/pickle loading can execute arbitrary Python code. Use the same dependency versions as training (recorded in `metrics.json`) when reloading a model. The installed `fraud-detection` command is equivalent to `python -m fraud_detection`.

## Modeling choices and limitations

- Left-join identity onto transactions by `TransactionID`, preserving transactions without identities. Normalize test identity names such as `id-01` to `id_01` before joining.
- Read known IEEE categorical fields as strings and numeric features as float32 for consistent chunk schemas and lower memory. Other feature names are assumed numeric. Transaction IDs remain int64.
- Exclude `TransactionID` and `isFraud` from features. Sort by `TransactionDT`; the latest fraction is held out. Keep equal timestamps on the same side of the split.
- Fit categorical encoding only on the training partition during evaluation. Unknown and missing categories map to `-1`; numeric missing values are handled natively by the classifier. Entirely missing columns are supported.
- Ordinal category codes are treated as ordered numeric features: a simple baseline, not native categorical boosting. Numeric-coded cards and addresses also remain numeric.
- Disable automatic early stopping to avoid adding a random internal validation split. After reporting holdout metrics, clone and refit the whole pipeline (including encoding) on all loaded training rows.
- Use ROC-AUC and average precision rather than accuracy for the imbalanced target. Constant-score average precision equals validation fraud prevalence. No test labels, submission placeholders, or validation labels are used to fit the holdout model.

This is a starting point, not a production fraud decision system or a promise of competition performance. A single time holdout does not establish robustness across future periods. Further work could include rolling temporal validation, feature engineering, native categorical boosting, probability calibration, and threshold/cost analysis. There is no API or dashboard in this scope.

## Tests

```shell
python -m pytest -q
```

Tests use small synthetic IEEE-shaped CSVs (no competition data required), including the actual CLI and saved-model prediction path. They cover identity naming/joins, missing/unseen categories, train-only encoding, chronological separation, reproducibility, row-limited training with full test coverage, submission order, and invalid inputs.

## Verified baseline run

The default full-data command above completed locally with Python 3.13.15, scikit-learn 1.5.2, pandas 2.2.3, and NumPy 2.1.1:

- 590,540 training rows; holdout model trained on 472,432 and evaluated on 118,108 later rows.
- Holdout ROC-AUC: **0.898514**; average precision: **0.516678**; log loss: **0.092424**.
- Holdout fraud prevalence / constant-score average precision: **0.034409**.
- Final model refitted on all 590,540 rows with 432 features and 100 boosting iterations.
- All **506,691** test transactions predicted. Submission IDs/order and probability bounds were independently checked.
- Reloading `model.joblib` and predicting with 10,000-row chunks reproduced the submission exactly. Holdout metrics were independently recomputed from the saved validation predictions.
- **31 tests passed**; editable package installation and wheel build succeeded (`python -m pip wheel . --no-deps --wheel-dir dist`).

Environment notes: joblib warned that Windows physical-core detection returned zero and fell back to logical cores; training and prediction still completed with the configured thread limit. The standalone `fraud-detection.exe --help` launcher was blocked with Windows “Access denied,” so that launcher is **unverified** here; the documented `python -m fraud_detection` entry point was successfully exercised for both training and prediction. For Duo execution environments, missing tooling can be provisioned through `.gitlab/duo/agent-config.yml`; local Windows executable permissions may require administrator assistance.
