"""Load IEEE-CIS data, evaluate without future leakage, and predict fraud risk."""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from threadpoolctl import threadpool_limits

LOG = logging.getLogger(__name__)
ID = "TransactionID"
TARGET = "isFraud"
TIME = "TransactionDT"
CATEGORICAL = {
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo",
    *(f"M{i}" for i in range(1, 10)),
    *(f"id_{i:02}" for i in (12, 15, 16, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38)),
}


def normalize_name(name):
    return name.replace("id-", "id_", 1) if name.startswith("id-") else name


def read_table(path, *, nrows=None, chunksize=None):
    """Use the known IEEE schema to avoid chunk-dependent type inference."""
    columns = pd.read_csv(path, nrows=0).columns
    dtypes = {
        c: ("int64" if c == ID else str if normalize_name(c) in CATEGORICAL else "float32")
        for c in columns
    }
    return pd.read_csv(path, dtype=dtypes, nrows=nrows, chunksize=chunksize)


def check_ids(frame, name):
    if ID not in frame:
        raise ValueError(f"{name} must contain {ID}")
    if frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError(f"{name} must have unique, non-null {ID} values")


def load_identity(path):
    identity = read_table(path).rename(columns=normalize_name)
    if identity.columns.duplicated().any():
        raise ValueError("Identity columns collide after id- to id_ normalization")
    check_ids(identity, "identity")
    if TARGET in identity:
        raise ValueError("Identity data must not contain the target")
    return identity


def merge_identity(transactions, identity):
    """Left join; missing identity records must never remove transactions."""
    check_ids(transactions, "transactions")
    overlap = (set(transactions) & set(identity)) - {ID}
    if overlap:
        raise ValueError(f"Overlapping transaction/identity columns: {sorted(overlap)}")
    return transactions.merge(identity, on=ID, how="left", sort=False, validate="one_to_one")


def chronological_split(frame, validation_fraction):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if TIME not in frame or frame[TIME].isna().any() or not np.isfinite(frame[TIME]).all():
        raise ValueError(f"Training data must have finite, non-null {TIME}")
    if TARGET not in frame or frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("Training target isFraud must contain only non-null 0/1 values")
    ordered = frame.sort_values(TIME, kind="stable").reset_index(drop=True)
    boundary = int(len(ordered) * (1 - validation_fraction))
    if boundary == 0 or boundary >= len(ordered):
        raise ValueError("Not enough rows for a chronological validation split")
    # Keep equal timestamps together; no training timestamp reaches validation time.
    cutoff = ordered[TIME].iloc[boundary]
    train = ordered.loc[ordered[TIME] < cutoff]
    valid = ordered.loc[ordered[TIME] >= cutoff]
    if train[TARGET].nunique() != 2 or valid[TARGET].nunique() != 2:
        raise ValueError("Both chronological partitions need both target classes; use more rows or another validation fraction")
    return train, valid


def features(frame):
    return frame.drop(columns=[ID, TARGET], errors="ignore")


def category_values(frame):
    """Normalize all-missing and mixed string columns for OrdinalEncoder."""
    return frame.astype("string").to_numpy(dtype=object, na_value=np.nan)


def build_model(columns, *, max_iter=100, seed=42):
    categorical = [c for c in columns if c in CATEGORICAL]
    numeric = [c for c in columns if c not in CATEGORICAL]
    preprocessing = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", Pipeline([
            ("strings", FunctionTransformer(category_values)),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                      encoded_missing_value=-1, dtype=np.float32)),
        ]), categorical),
    ], sparse_threshold=0)
    return Pipeline([
        ("preprocess", preprocessing),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, random_state=seed,
        )),
    ])


def make_submission(template, predictions):
    """Align by ID, not row position, and reject partial or invalid predictions."""
    if list(template.columns) != [ID, TARGET]:
        raise ValueError("Sample submission must have columns TransactionID,isFraud in that order")
    check_ids(template, "sample submission")
    check_ids(predictions, "predictions")
    if len(template) == 0 or len(template) != len(predictions) or set(template[ID]) != set(predictions[ID]):
        raise ValueError("Prediction IDs must exactly match sample submission IDs")
    values = predictions.set_index(ID)[TARGET].reindex(template[ID]).to_numpy()
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Predictions must be finite probabilities between 0 and 1")
    result = template.copy()
    result[TARGET] = values
    return result


def generate_submission(model, data_dir, output_path, *, chunk_size=20000):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    data_dir, output_path = Path(data_dir), Path(output_path)
    identity = load_identity(data_dir / "test_identity.csv")
    predictions = []
    expected = list(model.feature_names_in_)
    for chunk in read_table(data_dir / "test_transaction.csv", chunksize=chunk_size):
        if TARGET in chunk:
            raise ValueError("Test transactions must not contain the target")
        test = features(merge_identity(chunk, identity))
        if set(test.columns) != set(expected):
            missing, extra = set(expected) - set(test), set(test) - set(expected)
            raise ValueError(f"Test feature schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        probabilities = model.predict_proba(test[expected])[:, 1]
        predictions.append(pd.DataFrame({ID: chunk[ID].to_numpy(), TARGET: probabilities}))
    if not predictions:
        raise ValueError("Test transactions are empty")
    result = make_submission(read_table(data_dir / "sample_submission.csv"), pd.concat(predictions, ignore_index=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    LOG.info("Wrote %s predictions to %s", len(result), output_path)
    return len(result)


def train_pipeline(data_dir, output_dir, *, train_rows=None, validation_fraction=0.2,
                   max_iter=100, seed=42, threads=4, chunk_size=20000):
    if train_rows is not None and train_rows < 1:
        raise ValueError("train_rows must be positive")
    if max_iter < 1 or threads < 1 or chunk_size < 1:
        raise ValueError("max_iter, threads, and chunk_size must be positive")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    LOG.info("Loading training transactions%s", f" (first {train_rows} rows)" if train_rows else " (all rows)")
    frame = merge_identity(
        read_table(data_dir / "train_transaction.csv", nrows=train_rows),
        load_identity(data_dir / "train_identity.csv"),
    )
    train, valid = chronological_split(frame, validation_fraction)
    model = build_model(features(frame).columns, max_iter=max_iter, seed=seed)
    with threadpool_limits(limits=threads):
        LOG.info("Fitting validation model on %s rows; validating on %s later rows", len(train), len(valid))
        model.fit(features(train), train[TARGET])
        probabilities = model.predict_proba(features(valid))[:, 1]
        report = {
            "validation": {
                "roc_auc": float(roc_auc_score(valid[TARGET], probabilities)),
                "average_precision": float(average_precision_score(valid[TARGET], probabilities)),
                "log_loss": float(log_loss(valid[TARGET], probabilities)),
                "fraud_rate": float(valid[TARGET].mean()),
                "constant_baseline_roc_auc": 0.5,
                "constant_baseline_average_precision": float(valid[TARGET].mean()),
                "train_rows": len(train), "validation_rows": len(valid),
                "train_max_time": float(train[TIME].max()),
                "validation_min_time": float(valid[TIME].min()),
            },
            "training": {
                "rows_used": len(frame), "requested_row_limit": train_rows,
                "validation_fraction": validation_fraction, "max_iter": max_iter,
                "seed": seed, "threads": threads, "feature_count": len(model.feature_names_in_),
                "split": "chronological", "refit_on_all_loaded_rows": True,
            },
            "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        }
        LOG.info("Validation ROC-AUC %.6f; average precision %.6f", report["validation"]["roc_auc"], report["validation"]["average_precision"])
        validation_predictions = valid[[ID, TARGET]].copy()
        validation_predictions["fraud_probability"] = probabilities
        del train, valid
        LOG.info("Refitting a fresh pipeline on all %s loaded training rows", len(frame))
        final_model = clone(model)
        del model
        final_model.fit(features(frame), frame[TARGET])
        del frame
        output_dir.mkdir(parents=True, exist_ok=True)
        report["submission_rows"] = generate_submission(final_model, data_dir, output_dir / "submission.csv", chunk_size=chunk_size)
        joblib.dump(final_model, output_dir / "model.joblib", compress=3)
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def predict_pipeline(model_path, data_dir, output_path, *, threads=4, chunk_size=20000):
    """Load only trusted model artifacts: joblib loading can execute Python code."""
    if threads < 1:
        raise ValueError("threads must be positive")
    model = joblib.load(model_path)
    with threadpool_limits(limits=threads):
        return generate_submission(model, data_dir, output_path, chunk_size=chunk_size)
