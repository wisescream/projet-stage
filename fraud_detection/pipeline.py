"""Load IEEE-CIS data, evaluate without future leakage, and predict fraud risk."""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from threadpoolctl import threadpool_limits

from fraud_detection.model_evaluation import (PlattCalibratedModel, apply_platt, calibration_bins,
                                              fit_platt, threshold_cost_analysis)
from fraud_detection.model_lifecycle import write_model_manifest

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
    if len(frame) >= 1000:
        fit_train, calibration = chronological_split(train, 0.05)
    else:
        fit_train, calibration = train, None
    model = build_model(features(frame).columns, max_iter=max_iter, seed=seed)
    with threadpool_limits(limits=threads):
        LOG.info("Fitting validation model on %s rows, calibrating on %s, validating on %s later rows",
                 len(fit_train), len(calibration) if calibration is not None else 0, len(valid))
        model.fit(features(fit_train), fit_train[TARGET])
        if calibration is not None:
            calibration_raw = model.predict_proba(features(calibration))[:, 1]
            calibrator = fit_platt(calibration_raw, calibration[TARGET])
            calibration_max_time = float(calibration[TIME].max())
        else:
            calibrator = fit_platt([0.5, 0.5], [0, 1])
            calibration_max_time = None
        raw_probabilities = model.predict_proba(features(valid))[:, 1]
        probabilities = apply_platt(calibrator, raw_probabilities)
        validation_model = PlattCalibratedModel(model, calibrator)
        report = {
            "validation": {
                "roc_auc": float(roc_auc_score(valid[TARGET], probabilities)),
                "average_precision": float(average_precision_score(valid[TARGET], probabilities)),
                "log_loss": float(log_loss(valid[TARGET], probabilities)),
                "brier_score_raw": float(brier_score_loss(valid[TARGET], raw_probabilities)),
                "brier_score_calibrated": float(brier_score_loss(valid[TARGET], probabilities)),
                "fraud_rate": float(valid[TARGET].mean()),
                "constant_baseline_roc_auc": 0.5,
                "constant_baseline_average_precision": float(valid[TARGET].mean()),
                "train_rows": len(fit_train), "calibration_rows": len(calibration) if calibration is not None else 0,
                "validation_rows": len(valid),
                "train_max_time": float(fit_train[TIME].max()),
                "calibration_max_time": calibration_max_time,
                "calibration_method": getattr(calibrator, "method", "platt_sigmoid"),
                "validation_min_time": float(valid[TIME].min()),
                "calibration_bins": calibration_bins(valid[TARGET], probabilities),
                "threshold_cost_analysis": threshold_cost_analysis(valid[TARGET], probabilities),
            },
            "training": {
                "rows_used": len(frame), "requested_row_limit": train_rows,
                "validation_fraction": validation_fraction, "max_iter": max_iter,
                "seed": seed, "threads": threads, "feature_count": len(model.feature_names_in_),
                "split": "chronological", "refit_on_all_loaded_rows": True,
                "training_time_min": float(frame[TIME].min()),
                "training_time_max": float(frame[TIME].max()),
            },
            "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        }
        LOG.info("Validation ROC-AUC %.6f; average precision %.6f", report["validation"]["roc_auc"], report["validation"]["average_precision"])
        validation_predictions = valid[[ID, TARGET]].copy()
        validation_predictions["fraud_probability"] = probabilities
        del train, fit_train, valid
        if calibration is not None:
            del calibration
        LOG.info("Refitting a fresh pipeline on all %s loaded training rows", len(frame))
        final_model = clone(model)
        # Replay must use the holdout model, never the model refitted on its labels.
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(validation_model, output_dir / "model-validation.joblib", compress=3)
        del model
        final_model.fit(features(frame), frame[TARGET])
        del frame
        output_dir.mkdir(parents=True, exist_ok=True)
        report["submission_rows"] = generate_submission(final_model, data_dir, output_dir / "submission.csv", chunk_size=chunk_size)
        joblib.dump(final_model, output_dir / "model.joblib", compress=3)
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_model_manifest(output_dir / "model.joblib", output_dir, metrics=report["validation"],
                         feature_names=final_model.feature_names_in_,
                         training_period={"start": report["training"]["training_time_min"],
                                          "end": report["training"]["training_time_max"],
                                          "rows": report["training"]["rows_used"]},
                         calibration_method=report["validation"]["calibration_method"])
    return report


def evaluate_rolling_pipeline(data_dir, output_dir, *, train_rows=None, folds=3,
                              max_iter=100, seed=42, threads=4):
    from fraud_detection.model_evaluation import rolling_temporal_evaluation

    if train_rows is not None and train_rows < 1:
        raise ValueError("train_rows must be positive")
    if max_iter < 1 or threads < 1:
        raise ValueError("max_iter and threads must be positive")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    frame = merge_identity(
        read_table(data_dir / "train_transaction.csv", nrows=train_rows),
        load_identity(data_dir / "train_identity.csv"),
    )
    report = rolling_temporal_evaluation(frame, folds=folds, max_iter=max_iter, seed=seed, threads=threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rolling-evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    periods = []
    for fold in report["folds"]:
        periods.append({"fold": fold["fold"], "validation_start_time": fold["validation_start_time"],
                        "validation_end_time": fold["validation_end_time"], "validation_rows": fold["validation_rows"],
                        "fraud_rate": fold["calibrated"]["fraud_rate"],
                        "raw_average_precision": fold["raw"]["average_precision"],
                        "calibrated_average_precision": fold["calibrated"]["average_precision"],
                        "raw_brier_score": fold["raw"]["brier_score"],
                        "calibrated_brier_score": fold["calibrated"]["brier_score"]})
    pd.DataFrame(periods).to_csv(output_dir / "rolling-periods.csv", index=False)
    return report


def predict_pipeline(model_path, data_dir, output_path, *, threads=4, chunk_size=20000):
    """Load only trusted model artifacts: joblib loading can execute Python code."""
    if threads < 1:
        raise ValueError("threads must be positive")
    model = joblib.load(model_path)
    with threadpool_limits(limits=threads):
        return generate_submission(model, data_dir, output_path, chunk_size=chunk_size)
