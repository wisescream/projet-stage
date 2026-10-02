"""Temporal model validation, probability calibration, and decision policy reports."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_score, recall_score, roc_auc_score)

EPSILON = 1e-6
DEFAULT_REVIEW_RATES = (0.01, 0.02, 0.05, 0.10)
DEFAULT_COSTS = {
    "low": {"missed_fraud": 50.0, "manual_review": 1.0, "false_decline": 10.0},
    "medium": {"missed_fraud": 250.0, "manual_review": 3.0, "false_decline": 25.0},
    "high": {"missed_fraud": 1000.0, "manual_review": 5.0, "false_decline": 100.0},
}


class PlattCalibratedModel:
    """Calibrated probability wrapper retaining the sklearn pipeline interface."""

    def __init__(self, estimator, calibrator):
        self.estimator = estimator
        self.calibrator = calibrator

    @property
    def feature_names_in_(self):
        return self.estimator.feature_names_in_

    @property
    def named_steps(self):
        return self.estimator.named_steps

    def raw_probabilities(self, frame):
        return self.estimator.predict_proba(frame)[:, 1]

    def predict_proba(self, frame):
        calibrated = apply_platt(self.calibrator, self.raw_probabilities(frame))
        return np.column_stack((1 - calibrated, calibrated))


class IdentityCalibrator:
    """Conservative fallback when a calibration window cannot support fitting."""

    method = "identity_insufficient_calibration_data"

    def predict_proba(self, logits):
        probabilities = 1 / (1 + np.exp(-np.asarray(logits, dtype=float).reshape(-1)))
        return np.column_stack((1 - probabilities, probabilities))


def _logit(probabilities):
    clipped = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_platt(probabilities, labels):
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("Calibration data must contain both target classes")
    if len(labels) < 20:
        return IdentityCalibrator()
    calibrator = LogisticRegression(solver="lbfgs", random_state=42)
    calibrator.fit(_logit(probabilities), labels)
    if calibrator.coef_[0, 0] <= 0:
        return IdentityCalibrator()
    return calibrator


def apply_platt(calibrator, probabilities):
    return calibrator.predict_proba(_logit(probabilities))[:, 1]


def probability_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
    result = {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "fraud_rate": float(labels.mean()),
        "rows": int(len(labels)),
    }
    result["roc_auc"] = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None
    return result


def calibration_bins(labels, probabilities, *, bins=10):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    result = []
    for index in range(bins):
        mask = ((probabilities >= edges[index]) & (probabilities <= edges[index + 1])
                if index == bins - 1 else
                (probabilities >= edges[index]) & (probabilities < edges[index + 1]))
        if mask.any():
            result.append({"lower": float(edges[index]), "upper": float(edges[index + 1]),
                           "rows": int(mask.sum()), "mean_probability": float(probabilities[mask].mean()),
                           "observed_fraud_rate": float(labels[mask].mean())})
    return result


def threshold_cost_analysis(labels, probabilities, *, review_rates=DEFAULT_REVIEW_RATES,
                            decline_rate=0.002, costs=DEFAULT_COSTS):
    """Rank by risk and allocate review/decline capacity without tuning on labels."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must have equal, non-zero lengths")
    if not 0 <= decline_rate < 1:
        raise ValueError("decline_rate must be in [0, 1)")
    if any(not 0 < rate <= 1 for rate in review_rates):
        raise ValueError("review rates must be in (0, 1]")

    ranking = np.argsort(-probabilities, kind="stable")
    total = len(labels)
    declines = min(int(np.ceil(total * decline_rate)), total) if decline_rate > 0 else 0
    rows = []
    for capacity in review_rates:
        review_total = min(max(int(np.ceil(total * capacity)), declines), total)
        outcomes = np.full(total, "approve", dtype=object)
        outcomes[ranking[:review_total]] = "manual_review"
        if declines:
            outcomes[ranking[:declines]] = "decline"
        review_mask = outcomes == "manual_review"
        decline_mask = outcomes == "decline"
        flagged = outcomes != "approve"
        missed = int(((outcomes == "approve") & (labels == 1)).sum())
        false_positive_reviews = int((review_mask & (labels == 0)).sum())
        false_declines = int((decline_mask & (labels == 0)).sum())
        true_fraud_flagged = int((flagged & (labels == 1)).sum())
        item = {
            "target_review_rate": float(capacity),
            "review_rate": float(review_mask.mean()),
            "decline_rate": float(decline_mask.mean()),
            "review_threshold": float(probabilities[ranking[review_total - 1]]),
            "decline_threshold": float(probabilities[ranking[declines - 1]]) if declines else None,
            "precision_flagged": float(precision_score(labels, flagged, zero_division=0)),
            "recall_flagged": float(recall_score(labels, flagged, zero_division=0)),
            "false_positive_reviews": false_positive_reviews,
            "missed_fraud_approved": missed,
            "false_positive_declines": false_declines,
            "transactions": total,
            "fraud_caught_by_review_or_decline": true_fraud_flagged,
            "cost_scenarios": {},
        }
        for name, assumption in costs.items():
            total_cost = (missed * assumption["missed_fraud"]
                          + int(review_mask.sum()) * assumption["manual_review"]
                          + false_declines * assumption["false_decline"])
            item["cost_scenarios"][name] = {
                "assumptions": assumption,
                "estimated_total_cost": float(total_cost),
                "estimated_cost_per_transaction": float(total_cost / total),
            }
        rows.append(item)
    return {"method": "score-rank capacity allocation; thresholds are not label-optimized",
            "decline_capacity": float(decline_rate), "scenarios": rows}


def rolling_temporal_evaluation(frame, *, folds=3, max_iter=100, seed=42, threads=4,
                               calibration_fraction=0.15):
    """Walk forward over later periods; each fold calibrates on its own earlier tail."""
    from threadpoolctl import threadpool_limits

    from fraud_detection.pipeline import TARGET, TIME, build_model, chronological_split, features

    if folds < 2:
        raise ValueError("folds must be at least 2")
    if not 0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be between 0 and 0.5")
    ordered = frame.sort_values(TIME, kind="stable").reset_index(drop=True)
    unique_times = np.sort(ordered[TIME].unique())
    warmup = max(2, len(unique_times) // 2)
    validation_times = np.array_split(unique_times[warmup:], folds)
    if len(validation_times) != folds or any(len(window) == 0 for window in validation_times):
        raise ValueError("Not enough distinct timestamps for the requested rolling folds")

    reports = []
    for fold_index, window in enumerate(validation_times, start=1):
        valid = ordered.loc[ordered[TIME].isin(window)].copy()
        train = ordered.loc[ordered[TIME] < window[0]].copy()
        if len(train) < 4 or train[TARGET].nunique() != 2 or valid[TARGET].nunique() != 2:
            raise ValueError(f"Fold {fold_index} needs both classes in training and validation")
        proper_train, calibration = chronological_split(train, calibration_fraction)
        model = build_model(features(frame).columns, max_iter=max_iter, seed=seed)
        with threadpool_limits(limits=threads):
            model.fit(features(proper_train), proper_train[TARGET])
            raw_calibration = model.predict_proba(features(calibration))[:, 1]
            calibrator = fit_platt(raw_calibration, calibration[TARGET])
            raw_validation = model.predict_proba(features(valid))[:, 1]
        calibrated = apply_platt(calibrator, raw_validation)
        reports.append({
            "fold": fold_index,
            "train_start_time": float(proper_train[TIME].min()),
            "train_end_time": float(proper_train[TIME].max()),
            "calibration_start_time": float(calibration[TIME].min()),
            "calibration_end_time": float(calibration[TIME].max()),
            "validation_start_time": float(valid[TIME].min()),
            "validation_end_time": float(valid[TIME].max()),
            "training_rows": int(len(proper_train)),
            "calibration_rows": int(len(calibration)),
            "validation_rows": int(len(valid)),
            "raw": probability_metrics(valid[TARGET], raw_validation),
            "calibrated": probability_metrics(valid[TARGET], calibrated),
            "calibration_bins": calibration_bins(valid[TARGET], calibrated),
            "threshold_cost_analysis": threshold_cost_analysis(valid[TARGET], calibrated),
        })
    return {"method": "expanding-window walk-forward with an inner chronological calibration tail",
            "folds": reports}