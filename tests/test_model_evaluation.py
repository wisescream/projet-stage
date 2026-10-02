import numpy as np
import pandas as pd

from fraud_detection.model_evaluation import (
    IdentityCalibrator,
    apply_platt,
    fit_platt,
    rolling_temporal_evaluation,
    threshold_cost_analysis,
)


def test_platt_calibration_is_monotonic_and_small_samples_fall_back():
    probabilities = np.linspace(0.05, 0.95, 100)
    labels = (probabilities > 0.7).astype(int)
    calibrator = fit_platt(probabilities, labels)
    calibrated = apply_platt(calibrator, probabilities)
    assert np.isfinite(calibrated).all()
    assert (np.diff(calibrated) >= 0).all()
    assert isinstance(fit_platt([0.1, 0.9], [0, 1]), IdentityCalibrator)


def test_threshold_report_is_capacity_based_and_reports_fraud_costs():
    labels = np.array([1] * 10 + [0] * 90)
    probabilities = np.linspace(0.99, 0.01, 100)
    report = threshold_cost_analysis(labels, probabilities, review_rates=(0.1,), decline_rate=0.01)
    scenario = report["scenarios"][0]
    assert report["method"].startswith("score-rank capacity")
    assert scenario["decline_rate"] == 0.01
    assert scenario["review_rate"] == 0.09
    assert scenario["precision_flagged"] == 1
    assert scenario["recall_flagged"] == 1
    assert scenario["missed_fraud_approved"] == 0
    assert set(scenario["cost_scenarios"]) == {"low", "medium", "high"}


def test_rolling_evaluation_reports_successive_periods_and_calibration():
    labels = (np.arange(240) % 10 == 0).astype(int)
    frame = pd.DataFrame({
        "TransactionDT": np.arange(240),
        "isFraud": labels,
        "TransactionAmt": labels * 100 + np.arange(240) % 17,
    })
    report = rolling_temporal_evaluation(frame, folds=3, max_iter=4, threads=1)
    folds = report["folds"]
    assert len(folds) == 3
    assert all(a["validation_end_time"] < b["validation_start_time"] for a, b in zip(folds, folds[1:]))
    assert all(fold["training_rows"] < fold["calibration_rows"] + fold["training_rows"]
               for fold in folds)
    assert all("average_precision" in fold["calibrated"] and "brier_score" in fold["raw"] for fold in folds)
    assert all("threshold_cost_analysis" in fold for fold in folds)
