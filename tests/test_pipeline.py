import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from fraud_detection.cli import main
from fraud_detection.pipeline import (
    CATEGORICAL, ID, TARGET, TIME, build_model, chronological_split, features,
    generate_submission, load_identity, make_submission, merge_identity, read_table,
)


@pytest.fixture
def data_dir(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    labels = (np.arange(120) % 5 == 0).astype(int)
    train = pd.DataFrame({
        ID: np.arange(1000, 1120), TIME: np.arange(120), TARGET: labels,
        "TransactionAmt": labels * 100 + 10,
        "ProductCD": np.where(labels, "C", "W"),
        "M1": np.nan, "D1": np.nan,
    })
    train.to_csv(root / "train_transaction.csv", index=False)
    pd.DataFrame({ID: [1000, 1001, 1100, 1110], "id_01": [1, 2, 3, 4],
                  "id_31": ["browser", "browser", "future", "future"]}).to_csv(root / "train_identity.csv", index=False)
    pd.DataFrame({ID: [2003, 2001, 2002], TIME: [201, 202, 203],
                  "TransactionAmt": [110, 10, np.nan], "ProductCD": ["C", "new", np.nan],
                  "M1": ["T", np.nan, "F"], "D1": [1, np.nan, 2]}).to_csv(root / "test_transaction.csv", index=False)
    pd.DataFrame({ID: [2001], "id-01": [9], "id-31": ["unseen"]}).to_csv(root / "test_identity.csv", index=False)
    pd.DataFrame({ID: [2002, 2003, 2001], TARGET: [0.5] * 3}).to_csv(root / "sample_submission.csv", index=False)
    return root


def test_identity_normalization_and_left_join(data_dir):
    identity = load_identity(data_dir / "test_identity.csv")
    assert "id_01" in identity and "id_31" in identity
    assert "id-01" not in identity
    transactions = read_table(data_dir / "test_transaction.csv")
    merged = merge_identity(transactions, identity)
    assert merged[ID].tolist() == [2003, 2001, 2002]
    assert merged["id_01"].isna().tolist() == [True, False, True]
    assert merged["id_01"].iloc[1] == 9
    assert transactions[ID].dtype == np.dtype("int64")
    assert transactions[TIME].dtype == np.dtype("float32")


@pytest.mark.parametrize("ids", [[1, 1], [1, np.nan]])
def test_invalid_transaction_ids_rejected(ids):
    with pytest.raises(ValueError, match="unique, non-null"):
        merge_identity(pd.DataFrame({ID: ids}), pd.DataFrame({ID: [1]}))


def test_duplicate_identity_rejected(data_dir):
    path = data_dir / "test_identity.csv"
    pd.DataFrame({ID: [1, 1], "id-01": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_identity(path)


def test_colliding_identity_columns_rejected(data_dir):
    path = data_dir / "test_identity.csv"
    pd.DataFrame({ID: [1], "id-01": [0], "id_01": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="collide"):
        load_identity(path)


def test_chronological_split_keeps_ties_together():
    frame = pd.DataFrame({TIME: np.repeat(np.arange(10), 2), TARGET: [0, 1] * 10})
    train, valid = chronological_split(frame.sample(frac=1, random_state=3), 0.25)
    assert train[TIME].max() < valid[TIME].min()
    assert len(train) == 14 and len(valid) == 6
    assert train[TIME].is_monotonic_increasing


@pytest.mark.parametrize("fraction", [0, 1, -0.2, float("nan")])
def test_invalid_split_fraction(fraction):
    with pytest.raises(ValueError, match="validation_fraction"):
        chronological_split(pd.DataFrame(), fraction)


@pytest.mark.parametrize("target", [[0, 0, 0, 0], [0, 1, 2, 0], [0, 1, np.nan, 0]])
def test_invalid_or_single_class_targets(target):
    with pytest.raises(ValueError, match="target|classes"):
        chronological_split(pd.DataFrame({TIME: range(4), TARGET: target}), 0.5)


def test_invalid_times_rejected():
    with pytest.raises(ValueError, match="finite"):
        chronological_split(pd.DataFrame({TIME: [0, np.nan, 2, 3], TARGET: [0, 1, 0, 1]}), 0.5)


def test_preprocessing_is_fitted_only_on_training(data_dir):
    frame = merge_identity(read_table(data_dir / "train_transaction.csv"), load_identity(data_dir / "train_identity.csv"))
    train, valid = chronological_split(frame, 0.2)
    model = build_model(features(frame).columns, max_iter=3)
    with threadpool_limits(limits=1):
        model.fit(features(train), train[TARGET])
        probabilities = model.predict_proba(features(valid))[:, 1]
        test = features(merge_identity(read_table(data_dir / "test_transaction.csv"), load_identity(data_dir / "test_identity.csv")))
        assert np.isfinite(model.predict_proba(test)).all()
    assert np.isfinite(probabilities).all()
    assert ID not in model.feature_names_in_ and TARGET not in model.feature_names_in_
    categories = model.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encode"].categories_
    cat_columns = [c for c in model.feature_names_in_ if c in CATEGORICAL]
    assert "future" not in categories[cat_columns.index("id_31")]


def test_submission_reorders_by_id():
    template = pd.DataFrame({ID: [3, 1, 2], TARGET: [0.5] * 3})
    predictions = pd.DataFrame({ID: [1, 2, 3], TARGET: [0.1, 0.2, 0.3]})
    result = make_submission(template, predictions)
    assert result[ID].tolist() == [3, 1, 2]
    assert result[TARGET].tolist() == [0.3, 0.1, 0.2]
    assert template[TARGET].tolist() == [0.5] * 3


@pytest.mark.parametrize("ids,values", [
    ([1], [0.1]), ([1, 3], [0.1, 0.2]), ([1, 1], [0.1, 0.2]),
    ([1, 2], [np.nan, 0.2]), ([1, 2], [np.inf, 0.2]),
    ([1, 2], [-0.1, 0.2]), ([1, 2], [0.1, 1.2]),
])
def test_invalid_submissions_rejected(ids, values):
    template = pd.DataFrame({ID: [1, 2], TARGET: [0.5, 0.5]})
    with pytest.raises(ValueError):
        make_submission(template, pd.DataFrame({ID: ids, TARGET: values}))


def test_train_predict_cli_end_to_end(data_dir, tmp_path):
    output = tmp_path / "output"
    command = [sys.executable, "-m", "fraud_detection", "train", "--data-dir", str(data_dir),
               "--output-dir", str(output), "--max-iter", "8", "--threads", "1", "--chunk-size", "2"]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    report = json.loads(completed.stdout)
    assert report == json.loads((output / "metrics.json").read_text())
    assert report["training"]["rows_used"] == 120
    assert report["training"]["requested_row_limit"] is None
    assert report["training"]["refit_on_all_loaded_rows"] is True
    assert report["validation"]["roc_auc"] > 0.9
    assert report["validation"]["average_precision"] > report["validation"]["fraud_rate"]
    assert report["validation"]["train_max_time"] < report["validation"]["validation_min_time"]
    assert report["validation"]["calibration_method"] in {"platt_sigmoid", "identity_insufficient_calibration_data"}
    assert "threshold_cost_analysis" in report["validation"]
    assert report["submission_rows"] == 3
    manifest = json.loads((output / "model-manifest.json").read_text())
    assert manifest["sha256"] and manifest["feature_count"] == 7
    assert manifest["training_period"]["rows"] == 120
    validation = pd.read_csv(output / "validation_predictions.csv")
    assert len(validation) == 24 and validation[ID].tolist() == list(range(1096, 1120))
    submission = pd.read_csv(output / "submission.csv")
    assert submission.columns.tolist() == [ID, TARGET]
    assert submission[ID].tolist() == [2002, 2003, 2001]
    assert submission[TARGET].between(0, 1).all()
    model = joblib.load(output / "model.joblib")
    categories = model.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encode"].categories_
    assert any("future" in c for c in categories)  # final refit includes validation rows
    holdout_model = joblib.load(output / "model-validation.joblib")
    holdout_categories = holdout_model.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encode"].categories_
    assert not any("future" in c for c in holdout_categories)
    holdout_frame = merge_identity(read_table(data_dir / "train_transaction.csv"), load_identity(data_dir / "train_identity.csv"))
    _, holdout_rows = chronological_split(holdout_frame, 0.2)
    with threadpool_limits(limits=1):
        np.testing.assert_allclose(holdout_model.predict_proba(features(holdout_rows))[:, 1], validation["fraud_probability"])
    second = tmp_path / "reloaded.csv"
    subprocess.run([sys.executable, "-m", "fraud_detection", "predict", "--data-dir", str(data_dir),
                    "--model", str(output / "model.joblib"), "--output", str(second),
                    "--threads", "1", "--chunk-size", "1"], check=True, capture_output=True, text=True)
    pd.testing.assert_frame_equal(submission, pd.read_csv(second))
    # Repeating the entire training command yields the same probabilities.
    subprocess.run(command, check=True, capture_output=True, text=True)
    pd.testing.assert_frame_equal(submission, pd.read_csv(output / "submission.csv"))
    malformed = pd.read_csv(data_dir / "test_transaction.csv").drop(columns="D1")
    malformed.to_csv(data_dir / "test_transaction.csv", index=False)
    with pytest.raises(ValueError, match="schema mismatch"):
        generate_submission(model, data_dir, tmp_path / "bad.csv", chunk_size=2)
    assert not (tmp_path / "bad.csv").exists()


def test_row_limit_still_predicts_all_test_rows(data_dir, tmp_path, capsys):
    main(["train", "--data-dir", str(data_dir), "--output-dir", str(tmp_path / "limited"),
          "--train-rows", "60", "--max-iter", "2", "--threads", "1"])
    report = json.loads(capsys.readouterr().out)
    assert report["training"]["rows_used"] == report["training"]["requested_row_limit"] == 60
    assert report["submission_rows"] == 3


@pytest.mark.parametrize("flag,value", [("--train-rows", "0"), ("--threads", "-1"),
                                         ("--chunk-size", "0"), ("--max-iter", "0"),
                                         ("--validation-fraction", "1")])
def test_invalid_cli_arguments(flag, value):
    with pytest.raises(SystemExit) as error:
        main(["train", flag, value])
    assert error.value.code == 2


def test_missing_files_fail_clearly(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["train", "--data-dir", str(tmp_path)])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
