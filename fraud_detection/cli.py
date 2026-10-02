"""Command-line entry points for training and prediction."""

import argparse
import json
import logging
from pathlib import Path

from fraud_detection.model_lifecycle import active_model_path, promote_model, rollback_model
from fraud_detection.pipeline import evaluate_rolling_pipeline, predict_pipeline, train_pipeline


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def fraction(value):
    number = float(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="IEEE-CIS fraud detection baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Evaluate, refit, save a model, and generate submission.csv")
    predict = commands.add_parser("predict", help="Generate a submission using a trusted saved model")
    rolling = commands.add_parser("evaluate-rolling", help="Run calibrated walk-forward evaluation and cost analysis")
    promote = commands.add_parser("model-promote", help="Promote a hash-validated model artifact into the local registry")
    rollback = commands.add_parser("model-rollback", help="Roll back the active model to the previous version")
    active = commands.add_parser("model-active", help="Print the active model artifact path")
    for command in (train, predict, rolling):
        command.add_argument("--data-dir", type=Path, default=Path("ieee-fraud-detection"))
        command.add_argument("--threads", type=positive_int, default=4)
    for command in (train, predict):
        command.add_argument("--chunk-size", type=positive_int, default=20000)
    train.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    train.add_argument("--train-rows", type=positive_int, help="Use the first N training rows for a smoke run; test predictions remain complete")
    train.add_argument("--validation-fraction", type=fraction, default=0.2)
    train.add_argument("--max-iter", type=positive_int, default=100)
    train.add_argument("--seed", type=int, default=42)
    rolling.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    rolling.add_argument("--train-rows", type=positive_int, help="Use the first N rows for a development evaluation")
    rolling.add_argument("--folds", type=positive_int, default=3)
    rolling.add_argument("--max-iter", type=positive_int, default=100)
    rolling.add_argument("--seed", type=int, default=42)
    promote.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    promote.add_argument("--manifest", type=Path, default=Path("artifacts/model-manifest.json"))
    promote.add_argument("--registry-dir", type=Path, default=Path("artifacts/model-registry"))
    rollback.add_argument("--registry-dir", type=Path, default=Path("artifacts/model-registry"))
    active.add_argument("--registry-dir", type=Path, default=Path("artifacts/model-registry"))
    predict.add_argument("--model", type=Path, required=True, help="Trusted model.joblib produced by train")
    predict.add_argument("--output", type=Path, default=Path("artifacts/submission.csv"))
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if command == "train":
            print(json.dumps(train_pipeline(**args), indent=2))
        elif command == "evaluate-rolling":
            print(json.dumps(evaluate_rolling_pipeline(**args), indent=2))
        elif command == "model-promote":
            print(json.dumps(promote_model(args["model"], args["manifest"], args["registry_dir"]), indent=2))
        elif command == "model-rollback":
            print(json.dumps(rollback_model(args["registry_dir"]), indent=2))
        elif command == "model-active":
            print(json.dumps({"active_model": str(active_model_path(args["registry_dir"]))}, indent=2))
        else:
            args["model_path"] = args.pop("model")
            args["output_path"] = args.pop("output")
            predict_pipeline(**args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
