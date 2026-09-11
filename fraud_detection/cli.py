"""Command-line entry points for training and prediction."""

import argparse
import json
import logging
from pathlib import Path

from fraud_detection.pipeline import predict_pipeline, train_pipeline


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
    for command in (train, predict):
        command.add_argument("--data-dir", type=Path, default=Path("ieee-fraud-detection"))
        command.add_argument("--threads", type=positive_int, default=4)
        command.add_argument("--chunk-size", type=positive_int, default=20000)
    train.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    train.add_argument("--train-rows", type=positive_int, help="Use the first N training rows for a smoke run; test predictions remain complete")
    train.add_argument("--validation-fraction", type=fraction, default=0.2)
    train.add_argument("--max-iter", type=positive_int, default=100)
    train.add_argument("--seed", type=int, default=42)
    predict.add_argument("--model", type=Path, required=True, help="Trusted model.joblib produced by train")
    predict.add_argument("--output", type=Path, default=Path("artifacts/submission.csv"))
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if command == "train":
            print(json.dumps(train_pipeline(**args), indent=2))
        else:
            args["model_path"] = args.pop("model")
            args["output_path"] = args.pop("output")
            predict_pipeline(**args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
