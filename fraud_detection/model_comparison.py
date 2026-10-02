"""Optional supervised-model comparison on a chronological validation split.

The baseline is always available. XGBoost and LightGBM are deliberately optional
because their native wheels are platform and CUDA dependent.
"""
import importlib.util

from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def available_models():
    names = ["hist_gradient_boosting"]
    if importlib.util.find_spec("xgboost"):
        names.append("xgboost")
    if importlib.util.find_spec("lightgbm"):
        names.append("lightgbm")
    return names


def build_optional_model(name, *, seed=42):
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=.08,
                             subsample=.8, colsample_bytree=.8, eval_metric="logloss",
                             random_state=seed, n_jobs=4)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=31,
                              subsample=.8, colsample_bytree=.8, random_state=seed,
                              verbosity=-1)
    raise ValueError(f"Optional model is unavailable or unsupported: {name}")


def evaluate_probabilities(labels, probabilities):
    return {"roc_auc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(average_precision_score(labels, probabilities)),
            "log_loss": float(log_loss(labels, probabilities))}