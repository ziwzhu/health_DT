"""Train and select the phenotype classifier.

This is the bridge between the real and synthetic worlds: it learns
characteristics -> phenotype on observed patients so synthetic patients, who
have no observed sequence, can be assigned a phenotype and therefore a
transition matrix. Its accuracy is a ceiling on how faithfully any scenario can
be simulated, which is why three candidates are compared rather than one
assumed.
"""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from health_dt import config

logger = logging.getLogger(__name__)


def build_models() -> dict:
    """Instantiate the candidate estimators named in config.CLASSIFIER_MODELS.

    To add a model, add an entry here. It needs only ``fit``/``predict``; wrap
    a non-sklearn estimator in a class exposing those two methods.
    """
    catalogue = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=config.RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_SEED
        ),
    }
    unknown = set(config.CLASSIFIER_MODELS) - set(catalogue)
    if unknown:
        raise ValueError(f"unknown model(s) {sorted(unknown)}; known: {sorted(catalogue)}")
    return {name: catalogue[name] for name in config.CLASSIFIER_MODELS}


def train_and_select(X: pd.DataFrame, y: pd.Series) -> tuple[str, object, dict]:
    """Fit every candidate on a stratified split, keep the best macro F1.

    Macro F1 rather than accuracy: the smallest phenotype is the clinically
    interesting one, and accuracy would let a model ignore it.

    Returns
    -------
    (best_name, best_model, scores)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )
    logger.info("Train %d, test %d, features %d", len(X_train), len(X_test), X.shape[1])

    models = build_models()
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores[name] = float(f1_score(y_test, y_pred, average="macro"))

        logger.info("\n--- %s ---", name)
        logger.info("%s", classification_report(y_test, y_pred, digits=3))
        logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))

    best_name = max(scores, key=scores.get)
    logger.info("\nMacro F1 by model:")
    for name, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        logger.info("  %-22s %.4f%s", name, score, "  <- selected" if name == best_name else "")

    return best_name, models[best_name], scores


def feature_importances(model, feature_names: list[str]) -> pd.Series | None:
    """Extract a comparable importance vector from whichever model won.

    Tree ensembles expose ``feature_importances_``; the logistic pipeline is
    summarised by the mean absolute coefficient across classes. Returns None for
    estimators offering neither, which is not an error.
    """
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf"), "coef_"):
        values = np.mean(np.abs(model.named_steps["clf"].coef_), axis=0)
    else:
        return None
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def save_model(model, feature_names: list[str], name: str, path) -> None:
    """Persist the classifier together with its training column order.

    The column order matters: scoring a synthetic cohort requires exactly these
    features in exactly this order, and a cohort missing a rare category would
    otherwise produce a differently shaped matrix and a silently wrong
    prediction. See level2.assign.
    """
    joblib.dump({"model": model, "model_name": name, "feature_names": feature_names}, path)
    logger.info("Saved %s (%s)", path, name)


def load_model(path) -> dict:
    """Load a saved classifier bundle."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(
            f"{path} is not a model bundle. Re-run run_level1.py to regenerate it."
        )
    return bundle
