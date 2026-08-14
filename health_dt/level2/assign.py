"""Assign phenotypes to synthetic cohorts using the Level 1 classifier.

Each synthpop output is one-hot encoded to match the training design matrix,
scored, and written back with a ``cluster`` column that the simulation uses to
pick a transition matrix.

The alignment step matters: if a synthetic cohort lacks a rare category — very
possible after an age shift — a plain ``get_dummies`` produces different columns
from training, and the prediction is silently wrong. The design matrix is
therefore reindexed to the classifier's stored column order.
"""

import logging

import numpy as np
import pandas as pd

from health_dt import config

logger = logging.getLogger(__name__)


def align_design_matrix(df: pd.DataFrame, feature_names: list[str], context: str = "") -> pd.DataFrame:
    """One-hot encode and reindex to the classifier's training columns.

    Missing indicator columns (a category this cohort happens not to contain)
    are filled with 0, which is the correct encoding for "no patient in that
    category". Columns unknown to the model are dropped with a warning — keeping
    them would shift the column order.
    """
    cat_cols = [c for c in config.PATIENT_CAT_COLUMNS if c in df.columns]
    encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    missing = [c for c in feature_names if c not in encoded.columns]
    extra = [c for c in encoded.columns if c not in feature_names]
    if missing:
        logger.warning("%s: %d training feature(s) absent, filled with 0: %s", context, len(missing), missing[:6])
    if extra:
        logger.warning("%s: %d unseen column(s) dropped: %s", context, len(extra), extra[:6])

    return encoded.reindex(columns=feature_names, fill_value=0)


def assign_clusters(df: pd.DataFrame, bundle: dict, context: str = "") -> pd.DataFrame:
    """Return a copy of df with a 0-indexed ``cluster`` column appended."""
    X = align_design_matrix(df, bundle["feature_names"], context)
    clusters = np.asarray(bundle["model"].predict(X)).astype(int)

    out = df.copy()
    out["cluster"] = clusters
    shares = pd.Series(clusters).value_counts(normalize=True).sort_index()
    logger.info("  phenotype mix: %s", {int(k): round(float(v), 3) for k, v in shares.items()})
    return out
