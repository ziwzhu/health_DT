"""Prepare the modelling matrices.

Two artefacts come out of this, from the same source frame:

``processed_data.csv``
    Fully encoded classifier input — one-hot categoricals, binarised
    comorbidities, integer targets.

``processed_synth_raw_data.csv``
    The same patients with categoricals left as labelled groups. synthpop needs
    real factor levels to build its CART sequence; one-hot encoding here would
    destroy the joint structure it exists to learn.
"""

import logging

import pandas as pd

from health_dt import config

logger = logging.getLogger(__name__)


def regroup_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse high-cardinality SEER codes into analysis groups.

    Each source column is replaced by a GRP_-prefixed grouped version. Unmapped
    and missing codes fall into the group's documented default rather than
    becoming NaN. Maps live in config.CATEGORICAL_REGROUPINGS.
    """
    out = df.copy()
    for source, (mapping, default, renamed) in config.CATEGORICAL_REGROUPINGS.items():
        if source not in out.columns:
            logger.warning("column %s not found; skipping regrouping", source)
            continue

        unmapped = set(out[source].dropna().unique()) - set(mapping)
        if unmapped:
            logger.warning("%s: codes %s unmapped, using %r", source, sorted(unmapped), default)

        out[renamed] = out[source].map(mapping).fillna(default)
        out = out.drop(columns=[source])
        logger.info("%s -> %s: %d groups", source, renamed, out[renamed].nunique())
    return out


def encode_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Map each cluster label column to contiguous integers 0..K-1.

    The primary label becomes ``target``; alternatives become
    ``target_<original name>``. Labels are sorted before enumeration, so R's
    1/2/3 becomes 0/1/2 and the low/medium/high ordering is preserved.

    Returns
    -------
    (frame, label_maps)
        The frame with target columns appended, and the original-to-integer
        mapping for each (worth recording in the run log).
    """
    out = df.copy()
    label_maps = {}

    for source in [config.LABEL_COLUMN, *config.ALT_LABEL_COLUMNS]:
        if source not in out.columns:
            continue
        mapping = {label: i for i, label in enumerate(sorted(out[source].dropna().unique()))}
        name = "target" if source == config.LABEL_COLUMN else f"target_{source}"
        out[name] = out[source].map(mapping)
        label_maps[name] = mapping
        logger.info("Target %s -> %s: %s", source, name, mapping)

    if not label_maps:
        raise ValueError(
            f"no label columns found; expected {config.LABEL_COLUMN} in the input"
        )
    return out, label_maps


def binarise_comorbidities(df: pd.DataFrame) -> pd.DataFrame:
    """Convert CCW comorbidity flags to 0/1 and add ``comorbid_count``.

    The criterion is ``== 3``, not ``> 0``. In SEER-Medicare, 3 means "met
    claims criteria with sufficient FFS coverage"; other non-zero codes mean the
    coverage requirement was not met, so ``> 0`` silently counts patients of
    unknown status as having the condition.
    """
    out = df.copy()
    present = [c for c in config.COMORBIDITY_COLUMNS if c in out.columns]
    missing = [c for c in config.COMORBIDITY_COLUMNS if c not in out.columns]
    if missing:
        logger.warning("comorbidity columns absent: %s", missing)

    for col in present:
        out[col] = (out[col] == config.COMORBIDITY_PRESENT_CODE).astype(int)
    out["comorbid_count"] = out[present].sum(axis=1)
    return out


def build_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Produce the fully encoded classifier input.

    Missing categoricals are filled with the literal string "Missing" — an
    informative level, not a dropped row — then one-hot encoded.
    """
    frame, _ = encode_targets(df)
    frame = binarise_comorbidities(frame)

    logger.info(
        "Comorbidity count by cluster:\n%s",
        frame.groupby(config.LABEL_COLUMN)["comorbid_count"].agg(["mean", "std"]).to_string(),
    )
    _audit_cardinality(frame)

    cat_cols = [c for c in config.PATIENT_CAT_COLUMNS if c in frame.columns]
    for col in cat_cols:
        frame[col] = frame[col].fillna("Missing").astype(str)
    frame = pd.get_dummies(frame, columns=cat_cols, drop_first=False)

    onehot = [c for c in frame.columns if any(c.startswith(f"{cat}_") for cat in cat_cols)]
    features = (
        [c for c in config.PATIENT_BIN_COLUMNS if c in frame.columns]
        + [c for c in config.PATIENT_NUM_COLUMNS if c in frame.columns]
        + [c for c in config.COMORBIDITY_COLUMNS if c in frame.columns]
        + ["comorbid_count"]
        + onehot
    )
    targets = [c for c in frame.columns if c.startswith("target")]

    out = frame[features + targets].copy()
    logger.info("Model matrix: %d rows, %d features", len(out), len(features))
    return out


def build_synthesis_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Produce the synthpop input: grouped categoricals, no one-hot encoding.

    synthpop's CART synthesis conditions each variable on the previously
    synthesised ones. Handing it 60 sparse indicators instead of 12 factors
    would fragment those conditional distributions.
    """
    out = binarise_comorbidities(df)
    cat_cols = [c for c in config.PATIENT_CAT_COLUMNS if c in out.columns]
    for col in cat_cols:
        out[col] = out[col].fillna("Missing").astype(str)

    keep = (
        cat_cols
        + [c for c in config.PATIENT_BIN_COLUMNS if c in out.columns]
        + [c for c in config.PATIENT_NUM_COLUMNS if c in out.columns]
        + [c for c in config.COMORBIDITY_COLUMNS if c in out.columns]
        + ["comorbid_count"]
    )
    logger.info("Synthesis frame: %d rows, %d columns", len(out), len(keep))
    return out[keep].copy()


def _audit_cardinality(df: pd.DataFrame, max_levels: int = 10) -> None:
    """Warn about categoricals with more levels than expected.

    High cardinality blows up the one-hot matrix and usually means a regrouping
    map in config needs extending.
    """
    for col in config.PATIENT_CAT_COLUMNS:
        if col in df.columns and df[col].nunique(dropna=False) > max_levels:
            logger.warning("%s has %d levels (> %d)", col, df[col].nunique(dropna=False), max_levels)
