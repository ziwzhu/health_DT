"""Decode weekly event strings into integer state codes.

The SAS export stores each follow-up week as a 6-digit binary string, one bit
per provider type. This turns ``event1..event52`` into an ``(n_patients, 52)``
integer array, which is the currency of every later stage.
"""

import logging

import numpy as np
import pandas as pd

from health_dt import config

logger = logging.getLogger(__name__)


def encode_sequences(df: pd.DataFrame) -> np.ndarray:
    """Map event pattern strings to weekly state codes.

    Any pattern outside ``config.PATTERN_TO_STATE`` — the literal string
    "rare", an unexpected combination, a malformed cell — becomes RARE_STATE.
    That catch-all is deliberate: the ten mapped patterns plus Rare cover 100%
    of observed weeks, and dropping the remainder would bias the transition
    matrices.

    Parameters
    ----------
    df:
        Frame containing the event columns.

    Returns
    -------
    np.ndarray
        ``(n_patients, n_weeks)`` array of state codes.
    """
    event_cols = [c for c in config.EVENT_COLUMNS if c in df.columns]
    if not event_cols:
        raise ValueError(
            f"no event columns found; expected event1..event{config.N_WEEKS}, "
            f"got columns like {list(df.columns)[:8]}"
        )
    if len(event_cols) < config.N_WEEKS:
        logger.warning(
            "only %d of %d event columns present", len(event_cols), config.N_WEEKS
        )

    sequences = np.empty((len(df), len(event_cols)), dtype=int)
    for j, col in enumerate(event_cols):
        cleaned = df[col].astype(str).str.strip()
        sequences[:, j] = (
            cleaned.map(config.PATTERN_TO_STATE).fillna(config.RARE_STATE).astype(int)
        )

    validate_sequences(sequences, n_weeks=len(event_cols))

    # Patterns are matched as zero-padded 6-character strings. If the CSV was
    # written with leading zeros stripped ("11" instead of "000011"), every week
    # collapses to Rare and the transition matrices become meaningless.
    rare_share = float((sequences == config.RARE_STATE).mean())
    if rare_share > 0.25:
        logger.warning(
            "%.1f%% of patient-weeks fell into Rare — check that event columns are "
            "zero-padded 6-character strings (e.g. '000011', not '11')",
            rare_share * 100,
        )

    logger.info("Encoded %d patients x %d weeks", *sequences.shape)
    for state in range(config.N_STATES):
        pct = (sequences == state).mean() * 100
        logger.info("  %2d %-30s %5.1f%%", state, config.STATE_LABELS[state], pct)
    return sequences


def validate_sequences(sequences: np.ndarray, n_weeks: int | None = None) -> np.ndarray:
    """Check a sequence array is well formed; return it unchanged.

    Called at every point sequences cross a module or file boundary.
    """
    n_weeks = config.N_WEEKS if n_weeks is None else n_weeks
    sequences = np.asarray(sequences)
    if sequences.ndim != 2:
        raise ValueError(f"sequences must be 2-D, got shape {sequences.shape}")
    if sequences.shape[1] != n_weeks:
        raise ValueError(f"sequences have {sequences.shape[1]} weeks, expected {n_weeks}")
    if sequences.size:
        lo, hi = int(sequences.min()), int(sequences.max())
        if lo < 0 or hi >= config.N_STATES:
            raise ValueError(f"state codes out of range [0, {config.N_STATES - 1}]: [{lo}, {hi}]")
    return sequences


def derive_features(sequences: np.ndarray) -> pd.DataFrame:
    """Patient-level features summarising a weekly sequence.

    Feeds the ABM's acuity score and the outcome validation tables.
    """
    validate_sequences(sequences, n_weeks=sequences.shape[1])
    is_hospital = np.isin(sequences, config.HOSPITAL_STATES)
    is_rare = sequences == config.RARE_STATE
    return pd.DataFrame(
        {
            "had_hospital_encounter": is_hospital.any(axis=1).astype(int),
            "n_hospital_weeks": is_hospital.sum(axis=1),
            "had_rare_pattern": is_rare.any(axis=1).astype(int),
            "n_rare_weeks": is_rare.sum(axis=1),
        }
    )


def load_labels(n_patients: int) -> np.ndarray:
    """Read the external clustering's labels and normalise to 0-indexed.

    The join is positional — row i of the label file must be the same patient as
    row i of the combined export — so the row counts are checked rather than
    assumed.
    """
    frame = pd.read_csv(config.CLUSTER_LABELS_CSV)
    if "cluster" not in frame.columns:
        raise ValueError(
            f"{config.CLUSTER_LABELS_CSV.name} has no 'cluster' column; "
            f"found {list(frame.columns)[:8]}"
        )
    if len(frame) != n_patients:
        raise ValueError(
            f"{config.CLUSTER_LABELS_CSV.name} has {len(frame)} rows but the combined "
            f"export has {n_patients}. These are joined positionally, so row order and "
            "count must match."
        )
    return normalise_labels(frame["cluster"].to_numpy())


def normalise_labels(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary cluster labels onto contiguous integers 0..K-1.

    The R clustering emits 1-indexed labels; everything in Python is 0-indexed.
    Sort order is preserved, so 1/2/3 (low/medium/high use) becomes 0/1/2 with
    the same meaning. This is the single place that translation happens.
    """
    labels = np.asarray(labels)
    unique = np.unique(labels)
    mapping = {value: index for index, value in enumerate(unique)}
    if not np.array_equal(unique, np.arange(len(unique))):
        logger.info("Remapped cluster labels %s -> %s", unique.tolist(), list(mapping.values()))
    return np.array([mapping[v] for v in labels], dtype=int)
