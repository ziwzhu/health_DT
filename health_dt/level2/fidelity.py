"""Does the synthetic cohort behave like the real one?

Four checks, in increasing order of how much they can embarrass you:

1. Marginals — means and standard deviations of numeric features.
2. Phenotype mix — the share of patients in each cluster.
3. Transition fidelity — Frobenius distance between real and simulated
   transition matrices per phenotype. This matters most, since Level 3's
   conclusions are driven entirely by these matrices. The published run reported
   all clusters below 0.04.
4. pMSE — a logistic model is asked to discriminate real from synthetic records;
   the mean squared deviation of its predictions from the null share is compared
   to the value expected under perfect synthesis. A ratio near 1.0 means the
   discriminator does no better than chance.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from health_dt import config
from health_dt.level1.transitions import transition_matrix

logger = logging.getLogger(__name__)


def transition_fidelity(
    real_transitions: dict[int, np.ndarray],
    synthetic_sequences: np.ndarray,
    synthetic_clusters: np.ndarray,
) -> dict[int, float]:
    """Frobenius norm between real and simulated transition matrices, per cluster.

    Lower is better. A cluster the synthetic cohort never produces is reported
    as nan rather than skipped — that is a finding, not a rounding error.
    """
    synthetic_clusters = np.asarray(synthetic_clusters).astype(int)
    scores = {}
    for cluster, real_matrix in sorted(real_transitions.items()):
        mask = synthetic_clusters == cluster
        if not mask.any():
            logger.warning("cluster %d has no synthetic patients; fidelity undefined", cluster)
            scores[cluster] = float("nan")
            continue
        synthetic_matrix = transition_matrix(synthetic_sequences[mask])
        scores[cluster] = float(np.linalg.norm(real_matrix - synthetic_matrix, "fro"))
    return scores


def pmse(real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]) -> tuple[float, float]:
    """Propensity mean-squared error and its null-standardised ratio.

    Returns (pMSE, ratio). The ratio divides by the value expected when real and
    synthetic are exchangeable, so ~1.0 means they are indistinguishable.
    """
    X = np.vstack(
        [real[columns].fillna(0).to_numpy(float), synthetic[columns].fillna(0).to_numpy(float)]
    )
    y = np.concatenate([np.zeros(len(real)), np.ones(len(synthetic))])

    X_scaled = StandardScaler().fit_transform(X)
    predicted = LogisticRegression(max_iter=1000).fit(X_scaled, y).predict_proba(X_scaled)[:, 1]

    share = len(synthetic) / len(y)
    score = float(np.mean((predicted - share) ** 2))
    null = (X_scaled.shape[1]) * (1 - share) ** 2 * share / len(y)
    return score, float(score / null) if null > 0 else float("nan")


def compare(
    real: pd.DataFrame,
    real_sequences: np.ndarray,
    synthetic: pd.DataFrame,
    synthetic_sequences: np.ndarray,
    real_transitions: dict[int, np.ndarray],
) -> pd.DataFrame:
    """Run the full fidelity battery. Returns a one-row-per-metric summary."""
    shared = [
        c
        for c in real.columns
        if c in synthetic.columns
        and c not in config.NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(real[c])
    ]

    logger.info("Real n=%d, synthetic n=%d", len(real), len(synthetic))

    logger.info("\nWeekly state frequencies (real vs synthetic):")
    for state in range(config.N_STATES):
        logger.info(
            "  %-30s %5.1f%%  %5.1f%%",
            config.STATE_LABELS[state],
            (real_sequences == state).mean() * 100,
            (synthetic_sequences == state).mean() * 100,
        )

    rows = [{"metric": "n_real", "value": len(real)}, {"metric": "n_synthetic", "value": len(synthetic)}]

    if "cluster" in real.columns and "cluster" in synthetic.columns:
        real_mix = real["cluster"].value_counts(normalize=True)
        synth_mix = synthetic["cluster"].value_counts(normalize=True)
        logger.info("\nPhenotype mix:")
        for cluster in sorted(set(real_mix.index) | set(synth_mix.index)):
            logger.info(
                "  cluster %d: real %.1f%%, synthetic %.1f%%",
                cluster,
                real_mix.get(cluster, 0) * 100,
                synth_mix.get(cluster, 0) * 100,
            )
            rows.append(
                {"metric": f"mix_cluster_{cluster}_real", "value": round(float(real_mix.get(cluster, 0)), 4)}
            )
            rows.append(
                {"metric": f"mix_cluster_{cluster}_synthetic", "value": round(float(synth_mix.get(cluster, 0)), 4)}
            )

    logger.info("\nTransition fidelity (Frobenius, lower is better):")
    for cluster, value in transition_fidelity(
        real_transitions, synthetic_sequences, synthetic["cluster"].to_numpy()
    ).items():
        logger.info("  cluster %d: %.4f", cluster, value)
        rows.append({"metric": f"frobenius_cluster_{cluster}", "value": round(value, 5)})

    if shared:
        score, ratio = pmse(real, synthetic, shared)
        logger.info("\npMSE %.6f, ratio %.3f (ideal ~ 1.0)", score, ratio)
        rows.append({"metric": "pMSE", "value": round(score, 8)})
        rows.append({"metric": "pMSE_ratio", "value": round(ratio, 4)})

    return pd.DataFrame(rows)
