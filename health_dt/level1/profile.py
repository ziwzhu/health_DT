"""Tables 1 and 2 — are the phenotypes distinct, and do they matter clinically?

Two questions. *Descriptively distinct?* — answered by the association tests and
the cluster profile. *Clinically meaningful?* — answered by testing whether ED
visits, inpatient stays, length of stay, and Medicare cost differ across
phenotypes.

A clustering that is statistically clean but shows no outcome separation is not
useful for a digital twin, so this is a gate, not a formality.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from health_dt import config

logger = logging.getLogger(__name__)


def numeric_associations(df: pd.DataFrame, target: str = "target") -> pd.DataFrame:
    """One-way ANOVA of each numeric/binary variable across phenotypes.

    One row per variable: per-cluster means plus the ANOVA p-value. Variables
    absent from the frame are skipped rather than raising — the same extract
    serves several cohorts and not every column is always present.
    """
    columns = [
        *config.PATIENT_NUM_COLUMNS,
        *config.COMORBIDITY_COLUMNS,
        *config.PATIENT_BIN_COLUMNS,
        "comorbid_count",
    ]
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        groups = [g.dropna() for _, g in df.groupby(target)[col]]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue

        f_stat, p_value = stats.f_oneway(*groups)
        row = {"variable": col}
        row.update(
            {f"cluster_{k}": round(float(v), 3) for k, v in df.groupby(target)[col].mean().items()}
        )
        row["f_statistic"] = round(float(f_stat), 3)
        row["p_value"] = float(p_value)
        rows.append(row)
    return pd.DataFrame(rows)


def categorical_associations(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Chi-square test plus within-cluster percentages for each categorical.

    Percentages are column-wise (share of each cluster in that level), the
    orientation used in the manuscript's Table 1.
    """
    rows = []
    for col in config.PATIENT_CAT_COLUMNS:
        if col not in df.columns:
            logger.warning("categorical column %s not found; skipping", col)
            continue

        table = pd.crosstab(df[col].fillna("Missing"), df[target])
        chi2, p_value, _, _ = stats.chi2_contingency(table)
        pct = table.div(table.sum(axis=0), axis=1) * 100

        for level, level_row in pct.iterrows():
            record = {"variable": col, "category": level}
            record.update({f"cluster_{k}": round(float(v), 2) for k, v in level_row.items()})
            record["chi2"] = round(float(chi2), 3)
            record["p_value"] = float(p_value)
            rows.append(record)
    return pd.DataFrame(rows)


def outcome_tests(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Test each clinical outcome for differences across phenotypes.

    Binary outcomes get chi-square; counts and costs get Kruskal-Wallis, which
    assumes no normality — appropriate for cost and length-of-stay distributions
    with long right tails.
    """
    frame = df.copy()
    frame["cluster"] = labels

    rows = []
    for outcome in config.OUTCOME_COLUMNS:
        if outcome not in frame.columns:
            logger.info("  %s: skipped (not present)", outcome)
            continue

        if frame[outcome].nunique() <= 2:
            statistic, p_value, _, _ = stats.chi2_contingency(
                pd.crosstab(frame["cluster"], frame[outcome])
            )
            test = "Chi-square"
        else:
            groups = [g[outcome].dropna().to_numpy() for _, g in frame.groupby("cluster")]
            statistic, p_value = stats.kruskal(*groups)
            test = "Kruskal-Wallis"

        row = {"outcome": outcome, "test": test}
        row.update(
            {
                f"cluster_{int(c)}_mean": round(float(g[outcome].mean()), 4)
                for c, g in frame.groupby("cluster")
            }
        )
        row["statistic"] = round(float(statistic), 4)
        row["p_value"] = float(p_value)
        row["significance"] = _stars(p_value)
        rows.append(row)

        logger.info("  %s: %s = %.2f, p = %.2e %s", outcome, test, statistic, p_value, row["significance"])
    return pd.DataFrame(rows)


def cluster_profiles(
    df: pd.DataFrame, labels: np.ndarray, sequences: np.ndarray
) -> pd.DataFrame:
    """Table 1: one row per phenotype.

    Numeric columns summarised as mean +/- sd, categoricals as the three most
    common levels with shares, plus the percentage of patient-weeks each
    phenotype spends in each state — which is what makes the phenotypes
    interpretable as low/medium/high use.
    """
    frame = df.copy()
    frame["cluster"] = labels

    rows = []
    for cluster in sorted(frame["cluster"].unique()):
        subset = frame[frame["cluster"] == cluster]
        row = {"Cluster": int(cluster), "N": len(subset)}

        for col in frame.columns:
            if col in config.NON_FEATURE_COLUMNS or col == "cluster":
                continue
            if pd.api.types.is_numeric_dtype(frame[col]):
                row[f"{col} (mean+/-sd)"] = f"{subset[col].mean():.2f} +/- {subset[col].std():.2f}"
            elif frame[col].dtype == object:
                top = subset[col].value_counts(normalize=True).head(3)
                row[col] = ", ".join(f"{v}:{p:.0%}" for v, p in top.items())

        cluster_sequences = sequences[np.asarray(labels) == cluster]
        for state in range(config.N_STATES):
            row[f"% {config.STATE_LABELS[state]}"] = f"{(cluster_sequences == state).mean() * 100:.1f}%"
        rows.append(row)

    return pd.DataFrame(rows).set_index("Cluster")


def _stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"
