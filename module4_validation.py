"""
module4_validation.py — Validate clusters and generate profiles.

Inputs:  df_patients, labels, sequences
Outputs: Profile table, heatmaps, statistical tests, plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency, kruskal

from config import (N_STATES, N_WEEKS, STATE_LABELS, OUTCOME_COLUMNS,
                    OUTPUT_DIR, FIGURE_DPI)


def validate_outcomes(df_patients, labels, outcomes=None):
    """
    Test whether clusters differ significantly on clinical outcomes.

    For binary columns: Chi-square test.
    For continuous/count columns: Kruskal-Wallis test.

    Returns
    -------
    results : dict of {outcome: {test, statistic, p_value}}
    """
    outcomes = outcomes or OUTCOME_COLUMNS
    if not outcomes:
        print("[Module 4] No outcome columns configured — skipping validation.")
        print("  Set OUTCOME_COLUMNS in config.py to enable.")
        return {}

    df = df_patients.copy()
    df["cluster"] = labels

    print("\n" + "=" * 65)
    print("OUTCOME-AWARE CLUSTER VALIDATION")
    print("=" * 65)

    results = {}
    for outcome in outcomes:
        if outcome not in df.columns:
            print(f"\n  {outcome}: SKIPPED (column not found)")
            continue

        if df[outcome].nunique() <= 2:
            ct = pd.crosstab(df["cluster"], df[outcome])
            chi2, p, dof, expected = chi2_contingency(ct)
            test_name = "Chi-square"
        else:
            groups = [g[outcome].values for _, g in df.groupby("cluster")]
            chi2, p = kruskal(*groups)
            test_name = "Kruskal-Wallis"

        results[outcome] = {"test": test_name, "statistic": chi2, "p_value": p}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"\n  {outcome}:")
        print(f"    {test_name} = {chi2:.2f}, p = {p:.2e} {sig}")

        for c in sorted(df["cluster"].unique()):
            sub = df[df["cluster"] == c]
            if df[outcome].nunique() <= 2:
                print(f"    Cluster {c}: {sub[outcome].mean():.1%} (n={len(sub)})")
            else:
                print(f"    Cluster {c}: mean={sub[outcome].mean():.2f} (n={len(sub)})")

    return results


def profile_clusters(df_patients, labels, sequences,
                     extra_columns=None):
    """
    Generate Table 1-style cluster profiles.

    Parameters
    ----------
    extra_columns : list of str
        Additional columns from df_patients to include in the profile.
        Numeric columns get mean±sd; categorical columns get mode.

    Returns
    -------
    profile_df : pd.DataFrame
    """
    df = df_patients.copy()
    df["cluster"] = labels

    print("\n" + "=" * 65)
    print("CLUSTER PROFILES")
    print("=" * 65)

    rows = []
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        row = {"Cluster": c, "N": len(sub)}

        # Add any numeric columns that exist
        for col in df.columns:
            if col in ("cluster", "patient_id", "PATIENT_ID", "latent_profile"):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                row[f"{col} (mean+/-sd)"] = (
                    f"{sub[col].mean():.2f} +/- {sub[col].std():.2f}")
            elif df[col].dtype == object and col not in ("patient_id",):
                # For categorical, show distribution of top values
                top = sub[col].value_counts(normalize=True).head(3)
                summary = ", ".join([f"{v}:{p:.0%}" for v, p in top.items()])
                row[col] = summary

        # Utilization summary from sequences
        mask = labels == c
        cluster_seqs = sequences[mask]
        for s in range(N_STATES):
            pct = (cluster_seqs == s).mean() * 100
            row[f"% {STATE_LABELS.get(s, f'State {s}')}"] = f"{pct:.1f}%"

        rows.append(row)

    profile_df = pd.DataFrame(rows).set_index("Cluster")
    print(profile_df.to_string())

    profile_df.to_csv(os.path.join(OUTPUT_DIR, "cluster_profiles.csv"))
    print(f"\n  Saved: cluster_profiles.csv")

    return profile_df


def plot_utilization_heatmaps(sequences, labels):
    """
    Heatmap of weekly state distributions per cluster.
    """
    unique_clusters = sorted(np.unique(labels))
    n_clusters = len(unique_clusters)

    fig, axes = plt.subplots(n_clusters, 1,
                              figsize=(16, 3 * n_clusters + 1),
                              sharex=True)
    if n_clusters == 1:
        axes = [axes]

    for idx, c in enumerate(unique_clusters):
        mask = labels == c
        cluster_seqs = sequences[mask]

        weekly_dist = np.zeros((N_STATES, N_WEEKS))
        for s in range(N_STATES):
            weekly_dist[s] = (cluster_seqs == s).mean(axis=0)

        im = axes[idx].imshow(weekly_dist, aspect="auto", cmap="YlOrRd",
                               interpolation="nearest", vmin=0, vmax=0.8)
        axes[idx].set_ylabel(f"Cluster {c}\n(n={mask.sum()})")
        axes[idx].set_yticks(range(N_STATES))
        axes[idx].set_yticklabels(
            [STATE_LABELS.get(s, str(s)) for s in range(N_STATES)],
            fontsize=8)
        if idx == n_clusters - 1:
            axes[idx].set_xlabel("Week")

    fig.suptitle("Weekly Utilization Patterns by Cluster", fontsize=14, y=1.01)
    fig.colorbar(im, ax=axes, label="Proportion of Patients", shrink=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "utilization_heatmaps.png"),
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: utilization_heatmaps.png")

def plot_state_distribution_stacked(sequences, labels, cluster_names=None):
    "Stacked area chart of weekly state proportions per cluster."
    unique_c = sorted(np.unique(labels))
    n_cols = min(2, len(unique_c))
    n_rows = (len(unique_c) + 1)//2
    
    # colors that matching R plot
    state_colors = [
        "#B2E8D4", #0 No Encounter
        "#FFFFB3", #1 PCP Only
        "#B3CDE3", #2 Other Spec only
        "#F4A0A0", #3 Other Spec + PCP
        "#6BAED6", #4 Onc/Hem Only
        "#FDAE61", #5 Musculoskeletal only
        "#B2DF8A", #6 Mental health only
        "#AAAAAA", #7 Hospital Only
        "#F5D0E2", #8 Hospital + Other Spec
        "#C994E0", #9 Hospital + Other Spec + PCP
        "#E2F5D0", #10 Rare
        ]
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (14, 5*n_rows),
                             sharex=True, sharey=True, squeeze=False)
    for idx, c in enumerate(unique_c):
        ax = axes[idx//n_cols][idx%n_cols]
        cseq = sequences[labels == c]
        props = np.array([(cseq==s).mean(axis=0) for s in range(N_STATES)])
        
        # # debug loop for state
        # for s in range(N_STATES):
        #     mean_pct = props[s].mean() * 100
        #     print(f"Cluster{c}|State{s}({STATE_LABELS.get(s, s)}): {mean_pct:.1f}%")
        
        ax.stackplot(range(1, cseq.shape[1]+1), props,
                     labels=[STATE_LABELS.get(s, str(s)) for s in range (N_STATES)],
                     colors=state_colors[:N_STATES], alpha = 0.85)
        ax.set_title(f"Cluster {c} (n={(labels==c).sum()}")
        ax.set_ylim(0, 1)
    handles, labs = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=4, fontsize = 9)
    plt.savefig(os.path.join(OUTPUT_DIR, "state_distribution_stacked.png"), 
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: state_distribution_stacked.png")

def plot_pca_scatter(X_reduced, labels, df_patients=None):
    """
    2D PCA scatter colored by cluster.
    Only works with feature-based approach (not LCS distance).
    """
    if X_reduced is None:
        print("  Skipping PCA scatter (no feature matrix available)")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
                          c=labels, cmap="tab10", alpha=0.4, s=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Clusters in PCA Space")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_scatter.png"), dpi=FIGURE_DPI)
    plt.close()
    print("  Saved: pca_scatter.png")


def analyze_half_year_shifts(sequences, labels, best_k, method="kmeans"):
    """
    Split sequences into H1 (weeks 1-26) and H2 (weeks 27-52),
    cluster each half, measure how many patients shift clusters.
    """
    from module2_features import build_feature_matrix, reduce_dimensions
    from module3_clustering import _fit_single_k
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from scipy.optimize import linear_sum_assignment

    half = N_WEEKS // 2
    results = {}

    for label, seq_half in [("H1", sequences[:, :half]),
                             ("H2", sequences[:, half:])]:
        feat, _ = build_feature_matrix(seq_half, method="transition")
        scaler = StandardScaler()
        feat_clean = np.nan_to_num(scaler.fit_transform(feat), nan=0.0)
        n_comp = min(10, feat_clean.shape[1], feat_clean.shape[0])
        X = PCA(n_components=n_comp, random_state=42).fit_transform(feat_clean)
        lab, _ = _fit_single_k(X, method, best_k)
        results[label] = lab

    # Hungarian algorithm for alignment before judging shift
    contigency = pd.crosstab(results["H1"], results["H2"])
    row_ind, col_ind = linear_sum_assignment(-contigency.values)
    label_map = {contigency.columns[c]: contigency.index[r]
                 for r, c in zip(row_ind, col_ind)}
    # Remap H2 labels before comparing
    results["H2"] = np.array([label_map[l] for l in results["H2"]])
    shifted = (results["H1"] != results["H2"]).sum()
    n = len(sequences)

    print(f"\n{'=' * 65}")
    print(f"HALF-YEAR CLUSTER SHIFT ANALYSIS")
    print(f"{'=' * 65}")
    print(f"  Shifted: {shifted}/{n} ({shifted/n*100:.1f}%)")

    shift_mat = pd.crosstab(
        pd.Series(results["H1"], name="H1"),
        pd.Series(results["H2"], name="H2"),
        normalize="index"
    )
    print(f"\n  H1 -> H2 transitions:\n{shift_mat.round(3)}")

    return shifted, results