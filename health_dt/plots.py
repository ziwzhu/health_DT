"""All figure generation.

Kept in one module so the analysis stays importable without a display, styling
stays consistent, and a caller who only wants numbers never pays the matplotlib
import cost. Every function writes to OUTPUT_DIR and returns the path.
"""

import logging

import numpy as np
import pandas as pd

from health_dt import config

logger = logging.getLogger(__name__)


def _plt():
    """Import pyplot with a non-interactive backend (works headless)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, filename: str):
    path = config.out(filename)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    _plt().close(fig)
    logger.info("  saved %s", filename)
    return path


def utilization_heatmaps(sequences: np.ndarray, labels: np.ndarray):
    """Heatmap of weekly state prevalence, one panel per phenotype."""
    plt = _plt()
    labels = np.asarray(labels)
    clusters = sorted(np.unique(labels))

    fig, axes = plt.subplots(
        len(clusters), 1, figsize=(16, 3 * len(clusters) + 1), sharex=True, squeeze=False
    )
    image = None
    for idx, cluster in enumerate(clusters):
        ax = axes[idx][0]
        subset = sequences[labels == cluster]
        weekly = np.vstack([(subset == s).mean(axis=0) for s in range(config.N_STATES)])
        image = ax.imshow(weekly, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.8)
        ax.set_ylabel(f"Cluster {cluster}\n(n={int((labels == cluster).sum())})")
        ax.set_yticks(range(config.N_STATES))
        ax.set_yticklabels([config.STATE_LABELS[s] for s in range(config.N_STATES)], fontsize=8)
    axes[-1][0].set_xlabel("Week")

    fig.suptitle("Weekly utilisation patterns by phenotype", fontsize=14, y=1.01)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Proportion of patients", shrink=0.6)
    return _save(fig, "utilization_heatmaps.png")


def state_distribution_stacked(sequences: np.ndarray, labels: np.ndarray):
    """Stacked area chart of weekly state composition, one panel per phenotype."""
    plt = _plt()
    labels = np.asarray(labels)
    clusters = sorted(np.unique(labels))
    n_cols = min(2, len(clusters))
    n_rows = (len(clusters) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(14, 5 * n_rows), sharex=True, sharey=True, squeeze=False
    )
    for idx, cluster in enumerate(clusters):
        ax = axes[idx // n_cols][idx % n_cols]
        subset = sequences[labels == cluster]
        proportions = np.vstack([(subset == s).mean(axis=0) for s in range(config.N_STATES)])
        ax.stackplot(
            range(1, subset.shape[1] + 1),
            proportions,
            labels=[config.STATE_LABELS[s] for s in range(config.N_STATES)],
            colors=config.STATE_COLORS[: config.N_STATES],
            alpha=0.85,
        )
        ax.set_title(f"Cluster {cluster} (n={int((labels == cluster).sum())})")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Week")

    for idx in range(len(clusters), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, "state_distribution_stacked.png")


def resource_summary(by_scenario: dict[str, pd.DataFrame]):
    """Grouped bars of mean utilisation and total unmet demand per resource.

    ``by_scenario`` maps scenario name to a resource summary frame; the first
    entry is drawn as the reference.
    """
    plt = _plt()
    names = list(by_scenario)
    resources = by_scenario[names[0]]["resource"].tolist()
    x = np.arange(len(resources))
    width = min(0.8 / len(names), 0.35)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for offset, name in enumerate(names):
        frame = by_scenario[name].set_index("resource").reindex(resources)
        axes[0].bar(x + offset * width, frame["mean_utilization"] * 100, width=width, alpha=0.85, label=name)
        axes[1].bar(x + offset * width, frame["total_unmet"], width=width, alpha=0.85, label=name)

    centre = x + width * (len(names) - 1) / 2
    for ax, ylabel, title in (
        (axes[0], "Mean utilisation (%)", "Resource utilisation"),
        (axes[1], "Total unmet encounters (52 weeks)", "Unmet demand"),
    ):
        ax.set_xticks(centre)
        ax.set_xticklabels(resources, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylim(0, 120)
    axes[0].axhline(y=100, color="red", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return _save(fig, "abm_resource_summary.png")
