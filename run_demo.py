"""End-to-end demo on generated data — no SEER-Medicare access needed.

    python run_demo.py

Generates a small SEER-shaped cohort, runs all three levels, and prints samples
of the key outputs. Takes about a minute. Use it to check an installation, to
see what the pipeline produces before wiring up real data, or as a worked
example of calling the library directly.

`quickstart.ipynb` is the same walkthrough as a notebook, with figures.

Two things differ from a real run and are called out again in the output:
  * Population synthesis uses the `marginal` method (no R needed), which samples
    each variable independently and so destroys the joint structure the twin
    depends on. Real runs use synthpop.
  * Resource capacities are scaled down to the demo cohort. At full capacity a
    500-patient cohort strains nothing and Level 3 reports zero unmet demand.
"""

import argparse
import logging

import pandas as pd

import run_level1
import run_level2
import run_level3
from health_dt import config
from health_dt.demo_data import write_demo_data
from health_dt.logging_setup import banner, setup

logger = logging.getLogger("run_demo")

#: (title, filename, row limit, column selection). Column selection keeps the
#: printed tables readable — the full versions are on disk.
SAMPLES = [
    ("Level 1 — classifier comparison (macro F1)", "model_comparison.csv", None, None),
    ("Level 1 — top predictive features", "feature_importances.csv", 8, None),
    (
        "Level 1 — phenotype profiles, Table 1 (excerpt)",
        "cluster_profiles.csv",
        None,
        ["Cluster", "N", "age_at_followup_y (mean+/-sd)", "n_hospital_weeks (mean+/-sd)",
         "% No Encounter", "% PCP Only", "% Hospital only"],
    ),
    (
        "Level 1 — outcome differences by phenotype, Table 2",
        "outcome_tests.csv",
        None,
        ["outcome", "test", "cluster_0_mean", "cluster_1_mean", "cluster_2_mean",
         "p_value", "significance"],
    ),
    ("Level 2 — twin fidelity, reference scenario", "fidelity_orig_age.csv", None, None),
    (
        "Level 3 — unmet demand by resource (reference scenario, middle threshold)",
        "abm_runs_summary.csv",
        None,
        ["resource", "capacity_mean", "total_demand_mean", "total_served_mean",
         "total_unmet_mean", "total_unmet_std", "mean_utilization_mean"],
    ),
]


def main() -> None:
    args = _parse_args()
    setup()

    banner(logger, f"Demo — {args.n} synthetic patients")
    logger.info(
        "Not real data. Synthesis uses the marginal method (no R), and capacities\n"
        "are scaled to %.1f%% so the demo cohort actually strains the system.\n",
        args.n / config.N_SYNTHETIC * 100,
    )

    write_demo_data(n=args.n, seed=args.seed)

    run_level1.main()
    run_level2.main(
        argv=["--method", "marginal", "--replications", str(args.replications),
              "--n-synthetic", str(args.n)]
    )
    run_level3.main(argv=["--capacity-scale", str(args.n / config.N_SYNTHETIC)])

    _print_samples()

    banner(logger, "Demo complete")
    logger.info(
        "Outputs in %s\n\n"
        "Next steps:\n"
        "  - Put real data in %s (see README) and run the three level scripts.\n"
        "  - Edit health_dt/config.py to change scenarios, capacities, or columns.\n"
        "  - See docs/MODULAR_UPDATE_GUIDE.md to replace a component.",
        config.OUTPUT_DIR,
        config.DATA_DIR,
    )


def _print_samples() -> None:
    """Show the headline tables so the run has a visible result."""
    banner(logger, "Output samples")
    reference_threshold = config.DROP_THRESHOLDS[len(config.DROP_THRESHOLDS) // 2]

    for title, filename, head, columns in SAMPLES:
        path = config.OUTPUT_DIR / filename
        if not path.is_file():
            continue
        frame = pd.read_csv(path)

        # The sweep table covers every scenario and threshold; show one slice.
        if "scenario" in frame.columns:
            frame = frame[
                (frame["scenario"] == config.BASELINE_SCENARIO)
                & (frame["drop_threshold"] == reference_threshold)
            ]
        if columns:
            frame = frame[[c for c in columns if c in frame.columns]]
        if head:
            frame = frame.head(head)

        logger.info("\n--- %s ---", title)
        logger.info("    (%s)", filename)
        with pd.option_context("display.max_columns", 12, "display.width", 170):
            logger.info("%s", frame.round(3).to_string(index=False, max_colwidth=26))

    figures = sorted(config.OUTPUT_DIR.glob("*.png"))
    if figures:
        logger.info("\n--- Figures ---")
        for figure in figures:
            logger.info("  %s", figure.name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=500, help="Demo cohort size")
    parser.add_argument("--replications", type=int, default=3, help="Stochastic replications")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
