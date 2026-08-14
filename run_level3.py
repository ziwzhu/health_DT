"""Level 3 — finite resource capacity applied to planned trajectories.

    python run_level3.py
    python run_level3.py --scenario orig_age --verbose      # single run, weekly detail
    python run_level3.py --capacity-scale 0.5               # half the capacity everywhere

Reads Level 2's outputs from outputs/. Levels 1 and 2 are not re-run.

Outputs (to outputs/):
    abm_runs_all_results.csv     one row per resource per (scenario, seed, threshold)
    abm_runs_summary.csv         means and standard deviations across replications
    abm_resource_summary.png
"""

import argparse
import logging

from health_dt import config, plots
from health_dt.level3 import scenarios as sweep_module
from health_dt.logging_setup import banner, setup

logger = logging.getLogger("run_level3")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    setup(log_file=config.out("run_level3.log"))

    specs = [config.scenario(n) for n in args.scenario] if args.scenario else config.SCENARIOS
    resources = (
        config.scale_resources(args.capacity_scale) if args.capacity_scale != 1.0 else None
    )

    banner(logger, "Level 3 — resource-constrained simulation")
    logger.info("%d scenario(s), drop thresholds %s", len(specs), config.DROP_THRESHOLDS)
    if resources is not None:
        logger.info("Capacities scaled by %.3g", args.capacity_scale)

    runs = sweep_module.sweep(specs, resources=resources)
    summary = sweep_module.aggregate(runs)

    runs.to_csv(config.out("abm_runs_all_results.csv"), index=False)
    summary.to_csv(config.out("abm_runs_summary.csv"), index=False)

    # Report at the middle constraint level.
    reference_threshold = config.DROP_THRESHOLDS[len(config.DROP_THRESHOLDS) // 2]
    banner(logger, f"Summary at drop threshold {reference_threshold}")
    with_pd_options(summary[summary["drop_threshold"] == reference_threshold])

    by_scenario = {
        name: group.groupby("resource", as_index=False)[["mean_utilization", "total_unmet"]].mean()
        for name, group in runs[runs["drop_threshold"] == reference_threshold].groupby("scenario")
    }
    # Draw the baseline first so it reads as the reference bar.
    ordered = {}
    if config.BASELINE_SCENARIO in by_scenario:
        ordered[config.BASELINE_SCENARIO] = by_scenario.pop(config.BASELINE_SCENARIO)
    ordered.update(by_scenario)
    plots.resource_summary(ordered)

    logger.info("\nLevel 3 complete: %d rows. Outputs in %s", len(runs), config.OUTPUT_DIR)


def with_pd_options(frame) -> None:
    """Log a frame without pandas truncating columns."""
    import pandas as pd

    with pd.option_context("display.max_columns", None, "display.width", 200):
        logger.info("%s", frame.to_string(index=False))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scenario", action="append",
        help="Limit to this scenario (repeatable). Default: all in config.SCENARIOS.",
    )
    parser.add_argument(
        "--capacity-scale", type=float, default=1.0,
        help="Multiply every resource capacity by this factor. Use for capacity "
             "what-ifs, or to scale the system down to a small demo cohort.",
    )
    parser.add_argument("--verbose", action="store_true", help="Log weekly progress")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
