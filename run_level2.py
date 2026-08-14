"""Level 2 — synthetic populations and their planned care trajectories.

    python run_level2.py
    python run_level2.py --method marginal        # no R installed
    python run_level2.py --scenario orig_age      # one scenario only
    python run_level2.py --replications 10        # quick run

Reads Level 1's outputs from outputs/. Level 1 is not re-run.

Outputs (to outputs/):
    synthpop_patients_*.csv, *_clustered.csv
    sequences_<scenario>_seed<nnnnn>.npy
    fidelity_<scenario>.csv
"""

import argparse
import logging

import numpy as np
import pandas as pd

from health_dt import config
from health_dt.level1 import classify, transitions
from health_dt.level2 import assign, fidelity, simulate, synthesis
from health_dt.logging_setup import banner, setup

logger = logging.getLogger("run_level2")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    setup(log_file=config.out("run_level2.log"))

    scenarios = [config.scenario(n) for n in args.scenario] if args.scenario else config.SCENARIOS
    method = args.method
    if method == "synthpop" and not synthesis.r_available():
        raise SystemExit(
            f"Rscript not found at {config.RSCRIPT!r}. Install R with synthpop and "
            "truncnorm, set config.RSCRIPT, or re-run with --method marginal."
        )

    # --- 2a. Synthesis -------------------------------------------------------
    banner(logger, f"Level 2a — population synthesis ({method})")
    default_source = config.out("processed_synth_raw_data.csv")

    for spec in scenarios:
        source = config.DATA_DIR / spec.source_csv if spec.source_csv else default_source
        if not source.is_file():
            logger.warning("%s: source %s not found, using %s",
                           spec.name, source.name, default_source.name)
            source = default_source

        logger.info("\n%s — %s", spec.name, spec.description)
        synthesis.synthesize(
            source, config.out(spec.synth_csv), args.n_synthetic,
            age_shift=spec.age_shift, seed=config.RANDOM_SEED, method=method,
        )

    # --- 2b. Phenotype assignment -------------------------------------------
    banner(logger, "Level 2b — phenotype assignment")
    bundle = classify.load_model(config.out("best_model.pkl"))
    logger.info("Classifier: %s", bundle["model_name"])

    for spec in scenarios:
        logger.info("\n%s", spec.name)
        cohort = pd.read_csv(config.out(spec.synth_csv))
        scored = assign.assign_clusters(cohort, bundle, context=spec.name)
        scored.to_csv(config.out(spec.clustered_csv), index=False)

    # --- 2c. Trajectory simulation ------------------------------------------
    banner(logger, "Level 2c — utilisation simulation")
    seeds = simulate.replication_seeds(args.replications)
    logger.info("%d scenarios x %d replications", len(scenarios), len(seeds))

    for index, spec in enumerate(scenarios):
        cohort = pd.read_csv(config.out(spec.clustered_csv))
        clusters = sorted(cohort["cluster"].unique())

        cluster_transitions = transitions.load_transitions(
            config.out("cluster_transitions.npz"), expected_clusters=clusters
        )
        initial = transitions.load_initial_distributions(
            config.out("initial_distributions.npz"), expected_clusters=clusters
        )

        for seed in seeds:
            # Offset by scenario so two scenarios never share a random stream.
            sequences = simulate.simulate_sequences(
                cohort, cluster_transitions, initial, seed=int(seed) + index
            )
            np.save(config.out(spec.sequence_csv(int(seed))), sequences)
        logger.info("%s: wrote %d replication files", spec.name, len(seeds))

    # --- 2d. Fidelity --------------------------------------------------------
    banner(logger, "Level 2d — twin fidelity")
    real = pd.read_csv(config.COMBINED_CSV)
    real["cluster"] = np.load(config.out("real_labels.npy"))
    real_sequences = np.load(config.out("real_sequences.npy"))

    for spec in scenarios:
        logger.info("\n--- %s ---", spec.name)
        cohort = pd.read_csv(config.out(spec.clustered_csv))
        cluster_transitions = transitions.load_transitions(
            config.out("cluster_transitions.npz"),
            expected_clusters=sorted(cohort["cluster"].unique()),
        )
        synthetic_sequences = np.load(config.out(spec.sequence_csv(int(seeds[0]))))

        summary = fidelity.compare(
            real, real_sequences, cohort, synthetic_sequences, cluster_transitions
        )
        summary.to_csv(config.out(f"fidelity_{spec.name}.csv"), index=False)

    logger.info("\nLevel 2 complete. Outputs in %s", config.OUTPUT_DIR)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--method", default="synthpop", choices=["synthpop", "marginal"],
        help="Synthesis method. 'marginal' needs no R but destroys joint structure "
             "— smoke tests and null comparisons only.",
    )
    parser.add_argument(
        "--scenario", action="append",
        help="Limit to this scenario (repeatable). Default: all in config.SCENARIOS.",
    )
    parser.add_argument(
        "--replications", type=int, default=config.N_REPLICATIONS,
        help="Number of stochastic replications (default: %(default)s).",
    )
    parser.add_argument(
        "--n-synthetic", type=int, default=config.N_SYNTHETIC,
        help="Synthetic cohort size (default: %(default)s).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
