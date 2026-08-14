"""Level 1 — phenotype discovery, classification, and transition extraction.

    python run_level1.py

Reads the SEER-Medicare extract and the external clustering's labels. Writes the
classifier and the Markov matrices that Level 2 needs, plus Tables 1 and 2.

Outputs (to outputs/):
    processed_data.csv, processed_synth_raw_data.csv
    best_model.pkl, model_comparison.csv, feature_importances.csv
    cluster_transitions.npz, initial_distributions.npz, transition_cluster_*.csv
    cluster_profiles.csv, outcome_tests.csv, table1_numeric.csv, table1_categorical.csv
    real_sequences.npy, real_labels.npy
    utilization_heatmaps.png, state_distribution_stacked.png
"""

import logging

import numpy as np
import pandas as pd

from health_dt import config, plots
from health_dt.level1 import classify, preprocess, profile, sequences, transitions
from health_dt.logging_setup import banner, setup

logger = logging.getLogger("run_level1")


def main() -> None:
    setup(log_file=config.out("run_level1.log"))

    # --- Preprocessing -------------------------------------------------------
    banner(logger, "Level 1a — preprocessing")
    patients = pd.read_csv(config.CLUSTERED_PATIENTS_CSV)
    logger.info("Loaded %d patients from %s", len(patients), config.CLUSTERED_PATIENTS_CSV.name)

    grouped = preprocess.regroup_categoricals(patients)
    grouped.to_csv(config.out("patients_cat_processed.csv"), index=False)

    model_matrix = preprocess.build_model_matrix(grouped)
    model_matrix.to_csv(config.out("processed_data.csv"), index=False)

    synthesis_frame = preprocess.build_synthesis_frame(grouped)
    synthesis_frame.to_csv(config.out("processed_synth_raw_data.csv"), index=False)

    # --- Classification ------------------------------------------------------
    banner(logger, "Level 1b — phenotype classifier")
    target_cols = [c for c in model_matrix.columns if c.startswith("target")]
    X = model_matrix.drop(columns=target_cols)
    y = model_matrix[config.TARGET_COLUMN]

    best_name, best_model, scores = classify.train_and_select(X, y)
    classify.save_model(best_model, list(X.columns), best_name, config.out("best_model.pkl"))

    pd.DataFrame({"model": list(scores), "macro_f1": list(scores.values())}).sort_values(
        "macro_f1", ascending=False
    ).to_csv(config.out("model_comparison.csv"), index=False)

    importances = classify.feature_importances(best_model, list(X.columns))
    if importances is not None:
        logger.info("\nTop %d features:\n%s", config.TOP_FEATURES_TO_REPORT,
                    importances.head(config.TOP_FEATURES_TO_REPORT).to_string())
        importances.rename("importance").rename_axis("feature").reset_index().to_csv(
            config.out("feature_importances.csv"), index=False
        )

    # --- Sequences and transition matrices -----------------------------------
    banner(logger, "Level 1c — transition matrices")
    raw = pd.read_csv(config.COMBINED_CSV)
    real_sequences = sequences.encode_sequences(raw)
    labels = sequences.load_labels(len(raw))

    cluster_transitions = transitions.extract_transitions(real_sequences, labels)
    initial = transitions.extract_initial_distributions(real_sequences, labels)
    transitions.save_transitions(cluster_transitions, config.out("cluster_transitions.npz"))
    transitions.save_initial_distributions(initial, config.out("initial_distributions.npz"))

    np.save(config.out("real_sequences.npy"), real_sequences)
    np.save(config.out("real_labels.npy"), labels)

    # --- Tables and figures --------------------------------------------------
    banner(logger, "Level 1d — profiles and outcome tests")
    patient_frame = _build_patient_frame(raw, real_sequences)

    profile.outcome_tests(patient_frame, labels).to_csv(
        config.out("outcome_tests.csv"), index=False
    )
    profile.cluster_profiles(patient_frame, labels, real_sequences).reset_index().to_csv(
        config.out("cluster_profiles.csv"), index=False
    )
    profile.numeric_associations(model_matrix).to_csv(
        config.out("table1_numeric.csv"), index=False
    )
    profile.categorical_associations(grouped, config.LABEL_COLUMN).to_csv(
        config.out("table1_categorical.csv"), index=False
    )

    plots.utilization_heatmaps(real_sequences, labels)
    plots.state_distribution_stacked(real_sequences, labels)

    logger.info("\nLevel 1 complete. Outputs in %s", config.OUTPUT_DIR)


def _build_patient_frame(raw: pd.DataFrame, real_sequences: np.ndarray) -> pd.DataFrame:
    """Assemble outcomes, sequence-derived features, and characteristics."""
    keep = [c for c in (config.PATIENT_ID_COLUMN, *config.OUTCOME_COLUMNS) if c in raw.columns]
    frame = pd.concat(
        [raw[keep].reset_index(drop=True), sequences.derive_features(real_sequences)], axis=1
    )
    for col in config.PROFILE_CHAR_COLUMNS:
        if col in raw.columns:
            frame[col] = raw[col].to_numpy()
    return frame


if __name__ == "__main__":
    main()
