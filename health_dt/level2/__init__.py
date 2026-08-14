"""Level 2 — digital twin initialisation.

    synthesis   generate synthetic cohorts, optionally demographically shifted
    assign      give each synthetic patient a phenotype via the Level 1 classifier
    simulate    turn phenotypes into 52-week planned care trajectories
    fidelity    check the synthetic cohort behaves like the real one

Reads from Level 1: ``best_model.pkl``, ``cluster_transitions.npz``,
``initial_distributions.npz``, ``processed_synth_raw_data.csv``. Level 1 does
not need to be re-run.

Outputs consumed by Level 3: ``*_clustered.csv``, ``sequences_*.npy``.
"""
