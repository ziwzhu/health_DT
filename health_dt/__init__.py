"""Digital twin pipeline for cancer survivorship care utilisation.

Three levels, each self-contained and reading the previous level's outputs from
disk:

    level1  phenotype discovery, classification, Markov transition matrices
    level2  synthetic populations and their planned care trajectories
    level3  finite resource capacity applied to those trajectories

Run them with the ``run_level1.py`` / ``run_level2.py`` / ``run_level3.py``
scripts at the project root. Configuration lives in :mod:`health_dt.config`.
"""

# 1.x was the original ad-hoc script pipeline. 2.0 is the reorganised package:
# same science, restructured into three levels with a single config.
__version__ = "2.0.0"
