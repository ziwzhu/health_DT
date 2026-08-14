"""Level 3 — resource-constrained care delivery.

    agents      survivor agents and their triage acuity
    abm         resource pools and the weekly allocation engine
    scenarios   replication sweep across populations and constraint levels

Reads from Level 2: ``*_clustered.csv``, ``sequences_*.npy``. Levels 1 and 2 do
not need to be re-run.

The engine is a synchronous weekly queueing approximation, not a behavioural
agent-based model — agents do not interact, learn, or reschedule. See
docs/MODULAR_UPDATE_GUIDE.md for what a fuller model would add.
"""
