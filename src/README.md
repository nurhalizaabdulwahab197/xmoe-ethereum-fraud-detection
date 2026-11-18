# Stable static model reference

The existing src/models modules are retained unchanged from the pre-consolidation main commit. They implement the established static baselines and X-MoE. Dataset-specific runners, environments, frozen hashes and result tables now live on benchmark; source-study reconstruction is on reproduction.

The planned common encoder interface, temporal/sequence adapters and general task-head framework are not implemented merely by reorganizing these modules. Follow the benchmark guide for runnable experiments.
