# ScamSweeper/RQ1 — strict

Locked CUDA series: 50 run records, ten seeds (100–109), 300 epochs. Exploratory study. `per_seed.csv` retains all aggregate/class/connectivity/routing fields recorded in the run table. `result_records.jsonl` preserves the original result records and per-run provenance. `macro_f1_summary.csv` uses sample SD. `paired_statistics.csv` contains only applicable published comparison families; sensitivity is descriptive and has no inferential family.

The combined table/statistics in ../tables are authoritative originals; these family files are generated views. Raw arrays, checkpoints, prediction files and training histories remain in the verified original snapshot, SHA256 documented in ../README.md; they are not newly redistributed here. Result output hashes identify those archived artifacts.

See [code and protocol](../../../experiments/scamsweeper/README.md), [interpretation](../../../docs/scamsweeper/RESULTS.md) and [cross-dataset inventory](../../../docs/EXPERIMENT_INVENTORY.md).
