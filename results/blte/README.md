# BLTE — dataset-labelled historical evidence

This directory provides byte-identical, dataset-labelled views of the historical BLTE results. Each file retains the original Git blob identity. Existing results/clean, strict, ablation, sensitivity, cpu_gpu, tables and metrics_completion paths remain for old commands, audit scripts and recorded hashes. They contain BLTE evidence only, not Sweeper runs.

Start with [tables](tables/) and [completed metric audit](metrics_completion/). The latter records provenance, original-score agreement and A0/A6 reuse. The canonical-to-legacy mapping is in ../../docs/DATASET_PATH_MAP.json. Relative path strings inside copied original JSON/Markdown remain historical; use the map and original paths for verification.

[BLTE code](../../experiments/blte/README.md) • [BLTE guide](../../docs/blte/README.md) • [cross-dataset inventory](../../docs/EXPERIMENT_INVENTORY.md).
