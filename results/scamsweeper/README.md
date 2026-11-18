# ScamSweeper / RQ1 — locked GPU results

230 complete CUDA runs: clean 50, strict 50, A0–A7 ablation 80, coefficient sensitivity 50; seeds 100–109, 300 epochs. These are separate from BLTE results.

- [Combined original tables](tables/), [clean](clean/), [strict](strict/), [ablation](ablation/), [sensitivity](sensitivity/).
- [Exact captured source and launch instructions](../../experiments/scamsweeper/README.md).
- [Protocol](../../docs/scamsweeper/PROTOCOL.md), [results](../../docs/scamsweeper/RESULTS.md), [paper guide](../../docs/scamsweeper/MANUSCRIPT_GUIDE.md).
- [Independent artifact audit](provenance/INDEPENDENT_AUDIT.json) and [original experiment lock](provenance/experiment_lock.json).

Original complete snapshot: `sweeper-rq1-gpu-230-20261117T172514Z`; complete TAR SHA256 `9bf81e5103a3c83db2ef7f2d5068af4cace8555a833753200097063e52dd181b`. Raw input/checkpoint/prediction/history access is not bundled in these public tables. The snapshot is retained by the project owner.

Historical completion flags gpu_included=false and gpu_parity_completed=false describe the disabled optional CPU/GPU comparison; every recorded run device is cuda. Preserve those originals. This was not a controlled CPU/GPU parity experiment. No training was performed during GitHub organization.
