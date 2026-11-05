# Publication validation — 5 November 2026

The complete study consists of 100 final CPU runs, 300 epochs each. This branch
packages their saved evidence and unchanged training implementation. Packaging
did not retrain performance models or change the reported outcomes.

Completed checks:

- All 100 completed attempts have their original checkpoint, history and
  prediction hashes verified against the original local archive.
- Published history bytes match the archive. Published JSON changes
  only absolute local path strings; original and public hashes are both recorded.
- Before packaging, all 100 locally archived prediction sets reproduce their confusion matrices, accuracy and
  per-class/Macro-F1. Each history has 300 rows and the recorded checkpoint epoch
  is the earliest maximum validation Macro-F1. Dataset summaries reproduce
  the ten-seed means and sample SDs from the run records.
- Per-node prediction files are not distributed. The public checker verifies
  aggregate confusion cells, metric consistency, histories and publication hashes;
  it does not claim to re-audit undistributed per-node probabilities.
- Shared configuration and model/training source hashes match the completed
  study. The model source files also match the current main-branch model modules
  after ignoring platform line endings.
- Sixteen existing model/adapter/selection/attempt tests pass in the recorded
  Python environment. These use synthetic fixtures, not additional final runs.
- Twelve archive, primitive-index, feature aggregation and real-context tests pass,
  including parallel edges, malformed endpoints, time handling and label checks.
- The fresh CPU check revealed that the author's RQ1 ZIP contains three nested
  group ZIPs. The extraction wrapper was corrected to support that actual layout.
  Added tests cover direct/nested members, canonical output paths, missing or
  ambiguous members and raw-size checks. This does not change training code.
- The corrected wrapper was then run on the actual outer RQ1 ZIP: all three
  nested pickles were fully extracted and matched their pinned SHA-256 values;
  the RQ4 lists produced the frozen 4,913-account cohort. This was a raw-only
  preflight, not a complete graph/feature rebuild or model training run.
- The raw-input preflight verifies all three complete pickle hashes and each
  explicit source account-list hash. Its reconstructed cohort matches all 4,913
  original canonical address/class pairs after the frozen 13 exclusions.
- Relocated copies of all ten prepared RQ1 seed arrays match the frozen hashes.
  Training-only scaler and split arrays were independently recomputed and found
  equal. Both BLTE and RQ1 pass the shared runner's all-seed verification using
  the new machine-local registry. No test inference is performed in this check.
- The user subsequently completed the full outer-ZIP reconstruction in a fresh
  CPU checkout/environment on the reference PC, using Node.js 26.13.0. All four
  rebuilt graph arrays and all 40 seed arrays independently matched the original
  study hashes. The public configuration checker confirmed exact masks and
  training-only normalized values/mean/scale/variance for every seed. BLTE input
  hashes also passed. See `provenance/fresh_cpu_input_validation.json`.
- That fresh run measured 985.531 seconds for cohort aggregation and 1280.125
  seconds for context/graph building. A full end-to-end wall time was not recorded.
- Five-epoch validation-only checks then passed for all five models on each
  dataset: selected checkpoints reloaded to identical validation logits and test
  evaluation was excluded.
- Selected full reproduction subsequently completed for seed 100: five models
  on BLTE and five on the rebuilt RQ1, each at 300 epochs. All 250 predictive
  metric comparisons (25 per run), all ten prediction-file hashes and all ten
  best-epoch selections matched the original seed-100 reference exactly. New
  checkpoint/history/prediction integrity hashes passed. This does not assert
  equality of serialized checkpoint files, which include run-specific metadata.
  See `provenance/fresh_cpu_seed100_reproduction.json`.

Validation limits:

- An internet dependency installation on a fresh second machine has not been
  exercised. The complete observed freeze and exact principal versions are
  provided. Do not claim every future package index will supply the same wheels.
- Raw ZIP-to-input reconstruction and selected seed-100 final-training reproduction
  are verified on this PC. The other nine seeds were not rerun in the fresh
  checkout. Retain the primary 100-run study separately from these ten additional
  verification runs; do not treat repeated identical results as independent data.
- The raw data, prepared graph arrays, per-node predictions and checkpoints are not hosted in Git.
  Provider access and dataset terms remain separate from the code license.
- Exact numerical equality across different hardware, libraries and thread
  settings is not guaranteed. All final results, including the RQ1 seed where
  GraphSAGE has higher Macro-F1 than X-MoE, are retained.

This branch is independent of the BLTE reconstruction and demo branches. Its
documentation does not change their study configurations or deployment state.
