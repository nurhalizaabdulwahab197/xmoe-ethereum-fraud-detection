# RQ1 journal experiment parity

## Scientific execution status — 12 November 2026

**In progress: the full 230-run scientific RQ1 journal suite is not yet verified
complete.** No completed-run count is asserted from this checkout. Smoke and
synthetic implementation tests are not final performance results. The completed
100-run BLTE/RQ1 clean study is separate; its newly audited metric tables are in
[COMPLETED_METRICS.md](COMPLETED_METRICS.md). Keep the prespecified matrix below
and retain all completed/partial local evidence.

This branch adds fresh **exploratory** experiments after existing RQ1 outcomes
were inspected. It does not overwrite reference runs, retrain the deployed demo,
or turn existing benchmark predictions into new experimental results.
The live platform and private repository are unchanged.

## Experiment inventory audited against BLTE

| BLTE evidence / analysis | RQ1 extension | CPU runs |
|---|---|---:|
| Five-model clean comparison, seeds 100–109 | MLP, GCN, GAT, GraphSAGE, X-MoE | 50 |
| Five-model strict evaluation, seeds 100–109 | Same five models, visibility audits | 50 |
| Eight X-MoE ablations A0–A7 | Identical variant definitions and objectives | 80 |
| Normal–Phishing objective sensitivity | λ = 0, 0.10, 0.30, 0.50, 1.00 | 50 |
| Connected / isolated performance | Overall and per-class counts and metrics in every run | included |
| Class-wise precision/recall/F1, minority AP, confusion matrices | Frozen BLTE score function | included |
| Validation convergence, timing, RAM | Per-epoch history, worker time and RAM monitor | included |
| Routing diagnostics | Per-class expert weights / graph gate where defined | included |
| Mean ± sample SD, paired statistics | Separate signed-rank / sign-flip + Holm families | included |
| CPU/GPU comparison (four GNNs in original BLTE CUDA evidence) | Optional CUDA execution, 10 seeds | +40 GPU |

Default full CPU matrix: **230 runs**, each 300 epochs. This includes fresh clean
runs to give strict/clean comparisons the same runner, machine and thread policy.
Archived results remain preserved for comparison; no seed or configuration is
selected based on RQ1 test performance. There is no requirement to make X-MoE win.
GPU parity is **pending unless separately executed on CUDA hardware**. The existing
`.venv-cpu` cannot execute CUDA runs; use a separate compatible GPU environment,
invoke Python directly, and choose a new output directory for a GPU-inclusive matrix.
Do not replace CPU torch in the existing reproduction environment.

## Scientific protocol and limitations

- Exact original raw RQ1 artifact hashes are checked against
  `provenance/rq1_expected_arrays.json`. Features cannot be reconstructed from the
  rounded float32 demo inference inputs for this purpose.
- Split only the 4,913 labeled accounts in global node-ID order: stratified
  70/15/15, seeds 100–109, same two sklearn calls as BLTE.
- Refit StandardScaler on labeled training rows only, without added log1p.
  Rebuilt float32 features and all three masks must match the frozen seed hashes.
- Native five features remain five; they are not semantically identical to BLTE's 23.
- Clean: original full graph, including unlabeled context, retained.
- Strict train: training features and train→train edges only.
- Strict validation: train + validation features; train→train and cross
  train↔validation edges only. No validation→validation edges; no test/context.
- Strict test: train + test features; train→train and cross train↔test edges only.
  No test→test edges; no validation/context.
- All unknown/unlabeled context nodes are masked and all their incident edges
  excluded in strict views. Assigning them to training would silently change the
  meaning of the BLTE protocol; no temporal availability is documented for them.
- Original direction, self-loops and edge multiplicity are retained where allowed.
  Features for forbidden nodes are zero; the full node dimension is retained,
  matching BLTE's feature masking and dropout execution approach.
- **Important scope:** raw RQ1 count/time features were computed from the fixed
  full observation snapshot. Restricting message-passing edges does not remove
  that historical information from a visible node's feature values. This is a
  **restricted-topology node-holdout analysis with fixed snapshot features**, not
  proof of temporal generalization or complete isolation from held-out activity.
  The corresponding limitation also needs discussion for BLTE's released features.
  A future time-causal/feature-recomputed experiment would be a separate protocol.
- Adam lr .001, weight decay .0005; weighted CE; frozen auxiliary losses; PCGrad
  for X-MoE; no class-bias calibration; 300 epochs; highest validation Macro-F1,
  earliest checkpoint on ties; no early stopping or test-set model selection.
- GAT edge chunking and activation recomputation / X-MoE expert recomputation use
  the existing shared engine. No node/neighbor sampling or graph truncation.
- Final test evaluation occurs after checkpoint selection. A later diagnostic
  explanation pass uses the selected model; it cannot change checkpoint selection.
- Empty connectivity strata are undefined (`null`), never zero performance.
- Smoke tests are one epoch and cannot be included in manuscript tables.

## Statistical families

1. Clean X-MoE vs four baselines: paired conditional-exact signed-rank, Holm ×4.
2. Strict X-MoE vs four baselines: same, separate Holm ×4.
3. Clean→strict change: signed-rank, Holm across five models.
4. A0 vs seven ablations: exact paired sign-flip, Holm ×7 (matches BLTE ablation test).
5. Strict X-MoE vs GraphSAGE minority F1/AP: signed-rank, separate Holm ×4.
6. Optional CPU/GPU change: signed-rank, separate Holm ×4.

Paired mean differences, d_z (undefined if SD zero), descriptive percentile paired
bootstrap intervals (10,000 resamples, fixed RNG) and wins are included. These
intervals are not multiplicity-adjusted. Sensitivity curves are descriptive, not
a test-set tuning procedure. Means, sample SDs and nonempty-stratum counts are
reported. The report refuses final significance tables if planned runs are missing. It also
checks that identical controls (clean vs strict MLP, full X-MoE vs A0 vs λ=.30,
and A6 vs λ=0) have identical saved predictions within each seed.

## Windows / PC CS-NUR.HALIZA

Use a **new checkout** of `experiment/rq1-journal-parity`. The existing research
checkout, assets, checkpoints, secrets and live platform do not need editing.
The launcher uses the existing Python executable at
`%USERPROFILE%\Documents\xmoe-cpu-reproduction\.venv-cpu\Scripts\python.exe`.

From the new checkout, in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rq1_journal.ps1 -Stage preflight
powershell -ExecutionPolicy Bypass -File .\scripts\run_rq1_journal.ps1 -Stage smoke
```

Auto-discovery searches Documents and Desktop for the exact original raw graph
files. If needed specify `-Graph 'C:\path\to\original\prepared_graph'` on each
command. The demo inference package alone is insufficient. Preflight reconstructs
and verifies all ten splits/scalers without training or testing a model.
Smoke executes all 23 CPU configurations once with seed 100 and one epoch.
Read the timing and memory logs before starting full training:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_rq1_journal.ps1 -Stage full
```

Keep the PC awake. Each run executes in a separate process, writes its log, and
stops at >20 GiB process RAM, <4 GiB available host RAM, or a six-hour run timeout.
An interrupted run is never treated as complete. Preserve its folder/log and move
only that incomplete folder aside to retry. Verified completed runs resume without
retraining; modified source/configuration requires a **new output directory**.

Results: `results/new_runs/rq1_journal_parity/`. Only `full/` is scientific output.
`tables/completion.json` declares exactly what is complete/missing. Dataset arrays,
checkpoint binaries and local run directories must not be committed accidentally.
Manuscript tables and demo results should be updated only after full result audit.

## Developer checks (no scientific performance claims)

```powershell
python .\tests\journal\test_rq1_parity.py
python .\tests\journal\check_blte_equivalence.py
```

The first verifies context exclusion, actual feature/topology visibility, split
coverage, statistics, matrix inventory, output tamper detection and training of
all 23 configurations on a tiny synthetic graph. The second compares the extension
against pinned BLTE clean, strict and eight-variant ablation implementations on
synthetic labeled data. Pinned BLTE reference scripts and source hashes are kept
for review; they do not constitute RQ1 experiments.
