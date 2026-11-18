# Completed metric evaluation and provenance

Audit date: 12 November 2026. These files extend the recorded evaluations with
metrics recomputed from saved test probabilities. No training or inference is
performed by the publication checks. Training for the attributed supplementary
series was performed on the author's PCs before this audit. Frozen training
sources, historical result tables and model checkpoints are not replaced.

## Definitions

- Classification: Normal=0, Phishing=1, Scamming=2; predicted class is probability
  argmax. All metrics in a run refer to the same saved prediction file.
- Precision/recall/F1: report each class and unweighted macro averages. Weighted
  averages use true-class test support. A fixed three-label convention assigns
  zero precision/recall/F1 where the relevant denominator is zero.
- ROC-AUC: calculate one-versus-rest separately for all three classes, assigning
  half-credit to tied scores. Macro ROC-AUC is their unweighted mean; weighted
  ROC-AUC uses true-class support. Do not calculate AUC from hard predictions.
- Average precision (AP): non-interpolated precision weighted by recall increments,
  with equal-score observations grouped at one threshold. AP is not ROC-AUC or
  trapezoidal PR-AUC. An absent positive class has undefined AP.
- Balanced accuracy: macro recall over the three named classes. MCC uses the
  multiclass confusion-matrix definition, with zero for a zero denominator.
- Statistics: calculate every metric per seed, then the mean and sample standard
  deviation (`ddof=1`) over seeds 100-109. Do not pool predictions across seeds.
  Precision/recall/F1/AUC/AP values are fractions in CSV and percentages in the
  displayed tables; support is a count and MCC ranges from -1 to 1.
- Undefined ROC-AUC: if a class lacks positives or negatives, report null/N/A,
  never zero. The portable utility returns null macro/weighted ROC-AUC when any
  required class ROC-AUC is undefined. Isolated BLTE subsets containing only
  Normal cannot support ROC-AUC; report Normal performance/false-positive rate.

## Evidence and limits

Only aggregate metrics, confusion counts where available, and prediction hashes
are published. Per-address predictions, raw input graphs, model weights,
personal machine paths and private repository contents are not part of this
update. Matching saved probabilities to the original recorded scores is stronger
than matching an aggregate mean alone. The provenance for each series is stated
in its manifest entry and validation record.

A0-A7 and three historical GPU baseline series are supplementary runs that
reproduce the original recorded classification scores. Their new AUC and F1
are calculated from the same supplementary predictions. Equality of old scores
does not prove byte-identical old probabilities or checkpoints. Sensitivity
reuses A0 at lambda=0.30 and A6 at lambda=0.00; those 20 rows are shared evidence,
not independent additional runs. Reproduction PCGrad is a different ablation
from A0-A7. Checkpoint files were excluded from the uploaded audit evidence, so
this publication does not establish a new independent checkpoint-reload audit.

CPU/GPU stacks and protocols differ between historical `main` and reconstructed
`reproduction` experiments. Do not combine their scores into one mean or compare
concurrently measured runtimes as an equal-load efficiency benchmark. Existing
paired statistical tests concern their prespecified Macro-F1 families; this
update adds no AUC significance claims and does not select a new model/penalty
from test AUC. Lambda=0.30 remains the reference coefficient.

## Read-only verification

From the repository root, with Python's standard library only:

```console
python -B scripts/check_completed_metrics.py
python -B scripts/test_completed_metrics.py
```

The checker verifies artifact hashes, exact group/seed coverage, required metrics,
class supports and every published mean/sample SD. It does not replace a raw-data
or training audit. To evaluate a locally retained probability CSV without loading
a model or modifying it:

```console
python -B scripts/evaluate_saved_predictions.py --predictions PATH_TO_PREDICTIONS.csv --output NEW_METRICS.json
```

Supported headers include the historical `true_label`/`normal_probability` and
WSL `y_true`/`prob_normal` forms, plus the audited alternate probability naming.
Stored argmax predictions and unique node IDs are checked when present. The
output includes the input file hash and aggregate scores, never per-node data.
