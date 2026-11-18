# ScamSweeper reproducibility status

This page states only what can and cannot be reproduced from the authors'
released artifacts. It does not publish exploratory substitute experiments.

## Released data

| Paper question | Released artifact | Contents relevant to reproduction |
|---|---|---|
| RQ1 | `RQ1.zip` | Normal, phishing and Web3-scam NetworkX transaction graphs |
| RQ2/RQ3 | `RQ2&3.zip` | Processed windows 5, 10 and 15, plus no-limit split pickles |
| RQ4 | `RQ4.zip` | EPTransNet graph, 1,165 phishing labels, 636 normal labels and 3,125 Web3-scam labels |
| Case study | `Cases.zip` | Two detected-pattern CSV examples |

The accompanying README says the original transaction dataset is too large to
release and asks users to contact the authors when it is needed.

## Reproducibility decision

| Paper result | Status | Reason |
|---|---|---|
| RQ1 Table I / Figure 6 statistics | Reproduced from released artifacts | Complete graph statistics are present and the reported node, edge and degree-SD values can be recomputed deterministically |
| RQ2 Figure 7 | Not exactly reproducible | Paper uses 3,125 Web3 scams plus 3,125 randomly selected negatives; released windows contain only 3,076-3,125 total sessions and 710-716 negatives |
| RQ3 Table II | Not exactly reproducible | It depends on the same missing balanced cohort and the unreleased ScamSweeper/ablation implementation |
| RQ4 graph and cohort identity | Reproduced | All 1,165 phishing labels are in the released RQ4 graph; all 636 normal labels exactly match the RQ1-normal labelled source set |
| RQ4 Table III | Under investigation | The paper reports the cohort and several parameters, but not the split seed/account IDs, complete model code, baseline adaptations or metric implementation |

## Why RQ2/RQ3 comparisons are not reported

The audited RQ2/RQ3 archive contains no executable STRWalk, ScamSweeper,
baseline, split or metric code. Its released cohort also contradicts the
balanced sample count described in the paper. Training another model on those
pickles would use a different population and protocol. Such results cannot be
presented as a reproduction or as a fair ScamSweeper-versus-X-MoE comparison.

No RQ2/RQ3 substitute model code or exploratory result is published in this
branch. If the missing cohort and author implementation become available, the
decision can be revisited.

## RQ4 next step

RQ4 is evaluated separately because its 1,165 phishing and 636 normal account
identities can be reconstructed from the released RQ4 and RQ1 artifacts. The
next stage is to test whether the paper's Table III model and baselines can be
faithfully executed. Any independently implemented method will be labelled
`paper-based reimplementation`, not `exact reproduction`.
