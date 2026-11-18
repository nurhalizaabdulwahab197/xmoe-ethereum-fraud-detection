# ScamSweeper RQ4 status

RQ4 evaluates phishing detection using 1,165 phishing accounts from EPTransNet
and 636 normal accounts. The paper reports the following Table III targets:

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Yuan's method + STRWalk | 0.82 | 0.77 | 0.79 |
| DGTSD + STRWalk | 0.82 | 0.79 | 0.80 |
| BERT4ETH + STRWalk + DNN | 0.76 | 0.74 | 0.75 |
| BERT4ETH + STRWalk + random forest | 0.81 | 0.64 | 0.71 |
| ScamSweeper + TRWalk | 0.86 | 0.84 | 0.85 |
| ScamSweeper + STRWalk | 0.95 | 0.92 | 0.94 |

## What has been verified

- the released RQ4 graph has 2,973,489 nodes and 13,551,303 directed edges;
- all 1,165 released phishing labels are present;
- only 214 of the 636 normal labels occur in that graph;
- the complete 636-address normal list exactly matches the labelled source set
  in the released RQ1 normal graph;
- a static cross-release scan found transaction context for every normal target.

## Current limitation

The paper specifies a random 70%/20%/10% split, walk length 20, window size 4,
embedding dimension 128 and BERT4ETH maximum sequence length 100. It does not
release the split seed/account IDs, complete ScamSweeper/STRWalk/TRWalk code,
baseline adaptations or exact metric implementation.

The next work is therefore a controlled feasibility test of the RQ4 baselines.
An independently implemented baseline will be reported as a `paper-based
reimplementation`; only author-equivalent executable code and data may be
called an exact reproduction.
