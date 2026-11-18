# ScamSweeper / RQ1 experiments

Dataset key `scamsweeper`; the exact archived source and result records retain their original internal key `rq1`. This is the 1,322,252-node cohort-incident graph with five features and 4,913 labelled accounts, NOT the separate window5/session graph.

| Family | Source modules under source/experiments/rq1_parity | Locked runs | Results |
|---|---|---:|---|
| Clean | plan.py, train.py | 50 | [clean](../../results/scamsweeper/clean/) |
| Strict | protocol.py, train.py | 50 | [strict](../../results/scamsweeper/strict/) |
| A0–A7 ablation | ablation_models.py, train.py | 80 | [ablation](../../results/scamsweeper/ablation/) |
| Lambda sensitivity | plan.py, train.py | 50 | [sensitivity](../../results/scamsweeper/sensitivity/) |

The original source is nested to preserve ROOT-relative imports, frozen manifests and exact code hashes. Its pinned BLTE-derived reference modules are dependencies of the Sweeper engine, not additional Sweeper runs or current BLTE results. Original engine files are byte-identical to the locked CUDA snapshot.

## Reproduction (Linux, CUDA)

Use the [recorded GPU setup](../../docs/scamsweeper/REPRODUCE_GPU.md). From repository root, after environment setup:

```bash
GRAPH=/absolute/path/to/original/sweeper/graph
OUT=/absolute/path/to/NEW/scamsweeper-gpu-results
python experiments/scamsweeper/run.py --stage preflight --graph "$GRAPH" --output "$OUT" --threads 24
python experiments/scamsweeper/run.py --stage smoke --graph "$GRAPH" --output "$OUT" --threads 24
python experiments/scamsweeper/run.py --stage full --graph "$GRAPH" --output "$OUT" --threads 24
```

Proceed only after the preceding stage passes. This entry point requires an explicit output folder to protect the locked evidence. It executes the archived suite in a fresh process with its original working directory. It does not change training. Do NOT use --include-gpu: the captured matrix already contains 230 CUDA rows. No family-only selector exists in the archived runner. Smoke scores are not paper results.

Source-only description without scientific dependencies: `python experiments/scamsweeper/run.py --describe`.

[Protocol](../../docs/scamsweeper/PROTOCOL.md) • [Results](../../docs/scamsweeper/RESULTS.md) • [Paper guidance](../../docs/scamsweeper/MANUSCRIPT_GUIDE.md).
