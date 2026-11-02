# BLTE five-model alignment

Local integration package, 2 November 2026. Publication and repository locking are pending.

| Surface | Prepared change | Remaining verification |
|---|---|---|
| Local code | Training/model code preserved; five-model summary script added | Apply guarded installer and run checks |
| Results | 50 clean and 50 strict records; full-precision summaries | Strict predictions were audited on the user's PC; not bundled here |
| Demo | Five-model clean/strict results, minority metrics and protocol statistics | Test updated Streamlit app locally and after deployment |
| Manuscript | MLP strict, revised Holm families, tables, Figure 4 | Compile with the complete original LaTeX project |
| GitHub | Patch ready for commit | Commit, push and verify deployed commit |
| Freeze | No freeze claimed | Tag/release, protection and live demo verification |

The main benchmark remains BLTE only. ScamSweeper and Elliptic are separate projects.

The Try a Prediction page uses clean seed-100 checkpoints. Strict charts show ten-seed aggregate results; they do not select or replace inference checkpoints.

## Statistical families

- Clean Macro-F1: four X-MoE-versus-baseline comparisons.
- Strict Macro-F1: four X-MoE-versus-baseline comparisons.
- Clean-to-strict Macro-F1: five within-model comparisons.
- Strict minority comparison against GraphSAGE: four metrics (Phishing F1/AP, Scamming F1/AP).
- Ablation: seven exact paired sign-flip comparisons; unchanged.

All-zero MLP clean-to-strict differences are recorded as p=1 (no observed change); this is not a general equivalence test. Scamming F1 adjusted p is 0.275390625 when recomputed from the supplied strict records, replacing the old 0.283203. The non-significant conclusion is unchanged.

The 50 strict records reproduce the old four-model means and SDs; this does not recover the missing historical raw CSVs. Preserve the new files as a reproduction series with its own provenance.

## Existing evidence limitation

Efficiency summaries remain descriptive and hardware-dependent. This patch does not supply missing original ten-seed timing/parameter logs, regenerate connectivity evidence, or independently rerun model training. Consult `results/paper/AUDIT_STATUS.md` before freezing.

The curated strict CSV uses repository-relative evidence/prediction paths for portability. Original per-run CSV hashes are preserved. The generated per-run files remain under ignored local results folders and should be archived as release evidence before freezing.
