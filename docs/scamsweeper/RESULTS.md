# Locked results

Mean ± sample SD; ten seeds.

| family | model | Macro-F1 | count |
|---|---|---|---|
| clean | gat | 0.5775 ± 0.0263 | 10 |
| clean | gcn | 0.7426 ± 0.0159 | 10 |
| clean | graphsage | 0.8618 ± 0.0078 | 10 |
| clean | mlp | 0.6584 ± 0.0160 | 10 |
| clean | x_moe_gnn | 0.8910 ± 0.0098 | 10 |
| strict | gat | 0.5385 ± 0.0500 | 10 |
| strict | gcn | 0.6372 ± 0.0273 | 10 |
| strict | graphsage | 0.7526 ± 0.0162 | 10 |
| strict | mlp | 0.6584 ± 0.0160 | 10 |
| strict | x_moe_gnn | 0.7828 ± 0.0158 | 10 |

## Clean ablation

| ID | Configuration | Macro-F1 | Delta vs A0 (pp) | Holm p |
|---|---|---|---|---|
| A0 | Full X-MoE | 0.8910 ± 0.0098 | — | — |
| A1 | Conflict-aware expert only | 0.8886 ± 0.0109 | -0.24 | 1.000000 |
| A2 | GraphSAGE-expert configuration | 0.8438 ± 0.0091 | -4.72 | 0.013672 |
| A3 | Shared inner and outer routing | 0.8919 ± 0.0043 | +0.09 | 1.000000 |
| A4 | No inner or outer isolate masking | 0.8910 ± 0.0098 | +0.00 | 1.000000 |
| A5 | No auxiliary Scamming-specialist loss | 0.8901 ± 0.0086 | -0.09 | 1.000000 |
| A6 | No Normal–Phishing penalty | 0.8920 ± 0.0057 | +0.10 | 1.000000 |
| A7 | Fixed 50:50 outer mixture on connected nodes | 0.8912 ± 0.0087 | +0.02 | 1.000000 |

Delta = variant minus A0, in percentage points, computed before rounding. Positive values mean a numerically higher variant score. A0 is the prespecified reference and is bolded as a reference in paper tables, not as the numerically best variant. A6 has the highest ablation mean, without a significant difference from A0.

A0 vs GraphSAGE baseline: clean +.029262 Macro-F1 (9/10 wins), strict +.030272 (10/10 wins); Holm p=.0078125 in each family. A2 is an expert configuration with its applicable auxiliary objectives, not the standalone GraphSAGE baseline; do not merge their rows.

Only A0 vs A2 is significant in the seven-variant ablation family: +.047190 Macro-F1 for A0, Holm p=.013671875. A0 vs A1 and all other variants are non-significant. The evidence does not isolate a significant incremental benefit of adding GraphSAGE to the conflict-aware expert.

X-MoE clean→strict declines by .108193, Holm p=.009765625. Strict X-MoE vs GraphSAGE improves phishing F1 by .049603, phishing AP by .044099 and scamming AP by .014550 (all Holm p=.0078125). Scamming F1 difference is -.013239, p=.130859375: no demonstrated improvement in that metric.

A4 and A0 have identical reported means and SDs; the clean graph contains no degree-zero nodes. A4 removes both the conflict expert's inner connectivity mask and the outer mixture mask. A3 shares routing across classes in both inner and outer gates. A7 fixes only the outer mixture to 50:50 for connected nodes and retains isolate-safe routing. A5 removes an auxiliary specialist loss, not the entire GraphSAGE expert. Consult the exact architecture and loss implementations in experiments/scamsweeper/source/experiments/rq1_parity/ablation_models.py and train.py.

Full-precision values and original tests: ../../results/scamsweeper/tables/completed_runs.csv, paired_statistics.csv and summary_mean_sd.csv. The formatted table is not a replacement for these source tables.
