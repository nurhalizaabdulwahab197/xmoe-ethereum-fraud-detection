# Manuscript guidance

## Claims supported by this locked exploratory matrix

1. X-MoE has the highest mean Macro-F1 among the five tested models in both clean and restricted-topology settings and significantly outperforms the four baselines within the prespecified comparison families.
2. The margin over GraphSAGE is 2.93 percentage points clean and 3.03 strict. Use percentage points, not percent, for these differences.
3. Strict minority findings support phishing F1/AP and scamming AP gains, not a scamming F1 gain.
4. A0 significantly outperforms the GraphSAGE-expert configuration A2. It does not significantly outperform the conflict-aware-only A1 configuration or A3–A7.
5. This clean graph has no isolated nodes; it cannot test isolate-mask benefit on isolated accounts. A4 is retained and transparently reported.

## Avoid

- Claiming every routing/loss component is necessary or significantly improves Sweeper.
- Claiming class-specific routing outperforms shared or fixed routing here.
- Selecting lambda=0 from the observed test means and presenting it as prespecified.
- Calling strict a temporal or fully leakage-free experiment. Fixed-snapshot features remain a limitation.
- Calling A2 identical to the independently trained GraphSAGE baseline.
- Treating non-significance as equivalence, calling post-outcome extensions confirmatory, or claiming controlled CPU/GPU equivalence.

## Recommended paper inventory

Main table: clean/strict mean ± SD for all five models; include seed count and selection rule. Ablation table: A0–A7, mean ± SD, signed delta and Holm-adjusted p. Plot: lambda sensitivity, descriptive, with SD. Supplement: class-wise scores/AP, connectivity strata including null values, paired tests/effect sizes/intervals, all seeds, environment and input hashes. Use a shared A0–A7 definition across datasets and state dataset-specific support.

## Draft findings for later adaptation

On the Sweeper/RQ1 cohort-incident graph, X-MoE achieved a Macro-F1 of 0.8910 ± 0.0098 under the clean protocol and 0.7828 ± 0.0158 under restricted-topology evaluation. It exceeded GraphSAGE by 2.93 and 3.03 percentage points, respectively, with Holm-adjusted p=0.0078125 in both comparison families. The ablation analysis showed a significant advantage over the GraphSAGE-expert configuration, but did not establish significant incremental gains over the conflict-aware-only configuration or the remaining routing and objective variants. Because the clean graph contained no isolated nodes, this experiment could not establish the benefit of isolate masking for isolated accounts.

Use these findings with the explicit exploratory designation and fixed-snapshot feature limitation. Complete the canonical dataset citation and access statement before submission. Do not claim Q1 acceptance is guaranteed by these results.
