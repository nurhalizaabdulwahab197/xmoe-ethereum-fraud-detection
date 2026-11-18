# Three-branch organization

| Branch | Purpose | What belongs here |
|---|---|---|
| main | General project overview and stable model reference | Architecture, status, navigation, basic setup choices and roadmap |
| reproduction | Reproduce source studies / reference implementations | Separate BLTE, Sweeper and BERT4ETH evidence, original settings and deviations |
| benchmark | Evaluate X-MoE and baselines under documented benchmark protocols | Dataset adapters, clean/strict/A0–A7/sensitivity, multi-benchmark comparisons and records |

## Status agreed with the project owner

- BLTE reproduction: completed archived evidence; existing public-wrapper validation caveats remain documented.
- Sweeper source-study reproduction: IN PROGRESS. The completed RQ1 clean and 230-run benchmark studies are NOT used to change this status to complete.
- BERT4ETH reproduction: completed according to the project owner; code, configuration and numerical evidence have not yet been imported into this repository. Its folder is a documentation scaffold, not executable reproduction code.
- RQ2 temporal encoder and cross-benchmark X-MoE extensions: planned/in development; no completed result is inferred from the architecture diagram.

## Evidence preserved during consolidation

The old main study guide and all its scientific implementations/results are retained on benchmark. The old two-dataset-comparison tree is retained byte-for-byte under benchmark/studies/matched_two_dataset_cpu. The old reproduction tree is retained byte-for-byte under reproduction/blte. Original heads and tree SHAs are listed in BRANCH_MIGRATION.json on main.

The earlier descriptive study/* branches were aliases of original studies, not additional experiments. Their original commits remain reachable through the preserved branch histories and snapshots. The target branch set is exactly main, reproduction, benchmark; obsolete branch references are removed only after preservation is verified.

## No numerical merging

The matched CPU study (BLTE 50 + RQ1 50 clean runs, plus separately attributed verification) and the later Sweeper 230-run GPU extension retain separate identifiers, environments and evidence. Reused controls are attributed; overlapping run series are not pooled as independent evidence. Original data preparation and training source bytes remain preserved.

All scientific guides retain detailed setup, input checks, smoke commands, full commands, monitoring, output interpretation and limitations. Branch names describe the purpose of a study, not one model or one ablation.
