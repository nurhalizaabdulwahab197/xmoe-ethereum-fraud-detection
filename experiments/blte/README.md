# BLTE experiments

These launchers use the unchanged historical main-branch implementations. Defaults are explicit: 300 epochs, seeds 100–109, CPU; outputs go to results/new_runs/blte/<family>. Old entry points remain for compatibility with published commands and provenance hashes.

| Family | Dataset-labelled entry | Scientific implementation | Existing evidence |
|---|---|---|---|
| Clean | clean.py | ../../src/runner.py, protocol clean | ../../results/blte/clean/ |
| Strict | strict.py | ../strict_inductive.py | ../../results/blte/strict/ |
| A0–A7 | ablation.py | ../ablation.py | ../../results/blte/ablation/ |
| Lambda sensitivity | sensitivity.py | ../../src/runner.py, explicit penalty | ../../results/blte/sensitivity/ |

Run from repository root, for example:

```bash
python experiments/blte/clean.py --model x_moe_gnn --upstream /path/to/upstream --save-checkpoints --save-predictions
python experiments/blte/strict.py --model x_moe_gnn --upstream /path/to/upstream --save-predictions
python experiments/blte/ablation.py --variant all --upstream /path/to/upstream
python experiments/blte/sensitivity.py --x-moe-normal-phishing-penalty 0.10 --upstream /path/to/upstream --output-dir results/new_runs/blte/sensitivity/lambda_0.10 --save-predictions
```

Use a separate output folder for each model/coefficient/device invocation; examples are commands for new experiments, not evidence of reruns during organization. Sensitivity takes one explicit lambda per invocation. Values: 0, .1, .3, .5, 1. --device cuda selects a compatible GPU environment; do not imply it reproduces a historical CPU value exactly.

The recorded sensitivity table has 50 seed/coefficient rows; its metric-completion evidence uses 30 fresh runs plus 20 attributed A0/A6 reused rows. Counts of reporting rows are not counts of additional independent trainings. Historical GPU coverage is four GNNs, 40 rows; it is not the new Sweeper H200 suite.

The reconstruction and PCGrad ON/OFF study remains on the reproduction branch (120 primary reconstruction runs, 40 PCGrad runs, six separate fixed-mask pilots). It must not be called the A0–A7 component ablation.

[Full BLTE reproduction guide](../../docs/blte/REPRODUCTION.md) • [Cross-dataset inventory](../../docs/EXPERIMENT_INVENTORY.md).
