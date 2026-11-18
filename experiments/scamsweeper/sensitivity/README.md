# ScamSweeper/RQ1 — sensitivity code

This family is implemented by the exact captured unified runner in ../source/experiments/rq1_parity, not the BLTE engine. The locked matrix includes 50 sensitivity rows. Definitions: plan.py and train.py; strict visibility: protocol.py; A0–A7: ablation_models.py; summaries: report.py.

Use [../README.md](../README.md) to run preflight → smoke → full. The exact archived runner executes all four families; it has no family-only CLI selector. Do not substitute BLTE scripts or invent a --family option. The result family is [results/scamsweeper/sensitivity](../../../results/scamsweeper/sensitivity).
