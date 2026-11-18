# Dataset and experimental protocol

## Identity and scope

User project name: Sweeper; internal experiment key: `rq1`. This record concerns the prepared cohort-incident address graph, not the separate window5/session graph. The snapshot does not establish an external dataset landing page, canonical paper citation or redistribution licence; these must be supplied/verified before public data release.

- Nodes: 1,322,252; directed edge entries: 3,856,855.
- Labelled cohort: 4,913; Normal 636, Phishing 1,165, Scamming 3,112.
- Unlabelled context: 1,317,339, label -1; never relabelled Normal or supervised.
- Source label lists are recorded under the original `(3) RQ4.zip` naming in `source/provenance/rq1_label_policy.json`; the policy records 13 historical exclusions from the 3,125 scamming addresses. Preserve this provenance; do not silently rename original archive members.
- Five raw features: distinct_in_counterparties, distinct_out_counterparties, reciprocal_counterparties, active_days, activity_span_seconds.
- Features describe the observed cohort-incident graph, not full blockchain history. Temporal units are not independently verified in the source report; amount features are disabled. Inspect `graph_inputs_report.json` in the private snapshot for preparation details.
- Original direction, observed self-loops and multiplicity are retained. Degree for isolation sums both endpoints of original edges, as implemented by node_degree; self-loops contribute to this degree.
- Independent audit: zero degree-zero nodes in the clean released graph, including the labelled cohort. Thus clean A4 has no isolated-node cases on which its mask removal could act. This does not establish mask usefulness on other graphs or under strict visibility.

## Split and scaling

Labelled global IDs are stratified using two sklearn train_test_split calls: first test_size=.30, then .50 of the remainder, each with the run seed. Counts are 3,439 training, 737 validation and 737 test. Seeds are 100–109. StandardScaler is fitted on labelled training rows only in float64; transformed full features are cast to float32. No log1p feature transformation is added. Exact raw artifacts, all ten masks and normalized feature arrays must pass frozen SHA256 checks.

## Visibility

Clean retains the complete released graph including unlabelled context. Strict retains full node dimension but zeros forbidden features. Train uses only training features and train→train edges. Validation exposes train+validation features and train→train plus train↔validation edges. Test exposes train+test features and train→train plus train↔test edges. Query→query edges and all unlabelled-context edges are excluded. Validation nodes are absent from test visibility.

Strict uses the same fixed-snapshot input features as clean. Restricting message passing does not recompute historical features. Describe it as restricted-topology node holdout, not temporal evaluation or complete isolation from held-out activity.

## Training and selection

Adam lr=.001, weight_decay=.0005; class-weighted CE, weights computed from training labels. X-MoE uses the frozen multitask objectives and PCGrad. 300 epochs, no early stopping; highest validation Macro-F1 selects the checkpoint, earliest epoch on ties. Final test scoring follows checkpoint selection. Auxiliary diagnostics cannot change checkpoint selection. No neighbour sampling or graph truncation.

Inventory: 50 clean, 50 strict, 80 ablation, 50 sensitivity = 230 runs, all CUDA. Five models: MLP, GCN, GAT, GraphSAGE, X-MoE. Sensitivity coefficients: 0, .1, .3, .5, 1. A0 uses .3; sensitivity findings do not authorize test-based retuning.

## Statistics

Report means and sample SDs over ten seeds. Baseline comparisons use conditional-exact signed-rank tests, separate Holm families of four for clean and strict. Clean→strict uses Holm over five models. A0 vs A1–A7 uses exact paired sign-flip, Holm over seven. Strict minority metrics use a separate Holm family of four. Bootstrap intervals in the original CSV are descriptive paired intervals, not multiplicity-adjusted. Non-significance does not demonstrate equivalence. These seed-paired results are conditional on this dataset and split design, not independent real-world replications.
