# BLTE reproduction and X-MoE evaluation report

**Review draft, 4 November 2026. Proposed for branch `reproduction`; not yet published.**

## 1. Purpose and scope

We examined whether the experiments in the multiclass BLTE reference could be reconstructed from its paper, source code and processed artifacts. We include an explicitly documented MLP baseline and evaluate Full X-MoE under the reconstructed data/evaluation conditions. X-MoE–CE serves as a training-control variant for the combined additional training package.

This report separates source fidelity from new-method evaluation. Running an implementation successfully is not equivalent to recovering its published numerical results. Likewise, a proposed model is not an author model simply because it uses the same evaluation protocol.

The primary comparison contains **120 runs**: six method entries, two configurations and ten reconstructed partitions. Six released-mask pilots are separate from these primary summaries. Additional implementation checks are not included in the 120-run count.

The logical presentation is:

**BLTE reconstruction → additional MLP → Full X-MoE evaluation → X-MoE–CE training control → clean/strict evaluation → component ablation.**

Our clean/strict experiments on `main` and later component ablations are separate research stages. The order of presentation is not a claim that every development decision was caused by the audit findings.

## 2. Sources

Multiclass reference: **Explainable Graph Neural Networks for Multi-Class Fraudulent Address Classification in Ethereum**, *Future Internet* 2026, 18, 461. [DOI: 10.3390/fi18090461](https://doi.org/10.3390/fi18090461).

Upstream code: [Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network](https://github.com/salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network), pinned commit **`56afa30511fec8bf53d1f8270d47e5793186aadb`**. Statements about available author code refer to this audited snapshot, not an assertion about future repository versions.

Relevant author scripts:

- `GCN_training_GNNExplainer.py`
- `GAT_training_GNNExplainer.py`
- `GraphSAGE_training_GNNExplainer.py`

The original BLTE dataset publication is distinct from this multiclass reference. Dataset attribution should also follow the upstream references, including [A Labeled Transactions-Based Dataset on the Ethereum Network](https://doi.org/10.1007/978-981-33-6835-4_5).

## 3. Verified input artifacts

| Artifact | SHA-256 |
|---|---|
| `BLTE_multiclass_graph.pt` | `2859a8a3a815edd02b5cd85d46c00d730b2b66c4e8d047145db2e843c9e38456` |
| `BLTE_multiclass_v2.csv` | `02e376ddd98ba595372f19731fe5e51bbc90bf897b2a52b3c8bb98e786423b11` |

The graph contains 73,033 nodes, 23 features and 6,588 directed edges. CSV labels match graph node order. Features were finite in the input check. Label counts are Normal 72,865, Phishing 69 and Scamming 99.

Released masks:

| Partition | Nodes | Normal | Phishing | Scamming |
|---|---:|---:|---:|---:|
| Train | 51,123 | 51,006 | 48 | 69 |
| Validation | 10,955 | 10,929 | 11 | 15 |
| Test | 10,955 | 10,930 | 10 | 15 |

No overlap or unassigned nodes was detected in the released masks. These fixed masks were used in the pilots, not substituted for the ten reconstructed primary partitions.

### Preprocessing observation

Under the verified CSV feature order, graph features matched a StandardScaler transformation fitted on **all nodes** exactly in the numerical comparison. Train-only scaling did not match. This establishes numerical equivalence for the released artifact, not a recovered history of every author run.

The primary reconstruction retains the released feature matrix to preserve that artifact. It is not a train-only preprocessing experiment, and this choice should not be presented as leakage-free inductive preprocessing. We do not attribute any particular score gap solely to scaling without a controlled comparison.

## 4. Environment

| Component | Recorded audit environment |
|---|---|
| OS | Ubuntu 22.04.3 LTS under WSL 2 |
| Python | 3.10.12 |
| PyTorch | 2.0.1+cu118 |
| CUDA runtime | 11.8 |
| PyTorch Geometric | 2.3.1 |
| torch_scatter | 2.1.1+pt20cu118 |
| torch_sparse | 0.6.17+pt20cu118 |
| NumPy / SciPy | 1.24.4 / 1.10.1 |
| pandas / scikit-learn | 2.0.3 / 1.3.2 |
| Matplotlib / seaborn | 3.7.3 / 0.12.2 |
| NetworkX | 3.1 |
| GPU | NVIDIA GeForce GTX 1660 SUPER, 6 GB |
| Host driver | 595.95 |

The paper reports Ubuntu 22.04, Python 3.10, PyTorch 2.0.1, PyG 2.3.1 and an RTX 3090 with 24 GB. We matched principal software versions but do not claim identical hardware, OS execution context or every dependency. The audit's CUDA build is documented; it is not asserted to be a recovered author CUDA version.

## 5. Paper, released code and our reconstruction

| Item | Paper | Audited released code | Action in our evaluation |
|---|---|---|---|
| Learning rate | 0.01, Table 8 | 0.001 | Retain separate configurations |
| GAT hidden dimension | 64 | 32 | 32 in released-code; 64 per first-layer head in reconstruction |
| GAT heads | 8 | First layer 4 | First layer changed to 8 in paper-described reconstruction |
| GAT activation | ReLU in Table 8 | ELU | ReLU in paper-described reconstruction |
| Remaining GAT structure | Not fully recoverable from summary table | Second layer one head, `concat=False`, final linear classifier | Inherited and disclosed as assumptions |
| GCN / GraphSAGE | Two graph layers, hidden 64 | Corresponding released architectures | Keep architecture; LR distinguishes the two configurations |
| Epochs / optimizer / decay | 300 / Adam / 0.0005 | Same | Retained |
| Class imbalance | Class-weighted CE | Normalized inverse training-class frequency | Retained for CE models and Full base loss |
| Model selection | Repeated assessment described | Highest validation Macro-F1 | Retained |
| Repeated partitions | Ten stratified seeds described | Scripts load fixed released masks; no recovered ten-split procedure | Explicit reconstructed split procedure |
| Numerical seeds / exact partitions | Not recovered | No training-seed specification recovered from scripts | Our seeds 100–109, saved indices shared across methods |
| MLP | Included in baseline results | Verified MLP implementation not found in audited code | Our MLP, clearly labeled |

These discrepancies justify documenting two configurations. They do not prove that a particular discrepancy caused all differences from the publication or identify which implementation generated the published tables.

## 6. Reconstructed repeated evaluation

For seeds 100–109, scikit-learn 1.3.2 performs two stratified `train_test_split` calls: 70/30, followed by a 50/50 split of the remainder. The same seed is used in both calls. Indices are sorted, validated and saved. Initialization uses the corresponding seed.

Every method/configuration uses the exact saved partition for a given seed. Ten partitions were checked as unique, complete and non-overlapping internally, with all classes represented. Different repetitions can overlap; they are not independent datasets. None matched the released mask combination.

Features, labels and edge directions are retained from the released graph; no synthetic reverse edges are introduced by this evaluation. MLP ignores edges. Other models receive the retained graph, so the audit does not claim the strict visibility restrictions used in the main-branch strict experiment.

We use validation Macro-F1 for checkpoint selection and test predictions for final reporting. We do not select seeds or replace reference runs according to whether a proposed model wins. The primary series uses 300 epochs throughout.

## 7. Method definitions

### Additional MLP

Our MLP has 23 inputs, two hidden layers of width 64, LayerNorm, ReLU, dropout 0.5 and a three-class classifier. It accepts an edge argument for interface compatibility but does not use it. Its architecture is not verified as the author's MLP.

### Full X-MoE

The proposed architecture is unchanged, but its complete training package is retained:

| Objective / procedure | Recorded setting |
|---|---|
| Final multiclass objective | Training-frequency weighted CE |
| Auxiliary multiclass CE for each expert | Coefficient 0.25 each |
| Conflict-aware one-versus-rest class tasks | Coefficient 1.0 each |
| GraphSAGE Scamming-versus-rest specialist | Coefficient 0.5 |
| Normal–Phishing penalty on connected Normal training nodes | Coefficient 0.30 |
| Gradient combination | Existing fixed-task-order PCGrad; mean projected task gradients |
| Class-bias calibration | Off |

The coefficients were taken from the existing study configuration, not selected by searching these new test results. Test outcomes from earlier work were already known, so this extension is exploratory.

The Full runner adaptation adds a learning-rate parameter while preserving the existing training logic. Its internal `artifact` loader receives a cloned graph with the reconstructed masks. Results are labeled `blte_reconstructed_full_xmoe` to avoid implying original fixed-mask evaluation.

Full-versus-CE comparisons measure the combined additional training package. They do **not** isolate PCGrad, individual task losses or the penalty. PCGrad is an existing technique; the fixed-order implementation must be described rather than assumed identical to every published PCGrad variant.

### X-MoE–CE training control

The CE-only control uses the same proposed architecture. This architecture combines a conflict-aware expert containing feature/graph pathways with an independently parameterized GraphSAGE expert, connectivity-aware routing, disagreement inputs and class-specific mixture weights. The CE-only version trains its final logits with weighted multiclass CE and ordinary backpropagation. No PCGrad, auxiliary objectives, penalty or class-bias calibration is applied.

This control assesses the complete additional training package; it is not a separate proposed final method.

## 8. Fixed-mask pilots

These six runs are separate from the primary repeated comparison:

| Model | Released-code pilot Macro-F1 | Paper-described pilot Macro-F1 |
|---|---:|---:|
| GCN | 34.8166% | 44.8670% |
| GAT | 24.7526% | 33.0145% |
| GraphSAGE | 47.6724% | 62.7779% |

All use initialization seed 100 and the released masks. Released-script pilots completed training, evaluation, plotting and GNNExplainer. The seed addition, input/source hashes and adaptation differences were recorded. Successful explanation execution is not a separate validation of explanation faithfulness.

## 9. Primary results: six method entries

Mean test Macro-F1 ± sample SD (`ddof=1`), percent:

| Method | Released-code settings | Paper-described settings |
|---|---:|---:|
| Our MLP | 37.4544 ± 1.3567 | 37.7082 ± 1.1833 |
| GCN | 36.5333 ± 2.1597 | 41.4269 ± 2.2639 |
| GAT | 26.6442 ± 2.4989 | 28.5816 ± 4.5819 |
| GraphSAGE | 48.9195 ± 2.7373 | 52.0641 ± 3.1524 |
| X-MoE–CE | 48.5667 ± 3.3531 | 60.7443 ± 5.3174 |
| **Full X-MoE** | **67.6952 ± 6.9504** | **71.1905 ± 4.9837** |

The three conventional GNNs retain the mean ordering GraphSAGE > GCN > GAT in both configurations. X-MoE–CE exceeds the other methods under paper-described settings, but not GraphSAGE under released-code settings. Full X-MoE has the highest mean Macro-F1 in both evaluated configurations.

Full X-MoE has greater run variation than GraphSAGE. A higher mean is not a claim of lower variance, uniformly better calibration or lower computation cost.

### Minority-class F1

Values below are means in percent, not F1 computed from a pooled confusion matrix:

| Configuration | Method | Normal F1 | Phishing F1 | Scamming F1 |
|---|---|---:|---:|---:|
| Released-code | MLP | 98.5308 | 6.5858 | 7.2466 |
| Released-code | GCN | 94.0857 | 1.3722 | 14.1421 |
| Released-code | GAT | 75.9381 | 0.3054 | 3.6890 |
| Released-code | GraphSAGE | 99.4447 | 10.9296 | 36.3842 |
| Released-code | X-MoE–CE | 99.5118 | 14.0513 | 32.1371 |
| Released-code | Full X-MoE | 99.8676 | 58.7740 | 44.4440 |
| Paper-described | MLP | 98.3828 | 4.7828 | 9.9589 |
| Paper-described | GCN | 98.0344 | 4.4582 | 21.7881 |
| Paper-described | GAT | 82.6932 | 0.4568 | 2.5947 |
| Paper-described | GraphSAGE | 99.5218 | 11.9130 | 44.7574 |
| Paper-described | X-MoE–CE | 99.8562 | 42.0710 | 40.3058 |
| Paper-described | Full X-MoE | 99.9419 | 56.6760 | 56.9536 |

Full X-MoE improves mean Phishing and Scamming F1 over GraphSAGE in both configurations. Minority test support is small, so individual errors can substantially change these values. No class-specific significance tests are claimed here.

## 10. Statistical comparisons

The Full-X-MoE analysis uses ten matched seed-level differences per comparison, two-sided exact signed-rank sign enumeration and Holm correction jointly across the following ten comparisons. Differences were rounded to 12 decimal places for explicit zero/tie handling.

| Configuration | Full X-MoE minus comparator | Mean difference (pp) | Full higher / 10 | Holm-adjusted p |
|---|---|---:|---:|---:|
| Released-code | MLP | 30.240796 | 10 | 0.01953125 |
| Released-code | GCN | 31.161877 | 10 | 0.01953125 |
| Released-code | GAT | 41.051034 | 10 | 0.01953125 |
| Released-code | GraphSAGE | 18.775684 | 10 | 0.01953125 |
| Released-code | X-MoE–CE | 19.128460 | 10 | 0.01953125 |
| Paper-described | MLP | 33.482325 | 10 | 0.01953125 |
| Paper-described | GCN | 29.763600 | 10 | 0.01953125 |
| Paper-described | GAT | 42.608927 | 10 | 0.01953125 |
| Paper-described | GraphSAGE | 19.126419 | 10 | 0.01953125 |
| Paper-described | X-MoE–CE | 10.446181 | 10 | 0.01953125 |

Raw p is 0.001953125 in all ten comparisons. Their identical adjusted p-values result from the same raw values and joint Holm family; they are not independent replications of significance.

Earlier CE-only analyses are separately recorded: X-MoE–CE versus GraphSAGE had adjusted p = 0.6953125 in the released-code configuration and 0.01953125 in the paper-described configuration, with Holm correction across eight CE comparisons. These families were not one prospective global confirmatory plan; no global correction across every development analysis is claimed.

The tests are exploratory. Repeated partitions overlap, and the resulting differences are not independent-dataset replications. Nominal exact signed-rank calculations do not remove these dependencies. Prior test exposure and subsequent method extensions also preclude treating this as untouched-test confirmation. Non-significance does not demonstrate equivalence.

## 11. Descriptive comparison with the published paper

Table 11 reports Macro-F1 24.93 ± 1.31% for GAT, 38.68 ± 2.64% for GCN, 51.08 ± 1.54% for GraphSAGE and 49.03 ± 2.16% for MLP.

| Method | Published mean (%) | Our released-code mean (%) | Our paper-described mean (%) |
|---|---:|---:|---:|
| GAT | 24.93 | 26.6442 | 28.5816 |
| GCN | 38.68 | 36.5333 | 41.4269 |
| GraphSAGE | 51.08 | 48.9195 | 52.0641 |
| MLP | 49.03 | 37.4544 | 37.7082 |

This is descriptive: original partitions and full MLP architecture were not recovered. No paired significance test against paper runs is performed. The MLP gap does not establish an error in the paper's result. Numerical closeness of one metric does not establish exact reproduction or model equivalence.

The other metrics also differ. For example, released-code GraphSAGE accuracy was approximately 98.8288%, close to the paper's 98.84%, while its Macro Recall was 75.0509% versus 80.42%. A similar accuracy therefore does not imply matching minority predictions. The complete five-metric comparison CSV from the earlier audit should accompany the final evidence package.

## 12. Evidence and verification

The local workflow recorded source/input hashes, saved partition indices, per-run metadata, validation-selected checkpoints and test predictions. Prediction metrics were recomputed and test node identity aligned across methods. Full-X-MoE checkpoint reload validation was recorded. Available training histories belong to the relevant GNN/CE scripts; do not imply that the Full runner saved an epoch history when it did not.

The canonical 120-run report was completed at `six-model-blte-summary-20261104-004230-843901`. A documentation packaging manifest should index the exact three contributing batches and all included files. Files with absolute paths must be made portable in the public package without changing scientific arrays or reported values.

Hash manifests identify content; they are not filesystem write locks and do not replace independent evaluation of methodology.

## 13. Relationship to our main-branch experiments

The reconstruction deliberately retains the released features and topology to preserve source comparability. It cannot by itself establish that performance persists under training-only preprocessing or restricted graph visibility. The separate `main` workflow addresses these additional questions: clean fits preprocessing on training nodes, and strict further restricts connections according to its documented visibility rules. This provides a complementary assessment of information access, rather than replacing the reference reconstruction.

| Item | Present reconstruction | Existing `main` research workflow |
|---|---|---|
| Features | Released graph matrix retained | Clean scaler fitted on training nodes |
| Partitions | Saved reconstructed splits | Clean runner regenerates stratified splits per seed |
| Visibility | Released topology retained | Clean graph plus separate strict visibility experiment |
| Runtime | Audit stack matched to principal paper versions | Historical CPU/GPU stack documented separately |
| Method | CE-only and Full separately labeled | Full multitask training plus component studies |

The clean split-generation algorithm is similar, but exact node identities across software environments must be checked rather than inferred from seed labels. Historical result provenance still needs to be mapped before any direct cross-series attribution.

### PCGrad ablation

We ran 40 Full X-MoE ablation runs at 300 epochs: two gradient conditions across ten matched seeds in each of two configurations. Both conditions retain the eight training objectives, their weights, and mean aggregation of task gradients. The off condition disables gradient projection. The audit verified the run metadata and output hashes and confirmed matching initialization hashes for all 20 pairs.

| Configuration | PCGrad on, mean ± SD (%) | PCGrad off, mean ± SD (%) | Mean paired difference (pp) | On wins / ties / off wins |
|---|---:|---:|---:|---:|
| Released-code | 67.7787 ± 6.8333 | 58.0850 ± 4.3224 | +9.6937 | 10 / 0 / 0 |
| Paper-described | 72.1943 ± 5.6479 | 64.1750 ± 5.2227 | +8.0193 | 9 / 0 / 1 |

SD denotes the sample standard deviation across seeds. These 40 runs are reported separately from the 120 runs in the primary comparison.

For the released-code configuration, paired differences ranged from +2.7811 to +21.4722 percentage points. For the paper-described configuration, they ranged from −4.3562 to +23.5010 percentage points. Seed 103 was the only pair favoring the off condition.

Two-sided exact Wilcoxon signed-rank tests computed from the supplied seed differences gave raw p-values of 0.001953 and 0.005859, respectively. Holm adjustment across the two configurations gave 0.003906 and 0.005859. The differences supplied for analysis were rounded to four decimal places in percentage points and retained distinct absolute ranks within each configuration. Treat the tests as exploratory because repeated partitions can overlap.

The local evidence index is the completed ablation BATCH_STATUS.json in reports/20261104-093820-336312/. The tables here use the supplied audit summary. The 40 per-run artifacts reside in the local ablation workspace.

## 14. What is and is not established

**Established within this evaluation:** released scripts executed; both documented configurations were evaluated; the primary comparison covered 120 runs; Full X-MoE had the highest mean Macro-F1 and exceeded each comparator on ten matched splits in both configurations; verified prediction metrics support the reported tables.

**Not established:** exact recovery of the author's original repeated experiment; reproduction of every paper baseline; identity with the author's MLP architecture; attribution of the entire Full-versus-CE gap to PCGrad; generalization to unseen chains/times/datasets; superiority under every training configuration; equal compute cost; exact cross-machine GPU reproducibility.

## 15. Proposed concluding statement

We successfully executed the released BLTE GNN implementations and evaluated documented reconstructions of released-code and paper-described settings. Exact recovery of the original repeated experiment was not established because original numerical seeds/partitions and some implementation details were unavailable, and the paper and code specify different settings. Under the reconstructed data/evaluation conditions, Full X-MoE achieved the highest mean Macro-F1 in both configurations and exceeded every evaluated comparator on all ten matched splits. The CE-only architecture variant showed configuration-dependent benefits. The Full-versus-CE comparison supports the complete additional training package rather than any single component. These results remain exploratory, with overlapping repeated partitions and prior exposure to test outcomes.

## 16. Publication preparation status

This report is drafted from the validated outputs supplied in the working session. No new training, branch creation, commit or push was performed to create it. The local indexing helper can verify and index the underlying 120-run evidence on the user's WSL computer.

Before public release, complete a portable runner interface, validate its commands, include the recorded dependency freeze, confirm redistribution/attribution terms, map all reported numbers to immutable relative-path evidence, and review the beginner README. Exclude unrelated manuscript source and machine-specific backup material from the branch.
