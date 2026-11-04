# Source review and candidate validation

## Verified in the supplied ZIP

- All 110 exported file hashes match `EXPORT_MANIFEST.json`.
- Six primary GNN scripts match their preparation hashes.
- Four primary MLP/CE scripts match their preparation hashes.
- Full frozen source and adapter match their recorded hashes.
- CE and Full model modules are byte-identical.
- Ten saved partition NPZ hashes match the split manifest; arrays are complete, disjoint and unique.
- Three primary batches identify 120 unique method/configuration/seed combinations.
- All 96 summary metric rows and ten Full comparisons recompute from the supplied seed-level table within 1e-10 (p-values within 1e-12).

## Packaging changes

Training source bytes were copied unchanged. The new wrapper resolves relative package paths, takes an explicit input directory, isolates each seed in a child process, supplies saved masks, and prevents output overwrite. CE and Full retain their respective original seeding and CUDA policies; no blanket deterministic override is added. Full uses the same adapter, coefficients and checkpoint reload computation. Output filenames/metadata are normalized by the wrapper; historical result values are not edited.

Unlike the earlier proposed single-model-file layout, configuration-specific prepared scripts remain separate for source fidelity. PCGrad remains inside the Full runner; no new PCGrad implementation was introduced.

Public summary tables have machine-specific folder prefixes replaced by relative audit run identifiers. Numeric strings are copied without recomputing saved reference values. Original table/evidence hashes remain available for local verification. Personal absolute paths in original metadata are not included in the portable reference metadata.

## Pending local checks

The export omits historical predictions, checkpoints and input datasets. Therefore this review cannot independently recompute each historical run from predictions here. `verify_audit.py` must perform that check on the original WSL machine. It checks the original evidence SHA chain, metadata, outputs, saved test IDs/labels, probabilities, metrics, histories and recorded Full reload results. A historical reload report is not a new GPU reload.

This cloud runtime has no PyTorch/PyG or NVIDIA GPU. GPU execution of the new wrapper has NOT been tested. Do not call the package fully runnable/validated until local checks pass. The local model preflight performs no optimizer steps and does not certify an entire training execution. Any later controlled wrapper execution must retain its outputs and be labeled as a new execution, not replace primary runs.

## Publication boundary

This candidate is local review material, not a new Git branch. No training, Git tag, commit, push or baseline edit was performed. Upstream attribution/redistribution terms remain a publication check. The source export's broader development files are not copied into the candidate.
