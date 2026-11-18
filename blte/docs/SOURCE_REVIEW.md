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

## Local evidence verification

The archived evidence for 120 primary runs and 40 PCGrad ablation runs was verified in WSL. The 20 ablation pairs had matching initialization hashes. Public result tables report the audited results, and the full per-run evidence is stored in the local audit archive.

GPU execution of the portable public wrapper remains pending validation. A future wrapper run should be recorded as a separate execution.

## Published package scope

The `reproduction` branch contains source code, documentation, split indices, and result tables. See [upstream attribution](../provenance/UPSTREAM_ATTRIBUTION.md) for the source license and dataset terms.
