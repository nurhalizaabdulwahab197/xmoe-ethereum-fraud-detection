"""Verify prepared RQ1 arrays and create a machine-local registry; no training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'preparation'))
from utility import read, sha


def configure(blte, graph, inputs, output):
    import numpy as np
    from matched_data_adapter import adapt
    expected = read(ROOT / 'provenance/rq1_expected_arrays.json')
    for name, wanted in expected['graph_file_sha256'].items():
        if sha(graph / name) != wanted:
            raise ValueError('RQ1 graph hash mismatch: ' + name)
    raw = np.load(graph / 'features_raw.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(graph / 'labels.npy', mmap_mode='r', allow_pickle=False)
    candidates = np.load(graph / 'candidate_labeled_mask.npy', mmap_mode='r', allow_pickle=False)
    records = {}
    for seed in range(100,110):
        folder = inputs / f'seed{seed}'
        meta = read(folder / 'preprocessing_and_split.json')
        wanted = expected['seed_inputs'][str(seed)]
        for name, digest in wanted['array_sha256'].items():
            if sha(folder / name) != digest:
                raise ValueError(f'Seed {seed} array hash mismatch: {name}')
        # Recompute using raw labeled training rows; verify that prepared arrays
        # implement the declared split/scaler rather than merely trusting metadata.
        x, masks, scaler = adapt(raw, labels, candidates, seed)
        np.testing.assert_array_equal(np.load(folder/'features_normalized.npy',allow_pickle=False), x)
        for name, mask in masks.items():
            np.testing.assert_array_equal(np.load(folder/f'{name}_mask.npy',allow_pickle=False), mask)
        np.testing.assert_array_equal(meta['mean'], scaler.mean_)
        np.testing.assert_array_equal(meta['scale'], scaler.scale_)
        np.testing.assert_array_equal(meta['variance'], scaler.var_)
        if meta['feature_names'] != expected['feature_names']:
            raise ValueError('Prepared feature names/order differ')
        if meta['partition_counts'] != wanted['partition_counts']:
            raise ValueError('Partition class counts differ')
        records[str(seed)] = dict(wanted, manifest_sha256=sha(folder/'preprocessing_and_split.json'))
        del x
        print(f'PASS RQ1 seed {seed}: arrays, masks and training-only scaler',flush=True)
    blte_hashes = {
        'csv': '02e376ddd98ba595372f19731fe5e51bbc90bf897b2a52b3c8bb98e786423b11',
        'graph': '2859a8a3a815edd02b5cd85d46c00d730b2b66c4e8d047145db2e843c9e38456'}
    blte_paths = {'csv':blte/'BLTE_multiclass_v2.csv', 'graph':blte/'BLTE_multiclass_graph.pt'}
    for key, path in blte_paths.items():
        if sha(path) != blte_hashes[key]:
            raise ValueError('BLTE hash mismatch: '+key)
    lock_path = inputs / 'portable_study_lock.json'
    lock = {'status':'matched_supplement_data_audited_protocol_locked',
        'seeds':list(range(100,110)), 'graph':str(graph.resolve()), 'inputs':str(inputs.resolve()),
        'input_index_sha256':sha(inputs/'matched_input_report.json'),
        'graph_file_sha256':expected['graph_file_sha256'], 'seed_inputs':records,
        'scope':'Input integrity lock only. Training is controlled exclusively by common_config.json.'}
    if lock_path.exists() and read(lock_path) != lock:
        raise ValueError('Existing portable lock differs; preserve it and investigate')
    lock_path.write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
    registry = {'blte':{**{k:str(v.resolve()) for k,v in blte_paths.items()},
        **{k+'_sha256':v for k,v in blte_hashes.items()}, 'input_dim':23,
        'label_policy':'Processed BLTE annotation; Normal0 Phishing1 Scamming2'},
        'rq1':{'study_lock':str(lock_path.resolve()), 'study_lock_sha256':sha(lock_path),'input_dim':5,
        'label_policy':'Dataset-native explicit author lists; frozen 13 review exclusions; unknown context=-1'}}
    if output.exists() and read(output) != registry:
        raise ValueError('Registry exists with different paths; use a new --output')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
    print('PASS: registry created; no training or test evaluation performed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blte',type=Path,required=True)
    p.add_argument('--rq1-graph',type=Path,required=True)
    p.add_argument('--rq1-inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'datasets.local.json')
    a=p.parse_args()
    configure(a.blte.resolve(),a.rq1_graph.resolve(),a.rq1_inputs.resolve(),a.output.resolve())

if __name__=='__main__':
    main()
