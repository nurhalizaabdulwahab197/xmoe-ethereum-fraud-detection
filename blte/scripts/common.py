"""Shared validation utilities; no training at import time."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('mlp', 'gcn', 'gat', 'graphsage', 'xmoe-ce', 'full-xmoe')
CONFIGURATIONS = ('released-code', 'paper-described')

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str)+'\n', encoding='utf-8')

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def within(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f'Path outside root: {relative}')
    return path

def source_check():
    files = read(ROOT/'provenance/SOURCE_SHA256.json')
    for name, expected in files.items():
        require(sha(within(ROOT, name)) == expected, f'Source mismatch: {name}')
    return len(files)

def input_check(upstream):
    upstream = Path(upstream).resolve()
    for name, expected in read(ROOT/'provenance/INPUT_SHA256.json').items():
        require(sha(upstream/name) == expected, f'Input mismatch: {name}')
    return upstream

def environment(gpu=True):
    import sys
    import importlib.metadata as metadata
    expected = {'torch':'2.0.1+cu118','torch-geometric':'2.3.1',
        'torch-scatter':'2.1.1+pt20cu118','torch-sparse':'0.6.17+pt20cu118',
        'numpy':'1.24.4','scipy':'1.10.1','pandas':'2.0.3',
        'scikit-learn':'1.3.2','matplotlib':'3.7.3','seaborn':'0.12.2','networkx':'3.1'}
    require(sys.version_info[:3] == (3,10,12), 'Recorded Python is 3.10.12')
    actual = {name:metadata.version(name) for name in expected}
    for name,value in expected.items():
        require(actual[name] == value, f'{name}: expected {value}, found {actual[name]}')
    import torch
    require(torch.version.cuda == '11.8', 'Recorded CUDA runtime is 11.8')
    if gpu:
        require(torch.cuda.is_available(), 'CUDA unavailable; no CPU fallback')
        actual['gpu'] = torch.cuda.get_device_name(0)
    return actual

def splits(graph=None):
    import numpy as np
    manifest = read(ROOT/'splits/SPLIT_MANIFEST.json')
    result = {}
    signatures = set()
    for r in manifest['records']:
        seed = r['split_seed']; path = within(ROOT/'splits', r['file'])
        require(seed not in result, 'Duplicate split seed')
        require(sha(path) == r['file_sha256'], f'Split hash mismatch: {seed}')
        with np.load(path, allow_pickle=False) as z:
            arrays = {k:z[k].copy() for k in ('train','validation','test')}
        for key, a in arrays.items():
            require(np.issubdtype(a.dtype, np.integer) and a.ndim == 1, key)
            require(np.array_equal(a, np.sort(a)), f'Unsorted {key}: {seed}')
            require(len(a) == r[key+'_nodes'], f'Partition size: {key}')
            if graph is not None:
                y = graph.y.cpu().numpy()
                require(np.array_equal(np.bincount(y[a], minlength=3), r[key+'_class_counts']), 'Class counts')
        require(np.array_equal(np.sort(np.concatenate(list(arrays.values()))), np.arange(73033)), 'Incomplete/overlapping partition')
        signature = hashlib.sha256(b''.join(arrays[k].tobytes() for k in arrays)).hexdigest()
        require(signature not in signatures, 'Duplicate partition')
        signatures.add(signature); result[seed] = (path, arrays)
    require(set(result) == set(range(100,110)), 'Expected seeds 100–109')
    return result

def metrics(true, pred):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
    out = {'accuracy':float(accuracy_score(true,pred)),
      'macro_precision':float(precision_score(true,pred,labels=[0,1,2],average='macro',zero_division=0)),
      'macro_recall':float(recall_score(true,pred,labels=[0,1,2],average='macro',zero_division=0)),
      'macro_f1':float(f1_score(true,pred,labels=[0,1,2],average='macro',zero_division=0)),
      'weighted_f1':float(f1_score(true,pred,labels=[0,1,2],average='weighted',zero_division=0)),
      'confusion_matrix':confusion_matrix(true,pred,labels=[0,1,2]).tolist()}
    for key,val in zip(('normal_f1','phishing_f1','scamming_f1'),f1_score(true,pred,labels=[0,1,2],average=None,zero_division=0)):
        out[key] = float(val)
    return out

def check_prediction(path, ids, labels):
    import numpy as np
    import pandas as pd
    d = pd.read_csv(path)
    if 'node_index' in d:
        node, true, pred = 'node_index','true_label','predicted_label'
        cols = ['normal_probability','phishing_probability','scamming_probability']
    else:
        node,true,pred = 'node_id','y_true','y_pred'
        cols = ['prob_normal','prob_phishing','prob_scamming']
    require(np.array_equal(d[node].to_numpy(), ids), f'Test node mismatch: {path}')
    require(np.array_equal(d[true].to_numpy(), labels), f'Test label mismatch: {path}')
    p = d[cols].to_numpy()
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(), 'Invalid probabilities')
    require(np.allclose(p.sum(1),1,atol=1e-5,rtol=0), 'Probability sum')
    require(np.array_equal(p.argmax(1),d[pred].to_numpy()), 'Argmax mismatch')
    return metrics(labels,d[pred].to_numpy())
