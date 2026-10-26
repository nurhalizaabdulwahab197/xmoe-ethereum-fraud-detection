"""Inference for the frozen seed-100 BLTE benchmark; no model training."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from predictor_models import GCN, GAT, GraphSAGE, ExplainableMixtureOfExpertsGNN

ROOT = Path(__file__).resolve().parent
CLASSES = ['Normal', 'Phishing', 'Scamming']
MODELS = {
    'GCN': ('final_gcn', 'gcn', GCN),
    'GAT': ('final_gat', 'gat', GAT),
    'GraphSAGE': ('final_graphsage', 'graphsage', GraphSAGE),
    'X-MoE': ('final_xmoe', 'x_moe_gnn', ExplainableMixtureOfExpertsGNN),
}
PROB_COLS = ['normal_probability', 'phishing_probability', 'scamming_probability']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Predictor:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        manifest_path = self.root / 'manifest.json'
        self.manifest = json.loads(manifest_path.read_text())
        for relative, expected in self.manifest['sha256'].items():
            if sha256(self.root / relative) != expected:
                raise ValueError(f'File differs from validated package: {relative}')
        bundle = np.load(self.root / 'dataset' / 'benchmark_inputs.npz', allow_pickle=False)
        self.addresses = bundle['addresses']
        self.labels = bundle['labels']
        self.test_indices = np.flatnonzero(bundle['test_mask'])
        self.test_set = set(self.test_indices.tolist())
        self.feature_names = bundle['feature_names'].tolist()
        self.raw_features = bundle['raw_features']
        self.lookup = {str(self.addresses[i]).lower(): int(i) for i in self.test_indices}
        self.x = torch.from_numpy(bundle['x'])
        self.edges = torch.from_numpy(bundle['edge_index'])
        self.degree = torch.bincount(self.edges.flatten(), minlength=len(self.labels)).numpy()
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.models = {}
        self.biases = {}
        self.cache = {}
        self.explanation = None
        self.elapsed = {}
        for name, (folder, slug, cls) in MODELS.items():
            checkpoint = torch.load(
                self.root / 'checkpoints_reference' / folder /
                f'clean_{slug}_seed-100_checkpoint.pt',
                map_location='cpu', weights_only=True,
            )
            if checkpoint['protocol'] != 'clean' or checkpoint['seed'] != 100:
                raise ValueError('Expected clean-protocol seed-100 checkpoint')
            if checkpoint['feature_names'] != self.feature_names or checkpoint['class_names'] != CLASSES:
                raise ValueError('Checkpoint feature/class ordering differs from data')
            model = cls(checkpoint['input_dim'], checkpoint['output_dim'])
            weights = checkpoint['state_dict']
            if name == 'X-MoE':
                weights = {
                    ('conflict_aware_expert.' + k[len('v8_expert.'):]
                     if k.startswith('v8_expert.') else k): v
                    for k, v in weights.items()
                }
            model.load_state_dict(weights, strict=True)
            self.models[name] = model.eval()
            self.biases[name] = torch.tensor([
                checkpoint['metrics'].get(f'{c.lower()}_class_bias', 0.0)
                for c in CLASSES
            ], dtype=torch.float32)

    def probabilities(self, name):
        if name not in self.cache:
            started = time.perf_counter()
            with torch.inference_mode():
                model = self.models[name]
                if name == 'X-MoE':
                    explanation = model.explain(self.x, self.edges)
                    logits = torch.log(explanation['probabilities'].clamp_min(1e-12))
                    self.explanation = {k: v.numpy() for k, v in explanation.items()}
                else:
                    logits = model(self.x, self.edges)
                self.cache[name] = torch.softmax(logits + self.biases[name], dim=1).numpy()
            self.elapsed[name] = time.perf_counter() - started
        return self.cache[name]

    def resolve(self, address):
        return self.lookup.get(str(address).strip().lower())

    def compare(self, node, names):
        if node not in self.test_set:
            raise ValueError('Only held-out seed-100 test addresses are supported')
        rows = []
        for name in names:
            probs = self.probabilities(name)[node]
            rows.append({
                'Model': name,
                'Prediction': CLASSES[int(probs.argmax())],
                **{label: float(probs[i]) for i, label in enumerate(CLASSES)},
            })
        return pd.DataFrame(rows)

    def validate(self):
        rows = []
        for name, (folder, slug, _) in MODELS.items():
            ref = pd.read_csv(self.root / 'checkpoints_reference' / folder /
                              f'clean_{slug}_seed-100_predictions.csv')
            indices = ref.node_index.to_numpy(dtype=np.int64)
            if not np.array_equal(indices, self.test_indices):
                raise ValueError(f'{name}: saved test nodes differ from regenerated split')
            if not np.array_equal(ref.true_label.to_numpy(), self.labels[indices]):
                raise ValueError(f'{name}: true labels differ')
            actual = self.probabilities(name)[indices]
            expected = ref[PROB_COLS].to_numpy()
            difference = float(np.abs(actual - expected).max())
            same = int(np.sum(actual.argmax(axis=1) == ref.predicted_label.to_numpy()))
            passed = bool(np.allclose(actual, expected, atol=1e-5, rtol=1e-4)
                          and same == len(indices))
            rows.append({'model': name, 'nodes': len(indices),
                         'matching_predictions': same, 'max_probability_error': difference,
                         'passed': passed, 'full_graph_inference_seconds': self.elapsed[name]})
        return rows


if __name__ == '__main__':
    report = Predictor().validate()
    print(json.dumps(report, indent=2))
    output = ROOT / 'validation_local.json'
    output.write_text(json.dumps(report, indent=2))
    raise SystemExit(0 if all(r['passed'] for r in report) else 1)
