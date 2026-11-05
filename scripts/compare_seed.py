"""Compare one reproduced seed with published evidence, without model inference."""
import argparse
import csv
import json
from pathlib import Path

from audit_results import ROOT, MODELS, read, sha

FIELDS = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1',
          'phishing_ap', 'scamming_ap']
FIELDS += [f'{c}_{m}' for c in ['normal', 'phishing', 'scamming']
           for m in ['precision', 'recall', 'f1']]
FIELDS += [f'cm_{i}_{j}' for i in range(3) for j in range(3)]


def locate(output, model, seed):
    """Require one inventory-verified completion; never choose by test score."""
    inventory = read(output / 'attempt_inventory.json')['runs']
    entries = [r for r in inventory if r['model'] == model and r['seed'] == seed]
    if len(entries) != 1:
        raise ValueError(f'Missing/duplicated inventory entry: {model} seed {seed}')
    completed = [a for a in entries[0]['attempts'] if a['verified_complete']]
    if len(completed) != 1:
        raise ValueError(f'Expected one verified completion: {model} seed {seed}')
    folder = (output / completed[0]['folder']).resolve()
    if not folder.is_relative_to(output.resolve()):
        raise ValueError('Attempt folder escapes result root')
    return folder


def compare(output, dataset, seed):
    config = read(ROOT / 'common_config.json')
    rows = []
    for model in MODELS:
        folder = locate(output, model, seed)
        result = read(folder / 'result.json')
        resource = read(folder / 'resource_probe.json')
        reference = read(ROOT / f'results/reference/{dataset}/{model}/seed{seed}/result.json')
        if (result['dataset'], result['model'], result['seed']) != (dataset, model, seed):
            raise ValueError('Run identity differs')
        if (result['status'] != 'shared_pipeline_performance_run_completed'
                or result['epochs'] != 300 or resource['status'] != 'passed'
                or not result['test_evaluated']):
            raise ValueError('A completed 300-epoch final run is required')
        if (result['shared_config_sha256'] != sha(ROOT / 'common_config.json')
                or result['release_manifest_sha256'] != reference['release_manifest_sha256']
                or result['actual_torch_threads'] != config['threads']
                or result['optimizer'] != config['optimizer']
                or result['effective_training_options'] != config['training_options']):
            raise ValueError('Shared source/configuration/runtime options differ')
        for key in ['input_dim', 'feature_names', 'nodes', 'edges', 'partition_counts']:
            if result['dataset_metadata'][key] != reference['dataset_metadata'][key]:
                raise ValueError('Input metadata differs: ' + key)
        current_fp = {k: v for k, v in result['dataset_metadata']['input_fingerprints'].items()
                      if k != 'study_lock'}
        reference_fp = {k: v for k, v in reference['dataset_metadata']['input_fingerprints'].items()
                        if k != 'study_lock'}
        if current_fp != reference_fp:
            raise ValueError('Input-array/source hashes differ')
        for filename, field in [('best_checkpoint.pt', 'checkpoint_sha256'),
                                ('test_predictions.csv', 'prediction_sha256'),
                                ('training_history.csv', 'training_history_sha256')]:
            if sha(folder / filename) != result[field]:
                raise ValueError('Run artifact hash differs: ' + filename)
        with (folder / 'training_history.csv').open(encoding='utf-8-sig', newline='') as stream:
            history = list(csv.DictReader(stream))
        with (folder / 'test_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
            predictions = list(csv.DictReader(stream))
        expected_rows = 10955 if dataset == 'blte' else 737
        if len(history) != 300 or len(predictions) != expected_rows:
            raise ValueError('Wrong history or prediction population')
        best = max(float(r['validation_macro_f1']) for r in history)
        first = next(int(r['epoch']) for r in history if float(r['validation_macro_f1']) == best)
        if best != result['best_validation_macro_f1'] or first != result['best_epoch']:
            raise ValueError('Selected checkpoint is not the earliest validation maximum')
        differences = {k: {'reference': reference[k], 'new': result[k],
                           'delta': result[k] - reference[k]}
                       for k in FIELDS if result[k] != reference[k]}
        prediction_match = result['prediction_sha256'] == reference['prediction_sha256']
        epoch_match = result['best_epoch'] == reference['best_epoch']
        rows.append({'model': model, 'seed': seed, 'metrics_checked': len(FIELDS),
                     'metrics_exactly_equal': not differences, 'metric_differences': differences,
                     'prediction_bytes_exactly_equal': prediction_match,
                     'best_epoch_equal': epoch_match,
                     'reference_macro_f1': reference['macro_f1'], 'new_macro_f1': result['macro_f1']})
    exact = all(r['metrics_exactly_equal'] and r['prediction_bytes_exactly_equal']
                and r['best_epoch_equal'] for r in rows)
    return {'dataset': dataset, 'seed': seed, 'verified_completed_runs': len(rows),
            'status': 'exact_reference_match' if exact else 'reference_differences_require_review',
            'training_performed': False, 'model_inference_performed': False, 'runs': rows,
            'scope': 'Exact same-PC reproduction was observed for seed 100. Different hardware may differ; investigate rather than tune to reference test scores.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['blte', 'rq1'], required=True)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=range(100, 110), default=100)
    args = parser.parse_args()
    result = compare(args.results_dir.resolve(), args.dataset, args.seed)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result['status'] != 'exact_reference_match':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
