"""Shared engine: pinned original BLTE run_once/score and explicit execution wrappers."""
import importlib.util
from pathlib import Path
import sys
import itertools
import csv
from rq1_safe_io import atomic_json
from utility import sha, read
ROOT = Path(__file__).resolve().parent

def reference():
    source = ROOT / 'reference_src'
    if 'models' in sys.modules and not Path(sys.modules['models'].__file__).resolve().is_relative_to(source):
        raise ValueError('Another models package is already loaded; use a fresh process')
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('_pinned_blte_reference', source / 'reproduce.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def memory_options(model, name):
    if name == 'gat':
        from gat_chunked_messages import enable_gat_chunked_messages
        from gat_activation_checkpoint import enable_gat_activation_checkpointing
        enable_gat_chunked_messages(model, 32768)
        enable_gat_activation_checkpointing(model)
    if name == 'x_moe_gnn':
        from xmoe_memory_checkpoint import enable_xmoe_memory_checkpoint
        enable_xmoe_memory_checkpoint(model)
    return model

def run_reference(ref, model_name, seed, values, folder, epochs=300, feature_names=None, training_options=None):
    """Used by production and synthetic parity tests; never changes original source."""
    if training_options is None:
        training_options = read(ROOT / 'common_config.json')['training_options']
    cls = ref.MODELS[model_name]
    state = {'epoch': 0, 'best': -1.0, 'best_epoch': None}
    real_f1 = ref.f1_score

    def observe_validation(*args, **kwargs):
        result = real_f1(*args, **kwargs)
        if state['epoch'] < epochs:
            state['epoch'] += 1
            if float(result) > state['best']:
                state['best_epoch'] = state['epoch']
            state['best'] = max(state['best'], float(result))
            row = {'epoch':state['epoch'],'validation_macro_f1':float(result),
                   'best_validation_macro_f1':state['best']}
            with (folder/'training_history.csv').open('a',encoding='utf-8',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=list(row))
                if state['epoch'] == 1:
                    writer.writeheader()
                writer.writerow(row)
            atomic_json(folder/'progress.json',{**row,'test_evaluated':False},required=False)
            print(f'{model_name}: epoch {state["epoch"]}/{epochs}; validation Macro-F1={result:.4f}; best={state["best"]:.4f}', flush=True)
        return result

    def factory(*args, **kwargs):
        model = memory_options(cls(*args, **kwargs), model_name)

        def wrap(original):
            def call(*a, **kw):
                if not model.training and state['epoch'] >= epochs and not (folder / 'test_evaluation_started.json').exists():
                    atomic_json(folder / 'test_evaluation_started.json', {
                        'model': model_name, 'seed': seed, 'boundary': 'First post-training evaluation forward/explain in original run_once'})
                return original(*a, **kw)
            return call

        model.forward = wrap(model.forward)
        if hasattr(model, 'explain'):
            model.explain = wrap(model.explain)
        return model

    header = folder / 'feature_header.csv'
    # Original checkpoint writer reads nrows=0 to record names. Data tensors come from the adapter.
    if feature_names is None or len(feature_names) != values[0].shape[1]:
        raise ValueError('Dataset feature names must match actual input dimension')
    header.write_text('Address,Scam,attack,attack_encoded,' + ','.join(feature_names) + '\n', encoding='utf-8')
    original_loader = ref.clean_data
    ref.MODELS[model_name], ref.f1_score = factory, observe_validation
    ref.clean_data = lambda released_data, csv_path, current_seed: values
    try:
        result = ref.run_once(None, header, model_name, 'clean', seed, epochs, ref.torch.device('cpu'), True,
            checkpoint_path=folder / 'best_checkpoint.pt', prediction_path=folder / 'test_predictions.csv',
            **training_options)
        result.update(best_epoch=state['best_epoch'],checkpoint_filename='best_checkpoint.pt',
                      training_history_sha256=sha(folder/'training_history.csv'))
        return result
    finally:
        ref.MODELS[model_name], ref.f1_score, ref.clean_data = cls, real_f1, original_loader

def safe_result(result):
    import math
    result = dict(result)
    undefined = []
    for key, value in result.items():
        if hasattr(value, 'item'):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            # Empty connectivity strata can legitimately yield undefined source diagnostics.
            if key.endswith(('_f1', '_ap', '_precision', '_recall')) or key in ('accuracy', 'weighted_f1'):
                raise ValueError('Nonfinite predictive metric: ' + key)
            value = None
            undefined.append(key)
        result[key] = value
    result['undefined_source_diagnostics'] = undefined
    return result

def signed_rank_p(deltas):
    values = [v for v in deltas if v != 0]
    if not values:
        return 1.0
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    ranks = [0.] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and abs(values[order[end]]) == abs(values[order[start]]):
            end += 1
        rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    observed = abs(sum(r if v > 0 else -r for r, v in zip(ranks, values)))
    return sum(abs(sum(sign * rank for sign, rank in zip(signs, ranks))) >= observed
               for signs in itertools.product((-1, 1), repeat=len(values))) / (2 ** len(values))
