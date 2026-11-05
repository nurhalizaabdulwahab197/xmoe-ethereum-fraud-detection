"""Real-BLTE implementation bridge: original training loop, no final test phase."""
import ast
import hashlib
from pathlib import Path

import torch

from adapters import blte_graph, load_dataset, checked_entry
from shared_engine import reference, memory_options


def instrumented(ref):
    """Retain original initialization/training/validation; truncate before test code."""
    path = Path(ref.__file__)
    module = ast.parse(path.read_text())
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'run_once')
    index = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.For))
    fn.body = fn.body[:index+1]
    fn.body.insert(index, ast.parse('initial_state = snapshot(model)\ntraces = []').body[0])
    fn.body.insert(index+1, ast.parse('traces = []').body[0])
    loop = fn.body[-1]
    step_index = next(i for i,n in enumerate(loop.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                      and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'step')
    loop.body.insert(step_index+1, ast.parse('traces.append(capture_step(model, loss))').body[0])
    loop.body.extend(ast.parse('traces[-1]["validation_logits"] = val_logits.detach().cpu().clone()\ntraces[-1]["validation_f1"] = float(val_f1)').body)
    fn.body.append(ast.parse('return {"initial_state": initial_state, "traces": traces, "selected_state": best_state, "best_validation_f1": float(best_val_f1)}').body[0])
    ast.fix_missing_locations(fn)

    def snapshot(model):
        return {n:p.detach().cpu().clone() for n,p in model.state_dict().items()}

    def capture_step(model, loss):
        return {'state': snapshot(model), 'gradients': {n: p.grad.detach().cpu().clone() if p.grad is not None else None
            for n,p in model.named_parameters()}, 'base_loss': float(loss.detach().cpu()),
            'torch_rng': torch.get_rng_state().clone()}

    namespace = dict(vars(ref))
    namespace.update(snapshot=snapshot, capture_step=capture_step)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path) + ':pre_test_bridge', 'exec'), namespace)
    return namespace['run_once'], namespace


def compare_state(left, right, rtol=1e-5, atol=1e-6):
    if set(left) != set(right):
        raise ValueError('Parameter/gradient names differ')
    maximum = 0.
    for name in left:
        if left[name] is None or right[name] is None:
            if left[name] is not None or right[name] is not None:
                raise ValueError('Unused gradient differs: ' + name)
            continue
        torch.testing.assert_close(left[name], right[name], rtol=rtol, atol=atol)
        maximum = max(maximum, float((left[name]-right[name]).abs().max()))
    return maximum


def run_bridge(model, seed, registry, config):
    entry = checked_entry('blte', registry, seed)
    original = reference()
    graph = blte_graph(entry['graph'])
    baseline_values = original.clean_data(graph, Path(entry['csv']), seed)
    unified = reference()
    values, meta = load_dataset('blte', registry, seed, unified)
    for a,b in zip(baseline_values, values):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    baseline_fn, baseline_ns = instrumented(original)
    unified_fn, unified_ns = instrumented(unified)
    # The only dataset input replacement is the unified adapter's verified tensors.
    unified_ns['clean_data'] = lambda *args: values
    cls = unified_ns['MODELS'][model]
    unified_ns['MODELS'] = dict(unified_ns['MODELS'])
    unified_ns['MODELS'][model] = lambda *a, **kw: memory_options(cls(*a, **kw), model)
    options = dict(config['training_options'])
    args = (graph, Path(entry['csv']), model, 'clean', seed, 3, torch.device('cpu'), True)
    left = baseline_fn(*args, **options)
    right = unified_fn(*args, **options)
    compare_state(left['initial_state'], right['initial_state'], rtol=0, atol=0)
    records=[]
    for epoch, (a,b) in enumerate(zip(left['traces'], right['traces']), start=1):
        torch.testing.assert_close(torch.tensor(a['base_loss']), torch.tensor(b['base_loss']), rtol=1e-5,atol=1e-6)
        if not torch.equal(a['torch_rng'],b['torch_rng']):
            raise ValueError('Dropout/training RNG differs')
        if a['validation_f1'] != b['validation_f1']:
            raise ValueError('Validation F1 differs')
        torch.testing.assert_close(a['validation_logits'],b['validation_logits'],rtol=1e-5,atol=1e-6)
        if not torch.equal(a['validation_logits'].argmax(1),b['validation_logits'].argmax(1)):
            raise ValueError('Validation predictions differ')
        records.append({'epoch':epoch,'base_loss_absolute_difference':abs(a['base_loss']-b['base_loss']),
            'parameter_max_abs_difference':compare_state(a['state'],b['state']),
            'gradient_max_abs_difference':compare_state(a['gradients'],b['gradients']),
            'validation_logits_max_abs_difference':float((a['validation_logits']-b['validation_logits']).abs().max()),
            'validation_f1_equal':True,'rng_equal':True})
    selected_diff=compare_state(left['selected_state'],right['selected_state'])
    if left['best_validation_f1'] != right['best_validation_f1']:
        raise ValueError('Selected validation score differs')
    return {'status':'real_blte_pre_test_bridge_passed','model':model,'seed':seed,'epochs_per_path':3,
        'input_and_split_exact_match':True,'initial_parameters_exact_match':True,
        'epoch_checks':records,'selected_state_max_abs_difference':selected_diff,
        'float32_tolerance':{'rtol':1e-5,'atol':1e-6},'dataset_metadata':meta,
        'test_evaluated':False,'test_phase_excluded_from_instrumented_ast':True,
        'scope':'Three-epoch implementation bridge on the same BLTE input, not a predictive benchmark or proof for every training trajectory.'}
