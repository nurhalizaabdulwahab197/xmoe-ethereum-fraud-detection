"""Run the original training/validation prefix, excluding the entire test phase."""
import ast
import csv
import math
from pathlib import Path
import time
import uuid

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from rq1_safe_io import atomic_json, replace_retry
from shared_engine import memory_options
from utility import sha


def compare_checkpoint(current_path, reference_path):
    """Exact weight comparison only. Never score labels or load arbitrary pickle code."""
    current=torch.load(current_path,map_location='cpu',weights_only=True)
    with torch.serialization.safe_globals([np._core.multiarray.scalar,np.dtype,
            type(np.dtype('float64')),type(np.dtype('float32'))]):
        original=torch.load(reference_path,map_location='cpu',weights_only=True)
    def normalize(state):
        mapped={k.replace('v8_expert.','conflict_aware_expert.'):v for k,v in state.items()}
        if len(mapped)!=len(state):
            raise ValueError('Ambiguous expert parameter rename')
        return mapped
    left,right=normalize(current['state_dict']),normalize(original['state_dict'])
    if set(left)!=set(right) or current['input_dim']!=original['input_dim']:
        raise ValueError('Reference/current checkpoint layouts differ')
    same=sum(torch.equal(left[k],right[k]) for k in left)
    maximum=max(float((left[k]-right[k]).abs().max()) for k in left)
    comparison={'reference_sha256':sha(reference_path),'exact_equal':same==len(left),
                'matching_tensors':same,'total_tensors':len(left),'maximum_absolute_difference':maximum,
                'comparison':'Weights only, documented expert-name normalization; no test scoring'}
    if not comparison['exact_equal']:
        raise ValueError('Reproduction checkpoint is not exactly equal to the original: '+str(comparison))
    return comparison


def training_prefix(ref, values, model_name, observe):
    path = Path(ref.__file__)
    module = ast.parse(path.read_text(encoding='utf-8-sig'))
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'run_once')
    loops = [i for i, n in enumerate(fn.body) if isinstance(n, ast.For)]
    if len(loops) != 1:
        raise ValueError('Unexpected reference training-loop layout')
    index = loops[0]
    # Preserve initialization, objectives, PCGrad, optimizer, validation and strict > selection.
    fn.body = fn.body[:index + 1]
    loop = fn.body[-1]
    loop.body.extend(ast.parse('observe_epoch(model, loss, val_f1)').body)
    fn.body.extend(ast.parse('return model, best_state, float(best_val_f1)').body)
    ast.fix_missing_locations(fn)
    ns = dict(vars(ref))
    cls = ref.MODELS[model_name]
    ns['MODELS'] = dict(ref.MODELS)
    ns['MODELS'][model_name] = lambda *a, **kw: memory_options(cls(*a, **kw), model_name)
    ns['clean_data'] = lambda *a, **kw: values
    ns['observe_epoch'] = observe
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path) + ':validation_only', 'exec'), ns)
    return ns['run_once']


def run_validation(ref, model_name, seed, values, folder, epochs, feature_names, training_options):
    if epochs < 1 or len(feature_names) != values[0].shape[1]:
        raise ValueError('Invalid validation-only epoch budget or feature names')
    started = time.perf_counter()
    history = []
    best = {'score': -1., 'epoch': None}
    history_path = folder / 'training_history.csv'

    def observe(model, loss, val_f1):
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value) or not math.isfinite(val_f1):
            raise ValueError('Nonfinite technical training/validation value')
        for p in model.parameters():
            if not torch.isfinite(p).all() or (p.grad is not None and not torch.isfinite(p.grad).all()):
                raise ValueError('Nonfinite parameter/gradient')
        epoch = len(history) + 1
        if val_f1 > best['score']:
            best.update(score=val_f1, epoch=epoch)
        row = {'epoch': epoch, 'train_base_loss': loss_value,
               'validation_macro_f1': val_f1, 'best_validation_macro_f1': best['score']}
        with history_path.open('a', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not history:
                writer.writeheader()
            writer.writerow(row)
        history.append(row)
        atomic_json(folder / 'progress.json', {**row, 'validation_evaluated': True,
                                               'test_evaluated': False}, required=False)
        print(f'{model_name}: epoch {epoch}/{epochs}; validation Macro-F1={val_f1:.4f}; best={best["score"]:.4f}', flush=True)

    fn = training_prefix(ref, values, model_name, observe)
    model, best_state, best_score = fn(None, Path('.'), model_name, 'clean', seed, epochs,
                                      torch.device('cpu'), True, **training_options)
    if len(history) != epochs or best_state is None or best_score != best['score']:
        raise ValueError('Validation selection evidence inconsistent')
    model.load_state_dict(best_state)
    model.eval()
    x, edge, y, _, validation, _ = values
    with torch.no_grad():
        selected_logits = model(x, edge)[validation].detach().clone()
    true = y[validation].cpu().numpy()
    predicted = selected_logits.argmax(1).cpu().numpy()
    metrics = {'macro_f1': float(f1_score(true, predicted, labels=[0, 1, 2], average='macro', zero_division=0)),
               'accuracy': float(accuracy_score(true, predicted))}
    per_class = f1_score(true, predicted, labels=[0, 1, 2], average=None, zero_division=0)
    metrics.update({name + '_f1': float(value) for name, value in zip(('normal','phishing','scamming'), per_class)})
    if metrics['macro_f1'] != best_score:
        raise ValueError('Reloaded selected state has a different validation score')
    checkpoint = folder / 'best_validation_checkpoint.pt'
    temporary = folder / ('checkpoint.' + uuid.uuid4().hex + '.tmp')
    payload = {'state_dict': best_state, 'model_name': model_name, 'seed': seed, 'epochs': epochs,
               'best_epoch': best['epoch'], 'input_dim': x.shape[1], 'output_dim': 3,
               'feature_names': feature_names, 'training_options': training_options,
               'best_validation_metrics': metrics, 'test_evaluated': False,
               'purpose': 'Technical validation checkpoint; not a resumable optimizer checkpoint'}
    try:
        torch.save(payload, temporary)
        replace_retry(temporary, checkpoint)
    finally:
        if temporary.exists():
            temporary.unlink()
    # A fresh model verifies disk serialization and feature dimensions. Preserve Torch RNG.
    with torch.random.fork_rng(devices=[]):
        restored = memory_options(ref.MODELS[model_name](x.shape[1], 3), model_name)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        restored.load_state_dict(saved['state_dict'])
        restored.eval()
        with torch.no_grad():
            restored_logits = restored(x, edge)[validation]
        torch.testing.assert_close(selected_logits, restored_logits, rtol=0, atol=0)
    return {'status': 'validation_only_check_passed', 'epochs': epochs, 'best_epoch': best['epoch'],
            'best_validation_metrics': metrics, 'training_seconds': time.perf_counter() - started,
            'checkpoint_filename': checkpoint.name, 'checkpoint_sha256': sha(checkpoint),
            'checkpoint_reload_exact': True, 'training_history_sha256': sha(history_path),
            'validation_evaluated': True, 'test_evaluated': False,
            'test_phase_excluded_from_ast': True, 'resumable_training_checkpoint': False}
