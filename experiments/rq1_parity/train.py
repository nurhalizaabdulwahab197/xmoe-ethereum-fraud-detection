"""Fresh training, validation-only checkpoint selection, then one final test."""
import csv
import importlib.metadata
import platform
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from protocol import make_views
from data import sha

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'provenance/frozen_pipeline'
sys.path.insert(0, str(FROZEN))
from shared_engine import reference, memory_options, safe_result
ref = reference()
from ablation_models import build_model, get_training_outputs, node_degree


def code_inventory():
    paths = list(Path(__file__).parent.glob('*.py')) + list((FROZEN/'reference_src').rglob('*.py'))
    paths += [FROZEN/name for name in ['gat_chunked_messages.py', 'gat_activation_checkpoint.py',
                                    'xmoe_memory_checkpoint.py', 'shared_engine.py', 'common_config.json']]
    paths += [ROOT/'provenance/rq1_expected_arrays.json', FROZEN/'release_manifest.json']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def train(values, metadata, row, folder, epochs=300, threads=24):
    folder = Path(folder);folder.mkdir(parents=True, exist_ok=False)
    device = torch.device(row['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; GPU experiments cannot be reported as completed.')
    torch.set_num_threads(threads)
    ref.set_seed(row['seed'], True)
    x, edges, y, tm, vm, qm = values
    audits = None
    if row['family'] == 'strict':
        features, topology, audits = make_views(x, edges, y, (tm, vm, qm))
    else:
        features, topology = [x, x, x], [edges, edges, edges]
    xtrain, xval, xtest = [v.to(device) for v in features]
    etrain, eval_, etest = [v.to(device) for v in topology]
    y, tm, vm, qm = [v.to(device) for v in [y, tm, vm, qm]]
    variant = row['variant'];is_xmoe = row['model'] == 'x_moe_gnn'
    model = (build_model(variant, x.shape[1], 3) if is_xmoe else ref.MODELS[row['model']](x.shape[1], 3))
    # Execution-only recomputation/chunking, already used by the shared clean engine.
    if row['model'] == 'gat' or (is_xmoe and hasattr(model, 'conflict_aware_expert')):
        model = memory_options(model, row['model'])
    model = model.to(device)
    counts = torch.bincount(y[tm], minlength=3)
    weights = counts.sum()/counts;weights = weights/weights.sum()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0005)
    degree = node_degree(etrain, len(xtrain)).to(device)
    connected_normal = tm & (y == 0) & (degree > 0)
    if row['family']=='strict' and is_xmoe and row['coefficient']>0 and not connected_normal.any():
        raise ValueError('No connected Normal training nodes; the BLTE strict objective cannot be applied unchanged.')
    best, state, best_epoch = -1.0, None, None
    started = time.perf_counter();history = []
    for epoch in range(1, epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        if is_xmoe:
            logits, conflict, sage = get_training_outputs(model, variant, xtrain, etrain)
        else:
            logits, conflict, sage = model(xtrain, etrain), None, None
        target = y[tm]
        losses = [F.cross_entropy(logits[tm], target, weight=weights)]
        if is_xmoe:
            if conflict is not None and variant != 'a1_conflict_only':
                losses.append(.25*F.cross_entropy(conflict[tm], target, weight=weights))
            if sage is not None and variant != 'a2_graphsage_only':
                losses.append(.25*F.cross_entropy(sage[tm], target, weight=weights))
            if conflict is not None:
                for index in range(3):
                    losses.append(ref.balanced_binary_loss(conflict[tm,index:index+1], (target == index).to(conflict.dtype)))
            if sage is not None and variant != 'a5_no_scam_specialist':
                losses.append(.5*ref.balanced_binary_loss(sage[tm,2:3], (target == 2).to(sage.dtype)))
            coefficient = 0.0 if variant == 'a6_no_normal_phishing_penalty' else row['coefficient']
            if coefficient > 0 and connected_normal.any():
                losses.append(coefficient*logits[connected_normal].softmax(1)[:,1].mean())
            ref.pcgrad_backward(losses, list(model.parameters()))
        else:
            losses[0].backward()
        optimizer.step()
        # Discard graphs before validation. No test mask/labels are read here.
        del logits, conflict, sage, losses
        model.eval()
        with torch.no_grad():
            predicted = model(xval, eval_)[vm].argmax(1).cpu().numpy()
        score = f1_score(y[vm].cpu().numpy(), predicted, labels=[0,1,2], average='macro', zero_division=0)
        if score > best:
            best, best_epoch = float(score), epoch
            state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append({'epoch': epoch, 'validation_macro_f1': score, 'best_validation_macro_f1': best})
        print(f"{row['family']} {row['model']} {variant} seed{row['seed']} epoch {epoch}/{epochs} val={score:.6f}", flush=True)
        (folder/'progress.json').write_text(json.dumps({**history[-1], 'test_evaluated': False})+'\n')
    if state is None:
        raise ValueError('No validation-selected checkpoint.')
    model.load_state_dict(state);model.eval()
    provenance = {'configuration': row, 'epochs': epochs, 'threads': threads,
                  'source_sha256': code_inventory(), 'input': metadata,
                  'selection': 'Highest validation Macro-F1, earliest on ties; no test selection.',
                  'runtime_versions': {name: importlib.metadata.version(name) for name in ['torch','torch-geometric','numpy','pandas','scikit-learn','scipy','psutil']},
                  'python_version': platform.python_version(), 'torch_runtime': torch.__version__,
                  'strict_audit': audits, 'feature_visibility_scope': 'Frozen full-observation features; no temporal or raw-feature isolation claim.'}
    torch.save({'state_dict': state, 'metadata': provenance, 'best_epoch': best_epoch}, folder/'best_checkpoint.pt')
    (folder/'test_evaluation_started.json').write_text(json.dumps({'best_epoch':best_epoch,'training_complete':True})+'\n')
    with torch.no_grad():
        probability = model(xtest, etest)[qm].softmax(1).cpu().numpy()
    true = y[qm].cpu().numpy()
    result = dict(ref.score(true, probability))
    indices = qm.nonzero(as_tuple=False).flatten().cpu().numpy()
    test_degree = node_degree(etest, len(xtest))[qm].cpu().numpy()
    for prefix, keep in [('connected', test_degree > 0), ('isolated', test_degree == 0)]:
        result[prefix+'_nodes'] = int(keep.sum())
        for label, name in enumerate(['normal','phishing','scamming']):
            result[f'{prefix}_{name}_nodes'] = int(np.count_nonzero(true[keep] == label))
        if keep.any():
            result.update({prefix+'_'+k:v for k,v in ref.score(true[keep], probability[keep]).items()})
        else:
            result[prefix+'_macro_f1'] = None
    # Routing diagnostics are descriptive, never interpreted as causal explanations.
    if hasattr(model, 'explain'):
        with torch.no_grad():
            explanation = model.explain(xtest, etest)
        if isinstance(explanation, dict):
            for key, value in explanation.items():
                if torch.is_tensor(value) and ('gate' in key or 'weight' in key):
                    if value.shape == (len(xtest), 3):
                        for index, name in enumerate(['normal','phishing','scamming']):
                            result[f'{key}_{name}_test_mean'] = float(value[qm,index].mean())
                    elif value.shape == (len(xtest),):
                        result[key+'_test_mean'] = float(value[qm].mean())
                        for prefix, keep in [('connected', test_degree>0), ('isolated', test_degree==0)]:
                            if keep.any():result[key+'_'+prefix+'_test_mean'] = float(value[qm][torch.from_numpy(keep).to(device)].mean())
        del explanation
    prediction = {'node_index':indices, 'true_label':true, 'predicted_label':probability.argmax(1),
                  **{name+'_probability':probability[:,i] for i,name in enumerate(['normal','phishing','scamming'])}}
    import pandas as pd
    pd.DataFrame(prediction).to_csv(folder/'test_predictions.csv',index=False)
    pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
    result.update(row, dataset='rq1', epochs=epochs, best_epoch=best_epoch,
                  best_validation_macro_f1=best, training_seconds=time.perf_counter()-started,
                  train_nodes=int(tm.sum()), validation_nodes=int(vm.sum()), test_nodes=int(qm.sum()),
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  effective_normal_phishing_penalty=0.0 if variant=='a6_no_normal_phishing_penalty' else row['coefficient'],
                  effective_scam_specialist_weight=0.0 if variant in ['a5_no_scam_specialist','a1_conflict_only'] or not is_xmoe else .5,
                  training_performed=True, test_evaluated=True, neighbor_sampling=False,
                  inference_protocol='restricted_topology' if audits else 'full_released_graph',
                  strict_audit=audits, provenance=provenance)
    result = safe_result(result)
    result['output_sha256'] = {name:sha(folder/name) for name in ['best_checkpoint.pt','test_predictions.csv','training_history.csv']}
    (folder/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
