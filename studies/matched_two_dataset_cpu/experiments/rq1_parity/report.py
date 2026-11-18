"""Audit completed fresh runs; summaries never silently accept missing seeds."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from plan import matrix, run_id, MODELS, VARIANTS
from data import sha


def flip_p(deltas):
    values = np.asarray(deltas, dtype=float)
    observed = abs(values.mean())
    statistics = [abs(np.mean(values*np.array(sign))) for sign in itertools.product([-1,1], repeat=len(values))]
    return float(np.mean(np.array(statistics) >= observed - 1e-15))


def holm(rows):
    previous = 0.0
    for rank, index in enumerate(np.argsort([r['raw_p'] for r in rows])):
        previous = max(previous, (len(rows)-rank)*rows[index]['raw_p'])
        rows[index]['holm_adjusted_p'] = min(1.0, previous)
    return rows


def comparison(a, b, family, label, method='signed_rank', metric='macro_f1'):
    from shared_engine import signed_rank_p
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(20261107)
    boot = rng.choice(d, size=(10000,len(d)), replace=True).mean(1)
    sd = float(d.std(ddof=1))
    return {'family':family,'comparison':label,'metric':metric, 'test':'exact sign-flip' if method=='sign_flip' else 'conditional-exact signed-rank',
            'raw_p':flip_p(d) if method=='sign_flip' else signed_rank_p(d.tolist()),
            'mean_difference':float(d.mean()), 'effect_size_dz':float(d.mean()/sd) if sd > 0 else None,
            'paired_bootstrap_ci_low':float(np.quantile(boot,.025)), 'paired_bootstrap_ci_high':float(np.quantile(boot,.975)),
            'wins_out_of_10':int((d>0).sum()), 'seed_pairs':len(d)}


def audit_result(folder, row, epochs, source=None):
    folder = Path(folder)
    result = json.loads((folder/'result.json').read_text())
    if any(result.get(k) != v for k,v in row.items()) or result['epochs'] != epochs:
        raise ValueError('Run identity differs: '+str(folder))
    if not result.get('training_performed') or not result.get('test_evaluated'):
        raise ValueError('Run is not a completed training/test experiment.')
    if source is not None and result['provenance']['source_sha256'] != source:
        raise ValueError('Source changed since this run; select a new output folder.')
    for name, digest in result['output_sha256'].items():
        if sha(folder/name) != digest:
            raise ValueError('Completed run artifact changed: '+name)
    frame = pd.read_csv(folder/'test_predictions.csv')
    if len(frame) != result['test_nodes'] or frame.node_index.duplicated().any():
        raise ValueError('Prediction inventory mismatch.')
    probabilities=frame[['normal_probability','phishing_probability','scamming_probability']].to_numpy()
    if not np.isfinite(probabilities).all() or (probabilities<0).any() or (probabilities>1).any() or not np.allclose(probabilities.sum(1),1,atol=1e-6) or not np.array_equal(probabilities.argmax(1),frame.predicted_label):
        raise ValueError('Saved probabilities are invalid or disagree with class predictions.')
    from sklearn.metrics import f1_score
    score = f1_score(frame.true_label, frame.predicted_label, labels=[0,1,2], average='macro',zero_division=0)
    if abs(score-result['macro_f1']) > 1e-12:
        raise ValueError('Saved predictions do not reproduce reported Macro-F1.')
    cm = np.array([[result[f'cm_{i}_{j}'] for j in range(3)] for i in range(3)])
    if cm.sum() != len(frame):
        raise ValueError('Confusion matrix count mismatch.')
    if row['family'] == 'strict':
        if not result.get('strict_audit') or any(any(v[name] != 0 for name in ['forbidden_feature_values','forbidden_edges','unlabeled_context_edges']) for v in result['strict_audit'].values()):
            raise ValueError('Strict visibility audit failed.')
    return result


def summarize(root, gpu=False):
    root = Path(root);rows = [];missing=[]
    for item in matrix(gpu=gpu):
        folder = root/'full'/run_id(item)
        if not (folder/'result.json').is_file():
            missing.append(run_id(item));continue
        rows.append(audit_result(folder,item,300))
    table = root/'tables';table.mkdir(exist_ok=True)
    status = {'status':'auditing' if not missing else 'incomplete','completed_runs':len(rows),'expected_runs':len(matrix(gpu=gpu)),
              'missing':missing,'gpu_included':gpu,'exploratory':True,'cpu_tables_ready':False, 'gpu_parity_completed':False}
    (table/'completion.json').write_text(json.dumps(status,indent=2)+'\n')
    if rows:
        pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]).to_csv(table/'completed_runs.csv',index=False)
    if missing:
        print(f'INCOMPLETE: {len(rows)}/{len(matrix(gpu=gpu))} runs. No final significance tables generated.')
        return status
    # Mathematically identical controls must stay identical within this runner.
    indexed = {run_id(r): r for r in rows}
    for seed in range(100,110):
        controls = [
            [dict(family='clean',model='mlp',variant='a0_full',coefficient=.3,seed=seed,device='cpu'),
             dict(family='strict',model='mlp',variant='a0_full',coefficient=.3,seed=seed,device='cpu')],
            [dict(family=f,model='x_moe_gnn',variant='a0_full',coefficient=.3,seed=seed,device='cpu')
             for f in ['clean','ablation','sensitivity']],
            [dict(family='ablation',model='x_moe_gnn',variant='a6_no_normal_phishing_penalty',coefficient=.3,seed=seed,device='cpu'),
             dict(family='sensitivity',model='x_moe_gnn',variant='a0_full',coefficient=0.,seed=seed,device='cpu')]]
        for group in controls:
            hashes = {indexed[run_id(r)]['output_sha256']['test_predictions.csv'] for r in group}
            if len(hashes)!=1:
                raise ValueError('Equivalent-control predictions differ; investigate before reporting seed '+str(seed))
    frame = pd.DataFrame(rows)
    numeric = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in ['seed','epochs','coefficient']]
    frame.groupby(['family','device','model','variant','coefficient'])[numeric].agg(['mean','std','count']).to_csv(table/'summary_mean_sd.csv')
    def series(family,model='x_moe_gnn',variant='a0_full',coefficient=.3,device='cpu',metric='macro_f1'):
        selected=frame[(frame.family==family)&(frame.model==model)&(frame.variant==variant)&(frame.coefficient==coefficient)&(frame.device==device)].sort_values('seed')
        if selected.seed.tolist()!=list(range(100,110)):
            raise ValueError('Incomplete paired seed series.')
        return selected[metric].to_numpy(dtype=float)
    tests=[]
    for protocol in ['clean','strict']:
        family=[comparison(series(protocol),series(protocol,m),protocol+'_four_baselines','X-MoE vs '+m) for m in MODELS[:-1]]
        tests.extend(holm(family))
    tests.extend(holm([comparison(series('strict',m),series('clean',m),'clean_to_strict_five_models',m) for m in MODELS]))
    tests.extend(holm([comparison(series('ablation'),series('ablation',variant=v),'ablation_seven_variants','A0 vs '+v,'sign_flip') for v in VARIANTS[1:]]))
    metrics=['phishing_f1','phishing_ap','scamming_f1','scamming_ap']
    tests.extend(holm([comparison(series('strict',metric=m),series('strict','graphsage',metric=m),'strict_minority_four_metrics','X-MoE vs GraphSAGE',metric=m) for m in metrics]))
    if gpu:
        tests.extend(holm([comparison(series('clean',m,device='cuda'),series('clean',m),'cpu_gpu_four_models',m) for m in MODELS[1:]]))
    pd.DataFrame(tests).to_csv(table/'paired_statistics.csv',index=False)
    status.update(status='complete',cpu_tables_ready=True,gpu_parity_completed=gpu)
    (table/'completion.json').write_text(json.dumps(status,indent=2)+'\n')
    print('COMPLETE: audited seed-paired summaries and separate Holm families written.')
    return status
