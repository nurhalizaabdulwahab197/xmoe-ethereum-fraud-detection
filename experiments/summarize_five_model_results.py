"""Rebuild five-model BLTE tables from frozen per-seed CSVs; never train.

Default: print/check only. --apply writes manuscript and demo result tables.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, PermutationMethod

MODELS = ['mlp', 'gcn', 'gat', 'graphsage', 'x_moe_gnn']
NAMES = dict(zip(MODELS, ['MLP', 'GCN', 'GAT', 'GraphSAGE', 'X-MoE']))
MINORITY = ['phishing_f1', 'phishing_ap', 'scamming_f1', 'scamming_ap']

def holm(rows):
    previous = 0.0
    for rank, index in enumerate(np.argsort([r['raw_p'] for r in rows])):
        previous = max(previous, (len(rows)-rank)*rows[index]['raw_p'])
        rows[index]['holm_adjusted_p'] = min(previous, 1.0)
    return rows

def paired(d):
    d = np.asarray(d, dtype=float)
    if np.all(d == 0):
        return 1.0
    method = (PermutationMethod(n_resamples=np.inf)
              if np.any(d == 0) or len(np.unique(np.abs(d))) < len(d)
              else 'exact')
    return float(wilcoxon(d, alternative='two-sided', method=method).pvalue)

def indexed(frame, model, protocol):
    frame = frame.sort_values('seed').reset_index(drop=True)
    assert len(frame) == 10 and frame.seed.tolist() == list(range(100, 110))
    assert (frame.model == model).all() and (frame.protocol == protocol).all()
    assert (frame.epochs == 300).all()
    assert np.isfinite(frame.macro_f1).all()
    for _, r in frame.iterrows():
        cm = np.array([[r[f'cm_{i}_{j}'] for j in range(3)] for i in range(3)])
        assert cm.sum() == r.test_nodes
        denominator = cm.sum(0) + cm.sum(1)
        f1 = np.divide(2*cm.diagonal(), denominator, out=np.zeros(3), where=denominator != 0)
        assert abs(f1.mean()-r.macro_f1) < 1e-12
    return frame

def build(root):
    ref = root / 'results/reference'
    raw = pd.read_csv(ref / 'blte_strict_five_models/strict_50_runs.csv')
    assert len(raw) == 50 and not raw.duplicated(['model', 'seed']).any()
    assert set(raw.model) == set(MODELS)
    zero = [c for c in raw if 'leakage' in c] + ['validation_validation_edges', 'test_test_edges']
    assert (raw[zero] == 0).all().all()
    clean = {m: indexed(pd.read_csv(ref / f'blte_clean/{m}_10seed.csv'), m, 'clean') for m in MODELS}
    strict = {m: indexed(raw[raw.model == m], m, 'strict_inductive') for m in MODELS}
    overall, changes, minor, demo, tests = [], [], [], [], []
    for protocol, frames in [('clean', clean), ('strict', strict)]:
        family = []
        for m in MODELS[:-1]:
            d = frames['x_moe_gnn'].macro_f1.to_numpy() - frames[m].macro_f1.to_numpy()
            family.append(dict(analysis=protocol, comparison=f'X-MoE vs {NAMES[m]}',
                metric='macro_f1', raw_p=paired(d), mean_difference=d.mean(),
                effect_size_dz=d.mean()/d.std(ddof=1), wins_out_of_10=int((d > 0).sum()),
                correction_family=f'{protocol}_macro_f1_four_baselines', family_size=4))
        tests.extend(holm(family))
        for m in MODELS:
            frame = frames[m]
            minor.append(dict(protocol=protocol, model=NAMES[m], **{k: frame[k].mean() for k in ['normal_f1']+MINORITY}))
            for k in ['accuracy', 'macro_f1', 'normal_f1']+MINORITY:
                demo.append(dict(protocol=protocol, model=NAMES[m], metric=k,
                    mean=frame[k].mean(), std=frame[k].std(ddof=1)))
    for m in MODELS:
        s, c = strict[m].macro_f1.to_numpy(), clean[m].macro_f1.to_numpy()
        d = s-c
        changes.append(dict(analysis='clean_to_strict', comparison=NAMES[m], metric='macro_f1',
            model=NAMES[m], clean_macro_f1=c.mean(), strict_macro_f1=s.mean(),
            relative_change_percent=100*d.mean()/c.mean(), raw_p=paired(d),
            mean_difference=d.mean(), max_absolute_seed_difference=np.abs(d).max(),
            correction_family='clean_to_strict_five_models', family_size=5))
        t = next((r for r in tests if r['analysis']=='strict' and r['comparison']==f'X-MoE vs {NAMES[m]}'), {})
        overall.append(dict(model=NAMES[m], macro_f1_mean=s.mean(), macro_f1_std=s.std(ddof=1),
            xmoe_minus_model=strict['x_moe_gnn'].macro_f1.mean()-s.mean(),
            holm_adjusted_p_vs_xmoe=t.get('holm_adjusted_p', np.nan),
            xmoe_wins_out_of_10=t.get('wins_out_of_10', np.nan)))
    holm(changes)
    tests.extend(changes)
    family = []
    for k in MINORITY:
        d = strict['x_moe_gnn'][k].to_numpy()-strict['graphsage'][k].to_numpy()
        family.append(dict(analysis='strict_minority', comparison='X-MoE vs GraphSAGE', metric=k,
            raw_p=paired(d), mean_difference=d.mean(), correction_family='strict_minority_four_metrics', family_size=4))
    tests.extend(holm(family))
    ablation = pd.read_csv(root / 'results/paper/statistical_tests.csv')
    ablation = ablation[ablation.analysis == 'ablation'].copy()
    all_tests = pd.concat([ablation, pd.DataFrame(tests)], ignore_index=True)
    return {
        'results/paper/strict_inductive_10seed.csv': pd.DataFrame(overall),
        'results/paper/clean_to_strict_summary.csv': pd.DataFrame(changes)[['model','clean_macro_f1','strict_macro_f1','relative_change_percent','holm_adjusted_p']],
        'results/paper/minority_class_summary.csv': pd.DataFrame(minor),
        'results/paper/statistical_tests.csv': all_tests,
        'demo/data/verified_protocol_metrics.csv': pd.DataFrame(demo),
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    outputs = build(args.root)
    for name, frame in outputs.items():
        if args.apply:
            target = args.root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(target, index=False, float_format='%.15g')
        print(name)
        if 'protocol_metrics' not in name:
            print(frame.to_string(index=False))
    print(json.dumps({'status': 'five_model_tables_rebuilt' if args.apply else 'five_model_tables_checked',
        'training_performed': False, 'strict_runs': 50,
        'holm_families': {'clean_baselines':4, 'strict_baselines':4, 'clean_to_strict':5, 'strict_minority':4}}))

if __name__ == '__main__':
    main()
