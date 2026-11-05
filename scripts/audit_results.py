"""Audit saved predictions, selected epochs, hashes and all planned results."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
MODELS=['mlp','gcn','gat','graphsage','x_moe_gnn']

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):
            h.update(b)
    return h.hexdigest()

def close(actual,expected,label):
    if not math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12):
        raise ValueError(f'{label}: {actual} != {expected}')

def audit(root,dataset,published=False):
    if published:
        locations={(m,s):root/m/f'seed{s}' for m in MODELS for s in range(100,110)}
        evidence=read(ROOT/'provenance/evidence_inventory.json')['runs']
        expected={(r['model'],r['seed']):r for r in evidence if r['dataset']==dataset}
    else:
        inv=read(root/'attempt_inventory.json')['runs'];locations={}
        for row in inv:
            paths=[root/a['folder'] for a in row['attempts'] if a['verified_complete']]
            if len(paths)!=1:
                raise ValueError('Missing or ambiguous completion')
            locations[row['model'],row['seed']]=paths[0]
    if set(locations)!={(m,s) for m in MODELS for s in range(100,110)}:
        raise ValueError('Expected exactly five models and ten seeds')
    config=read(ROOT/'common_config.json')
    metrics={m:[] for m in MODELS}
    fingerprints={}
    for (model,seed),folder in locations.items():
        r=read(folder/'result.json');p=read(folder/'resource_probe.json')
        if (r['model'],r['seed'],r['dataset'])!=(model,seed,dataset):
            raise ValueError('Run identity mismatch')
        if r['epochs']!=300 or p['status']!='passed' or not r['test_evaluated']:
            raise ValueError('Incomplete final run')
        if r['shared_config_sha256']!=sha(ROOT/'common_config.json') or r['release_manifest_sha256']!=sha(ROOT/'release_manifest.json'):
            raise ValueError('Training source/config changed')
        if r['optimizer']!=config['optimizer'] or r['actual_torch_threads']!=24:
            raise ValueError('Execution differs from shared protocol')
        if r['effective_training_options']!=config['training_options']:
            raise ValueError('Training options differ')
        dimension=23 if dataset=='blte' else 5
        count=10955 if dataset=='blte' else 737
        if r['dataset_metadata']['input_dim']!=dimension:
            raise ValueError('Input adaptation differs')
        fp=r['dataset_metadata']['input_fingerprints']
        if seed in fingerprints and fp!=fingerprints[seed]:
            raise ValueError('Models do not share input/split fingerprints')
        fingerprints[seed]=fp
        if sha(folder/'training_history.csv')!=r['training_history_sha256']:
            raise ValueError('Saved evidence changed')
        if published:
            e=expected[model,seed]
            if sha(folder/'result.json')!=e['published_result_sha256']:
                raise ValueError('Published result changed')
        else:
            if sha(folder/'best_checkpoint.pt')!=r['checkpoint_sha256'] or sha(folder/'test_predictions.csv')!=r['prediction_sha256']:
                raise ValueError('Checkpoint or predictions changed')
        with (folder/'training_history.csv').open(encoding='utf-8-sig',newline='') as f:
            history=list(csv.DictReader(f))
        if len(history)!=300:
            raise ValueError('History length differs')
        best=max(float(h['validation_macro_f1']) for h in history)
        first=next(int(h['epoch']) for h in history if float(h['validation_macro_f1'])==best)
        close(best,r['best_validation_macro_f1'],'Validation selection')
        if first!=r['best_epoch']:
            raise ValueError('Earliest best epoch differs')
        if published:
            # Public evidence contains run-level confusion cells; per-node labels
            # and probabilities remain in the author's local archive.
            cm=[[r[f'cm_{i}_{j}'] for j in range(3)] for i in range(3)]
            if sum(map(sum,cm))!=count:
                raise ValueError('Wrong aggregate test population')
        else:
            with (folder/'test_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
                predictions=list(csv.DictReader(f))
            if len(predictions)!=count:
                raise ValueError('Wrong test population')
            cm=[[0]*3 for _ in range(3)];seen=set()
            for row in predictions:
                y=int(row['true_label']);pred=int(row['predicted_label'])
                node=int(row['node_index'])
                if node in seen or y not in range(3) or pred not in range(3):
                    raise ValueError('Invalid or duplicated test row')
                seen.add(node)
                prob=[float(row[k]) for k in ['normal_probability','phishing_probability','scamming_probability']]
                if any(not math.isfinite(v) or not 0<=v<=1 for v in prob) or abs(sum(prob)-1)>1e-5:
                    raise ValueError('Invalid probability vector')
                if max(range(3),key=prob.__getitem__)!=pred:
                    raise ValueError('Predicted label differs from probabilities')
                cm[y][pred]+=1
        f1=[]
        for c,name in enumerate(['normal','phishing','scamming']):
            tp=cm[c][c];actual=sum(cm[c]);called=sum(cm[i][c] for i in range(3))
            value=2*tp/(actual+called) if actual+called else 0
            close(value,r[name+'_f1'],name+' F1');f1.append(value)
            for j in range(3):
                if cm[c][j]!=r[f'cm_{c}_{j}']:
                    raise ValueError('Confusion cells differ')
        close(sum(f1)/3,r['macro_f1'],'Macro-F1')
        close(sum(cm[c][c] for c in range(3))/count,r['accuracy'],'Accuracy')
        metrics[model].append(r)
    summary=read(root/'summary.json')
    if summary['completed_runs']!=50 or not summary['all_complete']:
        raise ValueError('Summary is incomplete')
    for model,runs in metrics.items():
        for key,item in summary['models'][model]['metrics'].items():
            values=[r[key] for r in runs]
            close(statistics.mean(values),item['mean'],model+' '+key+' mean')
            close(statistics.stdev(values),item['sample_std'],model+' '+key+' SD')
    return {'dataset':dataset,'verified_runs':50,'predictions_recomputed_from_saved_rows':not published,
            'training_performed':False,'model_inference_performed':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['blte','rq1'],required=True)
    p.add_argument('--results-dir',type=Path,required=True)
    p.add_argument('--published',action='store_true')
    a=p.parse_args()
    print(json.dumps(audit(a.results_dir,a.dataset,a.published),indent=2))

if __name__=='__main__':
    main()
