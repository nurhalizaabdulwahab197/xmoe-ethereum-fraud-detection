"""Read-only summaries of new BLTE run CSVs; never trains or overwrites evidence."""
import argparse
import csv
import math
from pathlib import Path
import statistics

SEEDS = set(range(100, 110))

def collect(root, family):
    records = []
    for path in sorted(root.rglob('*.csv')):
        if path.name.endswith('_summary.csv'):
            continue
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or [])
            if not {'seed', 'epochs', 'macro_f1', 'protocol'}.issubset(fields):
                continue
            if family == 'ablation' and 'variant_id' not in fields:
                continue
            if family != 'ablation' and ('model' not in fields or 'variant_id' in fields):
                continue
            if family == 'sensitivity' and 'x_moe_normal_phishing_penalty' not in fields:
                continue
            for row in reader:
                if int(row['epochs']) != 300:
                    raise ValueError(f'Non-300-epoch run found: {path}; keep smoke outputs separate.')
                expected = 'strict_inductive' if family == 'strict' else 'clean'
                if row['protocol'] != expected:
                    raise ValueError(f'Unexpected protocol in {path}: {row["protocol"]}')
                row['_file'] = str(path)
                records.append(row)
    if not records:
        raise ValueError('No matching BLTE full-run CSVs found; select the new experiment output folder, not prediction files.')
    return records

def summarize(records, family, view):
    groups = {}
    seen = set()
    for row in records:
        label = row['variant_id'] if family == 'ablation' else row['model']
        if family == 'sensitivity':
            label += '/lambda_' + format(float(row['x_moe_normal_phishing_penalty']), '.2f')
        device = row.get('device', 'unspecified')
        key = (label, device)
        identity = (*key, int(row['seed']))
        if identity in seen:
            raise ValueError(f'Duplicate group/seed {identity}; do not mix all-variant and individual reruns or repeat invocations.')
        seen.add(identity)
        groups.setdefault(key, []).append(row)
    outputs = []
    for key, rows in sorted(groups.items()):
        seeds = {int(r['seed']) for r in rows}
        status = 'COMPLETE 10 SEEDS' if seeds == SEEDS and len(rows) == 10 else 'INCOMPLETE / DIFFERENT SEEDS'
        metrics = ['accuracy', 'macro_f1', 'phishing_f1', 'phishing_ap', 'scamming_f1', 'scamming_ap']
        if view == 'connectivity':
            metrics = sorted({k for r in rows for k in r if k.startswith(('connected_', 'isolated_', 'test_connected_', 'test_isolated_'))})
        summary = {}
        for metric in metrics:
            values = []
            for row in rows:
                value = row.get(metric, '')
                if value == '':continue
                number = float(value)
                if math.isfinite(number):values.append(number)
            if values:
                summary[metric] = (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None, len(values))
        if view == 'connectivity':
            fpr = []
            for row in rows:
                total = float(row.get('test_isolated_nodes', '0') or 0)
                normal = float(row.get('isolated_normal_nodes', '0') or 0)
                recall = row.get('isolated_normal_recall', '')
                if total > 0 and normal == total and recall != '':
                    fpr.append(1.0 - float(recall))
            if fpr:
                summary['isolated_normal_fpr_normal_only_stratum'] = (statistics.mean(fpr), statistics.stdev(fpr) if len(fpr)>1 else None, len(fpr))
        outputs.append((key,status,summary))
    return outputs

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--family', choices=['clean','strict','ablation','sensitivity'], required=True)
    parser.add_argument('--view', choices=['metrics','connectivity'], default='metrics')
    args = parser.parse_args()
    records = collect(args.root,args.family)
    print('BLTE only; read-only; mean +/- sample SD; counts reflect available finite values.')
    for (label,device),status,metrics in summarize(records,args.family,args.view):
        print(f'\n{label} | {device} | {status}')
        for name,(mean,sd,count) in metrics.items():
            sd_text = f'{sd:.6f}' if sd is not None else 'undefined'
            print(f'  {name}: {mean:.6f} +/- {sd_text} (n={count})')

if __name__ == '__main__':main()
