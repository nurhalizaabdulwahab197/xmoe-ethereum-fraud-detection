"""Evaluate saved three-class probabilities without training or model loading."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

CLASSES = ('normal', 'phishing', 'scamming')


def binary_auc_ap(labels, scores):
    """ROC-AUC has half-credit for ties; AP uses grouped thresholds."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives:
        return None, None
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    wins = ap = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        pos = sum(y for _, y in pairs[i:j])
        neg = j - i - pos
        wins += pos * (negatives - fp - neg) + 0.5 * pos * neg
        tp += pos
        fp += neg
        ap += (pos / positives) * (tp / (tp + fp))
        i = j
    return (wins / (positives * negatives) if negatives else None), ap


def evaluate(path):
    path = Path(path)
    aliases = {
        'label': ('true_label', 'y_true'),
        'prediction': ('predicted_label', 'y_pred', 'prediction'),
        'node': ('node_index', 'node_id'),
        'normal': ('normal_probability', 'probability_normal', 'prob_normal'),
        'phishing': ('phishing_probability', 'probability_phishing', 'prob_phishing'),
        'scamming': ('scamming_probability', 'probability_scamming', 'probability_scams', 'prob_scamming'),
    }
    labels, probabilities = [], []
    cm = [[0] * 3 for _ in range(3)]
    seen = set()
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = {name.strip().lower(): name for name in reader.fieldnames or []}
        columns = {key: next((fields[n] for n in names if n in fields), None)
                   for key, names in aliases.items()}
        if any(columns[k] is None for k in ('label', *CLASSES)):
            raise ValueError('Missing true labels or three class-probability columns')
        for row in reader:
            y = int(row[columns['label']])
            p = [float(row[columns[c]]) for c in CLASSES]
            if y not in range(3) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p):
                raise ValueError('Invalid class label or probability')
            if abs(sum(p) - 1) > 1e-5:
                raise ValueError('Class probabilities do not sum to one')
            predicted = max(range(3), key=p.__getitem__)
            if columns['prediction'] and int(row[columns['prediction']]) != predicted:
                raise ValueError('Stored prediction differs from probability argmax')
            if columns['node']:
                node = int(row[columns['node']])
                if node in seen:
                    raise ValueError('Duplicate test node')
                seen.add(node)
            cm[y][predicted] += 1
            labels.append(y)
            probabilities.append(p)
    if not labels:
        raise ValueError('Empty prediction file')
    n = len(labels)
    support = [sum(r) for r in cm]
    predicted_counts = [sum(cm[i][j] for i in range(3)) for j in range(3)]
    result = {'test_nodes': n, 'accuracy': sum(cm[i][i] for i in range(3)) / n}
    for i, c in enumerate(CLASSES):
        result[c + '_support'] = support[i]
        result[c + '_precision'] = cm[i][i] / predicted_counts[i] if predicted_counts[i] else 0.0
        result[c + '_recall'] = cm[i][i] / support[i] if support[i] else 0.0
        total = support[i] + predicted_counts[i]
        result[c + '_f1'] = 2 * cm[i][i] / total if total else 0.0
        auc, ap = binary_auc_ap([int(y == i) for y in labels], [p[i] for p in probabilities])
        result[c + '_roc_auc_ovr'] = auc
        result[c + '_ap'] = ap
        for j in range(3):
            result[f'cm_{i}_{j}'] = cm[i][j]
    for metric in ('precision', 'recall', 'f1'):
        result['macro_' + metric] = statistics.mean(result[c + '_' + metric] for c in CLASSES)
        result['weighted_' + metric] = sum(support[i] * result[c + '_' + metric] for i, c in enumerate(CLASSES)) / n
    result['balanced_accuracy'] = result['macro_recall']
    denominator = math.sqrt((n*n - sum(v*v for v in predicted_counts)) * (n*n - sum(v*v for v in support)))
    result['mcc'] = ((sum(cm[i][i] for i in range(3))*n - sum(a*b for a, b in zip(support, predicted_counts))) / denominator) if denominator else 0.0
    aucs = [result[c + '_roc_auc_ovr'] for c in CLASSES]
    result['macro_roc_auc_ovr'] = statistics.mean(aucs) if all(a is not None for a in aucs) else None
    result['weighted_roc_auc_ovr'] = sum(support[i]*aucs[i] for i in range(3))/n if all(a is not None for a in aucs) else None
    aps = [result[c + '_ap'] for c in CLASSES]
    result['macro_ap'] = statistics.mean(aps) if all(a is not None for a in aps) else None
    result['prediction_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = evaluate(args.predictions)
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        if args.output.exists():
            raise SystemExit('Output already exists; choose a new path')
        args.output.write_text(payload, encoding='utf-8')
    else:
        print(payload, end='')
