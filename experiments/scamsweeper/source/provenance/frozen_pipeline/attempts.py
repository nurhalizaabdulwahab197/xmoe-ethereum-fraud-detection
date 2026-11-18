"""Preserve failed attempts; reuse only a uniquely verified completed run."""
from pathlib import Path
from rq1_safe_io import atomic_json
from utility import read


def bind_study(output, identity):
    path = output / 'study_identity.json'
    if path.exists():
        if read(path) != identity:
            raise ValueError('Output belongs to another dataset/config/code/registry identity')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Existing output lacks study identity; audit it before reuse')
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(path, identity)


def resolve_attempt(output, model, seed, is_complete):
    base = output / f'{model}_seed{seed}'
    candidates = [base] + sorted(output.glob(f'{model}_seed{seed}_attempt[0-9]*'))
    candidates = [p for p in candidates if p.is_dir()]
    completed = [p for p in candidates if is_complete(p)]
    if len(completed) > 1:
        raise ValueError('Multiple verified completions; adjudicate without selecting by metrics')
    if completed:
        return completed[0], True
    if not base.exists():
        return base, False
    attempt = 2
    while True:
        path = output / f'{model}_seed{seed}_attempt{attempt:03d}'
        if not path.exists():
            return path, False
        attempt += 1


def inventory(output, planned, is_complete):
    records = []
    for model, seed in planned:
        paths = [output / f'{model}_seed{seed}'] + sorted(output.glob(f'{model}_seed{seed}_attempt[0-9]*'))
        paths = [p for p in paths if p.is_dir()]
        records.append({'model': model, 'seed': seed,
                        'attempts': [{'folder': p.name, 'verified_complete': bool(is_complete(p, model, seed))}
                                     for p in paths]})
    atomic_json(output / 'attempt_inventory.json', {'runs': records,
        'policy': 'Reuse verified completed outputs; preserve failed attempts. Interrupted seeds restart from original seed.'})
