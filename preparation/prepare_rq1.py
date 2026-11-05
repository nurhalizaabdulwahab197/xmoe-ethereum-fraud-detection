"""Prepare the frozen RQ1 cohort/graph from author archives, without unpickling."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'provenance/frozen_pipeline'))
from utility import read, sha


def named_members(archive, name):
    return [m for m in archive.infolist()
            if not m.is_dir() and m.filename.replace('\\','/').rsplit('/',1)[-1]==name]


def copy_member(archive, member, target, expected_size):
    if member.file_size != expected_size:
        raise ValueError('Unexpected uncompressed member size: '+member.filename)
    target.parent.mkdir(parents=True,exist_ok=True)
    with archive.open(member) as source, target.open('xb') as destination:
        shutil.copyfileobj(source,destination,8*1024**2)
    if target.stat().st_size != expected_size:
        raise ValueError('Extracted size differs: '+target.name)


def extract_graphs(archive_path, raw_root, boundaries):
    """Read either direct pickles or the author's outer ZIP -> group ZIP layout.

    Only selected members are streamed to explicitly chosen destination names;
    archive paths are never extracted as filesystem paths. Nested ZIPs are
    temporarily spooled to disk, not held as multi-gigabyte memory buffers.
    """
    with zipfile.ZipFile(archive_path) as outer:
        for group, entry in boundaries.items():
            name=Path(entry['file']).name
            direct=named_members(outer,name)
            nested=named_members(outer,group+'_pkl.zip')
            if len(direct)+len(nested)!=1:
                raise ValueError('Missing/ambiguous raw graph or nested archive: '+name)
            target=raw_root/entry['file']
            if direct:
                copy_member(outer,direct[0],target,entry['file_bytes'])
            else:
                print('Opening nested archive: '+group+'_pkl.zip',flush=True)
                with tempfile.TemporaryDirectory(prefix='nested_',dir=raw_root) as tmp:
                    inner_path=Path(tmp)/'member.zip'
                    copy_member(outer,nested[0],inner_path,nested[0].file_size)
                    with zipfile.ZipFile(inner_path) as inner:
                        matches=named_members(inner,name)
                        if len(matches)!=1:
                            raise ValueError('Missing/ambiguous member in nested archive: '+name)
                        copy_member(inner,matches[0],target,entry['file_bytes'])


def cohort(rq4, output):
    policy=read(ROOT/'provenance/rq1_label_policy.json')
    excluded=set(policy['excluded_address_sha256'])
    found=set(); accounts=[]
    with zipfile.ZipFile(rq4) as archive:
        for name, class_id in [('normal',0),('phish',1),('scams',2)]:
            source=policy['source_lists'][name]
            raw=archive.read(source['member'])
            if hashlib.sha256(raw).hexdigest()!=source['sha256']:
                raise ValueError('Source account-list hash differs: '+name)
            for address in raw.decode('utf-8-sig').splitlines():
                address=address.strip().lower()
                if not address:
                    continue
                if not re.fullmatch(r'0x[0-9a-f]{40}',address) or address in found:
                    raise ValueError('Invalid or conflicting source address')
                found.add(address)
                if hashlib.sha256(address.encode()).hexdigest() in excluded:
                    continue
                accounts.append(dict(address=address,class_id=class_id,class_name=name,
                    label_status='candidate_without_detected_conflict'))
    accounts.sort(key=lambda r:r['address'])
    if [sum(r['class_id']==c for r in accounts) for c in range(3)]!=policy['expected_class_counts']:
        raise ValueError('Frozen cohort counts differ')
    if len(found)-len(accounts)!=policy['excluded_count']:
        raise ValueError('Exclusion set differs')
    with (output/'candidate_accounts.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(accounts[0]));w.writeheader();w.writerows(accounts)
    labels={'accounts':[dict(address=r['address'],source_declared_class=r['class_name'],
        status=r['label_status']) for r in accounts]}
    (output/'provisional_target_manifest.json').write_text(json.dumps(labels,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--rq1-zip',type=Path)
    source.add_argument('--rq1-unpacked',type=Path)
    p.add_argument('--rq4-zip',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--node',default='node')
    p.add_argument('--verify-raw-only',action='store_true',
                   help='Verify raw bytes and frozen cohort only; skip indexing/features/splits')
    a=p.parse_args()
    out=a.output_dir.resolve()
    if out.exists():
        raise ValueError('Output exists; preserve it and choose a fresh directory')
    out.mkdir(parents=True)
    inputs=out/'inputs';inputs.mkdir()
    boundaries=read(ROOT/'provenance/rq1_node_boundaries.json')
    if a.rq1_zip:
        raw_root=out/'raw';raw_root.mkdir()
        extract_graphs(a.rq1_zip,raw_root,boundaries)
    else:
        raw_root=a.rq1_unpacked.resolve()
    expected={r['Name']:r['SHA256'] for r in read(ROOT/'provenance/rq1_raw_hashes.json')}
    for group,entry in boundaries.items():
        path=raw_root/entry['file']
        if sha(path)!=expected[path.name]:
            raise ValueError('Raw pickle hash mismatch: '+path.name)
        entry['file']=str(path.resolve())
        print('PASS raw graph: '+path.name,flush=True)
    cohort(a.rq4_zip,inputs)
    (inputs/'node_boundaries.json').write_text(json.dumps(boundaries,indent=2),encoding='utf-8')
    if a.verify_raw_only:
        print('PASS: pinned raw files and frozen 4913-account cohort; no graph extraction or training')
        return
    tasks=[
        [a.node,str(ROOT/'preparation/rq1_full_observation_index.mjs'),
         '--accounts',str(inputs/'candidate_accounts.csv'),'--labels',str(inputs/'provisional_target_manifest.json'),
         '--boundaries',str(inputs/'node_boundaries.json'),'--output-dir',str(out/'index')],
        [sys.executable,'-B',str(ROOT/'preparation/rq1_aggregate_features.py'),
         '--index',str(out/'index'),'--accounts',str(inputs/'candidate_accounts.csv'),'--output-dir',str(out/'aggregation')],
        [sys.executable,'-B',str(ROOT/'preparation/rq1_build_graph_inputs.py'),
         '--index',str(out/'index'),'--accounts',str(inputs/'candidate_accounts.csv'),
         '--aggregation',str(out/'aggregation'),'--output-dir',str(out/'graph')],
        [sys.executable,'-B',str(ROOT/'preparation/matched_data_adapter.py'),
         '--graph',str(out/'graph'),'--output-dir',str(out/'seed_inputs')]]
    for i,cmd in enumerate(tasks,1):
        print(f'Preparation stage {i}/{len(tasks)}',flush=True)
        subprocess.run(cmd,check=True)
        if i==3:
            expected_graph=read(ROOT/'provenance/rq1_expected_arrays.json')['graph_file_sha256']
            for name,digest in expected_graph.items():
                if sha(out/'graph'/name)!=digest:
                    raise ValueError('Rebuilt graph differs from frozen study: '+name)
    print('RQ1 graph prepared. Next run scripts/configure_datasets.py to verify all seed arrays and create the registry.')

if __name__=='__main__':
    main()
