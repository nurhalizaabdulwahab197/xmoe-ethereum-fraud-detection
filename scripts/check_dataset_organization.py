"""Read-only validation of dataset boundaries and published locked evidence."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics
ROOT = Path(__file__).resolve().parents[1]
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def check_sweeper():
    base=ROOT/'results/scamsweeper';source=ROOT/'experiments/scamsweeper/source'
    lock=json.loads((base/'provenance/experiment_lock.json').read_text())
    for name,h in lock['source_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h,name
    published=json.loads((ROOT/'docs/SCAMSWEEPER_PUBLISHED_HASHES.json').read_text())
    for name,h in published['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    status=json.loads((base/'tables/completion.json').read_text())
    assert status['status']=='complete' and status['completed_runs']==status['expected_runs']==230
    rows=read_csv(base/'tables/completed_runs.csv');assert len(rows)==230
    groups={}
    for r in rows:
        assert r['device']=='cuda' and int(r['epochs'])==300
        key=(r['family'],r['model'],r['variant'],float(r['coefficient']))
        groups.setdefault(key,[]).append(r)
    assert len(groups)==23
    for key,g in groups.items():assert sorted(int(r['seed']) for r in g)==list(range(100,110)),key
    for family,count in {'clean':50,'strict':50,'ablation':80,'sensitivity':50}.items():
        subset=read_csv(base/family/'per_seed.csv');assert len(subset)==count
        expected={(r['model'],r['variant'],float(r['coefficient']),int(r['seed'])):r for r in subset}
        records=[json.loads(line) for line in (base/family/'result_records.jsonl').read_text().splitlines()]
        assert len(records)==count and {r['family'] for r in subset}=={family}
        for r in records:
            assert r['device']=='cuda' and r['epochs']==300 and r['family']==family
            assert r['provenance']['source_sha256']==lock['source_sha256']
            key=(r['model'],r['variant'],r['coefficient'],r['seed'])
            assert abs(float(expected[key]['macro_f1'])-r['macro_f1'])<1e-12
    stats=read_csv(base/'tables/paired_statistics.csv')
    variants={k[2]:sorted(v,key=lambda r:int(r['seed'])) for k,v in groups.items() if k[0]=='ablation'}
    a0=[float(r['macro_f1']) for r in variants['a0_full']];raw=[]
    for variant,records in sorted(variants.items()):
        if variant=='a0_full':continue
        delta=[a-float(b['macro_f1']) for a,b in zip(a0,records)];obs=abs(statistics.mean(delta))
        p=sum(abs(statistics.mean(d*s for d,s in zip(delta,signs)))>=obs-1e-15 for signs in itertools.product([-1,1],repeat=10))/1024
        raw.append((variant,p))
    prev=0
    for rank,(variant,p) in enumerate(sorted(raw,key=lambda x:x[1])):
        prev=max(prev,(7-rank)*p)
        saved=next(r for r in stats if r['family']=='ablation_seven_variants' and r['comparison']=='A0 vs '+variant)
        assert abs(float(saved['holm_adjusted_p'])-min(1.,prev))<1e-12
    print('PASS ScamSweeper: locked source, 230 CUDA identities, ten-seed families, published hashes, result records and ablation Holm tests.')
def check_blte():
    mapping=json.loads((ROOT/'docs/DATASET_PATH_MAP.json').read_text())
    for item in mapping['blte']:
        old,new=ROOT/item['legacy'],ROOT/item['dataset_path']
        data=new.read_bytes();assert old.read_bytes()==data,item
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['git_blob_sha'],item
    print('PASS BLTE: dataset-labelled result views match original evidence bytes and Git blob SHAs.')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scope',choices=['all','scamsweeper','blte'],default='all');a=p.parse_args()
    if a.scope in ['all','scamsweeper']:check_sweeper()
    if a.scope in ['all','blte']:check_blte()
if __name__=='__main__':main()
