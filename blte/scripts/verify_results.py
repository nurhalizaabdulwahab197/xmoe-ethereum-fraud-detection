import argparse
from pathlib import Path
from common import ROOT,read,require,sha,within,splits,check_prediction

def main():
    p=argparse.ArgumentParser(description='Read-only verification of imported reference evidence or new runner outputs.')
    p.add_argument('--results-dir',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    a=p.parse_args()
    from common import input_check
    import pandas as pd
    import numpy as np
    labels=pd.read_csv(input_check(a.upstream)/'BLTE_multiclass_v2.csv',usecols=['attack_encoded'])['attack_encoded'].to_numpy()
    saved=splits();seen=set();count=0
    for path in sorted(a.results_dir.rglob('metadata.json')):
        m=read(path);require(m['status'] in ('PASS_NEW_RUN_VALIDATED','PASS_IMPORTED_EVIDENCE'),'Incomplete run')
        seed=m['split_seed'];key=(m['configuration'],m['model'],seed)
        require(key not in seen,'Duplicate run key');seen.add(key)
        require(m['epochs']==300 and m['split_sha256']==sha(saved[seed][0]),'Epoch/split identity')
        for n,h in m['output_sha256'].items():require(sha(within(path.parent,n))==h,'Output mismatch')
        computed=check_prediction(path.parent/'predictions.csv',saved[seed][1]['test'],labels[saved[seed][1]['test']])
        original=read(path.parent/'metrics.json')
        for k,v in computed.items():
            if k=='confusion_matrix':
                if k in original:require(v==original[k],k)
            elif k in original:require(np.isclose(v,original[k],atol=1e-10,rtol=0),k)
        count+=1
    require(count>0,'No metadata.json files found')
    index=a.results_dir/'IMPORT_VALIDATION.json'
    if index.is_file():
        info=read(index);require(info['status']=='PASS_120_IMPORTED_EVIDENCE' and count==120,'Import completeness')
        for r in info['runs']:require(sha(within(a.results_dir,r['path'])/'metadata.json')==r['metadata_sha256'],'Imported metadata changed')
    print('PASS_RESULTS:',count,'runs; no training or writes')
if __name__=='__main__':main()
