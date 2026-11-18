"""Summarize a complete 120-run primary series. Does not train or select repeats."""
import argparse
from pathlib import Path
import itertools
from common import ROOT,read,require,write

def signed_rank(differences):
    import numpy as np
    from scipy.stats import rankdata
    d=np.round(np.asarray(differences,dtype=float),12);d=d[d!=0]
    if not len(d):return 1.
    ranks=rankdata(abs(d),method='average');observed=float(ranks[d>0].sum())
    stats=np.array([np.dot(ranks,signs) for signs in itertools.product([0,1],repeat=len(d))])
    return min(1.,2*min(float(np.mean(stats<=observed+1e-12)),float(np.mean(stats>=observed-1e-12))))

def summarize(d):
    import numpy as np
    import pandas as pd
    needed={(c,m,s) for c in ('author','paper') for m in ('MLP','GCN','GAT','GraphSAGE','XMoE_CE','Full_XMoE') for s in range(100,110)}
    keys=list(zip(d.configuration,d.model,d.split_seed.astype(int)))
    require(len(keys)==120 and set(keys)==needed,'Expected exactly 120 unique primary identities')
    summaries=[]
    for (c,m),g in d.groupby(['configuration','model'],sort=True):
        for metric in ['accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','normal_f1','phishing_f1','scamming_f1']:
            x=g[metric].to_numpy(dtype=float)*100
            summaries.append(dict(configuration=c,model=m,metric=metric,n=len(x),mean_percent=x.mean(),sample_sd_percent=x.std(ddof=1),population_sd_percent=x.std(ddof=0)))
    tests=[]
    for c in ['author','paper']:
        wide=d[d.configuration==c].pivot(index='split_seed',columns='model',values='macro_f1')
        for m in ['MLP','GCN','GAT','GraphSAGE','XMoE_CE']:
            delta=wide.Full_XMoE-wide[m]
            tests.append(dict(configuration=c,comparison='Full_XMoE vs '+m,mean_difference_pp=float(delta.mean()*100),full_higher=int((delta>0).sum()),ties=int((delta==0).sum()),comparator_higher=int((delta<0).sum()),raw_p=signed_rank(delta),holm_adjusted_p=0.,holm_family_size=10))
    order=sorted(range(10),key=lambda i:tests[i]['raw_p']);last=0.
    for rank,i in enumerate(order):
        last=max(last,min(1.,(10-rank)*tests[i]['raw_p']));tests[i]['holm_adjusted_p']=last
    return pd.DataFrame(summaries),pd.DataFrame(tests)

def load_new(folder):
    import pandas as pd
    labels={'mlp':'MLP','gcn':'GCN','gat':'GAT','graphsage':'GraphSAGE','xmoe-ce':'XMoE_CE','full-xmoe':'Full_XMoE'}
    rows=[]
    for p in sorted(folder.rglob('metadata.json')):
        m=read(p);require(m['status']=='PASS_NEW_RUN_VALIDATED','New series includes incomplete/imported evidence')
        r={k:v for k,v in read(p.parent/'metrics.json').items() if isinstance(v,(int,float))}
        r.update(configuration={'released-code':'author','paper-described':'paper'}[m['configuration']],model=labels[m['model']],split_seed=m['split_seed'])
        rows.append(r)
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',action='store_true',help='Use bundled primary seed-level CSV')
    p.add_argument('--results-dir',type=Path,help='Complete NEW series, never mix with reference/repeats')
    p.add_argument('--output-dir',type=Path,help='Optional new directory; otherwise print only')
    a=p.parse_args();require(bool(a.reference)!=(a.results_dir is not None),'Choose --reference OR --results-dir')
    import pandas as pd
    d=pd.read_csv(ROOT/'results/tables/one_hundred_twenty_runs.csv') if a.reference else load_new(a.results_dir)
    summary,tests=summarize(d)
    print(summary[summary.metric=='macro_f1'].to_string(index=False));print(tests.to_string(index=False))
    if a.output_dir:
        a.output_dir.mkdir(parents=True,exist_ok=False)
        summary.to_csv(a.output_dir/'six_model_summary.csv',index=False)
        tests.to_csv(a.output_dir/'full_xmoe_paired_tests_holm10.csv',index=False)
    print('PASS_SUMMARY; exploratory comparisons; sample SD is not a confidence interval')
if __name__=='__main__':main()
