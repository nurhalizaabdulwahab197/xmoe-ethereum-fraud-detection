import argparse
from pathlib import Path
from common import input_check,splits,require

def main():
    ap=argparse.ArgumentParser(description='Recompute split arrays in memory and compare saved indices; no writes/training.')
    ap.add_argument('--upstream',type=Path,required=True);a=ap.parse_args()
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    g=torch.load(input_check(a.upstream)/'BLTE_multiclass_graph.pt',map_location='cpu',weights_only=False)
    for seed,(_,arrays) in splits(g).items():
        train,other=train_test_split(np.arange(g.num_nodes),test_size=.3,stratify=g.y.numpy(),random_state=seed)
        val,test=train_test_split(other,test_size=.5,stratify=g.y.numpy()[other],random_state=seed)
        for key,actual in [('train',train),('validation',val),('test',test)]:
            require(np.array_equal(np.sort(actual),arrays[key]),f'Reconstructed indices mismatch: {seed}/{key}')
        print('PASS split',seed)
    print('PASS_SPLIT_RECONSTRUCTION')
if __name__=='__main__':main()
