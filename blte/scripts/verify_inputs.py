import argparse
from pathlib import Path
from common import input_check,source_check,splits,require

def main():
    ap=argparse.ArgumentParser(description='Read-only input, alignment, source and saved-partition checks.')
    ap.add_argument('--upstream',type=Path,required=True); a=ap.parse_args()
    root=input_check(a.upstream)
    import numpy as np
    import pandas as pd
    import torch
    g=torch.load(root/'BLTE_multiclass_graph.pt',map_location='cpu',weights_only=False)
    d=pd.read_csv(root/'BLTE_multiclass_v2.csv')
    require(tuple(g.x.shape)==(73033,23),'Feature shape')
    require(tuple(g.edge_index.shape)==(2,6588),'Edge shape')
    require(bool(torch.isfinite(g.x).all()),'Non-finite features')
    require(np.array_equal(d['attack_encoded'].to_numpy(),g.y.numpy()),'CSV/graph label order')
    counts=torch.zeros(73033,dtype=torch.long)
    for name in ['train_mask','val_mask','test_mask']:
        mask=getattr(g,name); require(mask.dtype==torch.bool,'Mask dtype'); counts+=mask.long()
    require(bool((counts==1).all()),'Released masks incomplete or overlapping')
    print('PASS_INPUTS; sources:',source_check(),'; saved partitions:',len(splits(g)))
if __name__=='__main__': main()
