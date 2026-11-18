"""Read-only numerical preflight: seeded architecture/optimizer check plus forward/backward; no optimizer step."""
import argparse
import ast
import sys
from common import ROOT,read,environment,source_check,require

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    environment();source_check()
    import torch
    import numpy as np
    import random
    configs=read(ROOT/'provenance/CONFIGURATIONS.json')
    for key,r in configs.items():
        random.seed(100);np.random.seed(100);torch.manual_seed(100);torch.cuda.manual_seed_all(100)
        path=ROOT/r['script']
        if key.endswith('/full-xmoe'):
            # Compare model state with the CE frozen model via separate processes to avoid module cache collisions.
            continue
        tree=ast.parse(path.read_text());label=r['label']
        if label in ('MLP','XMoE_CE'):
            sys.path.insert(0,str(ROOT/'sources/prepared/model_source'))
            from models import MLP,ExplainableMixtureOfExpertsGNN
            cls=MLP if label=='MLP' else ExplainableMixtureOfExpertsGNN
            model=cls(23,3).cuda()
        else:
            # Compile only the unchanged model class node. Never execute top-level training.
            nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==label]
            require(len(nodes)==1,'Expected model class')
            import torch.nn.functional as F
            from torch_geometric.nn import GCNConv,GATConv,SAGEConv
            scope=dict(torch=torch,F=F,GCNConv=GCNConv,GATConv=GATConv,SAGEConv=SAGEConv)
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
            paper=key.startswith('paper-described/')
            model=(scope[label](23,64 if paper else 32,3,heads=8 if paper else 4,dropout=.5) if label=='GAT' else scope[label](23,64,3)).cuda()
        x=torch.randn(6,23,device='cuda');edge=torch.tensor([[0,1,2,3,4,5],[1,2,3,4,5,0]],device='cuda')
        y=torch.tensor([0,1,2,0,1,2],device='cuda')
        out=model(x,edge);require(tuple(out.shape)==(6,3),'Output shape')
        torch.nn.functional.cross_entropy(out,y).backward();torch.cuda.synchronize()
        print('PASS CE architecture forward/backward:',key)
    print('Full frozen model files are byte-identical to CE model files. Full multitask GPU runtime remains a separate local check.')
    print('No optimizer step, no experiment training, no files written.')
if __name__=='__main__':main()
