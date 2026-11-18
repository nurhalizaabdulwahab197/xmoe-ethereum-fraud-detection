import argparse
from common import environment

def main():
    argparse.ArgumentParser(description='Check recorded principal versions and GPU forward/backward; no optimizer step.').parse_args()
    import torch
    print(environment())
    from torch_geometric.nn import GCNConv,GATConv,SAGEConv
    for name,cls in [('GCN',GCNConv),('GAT',GATConv),('GraphSAGE',SAGEConv)]:
        x=torch.randn(6,23,device='cuda',requires_grad=True)
        edge=torch.tensor([[0,1,2,3,4,5],[1,2,3,4,5,0]],device='cuda')
        layer=(cls(23,64,heads=8) if name=='GAT' else cls(23,64)).cuda()
        layer(x,edge).square().mean().backward()
        torch.cuda.synchronize()
        print('PASS:',name,'forward/backward')
    print('PASS_ENVIRONMENT; no optimizer step or experiment training')
if __name__=='__main__': main()
