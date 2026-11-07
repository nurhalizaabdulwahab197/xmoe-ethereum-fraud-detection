"""Prespecified parity matrix; never infer scientific completion from file existence."""
MODELS = ['mlp', 'gcn', 'gat', 'graphsage', 'x_moe_gnn']
VARIANTS = ['a0_full', 'a1_conflict_only', 'a2_graphsage_only', 'a3_global_routing',
            'a4_no_isolate_mask', 'a5_no_scam_specialist', 'a6_no_normal_phishing_penalty', 'a7_fixed_50']
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 1.0]


def matrix(seeds=range(100, 110), gpu=False):
    rows = []
    for seed in seeds:
        for family in ['clean', 'strict']:
            for model in MODELS:
                rows.append(dict(family=family, model=model, variant='a0_full', coefficient=.3, seed=seed, device='cpu'))
        for variant in VARIANTS:
            rows.append(dict(family='ablation', model='x_moe_gnn', variant=variant, coefficient=.3, seed=seed, device='cpu'))
        for value in LAMBDAS:
            rows.append(dict(family='sensitivity', model='x_moe_gnn', variant='a0_full', coefficient=value, seed=seed, device='cpu'))
        if gpu:
            for model in MODELS[1:]:  # Original BLTE CUDA evidence has four GNNs, no MLP.
                rows.append(dict(family='clean', model=model, variant='a0_full', coefficient=.3, seed=seed, device='cuda'))
    return rows


def run_id(row):
    return f"{row['family']}/{row['device']}/{row['model']}/{row['variant']}/lambda_{row['coefficient']:.2f}/seed{row['seed']}"
