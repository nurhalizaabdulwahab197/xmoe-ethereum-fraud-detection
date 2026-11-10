"""Explicit BLTE-equivalent restricted topology, including RQ1 unknown nodes."""
import numpy as np
import torch


def make_views(x, edges, labels, masks):
    train, val, test = masks
    membership = torch.full((len(labels),), -1, dtype=torch.int8)
    total = sum(mask.to(torch.int8) for mask in masks)
    if (total > 1).any() or not torch.equal(total.bool(), labels >= 0):
        raise ValueError('Disjoint supervised masks must exactly cover labeled accounts.')
    for index, mask in enumerate(masks):
        membership[mask] = index
    src, dst = membership[edges[0]], membership[edges[1]]
    tt = (src == 0) & (dst == 0)
    allowed = [tt, tt | ((src == 0) & (dst == 1)) | ((src == 1) & (dst == 0)),
               tt | ((src == 0) & (dst == 2)) | ((src == 2) & (dst == 0))]
    features, topology = [], []
    audits = {}
    for name, keep_nodes, keep_edges in zip(('train', 'validation', 'test'),
                                            (train, train | val, train | test), allowed):
        values = x.clone()
        values[~keep_nodes] = 0
        edge = edges[:, keep_edges].contiguous()
        # These assertions refer to the actual tensors used by model.forward.
        if torch.count_nonzero(values[~keep_nodes]) or not keep_nodes[edge].all():
            raise ValueError('A forbidden node is visible in a strict forward view.')
        if name != 'train':
            query = val if name == 'validation' else test
            if (query[edge[0]] & query[edge[1]]).any():
                raise ValueError('Query-query edges are forbidden.')
        audits[name] = {'visible_feature_nodes': int(keep_nodes.sum()), 'edges': edge.shape[1],
                       'forbidden_feature_values': 0, 'forbidden_edges': 0,
                       'unlabeled_context_edges': int(((labels[edge[0]] < 0) | (labels[edge[1]] < 0)).sum())}
        features.append(values);topology.append(edge)
    return features, topology, audits


def masks_for(labels, seed):
    from sklearn.model_selection import train_test_split
    ids = np.flatnonzero(labels >= 0)
    train, other = train_test_split(ids, test_size=.30, random_state=seed, stratify=labels[ids])
    val, test = train_test_split(other, test_size=.50, random_state=seed, stratify=labels[other])
    masks = []
    for indices in (train, val, test):
        mask = np.zeros(len(labels), dtype=bool);mask[indices] = True;masks.append(mask)
    return masks
