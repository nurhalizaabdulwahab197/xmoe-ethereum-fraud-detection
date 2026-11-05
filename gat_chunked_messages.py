"""First-order CPU GAT weighted-sum aggregation with bounded edge temporaries.

Attention projection, softmax, dropout, self loops and all parameters remain in
the original GATConv. Only sum_j alpha[j,h] * x[j,h,c] is executed in chunks.
"""
import torch
from torch.autograd.function import once_differentiable


class ChunkedWeightedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, edge_index, alpha, num_target_nodes, chunk_size):
        if x.device.type != 'cpu' or edge_index.device != x.device or edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError('Dense CPU edge indices and CPU features required')
        if x.ndim != 3 or alpha.shape != (edge_index.shape[1], x.shape[1]) or alpha.dtype != x.dtype or alpha.device != x.device:
            raise ValueError('Unexpected GAT message/attention shape or dtype')
        if chunk_size < 1:
            raise ValueError('Positive edge chunk size required')
        ctx.chunk_size = chunk_size
        ctx.save_for_backward(x, edge_index, alpha)
        output = x.new_zeros((num_target_nodes, x.shape[1], x.shape[2]))
        for start in range(0, edge_index.shape[1], chunk_size):
            stop = min(start + chunk_size, edge_index.shape[1])
            source, target = edge_index[:, start:stop]
            output.index_add_(0, target, x[source] * alpha[start:stop].unsqueeze(-1))
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, edge_index, alpha = ctx.saved_tensors
        grad_x = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        grad_alpha = torch.empty_like(alpha) if ctx.needs_input_grad[2] else None
        for start in range(0, edge_index.shape[1], ctx.chunk_size):
            stop = min(start + ctx.chunk_size, edge_index.shape[1])
            source, target = edge_index[:, start:stop]
            gathered = grad_output[target]
            if grad_x is not None:
                grad_x.index_add_(0, source, gathered * alpha[start:stop].unsqueeze(-1))
            if grad_alpha is not None:
                grad_alpha[start:stop] = (gathered * x[source]).sum(dim=-1)
        return grad_x, None, grad_alpha, None, None


def enable_gat_chunked_messages(model, chunk_size=32768):
    from models import GAT
    if not isinstance(model, GAT) or chunk_size < 1:
        raise ValueError('Frozen GAT and a positive chunk size required')
    if getattr(model, '_rq1_gat_edge_chunk_size', 0):
        raise ValueError('Chunked messages already enabled')
    identities = [(name, id(parameter)) for name, parameter in model.named_parameters()]

    def wrapped(layer):
        def propagate(edge_index, x, alpha, size=None):
            if layer.explain or getattr(layer, '_edge_mask', None) is not None:
                raise ValueError('Edge-mask explanations require reloading the original GAT implementation')
            if not isinstance(x, tuple) or x[0] is None:
                raise ValueError('Expected original GAT projected source/target pair')
            count = x[1].shape[0] if x[1] is not None else size[1] if size and size[1] is not None else x[0].shape[0]
            return ChunkedWeightedSum.apply(x[0], edge_index, alpha, count, chunk_size)
        return propagate

    for layer in (model.gat1, model.gat2):
        if layer.flow != 'source_to_target' or layer.aggr not in ('add', 'sum'):
            raise ValueError('Only original source-to-target sum aggregation is supported')
        layer.propagate = wrapped(layer)
    model._rq1_gat_edge_chunk_size = chunk_size
    if identities != [(name, id(parameter)) for name, parameter in model.named_parameters()]:
        raise RuntimeError('GAT parameter names or identities changed')
    return model
