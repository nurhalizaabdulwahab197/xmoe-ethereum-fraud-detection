"""Optional activation recomputation; preserve GAT parameters and state keys."""
import torch
from torch.utils.checkpoint import checkpoint


def enable_gat_activation_checkpointing(model):
    from models import GAT
    if not isinstance(model, GAT):
        raise TypeError('This option is restricted to the frozen GAT baseline')
    if getattr(model, '_rq1_activation_checkpointing', False):
        raise ValueError('Activation checkpointing is already enabled')
    before = [(name, id(parameter)) for name, parameter in model.named_parameters()]

    def wrapped(layer):
        original_forward = layer.forward
        def forward(*args, **kwargs):
            if layer.training and torch.is_grad_enabled():
                return checkpoint(original_forward, *args, use_reentrant=False,
                                  preserve_rng_state=True, **kwargs)
            return original_forward(*args, **kwargs)
        return forward

    model.gat1.forward = wrapped(model.gat1)
    model.gat2.forward = wrapped(model.gat2)
    model._rq1_activation_checkpointing = True
    if before != [(name, id(parameter)) for name, parameter in model.named_parameters()]:
        raise RuntimeError('GAT parameter identities or state names changed')
    return model
