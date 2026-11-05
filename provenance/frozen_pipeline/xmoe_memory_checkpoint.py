"""Experimental execution-only memory helper. Not enabled in the final runner."""
import torch
from torch.utils.checkpoint import checkpoint


def enable_xmoe_memory_checkpoint(model):
    from models import ExplainableMixtureOfExpertsGNN
    if not isinstance(model, ExplainableMixtureOfExpertsGNN):
        raise TypeError('Only the frozen X-MoE model is supported')
    if getattr(model, '_rq1_expert_checkpoint', False):
        raise ValueError('Already enabled')
    before = [(name, id(p)) for name, p in model.named_parameters()]

    def wrap(expert):
        original = expert.forward

        def forward(*args, **kwargs):
            if expert.training and torch.is_grad_enabled():
                return checkpoint(original, *args, use_reentrant=False,
                                  preserve_rng_state=True, **kwargs)
            return original(*args, **kwargs)
        return forward

    for expert in (model.conflict_aware_expert, model.graphsage_expert):
        expert.forward = wrap(expert)
    model._rq1_expert_checkpoint = True
    if before != [(name, id(p)) for name, p in model.named_parameters()]:
        raise RuntimeError('Model parameter identities changed')
    return model
