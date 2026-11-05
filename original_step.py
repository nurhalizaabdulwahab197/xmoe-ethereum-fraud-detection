"""AST-extracted original optimizer step; no dataset or metric evaluation."""
import torch
import torch.nn.functional as F
from models import ExplainableMixtureOfExpertsGNN as ModularXMoE

def weighted_focal_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor, gamma: float) -> torch.Tensor:
    """Weighted focal loss with CE-equivalent normalization at gamma zero."""
    per_item_ce = F.cross_entropy(logits, labels, weight=class_weights, reduction='none')
    true_probability = F.softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).squeeze(1)
    denominator = class_weights[labels].sum().clamp_min(torch.finfo(logits.dtype).eps)
    return ((1.0 - true_probability) ** gamma * per_item_ce).sum() / denominator

def balanced_binary_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency weighted BCE normalized by the observed weights."""
    labels = labels.to(dtype=logits.dtype)
    counts = torch.bincount(labels.long(), minlength=2).to(dtype=logits.dtype)
    if (counts == 0).any():
        raise ValueError('balanced_binary_loss requires both binary classes')
    class_weights = counts.sum() / counts
    class_weights = class_weights / class_weights.sum()
    sample_weights = class_weights[labels.long()]
    per_item = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels, reduction='none')
    return (per_item * sample_weights).sum() / sample_weights.sum()

def project_conflicting_gradients(task_gradients: list[list[torch.Tensor]]) -> list[list[torch.Tensor]]:
    """Deterministic PCGrad projection in fixed task order."""
    projected = [[gradient.clone() for gradient in task] for task in task_gradients]
    for task_index, task in enumerate(projected):
        for other_index, other in enumerate(task_gradients):
            if task_index == other_index:
                continue
            dot = sum(((left * right).sum() for left, right in zip(task, other)))
            norm = sum(((gradient * gradient).sum() for gradient in other))
            if float(dot.detach()) < 0 and float(norm.detach()) > 0:
                scale = dot / norm
                task = [left - scale * right for left, right in zip(task, other)]
                projected[task_index] = task
    return projected

def pcgrad_backward(losses: list[torch.Tensor], parameters: list[torch.nn.Parameter]) -> None:
    """Populate parameter gradients with mean deterministic PCGrad gradients."""
    task_gradients: list[list[torch.Tensor]] = []
    for index, loss in enumerate(losses):
        raw = torch.autograd.grad(loss, parameters, retain_graph=index < len(losses) - 1, allow_unused=True)
        task_gradients.append([gradient if gradient is not None else torch.zeros_like(parameter) for gradient, parameter in zip(raw, parameters)])
    projected = project_conflicting_gradients(task_gradients)
    for parameter_index, parameter in enumerate(parameters):
        parameter.grad = torch.stack([task[parameter_index] for task in projected]).mean(dim=0)

def original_training_step(model, model_name, x, edge_index, y, train_mask, optimizer, class_weights,
    base_loss="weighted_ce", focal_gamma=2.0, expert_loss_weight=0.25,
    scam_specialist_loss_weight=0.5, x_moe_expert_task_loss_weight=1.0,
    x_moe_normal_phishing_penalty=0.30):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x_moe_expert_logits = None
    if isinstance(model, ModularXMoE):
        logits, conflict_aware_logits, graphsage_logits = model.training_outputs(x, edge_index)
        x_moe_expert_logits = (conflict_aware_logits, graphsage_logits)
    else:
        logits = model(x, edge_index)
    if base_loss == 'weighted_focal':
        loss = weighted_focal_loss(logits[train_mask], y[train_mask], class_weights, focal_gamma)
    else:
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=class_weights)
    if model_name == 'x_moe_gnn' and x_moe_expert_logits is not None:
        conflict_aware_logits, graphsage_logits = x_moe_expert_logits
        training_labels = y[train_mask]
        task_losses = [loss]
        task_losses.extend([expert_loss_weight * F.cross_entropy(expert_logits[train_mask], training_labels, weight=class_weights) for expert_logits in (conflict_aware_logits, graphsage_logits)])
        for class_id in range(3):
            binary_labels = (training_labels == class_id).to(dtype=conflict_aware_logits.dtype)
            task_losses.append(x_moe_expert_task_loss_weight * balanced_binary_loss(conflict_aware_logits[train_mask, class_id:class_id + 1], binary_labels))
        scamming_labels = (training_labels == 2).to(dtype=graphsage_logits.dtype)
        task_losses.append(scam_specialist_loss_weight * balanced_binary_loss(graphsage_logits[train_mask, 2:3], scamming_labels))
        if x_moe_normal_phishing_penalty > 0:
            degree = model.conflict_aware_expert.node_degree(edge_index, x.shape[0])
            connected_normal = train_mask & (y == 0) & (degree > 0)
            normal_phishing_probability = F.softmax(logits[connected_normal], dim=1)[:, 1].mean()
            task_losses.append(x_moe_normal_phishing_penalty * normal_phishing_probability)
        pcgrad_backward(task_losses, [parameter for parameter in model.parameters()])
    else:
        loss.backward()
    optimizer.step()
    return {"base_loss": float(loss.detach().cpu()), "task_count": len(task_losses) if model_name == "x_moe_gnn" else 1}
