"""BLTE ablation architectures, preserved for the RQ1 parity extension.

Extracted without changing variant implementations; see source_provenance.json.
"""
import torch
import torch.nn.functional as F
from torch import nn
from models.graphsage import GraphSAGE
from models.xmoe import ConflictAwareMultitaskExpert, ExplainableMixtureOfExpertsGNN

VARIANTS = {
    "a0_full": "Full X-MoE",
    "a1_conflict_only": "Conflict-Aware only",
    "a2_graphsage_only": "GraphSAGE-expert configuration",
    "a3_global_routing": "Shared routing",
    "a4_no_isolate_mask": "No isolate masking",
    "a5_no_scam_specialist": "No Scamming specialist",
    "a6_no_normal_phishing_penalty": "No Normal-Phishing objective",
    "a7_fixed_50": "Fixed 50:50 outer mixture",
}


def node_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return torch.bincount(edge_index.reshape(-1), minlength=num_nodes).float()


class GlobalRoutingXMoE(ExplainableMixtureOfExpertsGNN):
    """Use shared inner and outer routing while preserving A0 modules."""

    def _global_conflict_outputs(self, x: torch.Tensor, edge_index: torch.Tensor):
        expert = self.conflict_aware_expert
        feature_h = F.relu(expert.feature_fc1(x))
        feature_h = F.dropout(feature_h, p=0.5, training=self.training)
        feature_h = expert.feature_norm(F.relu(expert.feature_fc2(feature_h)))
        graph_h = F.relu(expert.graph_conv1(x, edge_index))
        graph_h = F.dropout(graph_h, p=0.5, training=self.training)
        graph_h = expert.graph_norm(F.relu(expert.graph_conv2(graph_h, edge_index)))
        degree = expert.node_degree(edge_index, x.shape[0])
        connected = (degree > 0).to(dtype=x.dtype).unsqueeze(1)
        gate_input = torch.cat([feature_h, graph_h, torch.log1p(degree).unsqueeze(1)], dim=1)
        gate_hidden = F.relu(expert.gate_fc1(gate_input))
        gate_logits = expert.gate_fc2(gate_hidden)
        scalar_gate = torch.sigmoid(gate_logits.mean(dim=1, keepdim=True)) * connected
        class_gate = scalar_gate.expand(-1, gate_logits.shape[1])
        feature_logits = expert.feature_heads(feature_h)
        graph_logits = expert.graph_heads(graph_h)
        logits = feature_logits + class_gate * (graph_logits - feature_logits)
        expert._last_class_gate = class_gate.detach()
        expert._last_degree = degree.detach()
        return logits, class_gate, degree

    def mix_experts(self, x: torch.Tensor, edge_index: torch.Tensor):
        conflict_aware_logits, _, degree = self._global_conflict_outputs(x, edge_index)
        graphsage_logits = self.graphsage_expert(x, edge_index)
        connected = (degree > 0).to(dtype=x.dtype).unsqueeze(1)
        gate_input = torch.cat([
            conflict_aware_logits,
            graphsage_logits,
            (conflict_aware_logits - graphsage_logits).abs(),
            torch.log1p(degree).unsqueeze(1),
        ], dim=1)
        gate_hidden = F.relu(self.expert_gate_fc1(gate_input))
        outer_gate_logits = self.expert_gate_fc2(gate_hidden)
        scalar_weight = torch.sigmoid(outer_gate_logits.mean(dim=1, keepdim=True)) * connected
        graphsage_weight = scalar_weight.expand(-1, conflict_aware_logits.shape[1])
        logits = conflict_aware_logits + graphsage_weight * (graphsage_logits - conflict_aware_logits)
        return logits, conflict_aware_logits, graphsage_logits, graphsage_weight, degree


class NoIsolateMaskXMoE(ExplainableMixtureOfExpertsGNN):
    """Remove isolate masking while preserving the A0 modules."""

    def _no_mask_conflict_outputs(self, x: torch.Tensor, edge_index: torch.Tensor):
        expert = self.conflict_aware_expert
        feature_h = F.relu(expert.feature_fc1(x))
        feature_h = F.dropout(feature_h, p=0.5, training=self.training)
        feature_h = expert.feature_norm(F.relu(expert.feature_fc2(feature_h)))
        graph_h = F.relu(expert.graph_conv1(x, edge_index))
        graph_h = F.dropout(graph_h, p=0.5, training=self.training)
        graph_h = expert.graph_norm(F.relu(expert.graph_conv2(graph_h, edge_index)))
        degree = expert.node_degree(edge_index, x.shape[0])
        gate_input = torch.cat([feature_h, graph_h, torch.log1p(degree).unsqueeze(1)], dim=1)
        class_gate = torch.sigmoid(expert.gate_fc2(F.relu(expert.gate_fc1(gate_input))))
        feature_logits = expert.feature_heads(feature_h)
        graph_logits = expert.graph_heads(graph_h)
        logits = feature_logits + class_gate * (graph_logits - feature_logits)
        expert._last_class_gate = class_gate.detach()
        expert._last_degree = degree.detach()
        return logits, class_gate, degree

    def mix_experts(self, x: torch.Tensor, edge_index: torch.Tensor):
        conflict_aware_logits, _, degree = self._no_mask_conflict_outputs(x, edge_index)
        graphsage_logits = self.graphsage_expert(x, edge_index)
        gate_input = torch.cat([
            conflict_aware_logits,
            graphsage_logits,
            (conflict_aware_logits - graphsage_logits).abs(),
            torch.log1p(degree).unsqueeze(1),
        ], dim=1)
        graphsage_weight = torch.sigmoid(self.expert_gate_fc2(F.relu(self.expert_gate_fc1(gate_input))))
        logits = conflict_aware_logits + graphsage_weight * (graphsage_logits - conflict_aware_logits)
        return logits, conflict_aware_logits, graphsage_logits, graphsage_weight, degree


class FixedFiftyXMoE(ExplainableMixtureOfExpertsGNN):
    """Use fixed 50:50 outer mixing on connected nodes."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(input_dim, output_dim)
        del self.expert_gate_fc1
        del self.expert_gate_fc2

    def mix_experts(self, x: torch.Tensor, edge_index: torch.Tensor):
        conflict_aware_logits = self.conflict_aware_expert(x, edge_index)
        graphsage_logits = self.graphsage_expert(x, edge_index)
        degree = self.conflict_aware_expert.node_degree(edge_index, x.shape[0])
        connected = (degree > 0).to(dtype=x.dtype).unsqueeze(1)
        graphsage_weight = 0.5 * connected.expand(-1, conflict_aware_logits.shape[1])
        logits = conflict_aware_logits + graphsage_weight * (graphsage_logits - conflict_aware_logits)
        return logits, conflict_aware_logits, graphsage_logits, graphsage_weight, degree


def build_model(variant: str, input_dim: int, output_dim: int) -> nn.Module:
    if variant in {"a0_full", "a5_no_scam_specialist", "a6_no_normal_phishing_penalty"}:
        return ExplainableMixtureOfExpertsGNN(input_dim, output_dim)
    if variant == "a1_conflict_only":
        return ConflictAwareMultitaskExpert(input_dim, output_dim)
    if variant == "a2_graphsage_only":
        return GraphSAGE(input_dim, output_dim)
    if variant == "a3_global_routing":
        return GlobalRoutingXMoE(input_dim, output_dim)
    if variant == "a4_no_isolate_mask":
        return NoIsolateMaskXMoE(input_dim, output_dim)
    if variant == "a7_fixed_50":
        return FixedFiftyXMoE(input_dim, output_dim)
    raise ValueError(f"Unknown ablation variant: {variant}")


def get_training_outputs(model: nn.Module, variant: str, x: torch.Tensor, edge_index: torch.Tensor):
    if variant == "a1_conflict_only":
        conflict_logits, _ = model.training_outputs(x, edge_index)
        return conflict_logits, conflict_logits, None
    if variant == "a2_graphsage_only":
        graphsage_logits = model(x, edge_index)
        return graphsage_logits, None, graphsage_logits
    logits, conflict_logits, graphsage_logits = model.training_outputs(x, edge_index)
    return logits, conflict_logits, graphsage_logits

