import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from .graphsage import GraphSAGE

class _IsolateAwareExpertBase(nn.Module):
    """Hybrid feature/graph encoder with an interpretable routing gate.

    The scalar gate is constrained to zero for isolated nodes. For connected
    nodes it learns a value in [0, 1], where zero routes the representation
    through the feature branch and one routes it through the GraphSAGE branch.
    It is a routing explanation, not a causal attribution by itself.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.feature_fc1 = nn.Linear(input_dim, 64)
        self.feature_fc2 = nn.Linear(64, 64)
        self.graph_conv1 = SAGEConv(input_dim, 64)
        self.graph_conv2 = SAGEConv(64, 64)
        self.feature_norm = nn.LayerNorm(64)
        self.graph_norm = nn.LayerNorm(64)
        self.gate_fc1 = nn.Linear(64 + 64 + 1, 32)
        self.gate_fc2 = nn.Linear(32, 1)
        self.classifier = nn.Linear(64, output_dim)
        self._last_gate: torch.Tensor | None = None
        self._last_degree: torch.Tensor | None = None

    @staticmethod
    def node_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        return torch.bincount(edge_index.reshape(-1), minlength=num_nodes).float()

    def encode(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        feature_h = F.relu(self.feature_fc1(x))
        feature_h = F.dropout(feature_h, p=0.5, training=self.training)
        feature_h = self.feature_norm(F.relu(self.feature_fc2(feature_h)))

        graph_h = F.relu(self.graph_conv1(x, edge_index))
        graph_h = F.dropout(graph_h, p=0.5, training=self.training)
        graph_h = self.graph_norm(F.relu(self.graph_conv2(graph_h, edge_index)))

        degree = self.node_degree(edge_index, x.shape[0])
        connected = (degree > 0).float().unsqueeze(1)
        degree_feature = torch.log1p(degree).unsqueeze(1)
        gate_input = torch.cat([feature_h, graph_h, degree_feature], dim=1)
        gate = torch.sigmoid(self.gate_fc2(F.relu(self.gate_fc1(gate_input))))
        gate = gate * connected
        fused = gate * graph_h + (1.0 - gate) * feature_h
        return fused, feature_h, graph_h, gate, degree, connected.squeeze(1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        fused, _, _, gate, degree, _ = self.encode(x, edge_index)
        self._last_gate = gate.detach().squeeze(1)
        self._last_degree = degree.detach()
        fused = F.dropout(fused, p=0.5, training=self.training)
        return self.classifier(fused)

    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return intrinsic routing and feature-only counterfactual evidence."""
        fused, feature_h, _, gate, degree, _ = self.encode(x, edge_index)
        full_probabilities = F.softmax(self.classifier(fused), dim=1)
        feature_probabilities = F.softmax(self.classifier(feature_h), dim=1)
        predicted = full_probabilities.argmax(dim=1)
        predicted_probability = full_probabilities.gather(
            1, predicted.unsqueeze(1)
        ).squeeze(1)
        feature_counterfactual_probability = feature_probabilities.gather(
            1, predicted.unsqueeze(1)
        ).squeeze(1)
        return {
            "probabilities": full_probabilities,
            "feature_probabilities": feature_probabilities,
            "prediction": predicted,
            "feature_prediction": feature_probabilities.argmax(dim=1),
            "graph_gate": gate.squeeze(1),
            "degree": degree,
            "graph_probability_delta": (
                predicted_probability - feature_counterfactual_probability
            ),
        }

    def gate_diagnostics(self) -> dict[str, float]:
        if self._last_gate is None or self._last_degree is None:
            raise RuntimeError("Run a forward pass before requesting gate diagnostics")
        connected = self._last_degree > 0
        isolated = ~connected
        return {
            "mean_graph_gate_all": float(self._last_gate.mean().cpu()),
            "mean_graph_gate_connected": float(
                self._last_gate[connected].mean().cpu()
            ),
            "mean_graph_gate_isolated": float(self._last_gate[isolated].mean().cpu()),
            "connected_fraction": float(connected.float().mean().cpu()),
        }



class ConflictAwareMultiTaskExplainableGNNV8(_IsolateAwareExpertBase):
    """Three one-vs-rest heads with class-specific isolate-safe graph gates."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        if output_dim != 3:
            raise ValueError("V8 requires exactly three output classes")
        super().__init__(input_dim, output_dim)
        del self.classifier
        self.gate_fc2 = nn.Linear(32, output_dim)
        self.feature_heads = nn.Linear(64, output_dim)
        self.graph_heads = nn.Linear(64, output_dim)
        self._last_class_gate: torch.Tensor | None = None

    def encode_multitask(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_h = F.relu(self.feature_fc1(x))
        feature_h = F.dropout(feature_h, p=0.5, training=self.training)
        feature_h = self.feature_norm(F.relu(self.feature_fc2(feature_h)))
        graph_h = F.relu(self.graph_conv1(x, edge_index))
        graph_h = F.dropout(graph_h, p=0.5, training=self.training)
        graph_h = self.graph_norm(F.relu(self.graph_conv2(graph_h, edge_index)))
        degree = self.node_degree(edge_index, x.shape[0])
        connected = (degree > 0).float().unsqueeze(1)
        gate_input = torch.cat(
            [feature_h, graph_h, torch.log1p(degree).unsqueeze(1)], dim=1
        )
        class_gate = torch.sigmoid(
            self.gate_fc2(F.relu(self.gate_fc1(gate_input)))
        ) * connected
        feature_logits = self.feature_heads(feature_h)
        graph_logits = self.graph_heads(graph_h)
        logits = feature_logits + class_gate * (graph_logits - feature_logits)
        return logits, feature_logits, class_gate, degree

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        logits, _, class_gate, degree = self.encode_multitask(x, edge_index)
        self._last_class_gate = class_gate.detach()
        self._last_degree = degree.detach()
        return logits

    def training_outputs(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, _, class_gate, degree = self.encode_multitask(x, edge_index)
        self._last_class_gate = class_gate.detach()
        self._last_degree = degree.detach()
        return logits, logits

    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        logits, feature_logits, class_gate, degree = self.encode_multitask(
            x, edge_index
        )
        probabilities = F.softmax(logits, dim=1)
        feature_probabilities = F.softmax(feature_logits, dim=1)
        predicted = probabilities.argmax(dim=1)
        selected_gate = class_gate.gather(1, predicted.unsqueeze(1)).squeeze(1)
        predicted_probability = probabilities.gather(
            1, predicted.unsqueeze(1)
        ).squeeze(1)
        feature_probability = feature_probabilities.gather(
            1, predicted.unsqueeze(1)
        ).squeeze(1)
        self._last_class_gate = class_gate.detach()
        self._last_degree = degree.detach()
        return {
            "probabilities": probabilities,
            "feature_probabilities": feature_probabilities,
            "prediction": predicted,
            "feature_prediction": feature_probabilities.argmax(dim=1),
            "graph_gate": selected_gate,
            "normal_graph_gate": class_gate[:, 0],
            "phishing_graph_gate": class_gate[:, 1],
            "scamming_graph_gate": class_gate[:, 2],
            "normal_head_logit": logits[:, 0],
            "phishing_head_logit": logits[:, 1],
            "scamming_head_logit": logits[:, 2],
            "degree": degree,
            "graph_probability_delta": predicted_probability - feature_probability,
        }

    def gate_diagnostics(self) -> dict[str, float]:
        if self._last_class_gate is None or self._last_degree is None:
            raise RuntimeError("Run a forward pass before requesting gate diagnostics")
        connected = self._last_degree > 0
        isolated = ~connected
        connected_gates = self._last_class_gate[connected]
        return {
            "mean_graph_gate_all": float(self._last_class_gate.mean().cpu()),
            "mean_graph_gate_connected": float(connected_gates.mean().cpu()),
            "mean_graph_gate_isolated": float(
                self._last_class_gate[isolated].mean().cpu()
            ),
            "mean_normal_graph_gate_connected": float(
                connected_gates[:, 0].mean().cpu()
            ),
            "mean_phishing_graph_gate_connected": float(
                connected_gates[:, 1].mean().cpu()
            ),
            "mean_scamming_graph_gate_connected": float(
                connected_gates[:, 2].mean().cpu()
            ),
            "connected_fraction": float(connected.float().mean().cpu()),
        }



class ExplainableMixtureOfExpertsGNN(nn.Module):
    """Class-conditional mixture of V8 and GraphSAGE experts.

    V8 supplies the safe default. A second GraphSAGE expert is routed in only
    for connected nodes. The three routing weights are directly inspectable,
    with one value per output class. This is an architecture for external
    validation, not evidence that it outperforms either expert on BLTE.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        if output_dim != 3:
            raise ValueError("X-MoE requires exactly three output classes")
        nn.Module.__init__(self)
        self.v8_expert = ConflictAwareMultiTaskExplainableGNNV8(
            input_dim, output_dim
        )
        self.graphsage_expert = GraphSAGE(input_dim, output_dim)
        # V8 logits, GraphSAGE logits, absolute disagreement, and node degree.
        self.expert_gate_fc1 = nn.Linear(output_dim * 3 + 1, 32)
        self.expert_gate_fc2 = nn.Linear(32, output_dim)
        self._last_expert_gate: torch.Tensor | None = None
        self._last_degree: torch.Tensor | None = None

    def mix_experts(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v8_logits = self.v8_expert(x, edge_index)
        graphsage_logits = self.graphsage_expert(x, edge_index)
        degree = self.v8_expert.node_degree(edge_index, x.shape[0])
        connected = (degree > 0).to(dtype=x.dtype).unsqueeze(1)
        gate_input = torch.cat(
            [
                v8_logits,
                graphsage_logits,
                (v8_logits - graphsage_logits).abs(),
                torch.log1p(degree).unsqueeze(1),
            ],
            dim=1,
        )
        graphsage_weight = torch.sigmoid(
            self.expert_gate_fc2(F.relu(self.expert_gate_fc1(gate_input)))
        ) * connected
        logits = v8_logits + graphsage_weight * (graphsage_logits - v8_logits)
        return logits, v8_logits, graphsage_logits, graphsage_weight, degree

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        logits, _, _, expert_gate, degree = self.mix_experts(x, edge_index)
        self._last_expert_gate = expert_gate.detach()
        self._last_degree = degree.detach()
        return logits

    def training_outputs(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, v8_logits, graphsage_logits, expert_gate, degree = self.mix_experts(
            x, edge_index
        )
        self._last_expert_gate = expert_gate.detach()
        self._last_degree = degree.detach()
        return logits, v8_logits, graphsage_logits

    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        logits, v8_logits, graphsage_logits, expert_gate, degree = self.mix_experts(
            x, edge_index
        )
        v8_explanation = self.v8_expert.explain(x, edge_index)
        probabilities = F.softmax(logits, dim=1)
        v8_probabilities = F.softmax(v8_logits, dim=1)
        graphsage_probabilities = F.softmax(graphsage_logits, dim=1)
        predicted = probabilities.argmax(dim=1)
        selected_expert_gate = expert_gate.gather(
            1, predicted.unsqueeze(1)
        ).squeeze(1)
        predicted_probability = probabilities.gather(
            1, predicted.unsqueeze(1)
        ).squeeze(1)
        v8_probability = v8_probabilities.gather(
            1, predicted.unsqueeze(1)
        ).squeeze(1)
        feature_probabilities = v8_explanation["feature_probabilities"]
        feature_probability = feature_probabilities.gather(
            1, predicted.unsqueeze(1)
        ).squeeze(1)
        self._last_expert_gate = expert_gate.detach()
        self._last_degree = degree.detach()
        return {
            "probabilities": probabilities,
            "feature_probabilities": feature_probabilities,
            "v8_probabilities": v8_probabilities,
            "graphsage_probabilities": graphsage_probabilities,
            "prediction": predicted,
            "feature_prediction": feature_probabilities.argmax(dim=1),
            "v8_prediction": v8_probabilities.argmax(dim=1),
            "graphsage_prediction": graphsage_probabilities.argmax(dim=1),
            "graph_gate": selected_expert_gate,
            "normal_graphsage_expert_weight": expert_gate[:, 0],
            "phishing_graphsage_expert_weight": expert_gate[:, 1],
            "scamming_graphsage_expert_weight": expert_gate[:, 2],
            "normal_v8_expert_weight": 1.0 - expert_gate[:, 0],
            "phishing_v8_expert_weight": 1.0 - expert_gate[:, 1],
            "scamming_v8_expert_weight": 1.0 - expert_gate[:, 2],
            "degree": degree,
            "graph_probability_delta": predicted_probability - feature_probability,
            "expert_probability_delta": predicted_probability - v8_probability,
        }

    def gate_diagnostics(self) -> dict[str, float]:
        if self._last_expert_gate is None or self._last_degree is None:
            raise RuntimeError("Run a forward pass before requesting gate diagnostics")
        connected = self._last_degree > 0
        isolated = ~connected
        connected_gates = self._last_expert_gate[connected]
        return {
            "mean_graph_gate_all": float(self._last_expert_gate.mean().cpu()),
            "mean_graph_gate_connected": float(connected_gates.mean().cpu()),
            "mean_graph_gate_isolated": float(
                self._last_expert_gate[isolated].mean().cpu()
            ),
            "mean_normal_graphsage_weight_connected": float(
                connected_gates[:, 0].mean().cpu()
            ),
            "mean_phishing_graphsage_weight_connected": float(
                connected_gates[:, 1].mean().cpu()
            ),
            "mean_scamming_graphsage_weight_connected": float(
                connected_gates[:, 2].mean().cpu()
            ),
            "connected_fraction": float(connected.float().mean().cpu()),
        }

