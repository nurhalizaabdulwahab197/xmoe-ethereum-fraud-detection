"""Deterministic BLTE multiclass reproduction runner.

Two protocols are supported:

``artifact`` uses the authors' released features and fixed masks. This is the
closest runnable equivalent of their public scripts.

``clean`` regenerates a 70/15/15 stratified split for every seed and fits the
feature scaler on training nodes only. It still uses the authors' released
edge set, whose construction is label-correlated (see REPRODUCIBILITY_AUDIT.md).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.nn import SAGEConv
from models import GAT as ModularGAT, GCN as ModularGCN


NON_FEATURE_COLUMNS = ["Address", "Scam", "attack", "attack_encoded"]


class GraphSAGE(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(input_dim, 64)
        self.conv2 = SAGEConv(64, 64)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc(x)


class MLP(nn.Module):
    """Feature-only comparator; not supplied by the authors' repository."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.out(x)


class IsolateAwareExplainableGNN(nn.Module):
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


class IsolateFixedGateGNN(IsolateAwareExplainableGNN):
    """Ablation using an equal feature/graph mix for connected nodes."""

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
        _, feature_h, graph_h, _, degree, connected = super().encode(x, edge_index)
        gate = 0.5 * connected.unsqueeze(1)
        fused = gate * graph_h + (1.0 - gate) * feature_h
        return fused, feature_h, graph_h, gate, degree, connected


class IsolateNoDegreeGateGNN(IsolateAwareExplainableGNN):
    """Ablation learning the gate without degree information."""

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
        _, feature_h, graph_h, _, degree, connected = super().encode(x, edge_index)
        zero_degree_feature = torch.zeros_like(degree).unsqueeze(1)
        gate_input = torch.cat([feature_h, graph_h, zero_degree_feature], dim=1)
        gate = torch.sigmoid(self.gate_fc2(F.relu(self.gate_fc1(gate_input))))
        gate = gate * connected.unsqueeze(1)
        fused = gate * graph_h + (1.0 - gate) * feature_h
        return fused, feature_h, graph_h, gate, degree, connected


class IsolateSoftGateGNN(IsolateAwareExplainableGNN):
    """Ablation allowing isolated nodes to use the root-only graph branch."""

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
        _, feature_h, graph_h, _, degree, connected = super().encode(x, edge_index)
        degree_feature = torch.log1p(degree).unsqueeze(1)
        gate_input = torch.cat([feature_h, graph_h, degree_feature], dim=1)
        gate = torch.sigmoid(self.gate_fc2(F.relu(self.gate_fc1(gate_input))))
        fused = gate * graph_h + (1.0 - gate) * feature_h
        return fused, feature_h, graph_h, gate, degree, connected


class IsolateBoundedLogitGNN(IsolateAwareExplainableGNN):
    """Precision-aware, bounded logit fusion proposed after the v1 ablation.

    v1's graph branch introduced most Normal-to-Scamming errors.  V2 therefore
    keeps the degree-aware learned gate and hard isolate mask, but applies graph
    information as a bounded, class-specific correction to feature-only logits.
    The correction scale is constrained to [0, 1] and remains inspectable.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(input_dim, output_dim)
        del self.classifier
        self.feature_classifier = nn.Linear(64, output_dim)
        self.graph_classifier = nn.Linear(64, output_dim)
        self.graph_classifier.load_state_dict(self.feature_classifier.state_dict())
        self.class_correction_logit = nn.Parameter(
            torch.full((output_dim,), -1.38629436112)
        )

    def encode_logits(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        _, feature_h, graph_h, gate, degree, _ = super().encode(x, edge_index)
        feature_logits = self.feature_classifier(feature_h)
        graph_logits = self.graph_classifier(graph_h)
        correction_scale = torch.sigmoid(self.class_correction_logit)
        logits = feature_logits + gate * correction_scale * (
            graph_logits - feature_logits
        )
        return logits, feature_logits, gate, degree, correction_scale

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        logits, _, gate, degree, _ = self.encode_logits(x, edge_index)
        self._last_gate = gate.detach().squeeze(1)
        self._last_degree = degree.detach()
        return logits

    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        logits, feature_logits, gate, degree, _ = self.encode_logits(x, edge_index)
        probabilities = F.softmax(logits, dim=1)
        feature_probabilities = F.softmax(feature_logits, dim=1)
        predicted = probabilities.argmax(dim=1)
        predicted_probability = probabilities.gather(1, predicted.unsqueeze(1)).squeeze(1)
        feature_probability = feature_probabilities.gather(
            1, predicted.unsqueeze(1)
        ).squeeze(1)
        return {
            "probabilities": probabilities,
            "feature_probabilities": feature_probabilities,
            "prediction": predicted,
            "feature_prediction": feature_probabilities.argmax(dim=1),
            "graph_gate": gate.squeeze(1),
            "degree": degree,
            "graph_probability_delta": predicted_probability - feature_probability,
        }


class IsolateAwareExplainableGNNV2(IsolateAwareExplainableGNN):
    """Selected v2: v1 architecture trained with a precision-aware loss."""


class IsolateAwareExplainableGNNV3(IsolateAwareExplainableGNN):
    """V3 candidate with a graph-induced false-Scamming training objective."""

    def training_outputs(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused, feature_h, _, gate, degree, _ = self.encode(x, edge_index)
        self._last_gate = gate.detach().squeeze(1)
        self._last_degree = degree.detach()
        logits = self.classifier(F.dropout(fused, p=0.5, training=self.training))
        feature_logits = self.classifier(feature_h)
        return logits, feature_logits, degree


class IsolateAwareExplainableGNNV4(IsolateAwareExplainableGNN):
    """Feature-normality-aware graph routing with a hard isolate constraint.

    The effective graph gate is the learned v1 gate multiplied by a transparent
    suppression factor derived from feature-only Normal confidence. During
    training, that confidence is detached for routing so the classifier cannot
    reduce the suppression by manipulating its Normal probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        normality_gate_alpha: float = 1.0,
    ) -> None:
        if normality_gate_alpha <= 0:
            raise ValueError("normality_gate_alpha must be positive")
        super().__init__(input_dim, output_dim)
        self.normality_gate_alpha = float(normality_gate_alpha)
        self._last_learned_gate: torch.Tensor | None = None

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
        _, feature_h, graph_h, learned_gate, degree, connected = super().encode(
            x, edge_index
        )
        feature_normality = F.softmax(self.classifier(feature_h), dim=1)[:, 0:1]
        routing_normality = (
            feature_normality.detach() if self.training else feature_normality
        )
        suppression = (1.0 - routing_normality).clamp_min(1e-6).pow(
            self.normality_gate_alpha
        )
        effective_gate = learned_gate * suppression
        fused = effective_gate * graph_h + (1.0 - effective_gate) * feature_h
        self._last_learned_gate = learned_gate.detach().squeeze(1)
        return fused, feature_h, graph_h, effective_gate, degree, connected

    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        explanation = super().explain(x, edge_index)
        if self._last_learned_gate is None:
            raise RuntimeError("V4 learned gate was not recorded")
        explanation["learned_graph_gate"] = self._last_learned_gate
        explanation["feature_normal_probability"] = explanation[
            "feature_probabilities"
        ][:, 0]
        return explanation

    def gate_diagnostics(self) -> dict[str, float]:
        diagnostics = super().gate_diagnostics()
        if self._last_learned_gate is None or self._last_degree is None:
            raise RuntimeError("Run a forward pass before requesting gate diagnostics")
        connected = self._last_degree > 0
        diagnostics["mean_learned_graph_gate_connected"] = float(
            self._last_learned_gate[connected].mean().cpu()
        )
        return diagnostics


class IsolateAwareExplainableGNNV5(IsolateAwareExplainableGNN):
    """Dynamic class-specific graph-logit routing for explainable fusion."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(input_dim, output_dim)
        self.gate_fc2 = nn.Linear(32, output_dim)
        self.output_dim = output_dim
        self._last_class_gate: torch.Tensor | None = None

    def encode_class_logits(
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
        gate_input = torch.cat(
            [feature_h, graph_h, torch.log1p(degree).unsqueeze(1)], dim=1
        )
        class_gate = torch.sigmoid(
            self.gate_fc2(F.relu(self.gate_fc1(gate_input)))
        ) * connected
        feature_logits = self.classifier(feature_h)
        graph_logits = self.classifier(graph_h)
        logits = feature_logits + class_gate * (graph_logits - feature_logits)
        return (
            logits,
            feature_logits,
            graph_logits,
            class_gate,
            degree,
            connected.squeeze(1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        logits, _, _, class_gate, degree, _ = self.encode_class_logits(x, edge_index)
        self._last_class_gate = class_gate.detach()
        self._last_gate = class_gate.detach().mean(dim=1)
        self._last_degree = degree.detach()
        return logits

    def training_outputs(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, _, _, class_gate, degree, _ = self.encode_class_logits(x, edge_index)
        self._last_class_gate = class_gate.detach()
        self._last_gate = class_gate.detach().mean(dim=1)
        self._last_degree = degree.detach()
        return logits, class_gate, degree

    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        logits, feature_logits, _, class_gate, degree, _ = self.encode_class_logits(
            x, edge_index
        )
        probabilities = F.softmax(logits, dim=1)
        feature_probabilities = F.softmax(feature_logits, dim=1)
        predicted = probabilities.argmax(dim=1)
        predicted_probability = probabilities.gather(1, predicted.unsqueeze(1)).squeeze(1)
        feature_probability = feature_probabilities.gather(
            1, predicted.unsqueeze(1)
        ).squeeze(1)
        selected_gate = class_gate.gather(1, predicted.unsqueeze(1)).squeeze(1)
        self._last_class_gate = class_gate.detach()
        self._last_gate = class_gate.detach().mean(dim=1)
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


class HierarchicalIsolateAwareExplainableGNNV6(IsolateAwareExplainableGNN):
    """Two-stage Normal/Fraud then Phishing/Scamming explainable GNN.

    Each binary decision has an independent graph gate. Both gates are hard
    masked to zero for isolated nodes, making their predictions exactly equal
    to the feature-only hierarchical counterfactual.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        if output_dim != 3:
            raise ValueError("V6 requires exactly Normal, Phishing and Scamming")
        super().__init__(input_dim, output_dim)
        del self.classifier
        self.gate_fc2 = nn.Linear(32, 2)
        self.feature_fraud_head = nn.Linear(64, 1)
        self.graph_fraud_head = nn.Linear(64, 1)
        self.feature_type_head = nn.Linear(64, 1)
        self.graph_type_head = nn.Linear(64, 1)
        self._last_stage_gate: torch.Tensor | None = None

    @staticmethod
    def hierarchical_log_probabilities(
        fraud_logit: torch.Tensor, type_logit: torch.Tensor
    ) -> torch.Tensor:
        """Return normalized log P(Normal, Phishing, Scamming)."""
        log_fraud = F.logsigmoid(fraud_logit)
        log_normal = F.logsigmoid(-fraud_logit)
        log_phishing_given_fraud = F.logsigmoid(type_logit)
        log_scamming_given_fraud = F.logsigmoid(-type_logit)
        return torch.cat(
            [
                log_normal,
                log_fraud + log_phishing_given_fraud,
                log_fraud + log_scamming_given_fraud,
            ],
            dim=1,
        )

    def encode_hierarchy(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[
        torch.Tensor,
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
        gate_input = torch.cat(
            [feature_h, graph_h, torch.log1p(degree).unsqueeze(1)], dim=1
        )
        stage_gate = torch.sigmoid(
            self.gate_fc2(F.relu(self.gate_fc1(gate_input)))
        ) * connected

        feature_fraud_logit = self.feature_fraud_head(feature_h)
        graph_fraud_logit = self.graph_fraud_head(graph_h)
        feature_type_logit = self.feature_type_head(feature_h)
        graph_type_logit = self.graph_type_head(graph_h)
        fraud_logit = feature_fraud_logit + stage_gate[:, 0:1] * (
            graph_fraud_logit - feature_fraud_logit
        )
        type_logit = feature_type_logit + stage_gate[:, 1:2] * (
            graph_type_logit - feature_type_logit
        )
        log_probabilities = self.hierarchical_log_probabilities(
            fraud_logit, type_logit
        )
        feature_log_probabilities = self.hierarchical_log_probabilities(
            feature_fraud_logit, feature_type_logit
        )
        return (
            log_probabilities,
            feature_log_probabilities,
            fraud_logit,
            type_logit,
            stage_gate,
            degree,
            connected.squeeze(1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        log_probabilities, _, _, _, stage_gate, degree, _ = self.encode_hierarchy(
            x, edge_index
        )
        self._last_stage_gate = stage_gate.detach()
        self._last_gate = stage_gate.detach().mean(dim=1)
        self._last_degree = degree.detach()
        return log_probabilities

    def training_outputs(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probabilities, _, fraud_logit, type_logit, stage_gate, degree, _ = (
            self.encode_hierarchy(x, edge_index)
        )
        self._last_stage_gate = stage_gate.detach()
        self._last_gate = stage_gate.detach().mean(dim=1)
        self._last_degree = degree.detach()
        return log_probabilities, fraud_logit, type_logit

    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        (
            log_probabilities,
            feature_log_probabilities,
            fraud_logit,
            type_logit,
            stage_gate,
            degree,
            _,
        ) = self.encode_hierarchy(x, edge_index)
        probabilities = log_probabilities.exp()
        feature_probabilities = feature_log_probabilities.exp()
        predicted = probabilities.argmax(dim=1)
        predicted_probability = probabilities.gather(
            1, predicted.unsqueeze(1)
        ).squeeze(1)
        feature_probability = feature_probabilities.gather(
            1, predicted.unsqueeze(1)
        ).squeeze(1)
        selected_gate = torch.where(
            predicted == 0, stage_gate[:, 0], stage_gate[:, 1]
        )
        self._last_stage_gate = stage_gate.detach()
        self._last_gate = stage_gate.detach().mean(dim=1)
        self._last_degree = degree.detach()
        return {
            "probabilities": probabilities,
            "feature_probabilities": feature_probabilities,
            "prediction": predicted,
            "feature_prediction": feature_probabilities.argmax(dim=1),
            "graph_gate": selected_gate,
            "fraud_graph_gate": stage_gate[:, 0],
            "type_graph_gate": stage_gate[:, 1],
            "fraud_probability": torch.sigmoid(fraud_logit).squeeze(1),
            "phishing_given_fraud_probability": torch.sigmoid(type_logit).squeeze(1),
            "degree": degree,
            "graph_probability_delta": predicted_probability - feature_probability,
        }

    def gate_diagnostics(self) -> dict[str, float]:
        if self._last_stage_gate is None or self._last_degree is None:
            raise RuntimeError("Run a forward pass before requesting gate diagnostics")
        connected = self._last_degree > 0
        isolated = ~connected
        connected_gates = self._last_stage_gate[connected]
        return {
            "mean_graph_gate_all": float(self._last_stage_gate.mean().cpu()),
            "mean_graph_gate_connected": float(connected_gates.mean().cpu()),
            "mean_graph_gate_isolated": float(
                self._last_stage_gate[isolated].mean().cpu()
            ),
            "mean_fraud_graph_gate_connected": float(
                connected_gates[:, 0].mean().cpu()
            ),
            "mean_type_graph_gate_connected": float(
                connected_gates[:, 1].mean().cpu()
            ),
            "connected_fraction": float(connected.float().mean().cpu()),
        }


class DegreeAwareScammingRescueGNNV7(HierarchicalIsolateAwareExplainableGNNV6):
    """V6 hierarchy plus a bounded degree-supported Scamming correction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_rescue: float = 0.5,
        degree_constant: float = 20.0,
    ) -> None:
        if max_rescue <= 0:
            raise ValueError("max_rescue must be positive")
        if degree_constant <= 0:
            raise ValueError("degree_constant must be positive")
        super().__init__(input_dim, output_dim)
        self.max_rescue = float(max_rescue)
        self.degree_constant = float(degree_constant)
        self.rescue_gate_fc1 = nn.Linear(64 + 64 + 1, 32)
        self.rescue_gate_fc2 = nn.Linear(32, 1)
        self.feature_rescue_head = nn.Linear(64, 1)
        self.graph_rescue_head = nn.Linear(64, 1)
        self._last_rescue_gate: torch.Tensor | None = None

    def encode_rescue(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
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
        gate_input = torch.cat(
            [feature_h, graph_h, torch.log1p(degree).unsqueeze(1)], dim=1
        )
        stage_gate = torch.sigmoid(
            self.gate_fc2(F.relu(self.gate_fc1(gate_input)))
        ) * connected
        rescue_gate = torch.sigmoid(
            self.rescue_gate_fc2(F.relu(self.rescue_gate_fc1(gate_input)))
        ) * connected

        feature_fraud_logit = self.feature_fraud_head(feature_h)
        graph_fraud_logit = self.graph_fraud_head(graph_h)
        feature_type_logit = self.feature_type_head(feature_h)
        graph_type_logit = self.graph_type_head(graph_h)
        fraud_logit = feature_fraud_logit + stage_gate[:, 0:1] * (
            graph_fraud_logit - feature_fraud_logit
        )
        type_logit = feature_type_logit + stage_gate[:, 1:2] * (
            graph_type_logit - feature_type_logit
        )
        base_log_probabilities = self.hierarchical_log_probabilities(
            fraud_logit, type_logit
        )
        feature_log_probabilities = self.hierarchical_log_probabilities(
            feature_fraud_logit, feature_type_logit
        )

        feature_rescue_logit = self.feature_rescue_head(feature_h)
        graph_rescue_logit = self.graph_rescue_head(graph_h)
        rescue_logit = feature_rescue_logit + rescue_gate * (
            graph_rescue_logit - feature_rescue_logit
        )
        degree_support = degree.unsqueeze(1) / (
            degree.unsqueeze(1) + self.degree_constant
        )
        rescue_correction = (
            self.max_rescue * degree_support * torch.sigmoid(rescue_logit)
        )
        final_scores = base_log_probabilities.clone()
        final_scores[:, 2:3] = final_scores[:, 2:3] + rescue_correction
        final_log_probabilities = F.log_softmax(final_scores, dim=1)
        return (
            final_log_probabilities,
            feature_log_probabilities,
            base_log_probabilities,
            fraud_logit,
            type_logit,
            rescue_logit,
            stage_gate,
            rescue_gate,
            degree_support.squeeze(1),
            degree,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        outputs = self.encode_rescue(x, edge_index)
        stage_gate, rescue_gate, degree = outputs[6], outputs[7], outputs[9]
        self._last_stage_gate = stage_gate.detach()
        self._last_rescue_gate = rescue_gate.detach().squeeze(1)
        self._last_degree = degree.detach()
        return outputs[0]

    def training_outputs(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.encode_rescue(x, edge_index)
        stage_gate, rescue_gate, degree = outputs[6], outputs[7], outputs[9]
        self._last_stage_gate = stage_gate.detach()
        self._last_rescue_gate = rescue_gate.detach().squeeze(1)
        self._last_degree = degree.detach()
        return outputs[0], outputs[3], outputs[4], outputs[5]

    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        outputs = self.encode_rescue(x, edge_index)
        (
            final_log_probabilities,
            feature_log_probabilities,
            base_log_probabilities,
            fraud_logit,
            type_logit,
            rescue_logit,
            stage_gate,
            rescue_gate,
            degree_support,
            degree,
        ) = outputs
        probabilities = final_log_probabilities.exp()
        feature_probabilities = feature_log_probabilities.exp()
        base_probabilities = base_log_probabilities.exp()
        predicted = probabilities.argmax(dim=1)
        predicted_probability = probabilities.gather(
            1, predicted.unsqueeze(1)
        ).squeeze(1)
        feature_probability = feature_probabilities.gather(
            1, predicted.unsqueeze(1)
        ).squeeze(1)
        rescue_correction = (
            self.max_rescue * degree_support * torch.sigmoid(rescue_logit.squeeze(1))
        )
        selected_gate = torch.where(
            predicted == 0, stage_gate[:, 0], stage_gate[:, 1]
        )
        self._last_stage_gate = stage_gate.detach()
        self._last_rescue_gate = rescue_gate.detach().squeeze(1)
        self._last_degree = degree.detach()
        return {
            "probabilities": probabilities,
            "feature_probabilities": feature_probabilities,
            "base_probabilities": base_probabilities,
            "prediction": predicted,
            "feature_prediction": feature_probabilities.argmax(dim=1),
            "graph_gate": selected_gate,
            "fraud_graph_gate": stage_gate[:, 0],
            "type_graph_gate": stage_gate[:, 1],
            "rescue_graph_gate": rescue_gate.squeeze(1),
            "fraud_probability": torch.sigmoid(fraud_logit).squeeze(1),
            "phishing_given_fraud_probability": torch.sigmoid(type_logit).squeeze(1),
            "rescue_probability": torch.sigmoid(rescue_logit).squeeze(1),
            "degree_support": degree_support,
            "rescue_correction": rescue_correction,
            "degree": degree,
            "graph_probability_delta": predicted_probability - feature_probability,
        }

    def gate_diagnostics(self) -> dict[str, float]:
        diagnostics = super().gate_diagnostics()
        if self._last_rescue_gate is None or self._last_degree is None:
            raise RuntimeError("Run a forward pass before requesting gate diagnostics")
        connected = self._last_degree > 0
        diagnostics["mean_rescue_graph_gate_connected"] = float(
            self._last_rescue_gate[connected].mean().cpu()
        )
        return diagnostics


class ConflictAwareMultiTaskExplainableGNNV8(IsolateAwareExplainableGNN):
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


class ExplainableMixtureOfExpertsGNN(IsolateAwareExplainableGNN):
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


MODELS = {
    "gcn": ModularGCN,
    "gat": ModularGAT,
    "graphsage": GraphSAGE,
    "x_moe_gnn": ExplainableMixtureOfExpertsGNN,
}


def set_seed(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def boolean_mask(indices: np.ndarray, size: int) -> torch.Tensor:
    mask = torch.zeros(size, dtype=torch.bool)
    mask[torch.from_numpy(indices)] = True
    return mask


def clean_data(released_data, csv_path: Path, seed: int):
    frame = pd.read_csv(csv_path)
    y = frame["attack_encoded"].to_numpy(dtype=np.int64, copy=True)
    indices = np.arange(len(frame))
    train_idx, remainder_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=y
    )
    val_idx, test_idx = train_test_split(
        remainder_idx,
        test_size=0.50,
        random_state=seed,
        stratify=y[remainder_idx],
    )

    raw_x = frame.drop(columns=NON_FEATURE_COLUMNS).to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(raw_x[train_idx])
    x = torch.tensor(scaler.transform(raw_x), dtype=torch.float32)
    size = len(frame)
    return (
        x,
        released_data.edge_index.cpu(),
        torch.from_numpy(y),
        boolean_mask(train_idx, size),
        boolean_mask(val_idx, size),
        boolean_mask(test_idx, size),
    )


def artifact_data(released_data):
    return (
        released_data.x.cpu(),
        released_data.edge_index.cpu(),
        released_data.y.cpu(),
        released_data.train_mask.cpu(),
        released_data.val_mask.cpu(),
        released_data.test_mask.cpu(),
    )


def score(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1, 2])
    result: dict[str, object] = {
        "accuracy": accuracy_score(y_true, predicted),
        "macro_precision": precision_score(
            y_true, predicted, labels=[0, 1, 2], average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, predicted, labels=[0, 1, 2], average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, predicted, labels=[0, 1, 2], average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, predicted, labels=[0, 1, 2], average="weighted", zero_division=0
        ),
        "phishing_ap": average_precision_score((y_true == 1), probabilities[:, 1]),
        "scamming_ap": average_precision_score((y_true == 2), probabilities[:, 2]),
    }
    for class_id, class_name in enumerate(("normal", "phishing", "scamming")):
        result[f"{class_name}_precision"] = precision_score(
            y_true, predicted, labels=[class_id], average="macro", zero_division=0
        )
        result[f"{class_name}_recall"] = recall_score(
            y_true, predicted, labels=[class_id], average="macro", zero_division=0
        )
        result[f"{class_name}_f1"] = f1_score(
            y_true, predicted, labels=[class_id], average="macro", zero_division=0
        )
    for actual in range(3):
        for predicted_class in range(3):
            result[f"cm_{actual}_{predicted_class}"] = int(
                matrix[actual, predicted_class]
            )
    return result


def weighted_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Weighted focal loss with CE-equivalent normalization at gamma zero."""
    per_item_ce = F.cross_entropy(
        logits, labels, weight=class_weights, reduction="none"
    )
    true_probability = F.softmax(logits, dim=1).gather(
        1, labels.unsqueeze(1)
    ).squeeze(1)
    denominator = class_weights[labels].sum().clamp_min(torch.finfo(logits.dtype).eps)
    return (((1.0 - true_probability) ** gamma) * per_item_ce).sum() / denominator


def balanced_binary_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Inverse-frequency weighted BCE normalized by the observed weights."""
    labels = labels.to(dtype=logits.dtype)
    counts = torch.bincount(labels.long(), minlength=2).to(dtype=logits.dtype)
    if (counts == 0).any():
        raise ValueError("balanced_binary_loss requires both binary classes")
    class_weights = counts.sum() / counts
    class_weights = class_weights / class_weights.sum()
    sample_weights = class_weights[labels.long()]
    per_item = F.binary_cross_entropy_with_logits(
        logits.squeeze(1), labels, reduction="none"
    )
    return (per_item * sample_weights).sum() / sample_weights.sum()


def project_conflicting_gradients(
    task_gradients: list[list[torch.Tensor]],
) -> list[list[torch.Tensor]]:
    """Deterministic PCGrad projection in fixed task order."""
    projected = [[gradient.clone() for gradient in task] for task in task_gradients]
    for task_index, task in enumerate(projected):
        for other_index, other in enumerate(task_gradients):
            if task_index == other_index:
                continue
            dot = sum((left * right).sum() for left, right in zip(task, other))
            norm = sum((gradient * gradient).sum() for gradient in other)
            if float(dot.detach()) < 0 and float(norm.detach()) > 0:
                scale = dot / norm
                task = [
                    left - scale * right for left, right in zip(task, other)
                ]
                projected[task_index] = task
    return projected


def pcgrad_backward(
    losses: list[torch.Tensor], parameters: list[torch.nn.Parameter]
) -> None:
    """Populate parameter gradients with mean deterministic PCGrad gradients."""
    task_gradients: list[list[torch.Tensor]] = []
    for index, loss in enumerate(losses):
        raw = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=index < len(losses) - 1,
            allow_unused=True,
        )
        task_gradients.append(
            [
                gradient if gradient is not None else torch.zeros_like(parameter)
                for gradient, parameter in zip(raw, parameters)
            ]
        )
    projected = project_conflicting_gradients(task_gradients)
    for parameter_index, parameter in enumerate(parameters):
        parameter.grad = torch.stack(
            [task[parameter_index] for task in projected]
        ).mean(dim=0)


def select_validation_class_bias(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Grid-select Phishing/Scamming logit biases using validation macro-F1."""
    best_score = -1.0
    best_norm = float("inf")
    best_bias = torch.zeros(3, device=logits.device)
    values = np.arange(-1.5, 1.5001, 0.25)
    true = labels.detach().cpu().numpy()
    for phishing_bias in values:
        for scamming_bias in values:
            bias = torch.tensor(
                [0.0, phishing_bias, scamming_bias],
                dtype=logits.dtype,
                device=logits.device,
            )
            predicted = (logits + bias).argmax(dim=1).detach().cpu().numpy()
            candidate_score = f1_score(
                true,
                predicted,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
            candidate_norm = abs(phishing_bias) + abs(scamming_bias)
            if candidate_score > best_score + 1e-15 or (
                abs(candidate_score - best_score) <= 1e-15
                and candidate_norm < best_norm
            ):
                best_score = candidate_score
                best_norm = candidate_norm
                best_bias = bias
    return best_bias


def run_once(
    released_data,
    csv_path: Path,
    model_name: str,
    protocol: str,
    seed: int,
    epochs: int,
    device: torch.device,
    deterministic: bool,
    explanation_path: Path | None = None,
    checkpoint_path: Path | None = None,
    prediction_path: Path | None = None,
    normal_scam_penalty: float = 0.0,
    graph_scam_uplift_penalty: float = 0.0,
    base_loss: str = "weighted_ce",
    focal_gamma: float = 2.0,
    calibrate_class_bias: bool = False,
    normality_gate_alpha: float = 1.0,
    normal_scamming_gate_penalty: float = 0.0,
    fraud_stage_loss_weight: float = 0.5,
    type_stage_loss_weight: float = 0.5,
    normal_fraud_penalty: float = 0.0,
    rescue_loss_weight: float = 0.5,
    max_rescue: float = 0.5,
    task_loss_weight: float = 0.5,
    expert_loss_weight: float = 0.25,
    scam_specialist_loss_weight: float = 0.5,
    x_moe_v8_task_loss_weight: float = 1.0,
    x_moe_normal_phishing_penalty: float = 0.0,
) -> dict[str, object]:
    set_seed(seed, deterministic)
    tensors = (
        artifact_data(released_data)
        if protocol == "artifact"
        else clean_data(released_data, csv_path, seed)
    )
    x, edge_index, y, train_mask, val_mask, test_mask = [
        tensor.to(device) for tensor in tensors
    ]
    model_kwargs = {}
    if model_name == "isolate_gnn_v4":
        model_kwargs["normality_gate_alpha"] = normality_gate_alpha
    if model_name == "isolate_gnn_v7":
        model_kwargs["max_rescue"] = max_rescue
    model = MODELS[model_name](
        x.shape[1], int(y.max().item()) + 1, **model_kwargs
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    counts = torch.bincount(y[train_mask], minlength=3)
    class_weights = counts.sum() / counts
    class_weights = class_weights / class_weights.sum()
    best_val_f1 = -1.0
    best_state = None
    started = time.perf_counter()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        feature_logits = None
        degree = None
        class_gate = None
        fraud_logit = None
        type_logit = None
        rescue_logit = None
        multitask_logits = None
        x_moe_expert_logits = None
        if isinstance(model, ExplainableMixtureOfExpertsGNN):
            logits, v8_logits, graphsage_logits = model.training_outputs(
                x, edge_index
            )
            x_moe_expert_logits = (v8_logits, graphsage_logits)
        elif isinstance(model, ConflictAwareMultiTaskExplainableGNNV8):
            logits, multitask_logits = model.training_outputs(x, edge_index)
        elif isinstance(model, DegreeAwareScammingRescueGNNV7):
            logits, fraud_logit, type_logit, rescue_logit = model.training_outputs(
                x, edge_index
            )
        elif isinstance(model, HierarchicalIsolateAwareExplainableGNNV6):
            logits, fraud_logit, type_logit = model.training_outputs(x, edge_index)
        elif isinstance(model, IsolateAwareExplainableGNNV5):
            logits, class_gate, degree = model.training_outputs(x, edge_index)
        elif isinstance(model, IsolateAwareExplainableGNNV3):
            logits, feature_logits, degree = model.training_outputs(x, edge_index)
        else:
            logits = model(x, edge_index)
        if model_name in {"isolate_gnn_v6", "isolate_gnn_v7"}:
            loss = F.nll_loss(
                logits[train_mask], y[train_mask], weight=class_weights
            )
        elif base_loss == "weighted_focal":
            loss = weighted_focal_loss(
                logits[train_mask],
                y[train_mask],
                class_weights,
                focal_gamma,
            )
        else:
            loss = F.cross_entropy(
                logits[train_mask], y[train_mask], weight=class_weights
            )
        if (
            model_name
            in {
                "isolate_bounded_logit",
                "isolate_gnn_v2",
                "isolate_gnn_v4",
                "isolate_gnn_v5",
            }
            and normal_scam_penalty > 0
        ):
            train_logits = logits[train_mask]
            train_labels = y[train_mask]
            normal_probabilities = F.softmax(
                train_logits[train_labels == 0], dim=1
            )[:, 2]
            loss = loss + normal_scam_penalty * normal_probabilities.mean()
        if (
            model_name == "isolate_gnn_v3"
            and graph_scam_uplift_penalty > 0
            and feature_logits is not None
            and degree is not None
        ):
            connected_normal = train_mask & (y == 0) & (degree > 0)
            full_scam_probability = F.softmax(logits[connected_normal], dim=1)[:, 2]
            feature_scam_probability = F.softmax(
                feature_logits[connected_normal], dim=1
            )[:, 2]
            graph_scam_uplift = F.relu(
                full_scam_probability - feature_scam_probability
            ).mean()
            loss = loss + graph_scam_uplift_penalty * graph_scam_uplift
        if (
            model_name == "isolate_gnn_v5"
            and normal_scamming_gate_penalty > 0
            and class_gate is not None
            and degree is not None
        ):
            connected_normal = train_mask & (y == 0) & (degree > 0)
            loss = loss + normal_scamming_gate_penalty * class_gate[
                connected_normal, 2
            ].mean()
        if (
            model_name in {"isolate_gnn_v6", "isolate_gnn_v7"}
            and fraud_logit is not None
            and type_logit is not None
        ):
            train_labels = y[train_mask]
            fraud_labels = (train_labels > 0).to(dtype=fraud_logit.dtype)
            fraud_loss = balanced_binary_loss(
                fraud_logit[train_mask], fraud_labels
            )
            fraud_train_mask = train_mask & (y > 0)
            phishing_labels = (y[fraud_train_mask] == 1).to(
                dtype=type_logit.dtype
            )
            type_loss = balanced_binary_loss(
                type_logit[fraud_train_mask], phishing_labels
            )
            loss = (
                loss
                + fraud_stage_loss_weight * fraud_loss
                + type_stage_loss_weight * type_loss
            )
            if normal_fraud_penalty > 0:
                normal_train_mask = train_mask & (y == 0)
                normal_fraud_probability = torch.sigmoid(
                    fraud_logit[normal_train_mask]
                ).mean()
                loss = loss + normal_fraud_penalty * normal_fraud_probability
        if (
            model_name == "isolate_gnn_v7"
            and rescue_logit is not None
            and rescue_loss_weight > 0
        ):
            rescue_train_mask = train_mask & ((y == 0) | (y == 2))
            rescue_labels = (y[rescue_train_mask] == 2).to(
                dtype=rescue_logit.dtype
            )
            rescue_loss = balanced_binary_loss(
                rescue_logit[rescue_train_mask], rescue_labels
            )
            loss = loss + rescue_loss_weight * rescue_loss
        if model_name == "x_moe_gnn" and x_moe_expert_logits is not None:
            v8_logits, graphsage_logits = x_moe_expert_logits
            training_labels = y[train_mask]
            task_losses = [loss]
            task_losses.extend(
                [
                    expert_loss_weight
                    * F.cross_entropy(
                        expert_logits[train_mask],
                        training_labels,
                        weight=class_weights,
                    )
                    for expert_logits in (v8_logits, graphsage_logits)
                ]
            )
            for class_id in range(3):
                binary_labels = (training_labels == class_id).to(
                    dtype=v8_logits.dtype
                )
                task_losses.append(
                    x_moe_v8_task_loss_weight
                    * balanced_binary_loss(
                        v8_logits[train_mask, class_id : class_id + 1],
                        binary_labels,
                    )
                )
            scamming_labels = (training_labels == 2).to(
                dtype=graphsage_logits.dtype
            )
            task_losses.append(
                scam_specialist_loss_weight
                * balanced_binary_loss(
                    graphsage_logits[train_mask, 2:3], scamming_labels
                )
            )
            if x_moe_normal_phishing_penalty > 0:
                degree = model.v8_expert.node_degree(edge_index, x.shape[0])
                connected_normal = train_mask & (y == 0) & (degree > 0)
                normal_phishing_probability = F.softmax(
                    logits[connected_normal], dim=1
                )[:, 1].mean()
                task_losses.append(
                    x_moe_normal_phishing_penalty
                    * normal_phishing_probability
                )
            pcgrad_backward(
                task_losses, [parameter for parameter in model.parameters()]
            )
        elif model_name == "isolate_gnn_v8" and multitask_logits is not None:
            task_losses = [loss]
            for class_id in range(3):
                binary_labels = (y[train_mask] == class_id).to(
                    dtype=multitask_logits.dtype
                )
                task_losses.append(
                    task_loss_weight
                    * balanced_binary_loss(
                        multitask_logits[train_mask, class_id : class_id + 1],
                        binary_labels,
                    )
                )
            pcgrad_backward(
                task_losses, [parameter for parameter in model.parameters()]
            )
        else:
            loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x, edge_index)[val_mask]
            val_predicted = val_logits.argmax(dim=1).cpu().numpy()
            val_true = y[val_mask].cpu().numpy()
        val_f1 = f1_score(
            val_true,
            val_predicted,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        class_bias = torch.zeros(3, device=device)
        if calibrate_class_bias:
            validation_logits = model(x, edge_index)[val_mask]
            class_bias = select_validation_class_bias(
                validation_logits, y[val_mask]
            )
        if isinstance(model, IsolateAwareExplainableGNN):
            explanation = model.explain(x, edge_index)
            full_probabilities = F.softmax(
                torch.log(explanation["probabilities"].clamp_min(1e-12))
                + class_bias,
                dim=1,
            )
            feature_probabilities = F.softmax(
                torch.log(explanation["feature_probabilities"].clamp_min(1e-12))
                + class_bias,
                dim=1,
            )
            calibrated_prediction = full_probabilities.argmax(dim=1)
            calibrated_probability = full_probabilities.gather(
                1, calibrated_prediction.unsqueeze(1)
            ).squeeze(1)
            calibrated_feature_probability = feature_probabilities.gather(
                1, calibrated_prediction.unsqueeze(1)
            ).squeeze(1)
            explanation.update(
                {
                    "probabilities": full_probabilities,
                    "feature_probabilities": feature_probabilities,
                    "prediction": calibrated_prediction,
                    "feature_prediction": feature_probabilities.argmax(dim=1),
                    "graph_probability_delta": (
                        calibrated_probability - calibrated_feature_probability
                    ),
                }
            )
            probabilities = full_probabilities[test_mask].cpu().numpy()
        else:
            probabilities = F.softmax(
                model(x, edge_index)[test_mask] + class_bias, dim=1
            ).cpu().numpy()
        y_true = y[test_mask].cpu().numpy()

    result = score(y_true, probabilities)
    if isinstance(model, IsolateAwareExplainableGNN):
        test_explanation = {
            key: value[test_mask].detach().cpu()
            for key, value in explanation.items()
        }
        test_connected = test_explanation["degree"] > 0
        test_flips = (
            test_explanation["prediction"]
            != test_explanation["feature_prediction"]
        )
        result.update(model.gate_diagnostics())
        result.update(
            {
                "test_connected_nodes": int(test_connected.sum()),
                "test_isolated_nodes": int((~test_connected).sum()),
                "mean_abs_graph_probability_delta_connected": float(
                    test_explanation["graph_probability_delta"][test_connected]
                    .abs()
                    .mean()
                ),
                "graph_counterfactual_flip_rate_connected": float(
                    test_flips[test_connected].float().mean()
                ),
                "graph_counterfactual_flip_rate_isolated": float(
                    test_flips[~test_connected].float().mean()
                ),
            }
        )
        if explanation_path is not None:
            test_indices = torch.where(test_mask)[0].cpu().numpy()
            explanation_frame = pd.DataFrame(
                {
                    "node_index": test_indices,
                    "true_label": y_true,
                    "predicted_label": test_explanation["prediction"].numpy(),
                    "feature_only_predicted_label": test_explanation[
                        "feature_prediction"
                    ].numpy(),
                    "normal_probability": probabilities[:, 0],
                    "phishing_probability": probabilities[:, 1],
                    "scamming_probability": probabilities[:, 2],
                    "feature_only_normal_probability": test_explanation[
                        "feature_probabilities"
                    ][:, 0].numpy(),
                    "feature_only_phishing_probability": test_explanation[
                        "feature_probabilities"
                    ][:, 1].numpy(),
                    "feature_only_scamming_probability": test_explanation[
                        "feature_probabilities"
                    ][:, 2].numpy(),
                    "degree": test_explanation["degree"].numpy(),
                    "is_isolated": (~test_connected).numpy(),
                    "graph_gate": test_explanation["graph_gate"].numpy(),
                    "graph_probability_delta": test_explanation[
                        "graph_probability_delta"
                    ].numpy(),
                    "decision_changed_without_graph": test_flips.numpy(),
                }
            )
            if "learned_graph_gate" in test_explanation:
                explanation_frame["learned_graph_gate"] = test_explanation[
                    "learned_graph_gate"
                ].numpy()
                explanation_frame["feature_normal_probability"] = test_explanation[
                    "feature_normal_probability"
                ].numpy()
            if "normal_graph_gate" in test_explanation:
                explanation_frame["normal_graph_gate"] = test_explanation[
                    "normal_graph_gate"
                ].numpy()
                explanation_frame["phishing_graph_gate"] = test_explanation[
                    "phishing_graph_gate"
                ].numpy()
                explanation_frame["scamming_graph_gate"] = test_explanation[
                    "scamming_graph_gate"
                ].numpy()
            if "fraud_graph_gate" in test_explanation:
                explanation_frame["fraud_graph_gate"] = test_explanation[
                    "fraud_graph_gate"
                ].numpy()
                explanation_frame["type_graph_gate"] = test_explanation[
                    "type_graph_gate"
                ].numpy()
                explanation_frame["fraud_probability"] = test_explanation[
                    "fraud_probability"
                ].numpy()
                explanation_frame["phishing_given_fraud_probability"] = (
                    test_explanation["phishing_given_fraud_probability"].numpy()
                )
            if "rescue_graph_gate" in test_explanation:
                explanation_frame["rescue_graph_gate"] = test_explanation[
                    "rescue_graph_gate"
                ].numpy()
                explanation_frame["rescue_probability"] = test_explanation[
                    "rescue_probability"
                ].numpy()
                explanation_frame["degree_support"] = test_explanation[
                    "degree_support"
                ].numpy()
                explanation_frame["rescue_correction"] = test_explanation[
                    "rescue_correction"
                ].numpy()
            if "normal_graphsage_expert_weight" in test_explanation:
                for class_name in ("normal", "phishing", "scamming"):
                    explanation_frame[f"{class_name}_graphsage_expert_weight"] = (
                        test_explanation[
                            f"{class_name}_graphsage_expert_weight"
                        ].numpy()
                    )
                    explanation_frame[f"{class_name}_v8_expert_weight"] = (
                        test_explanation[f"{class_name}_v8_expert_weight"].numpy()
                    )
                explanation_frame["expert_probability_delta"] = test_explanation[
                    "expert_probability_delta"
                ].numpy()
            explanation_path.parent.mkdir(parents=True, exist_ok=True)
            explanation_frame.to_csv(explanation_path, index=False)
    result.update(
        {
            "model": model_name,
            "protocol": protocol,
            "seed": seed,
            "epochs": epochs,
            "best_validation_macro_f1": best_val_f1,
            "training_seconds": time.perf_counter() - started,
            "device": str(device),
            "train_nodes": int(train_mask.sum()),
            "validation_nodes": int(val_mask.sum()),
            "test_nodes": int(test_mask.sum()),
            "normal_scam_penalty": normal_scam_penalty,
            "graph_scam_uplift_penalty": graph_scam_uplift_penalty,
            "base_loss": base_loss,
            "focal_gamma": focal_gamma,
            "calibrate_class_bias": calibrate_class_bias,
            "normal_class_bias": float(class_bias[0].cpu()),
            "phishing_class_bias": float(class_bias[1].cpu()),
            "scamming_class_bias": float(class_bias[2].cpu()),
            "normality_gate_alpha": normality_gate_alpha,
            "normal_scamming_gate_penalty": normal_scamming_gate_penalty,
            "fraud_stage_loss_weight": fraud_stage_loss_weight,
            "type_stage_loss_weight": type_stage_loss_weight,
            "normal_fraud_penalty": normal_fraud_penalty,
            "rescue_loss_weight": rescue_loss_weight,
            "max_rescue": max_rescue,
            "task_loss_weight": task_loss_weight,
            "expert_loss_weight": expert_loss_weight,
            "scam_specialist_loss_weight": scam_specialist_loss_weight,
            "x_moe_v8_task_loss_weight": x_moe_v8_task_loss_weight,
            "x_moe_normal_phishing_penalty": x_moe_normal_phishing_penalty,
        }
    )
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        feature_names = pd.read_csv(csv_path, nrows=0).drop(
            columns=NON_FEATURE_COLUMNS
        ).columns.tolist()
        torch.save(
            {
                "format_version": 1,
                "model_version": (
                    "isolate-aware-gnn-v1"
                    if model_name == "isolate_gnn"
                    else "isolate-aware-bounded-logit-prototype"
                    if model_name == "isolate_bounded_logit"
                    else "isolate-aware-gnn-v2"
                    if model_name == "isolate_gnn_v2"
                    else "isolate-aware-gnn-v3"
                    if model_name == "isolate_gnn_v3"
                    else "isolate-aware-gnn-v4"
                    if model_name == "isolate_gnn_v4"
                    else "isolate-aware-gnn-v5"
                    if model_name == "isolate_gnn_v5"
                    else "hierarchical-isolate-aware-explainable-gnn-v6"
                    if model_name == "isolate_gnn_v6"
                    else "degree-aware-scamming-rescue-gnn-v7"
                    if model_name == "isolate_gnn_v7"
                    else "conflict-aware-multitask-explainable-gnn-v8"
                    if model_name == "isolate_gnn_v8"
                    else "explainable-mixture-of-experts-gnn"
                    if model_name == "x_moe_gnn"
                    else model_name
                ),
                "model_name": model_name,
                "protocol": protocol,
                "seed": seed,
                "epochs": epochs,
                "input_dim": x.shape[1],
                "output_dim": int(y.max().item()) + 1,
                "feature_names": feature_names,
                "class_names": ["Normal", "Phishing", "Scamming"],
                "normality_gate_alpha": normality_gate_alpha,
                "normal_scamming_gate_penalty": normal_scamming_gate_penalty,
                "fraud_stage_loss_weight": fraud_stage_loss_weight,
                "type_stage_loss_weight": type_stage_loss_weight,
                "normal_fraud_penalty": normal_fraud_penalty,
                "rescue_loss_weight": rescue_loss_weight,
                "max_rescue": max_rescue,
                "task_loss_weight": task_loss_weight,
                "expert_loss_weight": expert_loss_weight,
                "scam_specialist_loss_weight": scam_specialist_loss_weight,
                "x_moe_v8_task_loss_weight": x_moe_v8_task_loss_weight,
                "x_moe_normal_phishing_penalty": x_moe_normal_phishing_penalty,
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "metrics": result,
            },
            checkpoint_path,
        )
        result["checkpoint_path"] = str(checkpoint_path)
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "node_index": torch.where(test_mask)[0].cpu().numpy(),
                "true_label": y_true,
                "predicted_label": probabilities.argmax(axis=1),
                "normal_probability": probabilities[:, 0],
                "phishing_probability": probabilities[:, 1],
                "scamming_probability": probabilities[:, 2],
            }
        ).to_csv(prediction_path, index=False)
        result["prediction_path"] = str(prediction_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="graphsage")
    parser.add_argument("--protocol", choices=["artifact", "clean"], default="artifact")
    parser.add_argument("--seeds", default="42", help="Comma-separated integer seeds")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upstream", type=Path, default=Path("upstream-paper"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--normal-scam-penalty",
        type=float,
        default=0.0,
        help="V2 penalty on Scamming probability for Normal training nodes.",
    )
    parser.add_argument(
        "--graph-scam-uplift-penalty",
        type=float,
        default=0.0,
        help="V3 penalty on graph-induced Scamming uplift for connected Normal nodes.",
    )
    parser.add_argument(
        "--base-loss",
        choices=["weighted_ce", "weighted_focal"],
        default="weighted_ce",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--normality-gate-alpha",
        type=float,
        default=1.0,
        help="V4 exponent controlling feature-normality graph suppression.",
    )
    parser.add_argument(
        "--normal-scamming-gate-penalty",
        type=float,
        default=0.0,
        help="V5 penalty on the Scamming gate for connected Normal nodes.",
    )
    parser.add_argument(
        "--fraud-stage-loss-weight",
        type=float,
        default=0.5,
        help="V6 auxiliary Normal/Fraud loss weight.",
    )
    parser.add_argument(
        "--type-stage-loss-weight",
        type=float,
        default=0.5,
        help="V6 auxiliary Phishing/Scamming loss weight.",
    )
    parser.add_argument(
        "--normal-fraud-penalty",
        type=float,
        default=0.0,
        help="V6 penalty on predicted Fraud probability for Normal nodes.",
    )
    parser.add_argument(
        "--rescue-loss-weight",
        type=float,
        default=0.5,
        help="V7 auxiliary Scamming-versus-Normal rescue loss weight.",
    )
    parser.add_argument(
        "--max-rescue",
        type=float,
        default=0.5,
        help="V7 maximum positive Scamming log-probability correction.",
    )
    parser.add_argument(
        "--task-loss-weight",
        type=float,
        default=0.5,
        help="V8 common one-versus-rest auxiliary loss weight.",
    )
    parser.add_argument(
        "--expert-loss-weight",
        type=float,
        default=0.25,
        help="X-MoE auxiliary multiclass loss weight for each expert.",
    )
    parser.add_argument(
        "--scam-specialist-loss-weight",
        type=float,
        default=0.5,
        help="X-MoE GraphSAGE Scamming-versus-rest specialist loss weight.",
    )
    parser.add_argument(
        "--x-moe-v8-task-loss-weight",
        type=float,
        default=1.0,
        help="X-MoE weight for each V8 one-versus-rest task.",
    )
    parser.add_argument(
        "--x-moe-normal-phishing-penalty",
        type=float,
        default=0.0,
        help="X-MoE penalty on Phishing probability for connected Normal nodes.",
    )
    parser.add_argument(
        "--calibrate-class-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Select Phishing and Scamming logit biases on the validation split.",
    )
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the validation-selected model state for every seed.",
    )
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save test-node labels, predictions, and class probabilities.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    seeds = [int(value.strip()) for value in args.seeds.split(",")]
    released_data = torch.load(
        args.upstream / "BLTE_multiclass_graph.pt", weights_only=False
    )
    csv_path = args.upstream / "BLTE_multiclass_v2.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in seeds:
        explanation_path = None
        if args.model in {
            "isolate_gnn",
            "isolate_gnn_v2",
            "isolate_gnn_v3",
            "isolate_gnn_v4",
            "isolate_gnn_v5",
            "isolate_gnn_v6",
            "isolate_gnn_v7",
            "isolate_gnn_v8",
            "x_moe_gnn",
        }:
            explanation_path = (
                args.output_dir
                / f"{args.protocol}_{args.model}_seed-{seed}_explanations.csv"
            )
        checkpoint_path = None
        if args.save_checkpoints:
            checkpoint_path = (
                args.output_dir
                / f"{args.protocol}_{args.model}_seed-{seed}_checkpoint.pt"
            )
        prediction_path = None
        if args.save_predictions:
            prediction_path = (
                args.output_dir
                / f"{args.protocol}_{args.model}_seed-{seed}_predictions.csv"
            )
        result = run_once(
            released_data,
            csv_path,
            args.model,
            args.protocol,
            seed,
            args.epochs,
            device,
            args.deterministic,
            explanation_path,
            checkpoint_path,
            prediction_path,
            args.normal_scam_penalty,
            args.graph_scam_uplift_penalty,
            args.base_loss,
            args.focal_gamma,
            args.calibrate_class_bias,
            args.normality_gate_alpha,
            args.normal_scamming_gate_penalty,
            args.fraud_stage_loss_weight,
            args.type_stage_loss_weight,
            args.normal_fraud_penalty,
            args.rescue_loss_weight,
            args.max_rescue,
            args.task_loss_weight,
            args.expert_loss_weight,
            args.scam_specialist_loss_weight,
            args.x_moe_v8_task_loss_weight,
            args.x_moe_normal_phishing_penalty,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    frame = pd.DataFrame(results)
    stem = f"{args.protocol}_{args.model}_seeds-{'-'.join(map(str, seeds))}"
    frame.to_csv(args.output_dir / f"{stem}.csv", index=False)
    numeric = frame.select_dtypes(include=np.number)
    summary = pd.DataFrame({"mean": numeric.mean(), "std": numeric.std(ddof=1)})
    summary.to_csv(args.output_dir / f"{stem}_summary.csv")
    print("\nSummary")
    print(summary.loc[["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]])


if __name__ == "__main__":
    main()
