"""Journal-validation ablation runner for X-MoE.

This file is intentionally separate from src/reproduce.py so that the
public reproduction runner and the four-model registry remain unchanged.

Ablations
---------
A0 full
    Original public X-MoE.

A1 conflict_only
    Removes the external GraphSAGE expert. The conflict-aware expert and
    its applicable auxiliary objectives remain.

A2 graphsage_only
    Removes the conflict-aware expert. The GraphSAGE expert and its
    applicable auxiliary objectives remain.

A3 global_routing
    Replaces class-specific routing in both the internal conflict-aware
    expert and the outer X-MoE mixture with one shared routing value per
    node while preserving the A0 parameter initialization.

A4 no_isolate_mask
    Removes isolate masking from both the internal conflict-aware routing
    and the outer X-MoE expert routing.

A5 no_scam_specialist
    Full architecture without the GraphSAGE Scamming-vs-rest specialist
    loss.

A6 no_normal_phishing_penalty
    Full architecture without the connected-Normal Phishing probability
    penalty.

A7 fixed_50
    Replaces learned outer expert routing with a fixed 50:50 mixture for
    connected nodes while retaining isolate masking.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import reproduce as public_runner
from models.graphsage import GraphSAGE
from models.xmoe import (
    ConflictAwareMultitaskExpert,
    ExplainableMixtureOfExpertsGNN,
)


VARIANTS = {
    "a0_full": "Full X-MoE",
    "a1_conflict_only": "Conflict-aware expert only",
    "a2_graphsage_only": "GraphSAGE expert only",
    "a3_global_routing": "Global inner + outer routing",
    "a4_no_isolate_mask": "No isolate masking",
    "a5_no_scam_specialist": "No scam-specialist loss",
    "a6_no_normal_phishing_penalty": "No normal-phishing penalty",
    "a7_fixed_50": "Fixed 50:50 outer mixture",
}


def node_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return torch.bincount(edge_index.reshape(-1), minlength=num_nodes).float()


class GlobalRoutingXMoE(ExplainableMixtureOfExpertsGNN):
    """Use shared inner and outer routing while preserving A0 modules."""

    def _global_conflict_outputs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        expert = self.conflict_aware_expert

        feature_h = F.relu(expert.feature_fc1(x))
        feature_h = F.dropout(
            feature_h,
            p=0.5,
            training=self.training,
        )
        feature_h = expert.feature_norm(
            F.relu(expert.feature_fc2(feature_h))
        )

        graph_h = F.relu(
            expert.graph_conv1(x, edge_index)
        )
        graph_h = F.dropout(
            graph_h,
            p=0.5,
            training=self.training,
        )
        graph_h = expert.graph_norm(
            F.relu(expert.graph_conv2(graph_h, edge_index))
        )

        degree = expert.node_degree(
            edge_index,
            x.shape[0],
        )

        connected = (
            (degree > 0)
            .to(dtype=x.dtype)
            .unsqueeze(1)
        )

        gate_input = torch.cat(
            [
                feature_h,
                graph_h,
                torch.log1p(degree).unsqueeze(1),
            ],
            dim=1,
        )

        # A3 ablation:
        # keep the A0 gate modules and initialization, but collapse the
        # three class-specific routing logits into one shared routing
        # decision for each node.
        gate_hidden = F.relu(
            expert.gate_fc1(gate_input)
        )

        gate_logits = expert.gate_fc2(
            gate_hidden
        )

        scalar_gate = (
            torch.sigmoid(
                gate_logits.mean(
                    dim=1,
                    keepdim=True,
                )
            )
            * connected
        )

        class_gate = scalar_gate.expand(
            -1,
            gate_logits.shape[1],
        )

        feature_logits = expert.feature_heads(
            feature_h
        )

        graph_logits = expert.graph_heads(
            graph_h
        )

        logits = (
            feature_logits
            + class_gate
            * (graph_logits - feature_logits)
        )

        expert._last_class_gate = class_gate.detach()
        expert._last_degree = degree.detach()

        return logits, class_gate, degree

    def mix_experts(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        (
            conflict_aware_logits,
            _,
            degree,
        ) = self._global_conflict_outputs(
            x,
            edge_index,
        )

        graphsage_logits = self.graphsage_expert(
            x,
            edge_index,
        )

        connected = (
            (degree > 0)
            .to(dtype=x.dtype)
            .unsqueeze(1)
        )

        gate_input = torch.cat(
            [
                conflict_aware_logits,
                graphsage_logits,
                (
                    conflict_aware_logits
                    - graphsage_logits
                ).abs(),
                torch.log1p(degree).unsqueeze(1),
            ],
            dim=1,
        )

        gate_hidden = F.relu(
            self.expert_gate_fc1(gate_input)
        )

        outer_gate_logits = self.expert_gate_fc2(
            gate_hidden
        )

        # One shared outer routing value for all classes.
        scalar_weight = (
            torch.sigmoid(
                outer_gate_logits.mean(
                    dim=1,
                    keepdim=True,
                )
            )
            * connected
        )

        graphsage_weight = scalar_weight.expand(
            -1,
            conflict_aware_logits.shape[1],
        )

        logits = (
            conflict_aware_logits
            + graphsage_weight
            * (
                graphsage_logits
                - conflict_aware_logits
            )
        )

        return (
            logits,
            conflict_aware_logits,
            graphsage_logits,
            graphsage_weight,
            degree,
        )


class NoIsolateMaskXMoE(ExplainableMixtureOfExpertsGNN):
    """Remove isolate masking while preserving the A0 modules."""

    def _no_mask_conflict_outputs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        expert = self.conflict_aware_expert

        feature_h = F.relu(expert.feature_fc1(x))
        feature_h = F.dropout(
            feature_h,
            p=0.5,
            training=self.training,
        )
        feature_h = expert.feature_norm(
            F.relu(expert.feature_fc2(feature_h))
        )

        graph_h = F.relu(
            expert.graph_conv1(x, edge_index)
        )
        graph_h = F.dropout(
            graph_h,
            p=0.5,
            training=self.training,
        )
        graph_h = expert.graph_norm(
            F.relu(expert.graph_conv2(graph_h, edge_index))
        )

        degree = expert.node_degree(
            edge_index,
            x.shape[0],
        )

        gate_input = torch.cat(
            [
                feature_h,
                graph_h,
                torch.log1p(degree).unsqueeze(1),
            ],
            dim=1,
        )

        # A4 ablation:
        # identical A0 inner routing except that isolate masking is removed.
        class_gate = torch.sigmoid(
            expert.gate_fc2(
                F.relu(
                    expert.gate_fc1(gate_input)
                )
            )
        )

        feature_logits = expert.feature_heads(
            feature_h
        )

        graph_logits = expert.graph_heads(
            graph_h
        )

        logits = (
            feature_logits
            + class_gate
            * (graph_logits - feature_logits)
        )

        expert._last_class_gate = class_gate.detach()
        expert._last_degree = degree.detach()

        return logits, class_gate, degree

    def mix_experts(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        (
            conflict_aware_logits,
            _,
            degree,
        ) = self._no_mask_conflict_outputs(
            x,
            edge_index,
        )

        graphsage_logits = self.graphsage_expert(
            x,
            edge_index,
        )

        gate_input = torch.cat(
            [
                conflict_aware_logits,
                graphsage_logits,
                (
                    conflict_aware_logits
                    - graphsage_logits
                ).abs(),
                torch.log1p(degree).unsqueeze(1),
            ],
            dim=1,
        )

        # A4 ablation:
        # identical A0 outer routing except that isolate masking is removed.
        graphsage_weight = torch.sigmoid(
            self.expert_gate_fc2(
                F.relu(
                    self.expert_gate_fc1(
                        gate_input
                    )
                )
            )
        )

        logits = (
            conflict_aware_logits
            + graphsage_weight
            * (
                graphsage_logits
                - conflict_aware_logits
            )
        )

        return (
            logits,
            conflict_aware_logits,
            graphsage_logits,
            graphsage_weight,
            degree,
        )


class FixedFiftyXMoE(ExplainableMixtureOfExpertsGNN):
    """Use fixed 50:50 outer mixing on connected nodes."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(input_dim, output_dim)

        # These learned routing layers are not active in this ablation.
        del self.expert_gate_fc1
        del self.expert_gate_fc2

    def mix_experts(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        conflict_aware_logits = self.conflict_aware_expert(
            x,
            edge_index,
        )

        graphsage_logits = self.graphsage_expert(
            x,
            edge_index,
        )

        degree = self.conflict_aware_expert.node_degree(
            edge_index,
            x.shape[0],
        )

        connected = (
            (degree > 0)
            .to(dtype=x.dtype)
            .unsqueeze(1)
        )

        graphsage_weight = (
            0.5
            * connected.expand(
                -1,
                conflict_aware_logits.shape[1],
            )
        )

        logits = (
            conflict_aware_logits
            + graphsage_weight
            * (graphsage_logits - conflict_aware_logits)
        )

        return (
            logits,
            conflict_aware_logits,
            graphsage_logits,
            graphsage_weight,
            degree,
        )


def build_model(
    variant: str,
    input_dim: int,
    output_dim: int,
) -> nn.Module:
    if variant in {
        "a0_full",
        "a5_no_scam_specialist",
        "a6_no_normal_phishing_penalty",
    }:
        return ExplainableMixtureOfExpertsGNN(
            input_dim,
            output_dim,
        )

    if variant == "a1_conflict_only":
        return ConflictAwareMultitaskExpert(
            input_dim,
            output_dim,
        )

    if variant == "a2_graphsage_only":
        return GraphSAGE(
            input_dim,
            output_dim,
        )

    if variant == "a3_global_routing":
        return GlobalRoutingXMoE(
            input_dim,
            output_dim,
        )

    if variant == "a4_no_isolate_mask":
        return NoIsolateMaskXMoE(
            input_dim,
            output_dim,
        )

    if variant == "a7_fixed_50":
        return FixedFiftyXMoE(
            input_dim,
            output_dim,
        )

    raise ValueError(f"Unknown ablation variant: {variant}")


def get_training_outputs(
    model: nn.Module,
    variant: str,
    x: torch.Tensor,
    edge_index: torch.Tensor,
):
    if variant == "a1_conflict_only":
        conflict_logits, _ = model.training_outputs(
            x,
            edge_index,
        )
        return (
            conflict_logits,
            conflict_logits,
            None,
        )

    if variant == "a2_graphsage_only":
        graphsage_logits = model(
            x,
            edge_index,
        )
        return (
            graphsage_logits,
            None,
            graphsage_logits,
        )

    logits, conflict_logits, graphsage_logits = (
        model.training_outputs(
            x,
            edge_index,
        )
    )

    return (
        logits,
        conflict_logits,
        graphsage_logits,
    )


def run_variant_once(
    released_data,
    csv_path: Path,
    variant: str,
    protocol: str,
    seed: int,
    epochs: int,
    device: torch.device,
    deterministic: bool,
    expert_loss_weight: float,
    scam_specialist_loss_weight: float,
    expert_task_loss_weight: float,
    normal_phishing_penalty: float,
) -> dict[str, object]:

    public_runner.set_seed(
        seed,
        deterministic,
    )

    tensors = (
        public_runner.artifact_data(released_data)
        if protocol == "artifact"
        else public_runner.clean_data(
            released_data,
            csv_path,
            seed,
        )
    )

    (
        x,
        edge_index,
        y,
        train_mask,
        val_mask,
        test_mask,
    ) = [
        tensor.to(device)
        for tensor in tensors
    ]

    model = build_model(
        variant,
        x.shape[1],
        int(y.max().item()) + 1,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=5e-4,
    )

    counts = torch.bincount(
        y[train_mask],
        minlength=3,
    )

    class_weights = counts.sum() / counts
    class_weights = (
        class_weights
        / class_weights.sum()
    )

    best_val_f1 = -1.0
    best_state = None

    started = time.perf_counter()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(
            set_to_none=True,
        )

        (
            logits,
            conflict_logits,
            graphsage_logits,
        ) = get_training_outputs(
            model,
            variant,
            x,
            edge_index,
        )

        training_labels = y[train_mask]

        base_loss = F.cross_entropy(
            logits[train_mask],
            training_labels,
            weight=class_weights,
        )

        task_losses = [base_loss]

        # Conflict-aware expert auxiliary supervision.
        # In A1, conflict_logits are also the final logits used by
        # base_loss. Repeating the expert CE would duplicate the same
        # multiclass objective and change its effective weighting.
        if (
            conflict_logits is not None
            and variant != "a1_conflict_only"
        ):
            task_losses.append(
                expert_loss_weight
                * F.cross_entropy(
                    conflict_logits[train_mask],
                    training_labels,
                    weight=class_weights,
                )
            )

        # GraphSAGE expert auxiliary supervision.
        # In A2, graphsage_logits are also the final logits used by
        # base_loss. Repeating the expert CE would duplicate the same
        # multiclass objective and change its effective weighting.
        if (
            graphsage_logits is not None
            and variant != "a2_graphsage_only"
        ):
            task_losses.append(
                expert_loss_weight
                * F.cross_entropy(
                    graphsage_logits[train_mask],
                    training_labels,
                    weight=class_weights,
                )
            )

        # Conflict-aware one-vs-rest tasks.
        if conflict_logits is not None:
            for class_id in range(3):
                binary_labels = (
                    training_labels == class_id
                ).to(
                    dtype=conflict_logits.dtype
                )

                task_losses.append(
                    expert_task_loss_weight
                    * public_runner.balanced_binary_loss(
                        conflict_logits[
                            train_mask,
                            class_id : class_id + 1,
                        ],
                        binary_labels,
                    )
                )

        # GraphSAGE scam-specialist objective.
        use_scam_specialist = (
            variant != "a5_no_scam_specialist"
            and graphsage_logits is not None
            and scam_specialist_loss_weight > 0
        )

        if use_scam_specialist:
            scamming_labels = (
                training_labels == 2
            ).to(
                dtype=graphsage_logits.dtype
            )

            task_losses.append(
                scam_specialist_loss_weight
                * public_runner.balanced_binary_loss(
                    graphsage_logits[
                        train_mask,
                        2:3,
                    ],
                    scamming_labels,
                )
            )

        # Connected-Normal Phishing penalty.
        use_normal_phishing_penalty = (
            variant
            != "a6_no_normal_phishing_penalty"
            and normal_phishing_penalty > 0
        )

        if use_normal_phishing_penalty:
            degree = node_degree(
                edge_index,
                x.shape[0],
            ).to(device)

            connected_normal = (
                train_mask
                & (y == 0)
                & (degree > 0)
            )

            if bool(connected_normal.any()):
                normal_phishing_probability = (
                    F.softmax(
                        logits[connected_normal],
                        dim=1,
                    )[:, 1].mean()
                )

                task_losses.append(
                    normal_phishing_penalty
                    * normal_phishing_probability
                )

        public_runner.pcgrad_backward(
            task_losses,
            [
                parameter
                for parameter in model.parameters()
            ],
        )

        optimizer.step()

        model.eval()

        with torch.no_grad():
            val_logits = model(
                x,
                edge_index,
            )[val_mask]

            val_predicted = (
                val_logits.argmax(dim=1)
                .cpu()
                .numpy()
            )

            val_true = (
                y[val_mask]
                .cpu()
                .numpy()
            )

        val_f1 = public_runner.f1_score(
            val_true,
            val_predicted,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "No validation-selected state was created."
        )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_logits = model(
            x,
            edge_index,
        )[test_mask]

        probabilities = (
            F.softmax(
                test_logits,
                dim=1,
            )
            .cpu()
            .numpy()
        )

        y_true = (
            y[test_mask]
            .cpu()
            .numpy()
        )

    result = public_runner.score(
        y_true,
        probabilities,
    )

    degree = node_degree(
        edge_index,
        x.shape[0],
    )

    test_degree = (
        degree[test_mask]
        .detach()
        .cpu()
    )

    result.update(
        {
            "variant_id": variant,
            "variant": VARIANTS[variant],
            "protocol": protocol,
            "seed": seed,
            "epochs": epochs,
            "device": str(device),
            "best_validation_macro_f1": best_val_f1,
            "training_seconds": (
                time.perf_counter()
                - started
            ),
            "parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "train_nodes": int(
                train_mask.sum()
            ),
            "validation_nodes": int(
                val_mask.sum()
            ),
            "test_nodes": int(
                test_mask.sum()
            ),
            "test_connected_nodes": int(
                (test_degree > 0).sum()
            ),
            "test_isolated_nodes": int(
                (test_degree == 0).sum()
            ),
            "expert_loss_weight": (
                expert_loss_weight
            ),
            "scam_specialist_loss_weight": (
                0.0
                if variant
                == "a5_no_scam_specialist"
                else scam_specialist_loss_weight
            ),
            "expert_task_loss_weight": (
                expert_task_loss_weight
                if conflict_logits is not None
                else 0.0
            ),
            "normal_phishing_penalty": (
                0.0
                if variant
                == "a6_no_normal_phishing_penalty"
                else normal_phishing_penalty
            ),
        }
    )

    return result


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        choices=["all", *VARIANTS.keys()],
        default="all",
    )

    parser.add_argument(
        "--protocol",
        choices=["artifact", "clean"],
        default="clean",
    )

    parser.add_argument(
        "--seeds",
        default="100",
        help="Comma-separated integer seeds.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
    )

    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path("upstream-paper"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/phase1_ablation"
        ),
    )

    parser.add_argument(
        "--expert-loss-weight",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--scam-specialist-loss-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--expert-task-loss-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--normal-phishing-penalty",
        type=float,
        default=0.30,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(
            args.device
        )

    seeds = [
        int(value.strip())
        for value in args.seeds.split(",")
    ]

    released_data = torch.load(
        args.upstream
        / "BLTE_multiclass_graph.pt",
        weights_only=False,
    )

    csv_path = (
        args.upstream
        / "BLTE_multiclass_v2.csv"
    )

    selected_variants = (
        list(VARIANTS.keys())
        if args.variant == "all"
        else [args.variant]
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for variant in selected_variants:
        print()
        print("=" * 72)
        print(
            f"{variant}: {VARIANTS[variant]}"
        )
        print("=" * 72)

        for seed in seeds:
            result = run_variant_once(
                released_data=released_data,
                csv_path=csv_path,
                variant=variant,
                protocol=args.protocol,
                seed=seed,
                epochs=args.epochs,
                device=device,
                deterministic=args.deterministic,
                expert_loss_weight=(
                    args.expert_loss_weight
                ),
                scam_specialist_loss_weight=(
                    args.scam_specialist_loss_weight
                ),
                expert_task_loss_weight=(
                    args.expert_task_loss_weight
                ),
                normal_phishing_penalty=(
                    args.normal_phishing_penalty
                ),
            )

            rows.append(result)

            print(
                json.dumps(
                    result,
                    sort_keys=True,
                ),
                flush=True,
            )

    frame = pd.DataFrame(rows)

    seed_stem = "-".join(
        map(str, seeds)
    )

    stem = (
        f"{args.protocol}_"
        f"{args.variant}_"
        f"seeds-{seed_stem}_"
        f"epochs-{args.epochs}"
    )

    raw_path = (
        args.output_dir
        / f"{stem}.csv"
    )

    frame.to_csv(
        raw_path,
        index=False,
    )

    summary_metrics = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "normal_precision",
        "normal_recall",
        "normal_f1",
        "phishing_precision",
        "phishing_recall",
        "phishing_f1",
        "phishing_ap",
        "scamming_precision",
        "scamming_recall",
        "scamming_f1",
        "scamming_ap",
        "best_validation_macro_f1",
        "training_seconds",
        "parameter_count",
    ]

    summary = (
        frame.groupby(
            ["variant_id", "variant"]
        )[summary_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(
            [
                str(value)
                for value in column
                if str(value)
            ]
        )
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    summary_path = (
        args.output_dir
        / f"{stem}_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 72)
    print("ABLATION SUMMARY")
    print("=" * 72)

    display = (
        frame.groupby(
            ["variant_id", "variant"]
        )[
            [
                "accuracy",
                "macro_f1",
                "phishing_f1",
                "phishing_ap",
                "scamming_f1",
                "scamming_ap",
            ]
        ]
        .mean()
    )

    print(display.to_string())

    print()
    print(
        f"Raw results: {raw_path}"
    )
    print(
        f"Summary:     {summary_path}"
    )


if __name__ == "__main__":
    main()