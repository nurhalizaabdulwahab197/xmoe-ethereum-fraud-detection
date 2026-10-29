"""
STRICT-INDUCTIVE evaluation.

Scientific delta relative to the frozen clean protocol:

TRAIN
    features: training nodes only
    topology: train-train edges only

VALIDATION
    features: train + validation
    topology:
        train-train
        train-validation
        validation-train
    excluded:
        validation-validation
        all test nodes/edges

TEST
    features: train + test
    topology:
        train-train
        train-test
        test-train
    excluded:
        test-test
        all validation nodes/edges

The original direction of released edges is preserved.
No reverse edges are synthesised.
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

from sklearn.metrics import f1_score


# =========================================================
# LOAD FROZEN PUBLIC IMPLEMENTATION
# =========================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_ROOT),
)

import reproduce as base  # noqa: E402


PROTOCOL = "strict_inductive"


# =========================================================
# GRAPH VIEW CONSTRUCTION
# =========================================================

def build_strict_views(
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
):
    """
    Construct three mutually controlled graph views.

    Split IDs:
        0 = training
        1 = validation
        2 = test
    """

    num_nodes = train_mask.numel()

    membership = torch.full(
        (num_nodes,),
        -1,
        dtype=torch.long,
        device=edge_index.device,
    )

    membership[train_mask] = 0
    membership[val_mask] = 1
    membership[test_mask] = 2

    if torch.any(membership < 0):
        raise RuntimeError(
            "Split membership is incomplete."
        )

    src = edge_index[0]
    dst = edge_index[1]

    src_split = membership[src]
    dst_split = membership[dst]

    train_edge_mask = (
        (src_split == 0)
        & (dst_split == 0)
    )

    validation_edge_mask = (
        (
            (src_split == 0)
            & (dst_split == 0)
        )
        |
        (
            (src_split == 0)
            & (dst_split == 1)
        )
        |
        (
            (src_split == 1)
            & (dst_split == 0)
        )
    )

    test_edge_mask = (
        (
            (src_split == 0)
            & (dst_split == 0)
        )
        |
        (
            (src_split == 0)
            & (dst_split == 2)
        )
        |
        (
            (src_split == 2)
            & (dst_split == 0)
        )
    )

    train_edge_index = edge_index[:, train_edge_mask]
    validation_edge_index = edge_index[:, validation_edge_mask]
    test_edge_index = edge_index[:, test_edge_mask]

    def split_pairs(view):
        source = membership[view[0]]
        target = membership[view[1]]
        return source, target

    a, b = split_pairs(train_edge_index)
    if not torch.all((a == 0) & (b == 0)):
        raise RuntimeError(
            "TRAIN topology leakage detected."
        )

    a, b = split_pairs(validation_edge_index)
    if torch.any((a == 2) | (b == 2)):
        raise RuntimeError(
            "TEST node leaked into validation topology."
        )
    if torch.any((a == 1) & (b == 1)):
        raise RuntimeError(
            "Validation-validation edge leakage detected."
        )

    a, b = split_pairs(test_edge_index)
    if torch.any((a == 1) | (b == 1)):
        raise RuntimeError(
            "VALIDATION node leaked into test topology."
        )
    if torch.any((a == 2) & (b == 2)):
        raise RuntimeError(
            "Test-test edge leakage detected."
        )

    return (
        train_edge_index,
        validation_edge_index,
        test_edge_index,
    )


# =========================================================
# STAGE-SPECIFIC FEATURE VISIBILITY
# =========================================================

def build_feature_views(
    x: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
):
    """
    The scaler has already been fitted on training nodes only
    by base.clean_data().

    We additionally hide feature values for nodes that are not
    permitted at each evaluation stage.
    """

    train_x = x.clone()
    train_x[~train_mask] = 0.0

    validation_x = x.clone()
    validation_x[test_mask] = 0.0

    test_x = x.clone()
    test_x[val_mask] = 0.0

    return (
        train_x,
        validation_x,
        test_x,
    )


# =========================================================
# COUNT FORBIDDEN EDGES
# =========================================================

def leakage_counts(
    edge_index,
    train_mask,
    val_mask,
    test_mask,
):
    membership = torch.full(
        (train_mask.numel(),),
        -1,
        dtype=torch.long,
        device=edge_index.device,
    )

    membership[train_mask] = 0
    membership[val_mask] = 1
    membership[test_mask] = 2

    src = membership[edge_index[0]]
    dst = membership[edge_index[1]]

    return {
        "train_validation": int((((src == 0) & (dst == 1)) | ((src == 1) & (dst == 0))).sum().item()),
        "train_test": int((((src == 0) & (dst == 2)) | ((src == 2) & (dst == 0))).sum().item()),
        "validation_test": int((((src == 1) & (dst == 2)) | ((src == 2) & (dst == 1))).sum().item()),
        "validation_validation": int(((src == 1) & (dst == 1)).sum().item()),
        "test_test": int(((src == 2) & (dst == 2)).sum().item()),
    }


# =========================================================
# ONE STRICT-INDUCTIVE RUN
# =========================================================

def run_once(
    released_data,
    csv_path: Path,
    model_name: str,
    seed: int,
    epochs: int,
    device: torch.device,
    deterministic: bool,
    base_loss: str,
    focal_gamma: float,
    calibrate_class_bias: bool,
    expert_loss_weight: float,
    scam_specialist_loss_weight: float,
    x_moe_expert_task_loss_weight: float,
    x_moe_normal_phishing_penalty: float,
    prediction_path: Path | None,
):
    base.set_seed(seed, deterministic)

    (
        x,
        full_edge_index,
        y,
        train_mask,
        val_mask,
        test_mask,
    ) = base.clean_data(
        released_data,
        csv_path,
        seed,
    )

    tensors = [
        x,
        full_edge_index,
        y,
        train_mask,
        val_mask,
        test_mask,
    ]

    (
        x,
        full_edge_index,
        y,
        train_mask,
        val_mask,
        test_mask,
    ) = [
        tensor.to(device)
        for tensor in tensors
    ]

    (
        train_edge_index,
        validation_edge_index,
        test_edge_index,
    ) = build_strict_views(
        full_edge_index,
        train_mask,
        val_mask,
        test_mask,
    )

    (
        train_x,
        validation_x,
        test_x,
    ) = build_feature_views(
        x,
        train_mask,
        val_mask,
        test_mask,
    )

    model = base.MODELS[model_name](
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
    class_weights = class_weights / class_weights.sum()

    best_val_f1 = -1.0
    best_state = None

    started = time.perf_counter()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        x_moe_expert_logits = None

        if isinstance(model, base.ModularXMoE):
            (
                logits,
                conflict_aware_logits,
                graphsage_logits,
            ) = model.training_outputs(
                train_x,
                train_edge_index,
            )

            x_moe_expert_logits = (
                conflict_aware_logits,
                graphsage_logits,
            )
        else:
            logits = model(
                train_x,
                train_edge_index,
            )

        if base_loss == "weighted_focal":
            loss = base.weighted_focal_loss(
                logits[train_mask],
                y[train_mask],
                class_weights,
                focal_gamma,
            )
        else:
            loss = F.cross_entropy(
                logits[train_mask],
                y[train_mask],
                weight=class_weights,
            )

        if (
            model_name == "x_moe_gnn"
            and x_moe_expert_logits is not None
        ):
            (
                conflict_aware_logits,
                graphsage_logits,
            ) = x_moe_expert_logits

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
                    for expert_logits in (
                        conflict_aware_logits,
                        graphsage_logits,
                    )
                ]
            )

            for class_id in range(3):
                binary_labels = (
                    training_labels == class_id
                ).to(
                    dtype=conflict_aware_logits.dtype
                )

                task_losses.append(
                    x_moe_expert_task_loss_weight
                    * base.balanced_binary_loss(
                        conflict_aware_logits[
                            train_mask,
                            class_id:class_id + 1,
                        ],
                        binary_labels,
                    )
                )

            scamming_labels = (
                training_labels == 2
            ).to(
                dtype=graphsage_logits.dtype
            )

            task_losses.append(
                scam_specialist_loss_weight
                * base.balanced_binary_loss(
                    graphsage_logits[
                        train_mask,
                        2:3,
                    ],
                    scamming_labels,
                )
            )

            if x_moe_normal_phishing_penalty > 0:
                degree = (
                    model
                    .conflict_aware_expert
                    .node_degree(
                        train_edge_index,
                        train_x.shape[0],
                    )
                )

                connected_normal = (
                    train_mask
                    & (y == 0)
                    & (degree > 0)
                )

                if not torch.any(connected_normal):
                    raise RuntimeError(
                        "No connected Normal training nodes."
                    )

                normal_phishing_probability = (
                    F.softmax(
                        logits[connected_normal],
                        dim=1,
                    )[:, 1]
                    .mean()
                )

                task_losses.append(
                    x_moe_normal_phishing_penalty
                    * normal_phishing_probability
                )

            base.pcgrad_backward(
                task_losses,
                [
                    parameter
                    for parameter in model.parameters()
                ],
            )
        else:
            loss.backward()

        optimizer.step()

        model.eval()

        with torch.no_grad():
            val_logits = model(
                validation_x,
                validation_edge_index,
            )[val_mask]

            val_predicted = (
                val_logits
                .argmax(dim=1)
                .cpu()
                .numpy()
            )

            val_true = (
                y[val_mask]
                .cpu()
                .numpy()
            )

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

    if best_state is None:
        raise RuntimeError(
            "No validation-selected state."
        )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        class_bias = torch.zeros(
            3,
            device=device,
        )

        if calibrate_class_bias:
            validation_logits = model(
                validation_x,
                validation_edge_index,
            )[val_mask]

            class_bias = base.select_validation_class_bias(
                validation_logits,
                y[val_mask],
            )

        if isinstance(model, base.ModularXMoE):
            explanation = model.explain(
                test_x,
                test_edge_index,
            )

            full_probabilities = F.softmax(
                torch.log(
                    explanation[
                        "probabilities"
                    ].clamp_min(1e-12)
                )
                + class_bias,
                dim=1,
            )

            probabilities = (
                full_probabilities[test_mask]
                .cpu()
                .numpy()
            )
        else:
            probabilities = (
                F.softmax(
                    model(
                        test_x,
                        test_edge_index,
                    )[test_mask]
                    + class_bias,
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

    result = base.score(
        y_true,
        probabilities,
    )

    train_leakage = leakage_counts(
        train_edge_index,
        train_mask,
        val_mask,
        test_mask,
    )

    validation_leakage = leakage_counts(
        validation_edge_index,
        train_mask,
        val_mask,
        test_mask,
    )

    test_leakage = leakage_counts(
        test_edge_index,
        train_mask,
        val_mask,
        test_mask,
    )

    if any(value != 0 for value in train_leakage.values()):
        raise RuntimeError(
            f"Training leakage: {train_leakage}"
        )

    if (
        validation_leakage["train_test"] != 0
        or validation_leakage["validation_test"] != 0
        or validation_leakage["validation_validation"] != 0
        or validation_leakage["test_test"] != 0
    ):
        raise RuntimeError(
            f"Validation leakage: {validation_leakage}"
        )

    if (
        test_leakage["train_validation"] != 0
        or test_leakage["validation_test"] != 0
        or test_leakage["validation_validation"] != 0
        or test_leakage["test_test"] != 0
    ):
        raise RuntimeError(
            f"Test leakage: {test_leakage}"
        )

    result.update(
        {
            "model": model_name,
            "protocol": PROTOCOL,
            "seed": seed,
            "epochs": epochs,
            "best_validation_macro_f1": best_val_f1,
            "training_seconds": time.perf_counter() - started,
            "device": str(device),
            "train_nodes": int(train_mask.sum()),
            "validation_nodes": int(val_mask.sum()),
            "test_nodes": int(test_mask.sum()),
            "released_edges": int(full_edge_index.shape[1]),
            "train_edges": int(train_edge_index.shape[1]),
            "validation_edges": int(validation_edge_index.shape[1]),
            "test_edges": int(test_edge_index.shape[1]),
            "train_train_validation_leakage": train_leakage["train_validation"],
            "train_train_test_leakage": train_leakage["train_test"],
            "train_validation_test_leakage": train_leakage["validation_test"],
            "validation_test_node_leakage": (
                validation_leakage["train_test"]
                + validation_leakage["validation_test"]
                + validation_leakage["test_test"]
            ),
            "validation_validation_edges": validation_leakage["validation_validation"],
            "test_validation_node_leakage": (
                test_leakage["train_validation"]
                + test_leakage["validation_test"]
                + test_leakage["validation_validation"]
            ),
            "test_test_edges": test_leakage["test_test"],
            "base_loss": base_loss,
            "focal_gamma": focal_gamma,
            "calibrate_class_bias": calibrate_class_bias,
            "expert_loss_weight": (
                expert_loss_weight
                if model_name == "x_moe_gnn"
                else np.nan
            ),
            "scam_specialist_loss_weight": (
                scam_specialist_loss_weight
                if model_name == "x_moe_gnn"
                else np.nan
            ),
            "x_moe_expert_task_loss_weight": (
                x_moe_expert_task_loss_weight
                if model_name == "x_moe_gnn"
                else np.nan
            ),
            "x_moe_normal_phishing_penalty": (
                x_moe_normal_phishing_penalty
                if model_name == "x_moe_gnn"
                else np.nan
            ),
        }
    )

    if prediction_path is not None:
        prediction_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            {
                "node_index": torch.where(test_mask)[0].cpu().numpy(),
                "true_label": y_true,
                "predicted_label": probabilities.argmax(axis=1),
                "normal_probability": probabilities[:, 0],
                "phishing_probability": probabilities[:, 1],
                "scamming_probability": probabilities[:, 2],
            }
        ).to_csv(
            prediction_path,
            index=False,
        )

        result["prediction_path"] = str(
            prediction_path
        )

    return result


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=base.MODELS,
        default="x_moe_gnn",
    )

    parser.add_argument(
        "--seeds",
        default="100",
        help="Comma-separated seeds.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
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
            "results/strict_inductive"
        ),
    )

    parser.add_argument(
        "--base-loss",
        choices=[
            "weighted_ce",
            "weighted_focal",
        ],
        default="weighted_ce",
    )

    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
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
        "--x-moe-expert-task-loss-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--x-moe-normal-phishing-penalty",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--calibrate-class-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args()

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

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for seed in seeds:
        prediction_path = None

        if args.save_predictions:
            prediction_path = (
                args.output_dir
                / (
                    f"{PROTOCOL}_"
                    f"{args.model}_"
                    f"seed-{seed}_"
                    f"predictions.csv"
                )
            )

        result = run_once(
            released_data=released_data,
            csv_path=csv_path,
            model_name=args.model,
            seed=seed,
            epochs=args.epochs,
            device=device,
            deterministic=args.deterministic,
            base_loss=args.base_loss,
            focal_gamma=args.focal_gamma,
            calibrate_class_bias=args.calibrate_class_bias,
            expert_loss_weight=args.expert_loss_weight,
            scam_specialist_loss_weight=args.scam_specialist_loss_weight,
            x_moe_expert_task_loss_weight=args.x_moe_expert_task_loss_weight,
            x_moe_normal_phishing_penalty=args.x_moe_normal_phishing_penalty,
            prediction_path=prediction_path,
        )

        results.append(result)
        print(
            json.dumps(
                result,
                sort_keys=True,
            ),
            flush=True,
        )

    frame = pd.DataFrame(results)

    stem = (
        f"{PROTOCOL}_"
        f"{args.model}_"
        f"seeds-"
        f"{'-'.join(map(str, seeds))}_"
        f"epochs-{args.epochs}"
    )

    raw_path = (
        args.output_dir
        / f"{stem}.csv"
    )

    summary_path = (
        args.output_dir
        / f"{stem}_summary.csv"
    )

    frame.to_csv(
        raw_path,
        index=False,
    )

    numeric = frame.select_dtypes(
        include=np.number
    )

    summary = pd.DataFrame(
        {
            "mean": numeric.mean(),
            "std": numeric.std(ddof=1),
        }
    )

    summary.to_csv(
        summary_path
    )

    print()
    print("Summary")
    print(
        summary.loc[
            [
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
            ]
        ]
    )

    print()
    print("Saved:")
    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()
