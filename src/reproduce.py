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
from models import GAT as ModularGAT, GCN as ModularGCN, GraphSAGE as ModularGraphSAGE, MLP as ModularMLP, ExplainableMixtureOfExpertsGNN as ModularXMoE


NON_FEATURE_COLUMNS = ["Address", "Scam", "attack", "attack_encoded"]



MODELS = {
    "gcn": ModularGCN,
    "gat": ModularGAT,
    "graphsage": ModularGraphSAGE,
    "mlp": ModularMLP,
    "x_moe_gnn": ModularXMoE,
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
    base_loss: str = "weighted_ce",
    focal_gamma: float = 2.0,
    calibrate_class_bias: bool = False,
    expert_loss_weight: float = 0.25,
    scam_specialist_loss_weight: float = 0.5,
    x_moe_expert_task_loss_weight: float = 1.0,
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
    model = MODELS[model_name](
        x.shape[1], int(y.max().item()) + 1
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
        x_moe_expert_logits = None
        if isinstance(model, ModularXMoE):
            logits, conflict_aware_logits, graphsage_logits = model.training_outputs(
                x, edge_index
            )
            x_moe_expert_logits = (conflict_aware_logits, graphsage_logits)
        else:
            logits = model(x, edge_index)
        if base_loss == "weighted_focal":
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
        if model_name == "x_moe_gnn" and x_moe_expert_logits is not None:
            conflict_aware_logits, graphsage_logits = x_moe_expert_logits
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
                    for expert_logits in (conflict_aware_logits, graphsage_logits)
                ]
            )
            for class_id in range(3):
                binary_labels = (training_labels == class_id).to(
                    dtype=conflict_aware_logits.dtype
                )
                task_losses.append(
                    x_moe_expert_task_loss_weight
                    * balanced_binary_loss(
                        conflict_aware_logits[train_mask, class_id : class_id + 1],
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
                degree = model.conflict_aware_expert.node_degree(edge_index, x.shape[0])
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
        if isinstance(model, ModularXMoE):
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
    if isinstance(model, ModularXMoE):
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
            if "normal_graphsage_expert_weight" in test_explanation:
                for class_name in ("normal", "phishing", "scamming"):
                    explanation_frame[f"{class_name}_graphsage_expert_weight"] = (
                        test_explanation[
                            f"{class_name}_graphsage_expert_weight"
                        ].numpy()
                    )
                    explanation_frame[f"{class_name}_conflict_aware_expert_weight"] = (
                        test_explanation[f"{class_name}_conflict_aware_expert_weight"].numpy()
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
            "base_loss": base_loss,
            "focal_gamma": focal_gamma,
            "calibrate_class_bias": calibrate_class_bias,
            "normal_class_bias": float(class_bias[0].cpu()),
            "phishing_class_bias": float(class_bias[1].cpu()),
            "scamming_class_bias": float(class_bias[2].cpu()),
            **(
                {
                    "expert_loss_weight": expert_loss_weight,
                    "scam_specialist_loss_weight": scam_specialist_loss_weight,
                    "x_moe_expert_task_loss_weight": x_moe_expert_task_loss_weight,
                    "x_moe_normal_phishing_penalty": x_moe_normal_phishing_penalty,
                }
                if model_name == "x_moe_gnn"
                else {}
            ),
        }
    )
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        feature_names = pd.read_csv(csv_path, nrows=0).drop(
            columns=NON_FEATURE_COLUMNS
        ).columns.tolist()
        torch.save(
            {
                "format_version": 2,
                "model_version": {
                    "gcn": "gcn",
                    "gat": "gat",
                    "graphsage": "graphsage",
                    "x_moe_gnn": "x-moe-gnn",
                }[model_name],
                "model_name": model_name,
                "protocol": protocol,
                "seed": seed,
                "epochs": epochs,
                "input_dim": x.shape[1],
                "output_dim": int(y.max().item()) + 1,
                "feature_names": feature_names,
                "class_names": ["Normal", "Phishing", "Scamming"],
                **(
                    {
                        "expert_loss_weight": expert_loss_weight,
                        "scam_specialist_loss_weight": scam_specialist_loss_weight,
                        "x_moe_expert_task_loss_weight": x_moe_expert_task_loss_weight,
                        "x_moe_normal_phishing_penalty": x_moe_normal_phishing_penalty,
                    }
                    if model_name == "x_moe_gnn"
                    else {}
                ),
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
        "--base-loss",
        choices=["weighted_ce", "weighted_focal"],
        default="weighted_ce",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
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
        "--x-moe-expert-task-loss-weight",
        type=float,
        default=1.0,
        help="X-MoE weight for each conflict-aware expert one-versus-rest task.",
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
        if args.model == "x_moe_gnn":
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
            args.base_loss,
            args.focal_gamma,
            args.calibrate_class_bias,
            args.expert_loss_weight,
            args.scam_specialist_loss_weight,
            args.x_moe_expert_task_loss_weight,
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



