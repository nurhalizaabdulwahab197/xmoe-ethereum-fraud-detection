# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 23:15:12 2026

@author: Salam
"""

# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

import networkx as nx

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================
# 1. Load graph
# =========================

data = torch.load("BLTE_multiclass_graph.pt", weights_only=False)


# Audit addition: use the prevalidated reconstructed partition.
# AUDIT_SPLIT_FILE is supplied by the audit executor.
with np.load(AUDIT_SPLIT_FILE, allow_pickle=False) as audit_partition:
    for audit_key, audit_attribute in (
        ("train", "train_mask"),
        ("validation", "val_mask"),
        ("test", "test_mask"),
    ):
        audit_indices = torch.as_tensor(
            audit_partition[audit_key].copy(), dtype=torch.long,
            device=data.y.device
        )
        audit_mask = torch.zeros(
            data.num_nodes, dtype=torch.bool, device=data.y.device
        )
        audit_mask[audit_indices] = True
        setattr(data, audit_attribute, audit_mask)
# Original model, training and evaluation code continues below.


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

class_names = ["Normal", "Phishing", "Scamming"]

print("Graph loaded successfully")
print(data)
print("Device:", device)


# =========================
# 2. Define GCN model
# =========================

import sys
from pathlib import Path as AuditPath
sys.path.insert(0, str(AuditPath(__file__).resolve().parents[1] / 'model_source'))
from models import ExplainableMixtureOfExpertsGNN as AuditModel


# =========================
# 3. Initialize model
# =========================

input_dim = data.num_node_features
hidden_dim = 64
output_dim = int(data.y.max().item()) + 1

model = AuditModel(input_dim, output_dim).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=5e-4
)

class_counts = torch.bincount(data.y[data.train_mask])
class_weights = class_counts.sum() / class_counts
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)

print("\nClass counts:", class_counts.cpu().numpy())
print("Class weights:", class_weights.cpu().numpy())


# =========================
# 4. Training history
# =========================

train_losses = []
val_losses = []
train_accs = []
val_accs = []
train_macro_f1s = []
val_macro_f1s = []


# =========================
# 5. Training loop
# =========================

epochs = 300
best_val_f1 = 0
best_model_path = 'best_xmoe_ce_model.pt'

for epoch in range(1, epochs + 1):

    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    loss = F.cross_entropy(
        out[data.train_mask],
        data.y[data.train_mask],
        weight=class_weights
    )

    loss.backward()
    optimizer.step()

    model.eval()

    with torch.no_grad():

        out_eval = model(data.x, data.edge_index)

        train_loss = F.cross_entropy(
            out_eval[data.train_mask],
            data.y[data.train_mask],
            weight=class_weights
        )

        val_loss = F.cross_entropy(
            out_eval[data.val_mask],
            data.y[data.val_mask],
            weight=class_weights
        )

        train_pred = out_eval[data.train_mask].argmax(dim=1)
        val_pred = out_eval[data.val_mask].argmax(dim=1)

        train_true = data.y[data.train_mask]
        val_true = data.y[data.val_mask]

        train_acc = accuracy_score(
            train_true.cpu().numpy(),
            train_pred.cpu().numpy()
        )

        val_acc = accuracy_score(
            val_true.cpu().numpy(),
            val_pred.cpu().numpy()
        )

        train_macro_f1 = f1_score(
            train_true.cpu().numpy(),
            train_pred.cpu().numpy(),
            average="macro",
            zero_division=0
        )

        val_macro_f1 = f1_score(
            val_true.cpu().numpy(),
            val_pred.cpu().numpy(),
            average="macro",
            zero_division=0
        )

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_macro_f1s.append(train_macro_f1)
    val_macro_f1s.append(val_macro_f1)

    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        torch.save(model.state_dict(), best_model_path)

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Train Macro-F1: {train_macro_f1:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f}"
        )


# =========================
# 6. Test evaluation
# =========================

model.load_state_dict(torch.load(best_model_path))
model.eval()

with torch.no_grad():

    out = model(data.x, data.edge_index)

    prob = F.softmax(out[data.test_mask], dim=1)

    y_score = prob.cpu().numpy()
    y_pred = prob.argmax(dim=1).cpu().numpy()
    y_true = data.y[data.test_mask].cpu().numpy()

accuracy = accuracy_score(y_true, y_pred)
macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n==============================")
print('Final XMoE_CE Test Results')
print("==============================")
print("Accuracy:", round(accuracy, 4))
print("Macro Precision:", round(macro_precision, 4))
print("Macro Recall:", round(macro_recall, 4))
print("Macro F1-score:", round(macro_f1, 4))
print("Weighted F1-score:", round(weighted_f1, 4))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
))

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)


# =========================
# 7. Save results
# =========================

results = pd.DataFrame({
    "Model": ['XMoE_CE'],
    "Accuracy": [accuracy],
    "Macro Precision": [macro_precision],
    "Macro Recall": [macro_recall],
    "Macro F1": [macro_f1],
    "Weighted F1": [weighted_f1],
    "Best Validation Macro F1": [best_val_f1]
})

results.to_csv('XMoE_CE_results.csv', index=False)

history = pd.DataFrame({
    "Epoch": np.arange(1, epochs + 1),
    "Train Loss": train_losses,
    "Validation Loss": val_losses,
    "Train Accuracy": train_accs,
    "Validation Accuracy": val_accs,
    "Train Macro F1": train_macro_f1s,
    "Validation Macro F1": val_macro_f1s
})

history.to_csv('XMoE_CE_training_history.csv', index=False)

