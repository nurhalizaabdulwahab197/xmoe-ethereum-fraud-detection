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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

class_names = ["Normal", "Phishing", "Scamming"]

print("Graph loaded successfully")
print(data)
print("Device:", device)


# =========================
# 2. Define GCN model
# =========================

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc(x)

        return x


# =========================
# 3. Initialize model
# =========================

input_dim = data.num_node_features
hidden_dim = 64
output_dim = int(data.y.max().item()) + 1

model = GCN(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    dropout=0.5
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
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
best_model_path = "best_gcn_model.pt"

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
print("Final GCN Test Results")
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
    "Model": ["GCN"],
    "Accuracy": [accuracy],
    "Macro Precision": [macro_precision],
    "Macro Recall": [macro_recall],
    "Macro F1": [macro_f1],
    "Weighted F1": [weighted_f1],
    "Best Validation Macro F1": [best_val_f1]
})

results.to_csv("GCN_results.csv", index=False)

history = pd.DataFrame({
    "Epoch": np.arange(1, epochs + 1),
    "Train Loss": train_losses,
    "Validation Loss": val_losses,
    "Train Accuracy": train_accs,
    "Validation Accuracy": val_accs,
    "Train Macro F1": train_macro_f1s,
    "Validation Macro F1": val_macro_f1s
})

history.to_csv("GCN_training_history.csv", index=False)


# =========================
# 8. Plots
# =========================

plt.figure(figsize=(7, 5))
plt.plot(history["Epoch"], history["Train Loss"], label="Training Loss",color="red")
plt.plot(history["Epoch"], history["Validation Loss"], label="Validation Loss",color="green")
plt.title("GCN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("Fig_GCN_Loss.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(history["Epoch"], history["Train Accuracy"], label="Training Accuracy",color="red")
plt.plot(history["Epoch"], history["Validation Accuracy"], label="Validation Accuracy",color="green")
plt.title("GCN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("Fig_GCN_Accuracy.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(history["Epoch"], history["Train Macro F1"], label="Training Macro F1",color="red")
plt.plot(history["Epoch"], history["Validation Macro F1"], label="Validation Macro F1",color="green")
plt.title("GCN Training and Validation Macro F1")
plt.xlabel("Epoch")
plt.ylabel("Macro F1-score")
plt.legend()
plt.tight_layout()
plt.savefig("Fig_GCN_Macro_F1.png", dpi=300)
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("GCN Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.savefig("Fig_GCN_Confusion_Matrix.png", dpi=300)
plt.show()


# =========================
# 9. GNNExplainer
# =========================

with torch.no_grad():
    out_all = model(data.x, data.edge_index)
    pred_all = out_all.argmax(dim=1)

test_nodes = torch.where(data.test_mask)[0]

correct_test_nodes = test_nodes[
    pred_all[test_nodes] == data.y[test_nodes]
]

selected_node = None
selected_class = None

for target_class in [2, 1, 0]:
    candidates = correct_test_nodes[
        data.y[correct_test_nodes] == target_class
    ]

    if len(candidates) > 0:
        selected_node = candidates[0].item()
        selected_class = target_class
        break

print("\nSelected node for GNNExplainer:", selected_node)
print("True class:", class_names[selected_class])
print("Predicted class:", class_names[pred_all[selected_node].item()])

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=ModelConfig(
        mode="multiclass_classification",
        task_level="node",
        return_type="raw"
    )
)

explanation = explainer(
    x=data.x,
    edge_index=data.edge_index,
    index=selected_node
)

print("\nGNNExplainer completed successfully.")


# =========================
# 10. Feature importance
# =========================

feature_cols = [f"Feature_{i}" for i in range(data.num_node_features)]

feature_importance = explanation.node_mask[selected_node].detach().cpu().numpy()

feature_importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 important features:")
print(feature_importance_df.head(10))

feature_importance_df.to_csv(
    "GCN_GNNExplainer_feature_importance.csv",
    index=False
)

top_features = feature_importance_df.head(10)

plt.figure(figsize=(9, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.gca().invert_yaxis()
plt.title("GNNExplainer Top 10 Features - GCN")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("Fig_GCN_GNNExplainer_Top_Features.png", dpi=300)
plt.show()


# =========================
# 11. Edge importance
# =========================

edge_mask = explanation.edge_mask.detach().cpu().numpy()

edge_importance_df = pd.DataFrame({
    "Source_Node": data.edge_index[0].detach().cpu().numpy(),
    "Target_Node": data.edge_index[1].detach().cpu().numpy(),
    "Importance": edge_mask
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 important edges:")
print(edge_importance_df.head(10))

edge_importance_df.to_csv(
    "GCN_GNNExplainer_edge_importance.csv",
    index=False
)


# =========================
# 12. Explanation subgraph
# =========================

num_hops = 2

subset, sub_edge_index, mapping, edge_mask_sub = k_hop_subgraph(
    selected_node,
    num_hops,
    data.edge_index,
    relabel_nodes=True
)

sub_data = Data(
    x=data.x[subset],
    edge_index=sub_edge_index,
    y=data.y[subset]
)

G = to_networkx(
    sub_data,
    to_undirected=True
)

plt.figure(figsize=(8, 6))

pos = nx.spring_layout(G, seed=42)

node_labels = {
    i: class_names[int(sub_data.y[i].item())]
    for i in range(sub_data.num_nodes)
}

nx.draw_networkx_nodes(G, pos, node_size=450)
nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.6)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

plt.title(
    f"GNNExplainer 2-Hop Explanation Subgraph - GCN\n"
    f"Explained Node Class: {class_names[selected_class]}"
)

plt.axis("off")
plt.tight_layout()
plt.savefig("Fig_GCN_GNNExplainer_Subgraph.png", dpi=300)
plt.show()


# =========================
# Publication-quality GNNExplainer subgraph
# =========================

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

num_hops = 2

subset, sub_edge_index, mapping, edge_mask_sub = k_hop_subgraph(
    selected_node,
    num_hops,
    data.edge_index,
    relabel_nodes=True
)

sub_data = Data(
    x=data.x[subset],
    edge_index=sub_edge_index,
    y=data.y[subset]
)

G = to_networkx(
    sub_data,
    to_undirected=True
)

# =========================
# Node colors
# =========================

class_color_map = {
    0: "#8FD694",   # Normal
    1: "#4A90E2",   # Phishing
    2: "#E74C3C"    # Scamming
}

node_colors = []
node_sizes = []

for i in range(sub_data.num_nodes):
    if i == int(mapping):
        node_colors.append("#FFD700")
        node_sizes.append(1300)
    else:
        label = int(sub_data.y[i].item())
        node_colors.append(class_color_map[label])
        node_sizes.append(650)

# =========================
# Layout
# =========================

pos = nx.spring_layout(
    G,
    k=0.9,
    iterations=300,
    seed=42
)

# =========================
# Edge importance
# =========================

edge_list = list(G.edges())

if edge_mask_sub is not None:
    edge_weights = edge_mask_sub.detach().cpu().numpy().astype(float)
else:
    edge_weights = np.ones(len(edge_list), dtype=float)

if len(edge_weights) > 0:
    edge_weights = (edge_weights - edge_weights.min()) / (
        edge_weights.max() - edge_weights.min() + 1e-8
    )
else:
    edge_weights = np.ones(len(edge_list), dtype=float)

if len(edge_weights) != len(edge_list):
    edge_weights = np.ones(len(edge_list), dtype=float) * 0.5

edge_widths = 1.0 + 4.0 * edge_weights
edge_alphas = 0.25 + 0.65 * edge_weights

# =========================
# Labels
# =========================

node_labels = {}

for i in range(sub_data.num_nodes):
    label = int(sub_data.y[i].item())

    if i == int(mapping):
        node_labels[i] = f"Target\n{class_names[label]}"

    elif label != 0:
        node_labels[i] = class_names[label]

# =========================
# Plot
# =========================

plt.figure(figsize=(13, 10))

for idx, edge in enumerate(edge_list):
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[edge],
        width=edge_widths[idx],
        alpha=edge_alphas[idx],
        edge_color="#555555",
        arrows=True,
        arrowsize=12
    )

nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=node_sizes,
    edgecolors="black",
    linewidths=1.0
)

nx.draw_networkx_labels(
    G,
    pos,
    labels=node_labels,
    font_size=9,
    font_weight="bold"
)

legend_elements = [
    Line2D([0], [0], marker="o", color="w", label="Normal",
           markerfacecolor="#8FD694", markeredgecolor="black", markersize=12),
    Line2D([0], [0], marker="o", color="w", label="Phishing",
           markerfacecolor="#4A90E2", markeredgecolor="black", markersize=12),
    Line2D([0], [0], marker="o", color="w", label="Scamming",
           markerfacecolor="#E74C3C", markeredgecolor="black", markersize=12),
    Line2D([0], [0], marker="o", color="w", label="Explained node",
           markerfacecolor="#FFD700", markeredgecolor="black", markersize=12),
    Line2D([0], [0], color="#555555", lw=4, label="High-importance edge"),
    Line2D([0], [0], color="#555555", lw=1, alpha=0.4, label="Low-importance edge")
]

plt.legend(
    handles=legend_elements,
    loc="upper left",
    frameon=True,
    fontsize=10
)

plt.title(
    "GNNExplainer-Based Local Explanation for GCN Prediction\n"
    f"Explained Node Class: {class_names[selected_class]}",
    fontsize=15,
    fontweight="bold"
)

plt.axis("off")
plt.tight_layout()

plt.savefig(
    "Fig_GCN_GNNExplainer_Publication_Quality_Subgraph.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "Fig_GCN_GNNExplainer_Publication_Quality_Subgraph.pdf",
    bbox_inches="tight"
)

plt.show()

print("Saved:")
print("Fig_GCN_GNNExplainer_Publication_Quality_Subgraph.png")
print("Fig_GCN_GNNExplainer_Publication_Quality_Subgraph.pdf")
print("\nAll GCN training, evaluation, plotting, and GNNExplainer completed.")