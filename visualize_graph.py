"""
Visualize the ibm01 heterogeneous netlist graph.
Pick the option that suits your needs (see comments).

Install: pip install matplotlib networkx plotly pandas
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from plc_to_hetero_graph import build_from_ibm01   # reuse your existing function

data = build_from_ibm01("ibm01.modified.txt")

cell_xy = data["cell"].x.numpy()          # shape (N_cells, 2)
net_deg  = data["net"].x.numpy().ravel()  # shape (N_nets,)
c2n = data["cell", "drives",      "net"].edge_index.numpy()
n2c = data["net",  "fans_out_to", "cell"].edge_index.numpy()

# ─────────────────────────────────────────────────────────────────────────────
# OPTION 1 — Spatial scatter plot of cell nodes (fast, always works)
#   Shows where cells sit on the 64×64 routing grid, colored by how many
#   nets they appear in (degree).
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatial(cell_xy, c2n, n2c):
    # Count per-cell degree (how many nets touch each cell)
    num_cells = len(cell_xy)
    degree = torch.zeros(num_cells)
    degree.scatter_add_(0, torch.tensor(c2n[0]), torch.ones(c2n.shape[1]))
    degree.scatter_add_(0, torch.tensor(n2c[1]), torch.ones(n2c.shape[1]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter of all cells
    sc = axes[0].scatter(cell_xy[:, 0], cell_xy[:, 1],
                         c=degree.numpy(), cmap="plasma", s=6, alpha=0.7)
    plt.colorbar(sc, ax=axes[0], label="Net degree")
    axes[0].set_title("Cell positions on 64×64 grid\n(color = net degree)")
    axes[0].set_xlabel("X"); axes[0].set_ylabel("Y")
    axes[0].set_aspect("equal")

    # Right: degree distribution histogram
    axes[1].hist(degree.numpy(), bins=40, color="steelblue", edgecolor="white")
    axes[1].set_title("Cell degree distribution")
    axes[1].set_xlabel("Degree (# nets)"); axes[1].set_ylabel("Count")
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig("spatial_view.png", dpi=150)
    plt.show()
    print("Saved: spatial_view.png")