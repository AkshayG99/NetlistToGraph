"""
Parse a .plc Protobuf netlist and build a heterogeneous graph for GNN training.

Node types:
  - 'cell'  : standard cells / macros
  - 'net'   : hyperedges (one node per net)

Edge types:
  - ('cell', 'in',  'net')  : cell drives a net (source pin)
  - ('net',  'out', 'cell') : net fans out to a cell (sink pin)

This bipartite cell↔net representation handles hyperedges naturally,
which is the standard approach for netlist GNNs (e.g., DeepPlace, CircuitGNN).

Install: pip install torch torch_geometric
"""

import re
import torch
from torch_geometric.data import HeteroData


# ── 1. Parse the .plc file ────────────────────────────────────────────────────

def parse_plc(plc_path):
    """
    Returns:
        nodes  : list of dicts with keys: name, type, width, height, x, y
        nets   : list of dicts with keys: name, pins (list of node indices)
    """
    nodes, nets = [], []
    name_to_idx = {}

    with open(plc_path) as f:
        content = f.read()

    # Each node block looks like:
    #   node {
    #     name: "..."
    #     width: 1.0  height: 1.0
    #     x: 0.0  y: 0.0
    #     type: STDCELL / MACRO / PORT
    #   }
    node_blocks = re.findall(r'node \{(.*?)\}', content, re.DOTALL)
    for blk in node_blocks:
        get = lambda key: re.search(rf'{key}:\s*"?([^\s"]+)"?', blk)
        name  = get("name").group(1)  if get("name")  else f"n{len(nodes)}"
        ntype = get("type").group(1)  if get("type")  else "STDCELL"
        w     = float(get("width").group(1))  if get("width")  else 1.0
        h     = float(get("height").group(1)) if get("height") else 1.0
        x     = float(get("x").group(1))      if get("x")      else 0.0
        y     = float(get("y").group(1))      if get("y")      else 0.0
        name_to_idx[name] = len(nodes)
        nodes.append({"name": name, "type": ntype, "w": w, "h": h, "x": x, "y": y})

    # Net blocks:
    #   net {
    #     name: "..."
    #     input_pin_names: ["node1", ...]
    #     output_pin_names: ["node2", ...]
    #   }
    net_blocks = re.findall(r'net \{(.*?)\}', content, re.DOTALL)
    for blk in net_blocks:
        name_m = re.search(r'name:\s*"([^"]+)"', blk)
        net_name = name_m.group(1) if name_m else f"net{len(nets)}"
        pins = re.findall(r'"([^"]+)"', blk)
        pin_idxs = [name_to_idx[p] for p in pins if p in name_to_idx]
        if pin_idxs:
            nets.append({"name": net_name, "pins": pin_idxs})

    return nodes, nets


# ── 2. Build HeteroData ───────────────────────────────────────────────────────

def build_hetero_graph(nodes, nets):
    """
    Build a bipartite heterogeneous graph: cell ↔ net.

    Node features for 'cell':  [width, height, x, y, is_macro]
    Node features for 'net':   [degree]  (fanout)
    """
    # Cell node features
    cell_feats = []
    for n in nodes:
        is_macro = 1.0 if "MACRO" in n["type"].upper() else 0.0
        cell_feats.append([n["w"], n["h"], n["x"], n["y"], is_macro])

    # Net node features + build edges
    net_feats = []
    cell_to_net_src, cell_to_net_dst = [], []  # cell → net (drives)
    net_to_cell_src, net_to_cell_dst = [], []  # net  → cell (fanout)

    for net_idx, net in enumerate(nets):
        degree = len(net["pins"])
        net_feats.append([float(degree)])

        pins = net["pins"]
        # Convention: first pin = driver (input), rest = sinks (output)
        if pins:
            cell_to_net_src.append(pins[0])
            cell_to_net_dst.append(net_idx)
        for sink in pins[1:]:
            net_to_cell_src.append(net_idx)
            net_to_cell_dst.append(sink)

    data = HeteroData()

    data["cell"].x = torch.tensor(cell_feats, dtype=torch.float)
    data["net"].x  = torch.tensor(net_feats,  dtype=torch.float)

    data["cell", "drives", "net"].edge_index = torch.tensor(
        [cell_to_net_src, cell_to_net_dst], dtype=torch.long
    )
    data["net", "fans_out_to", "cell"].edge_index = torch.tensor(
        [net_to_cell_src, net_to_cell_dst], dtype=torch.long
    )

    return data


# ── 3. Shortcut: build directly from ibm01.modified.txt ──────────────────────
#    (skip the .plc step entirely — useful for quick experiments)

def build_from_ibm01(filepath):
    """
    Parse ibm01.modified.txt directly and return a HeteroData graph.
    Treats each unique (x,y) grid point as a cell node.
    """
    import re

    nets_raw = []
    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'^(net\S+)\s+\d+\s+(\d+)', line)
        if m:
            num_pins = int(m.group(2))
            net_name = m.group(1)
            pins = []
            for _ in range(num_pins):
                i += 1
                px, py = map(int, lines[i].strip().split())
                pins.append((px, py))
            nets_raw.append(pins)
        i += 1

    # Map unique (x,y) → cell index
    coord_to_idx = {}
    for pins in nets_raw:
        for coord in pins:
            if coord not in coord_to_idx:
                coord_to_idx[coord] = len(coord_to_idx)

    # Cell features: [x, y]
    coords = sorted(coord_to_idx, key=coord_to_idx.get)
    cell_x = torch.tensor([[c[0], c[1]] for c in coords], dtype=torch.float)

    net_feats = []
    c2n_src, c2n_dst = [], []
    n2c_src, n2c_dst = [], []

    for net_idx, pins in enumerate(nets_raw):
        net_feats.append([float(len(pins))])
        cell_pins = [coord_to_idx[p] for p in pins]
        if cell_pins:
            c2n_src.append(cell_pins[0])
            c2n_dst.append(net_idx)
        for sink in cell_pins[1:]:
            n2c_src.append(net_idx)
            n2c_dst.append(sink)

    data = HeteroData()
    data["cell"].x = cell_x
    data["net"].x  = torch.tensor(net_feats, dtype=torch.float)
    data["cell", "drives", "net"].edge_index = torch.tensor([c2n_src, c2n_dst], dtype=torch.long)
    data["net", "fans_out_to", "cell"].edge_index = torch.tensor([n2c_src, n2c_dst], dtype=torch.long)

    return data


# ── 4. Simple Heterogeneous GNN skeleton ─────────────────────────────────────

from torch_geometric.nn import HeteroConv, SAGEConv
import torch.nn as nn

class NetlistHeteroGNN(nn.Module):
    def __init__(self, cell_in, net_in, hidden, out_dim, num_layers=3):
        super().__init__()
        self.cell_proj = nn.Linear(cell_in, hidden)
        self.net_proj  = nn.Linear(net_in,  hidden)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ("cell", "drives",      "net"):  SAGEConv(hidden, hidden),
                ("net",  "fans_out_to", "cell"): SAGEConv(hidden, hidden),
            }, aggr="sum")
            self.convs.append(conv)

        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            "cell": self.cell_proj(x_dict["cell"]).relu(),
            "net":  self.net_proj(x_dict["net"]).relu(),
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: v.relu() for k, v in x_dict.items()}
        return self.head(x_dict["cell"])  # per-cell predictions


# ── 5. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--plc",    default=None,              help="Path to .plc Protobuf file")
    p.add_argument("--ibm01",  default="ibm01.modified.txt", help="Path to ibm01.modified.txt")
    args = p.parse_args()

    if args.plc:
        print("Building graph from .plc ...")
        nodes, nets = parse_plc(args.plc)
        data = build_hetero_graph(nodes, nets)
    else:
        print("Building graph directly from ibm01.modified.txt ...")
        data = build_from_ibm01(args.ibm01)

    print(data)
    print(f"\nCell nodes : {data['cell'].x.shape}")
    print(f"Net  nodes : {data['net'].x.shape}")
    print(f"cell->net edges : {data['cell','drives','net'].edge_index.shape[1]}")
    print(f"net->cell edges : {data['net','fans_out_to','cell'].edge_index.shape[1]}")

    # Instantiate GNN
    cell_dim = data["cell"].x.shape[1]
    net_dim  = data["net"].x.shape[1]
    model = NetlistHeteroGNN(cell_in=cell_dim, net_in=net_dim, hidden=64, out_dim=1)
    print(f"\nModel: {model}")

    # Forward pass (untrained)
    out = model({"cell": data["cell"].x, "net": data["net"].x},
                {("cell","drives","net"):      data["cell","drives","net"].edge_index,
                 ("net","fans_out_to","cell"):  data["net","fans_out_to","cell"].edge_index})
    print(f"Output shape: {out.shape}")  # (num_cells, 1)