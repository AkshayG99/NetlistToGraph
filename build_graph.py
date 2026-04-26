"""
Build a heterogeneous cell↔net graph from ibm01.modified.txt.
No GNN — just parses the netlist and constructs the PyG HeteroData object.

Usage:
    python build_graph.py
    python build_graph.py --ibm01 path/to/ibm01.modified.txt
"""

import re
import argparse
import torch
from torch_geometric.data import HeteroData


def build_from_ibm01(filepath):
    nets_raw = []

    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'^(net\S+)\s+\d+\s+(\d+)', line)
        if m:
            num_pins = int(m.group(2))
            pins = []
            for _ in range(num_pins):
                i += 1
                px, py = map(int, lines[i].strip().split())
                pins.append((px, py))
            nets_raw.append(pins)
        i += 1

    # Map unique (x, y) coordinates → cell index
    coord_to_idx = {}
    for pins in nets_raw:
        for coord in pins:
            if coord not in coord_to_idx:
                coord_to_idx[coord] = len(coord_to_idx)

    # Cell features: [x, y]
    coords = sorted(coord_to_idx, key=coord_to_idx.get)
    cell_x = torch.tensor([[c[0], c[1]] for c in coords], dtype=torch.float)

    # Net features: [fanout degree]
    net_feats = []
    c2n_src, c2n_dst = [], []   # cell → net (driver)
    n2c_src, n2c_dst = [], []   # net  → cell (sinks)

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
    data["cell", "drives",      "net"].edge_index = torch.tensor([c2n_src, c2n_dst], dtype=torch.long)
    data["net",  "fans_out_to", "cell"].edge_index = torch.tensor([n2c_src, n2c_dst], dtype=torch.long)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ibm01", default="ibm01.modified.txt")
    args = parser.parse_args()

    print(f"Parsing {args.ibm01} ...")
    data = build_from_ibm01(args.ibm01)

    print(data)
    print(f"\nCell nodes : {data['cell'].x.shape}  (features: x, y)")
    print(f"Net  nodes : {data['net'].x.shape}   (feature: fanout degree)")
    print(f"cell→net edges : {data['cell','drives','net'].edge_index.shape[1]}")
    print(f"net→cell edges : {data['net','fans_out_to','cell'].edge_index.shape[1]}")