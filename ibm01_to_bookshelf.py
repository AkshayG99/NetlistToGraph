"""
Convert ibm01.modified.txt (Labyrinth/ISPD routing format) to Bookshelf format
(.nodes, .nets, .pl, .scl) suitable for the TILOS FormatTranslators.
"""

import os
import re

def parse_ibm01(filepath):
    """Parse ibm01.modified.txt into a list of nets with pins."""
    nets = []
    grid_w, grid_h = 64, 64
    with open(filepath) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("grid"):
            parts = line.split()
            grid_w, grid_h = int(parts[1]), int(parts[2])
        elif re.match(r'^net\S+\s+\d+\s+\d+', line):
            parts = line.split()
            net_name = parts[0]
            num_pins = int(parts[2])
            pins = []
            for _ in range(num_pins):
                i += 1
                pin_parts = lines[i].strip().split()
                x, y = int(pin_parts[0]), int(pin_parts[1])
                pins.append((x, y))
            nets.append({"name": net_name, "pins": pins})
        i += 1
    return nets, grid_w, grid_h


def write_bookshelf(nets, grid_w, grid_h, design, out_dir):
    """Write Bookshelf .nodes, .nets, .pl, .scl files."""
    os.makedirs(out_dir, exist_ok=True)

    # Collect unique pins as nodes (each unique (x,y) becomes a node)
    pin_to_node = {}
    node_list = []
    for net in nets:
        for (x, y) in net["pins"]:
            key = (x, y)
            if key not in pin_to_node:
                node_id = f"node_{x}_{y}"
                pin_to_node[key] = node_id
                node_list.append((node_id, x, y))

    cell_w, cell_h = 1, 1  # unit cell size in grid units

    # .nodes file
    with open(f"{out_dir}/{design}.nodes", "w") as f:
        f.write("UCLA nodes 1.0\n\n")
        f.write(f"NumNodes : {len(node_list)}\n")
        f.write(f"NumTerminals : 0\n\n")
        for (nid, x, y) in node_list:
            f.write(f"\t{nid}\t{cell_w}\t{cell_h}\n")

    # .nets file
    with open(f"{out_dir}/{design}.nets", "w") as f:
        f.write("UCLA nets 1.0\n\n")
        f.write(f"NumNets : {len(nets)}\n")
        total_pins = sum(len(n["pins"]) for n in nets)
        f.write(f"NumPins : {total_pins}\n\n")
        for net in nets:
            f.write(f"NetDegree : {len(net['pins'])} {net['name']}\n")
            for (x, y) in net["pins"]:
                nid = pin_to_node[(x, y)]
                # offset from cell center = 0, 0 (pins at center)
                f.write(f"\t{nid} I : 0.0 0.0\n")

    # .pl file (placement — use grid coordinates directly)
    with open(f"{out_dir}/{design}.pl", "w") as f:
        f.write("UCLA pl 1.0\n\n")
        for (nid, x, y) in node_list:
            f.write(f"{nid}\t{x}\t{y} : N\n")

    # .scl file (rows — one row per grid row)
    with open(f"{out_dir}/{design}.scl", "w") as f:
        f.write("UCLA scl 1.0\n\n")
        f.write(f"NumRows : {grid_h}\n\n")
        for row in range(grid_h):
            f.write(f"CoreRow Horizontal\n")
            f.write(f"  Coordinate    : {row}\n")
            f.write(f"  Height        : {cell_h}\n")
            f.write(f"  Sitewidth     : 1\n")
            f.write(f"  Sitespacing   : 1\n")
            f.write(f"  Siteorient    : 1\n")
            f.write(f"  Sitesymmetry  : 1\n")
            f.write(f"  SubrowOrigin  : 0  NumSites : {grid_w}\n")
            f.write(f"End\n")

    print(f"Bookshelf files written to: {out_dir}/")
    print(f"  Nodes: {len(node_list)}, Nets: {len(nets)}")


if __name__ == "__main__":
    INPUT_FILE = "ibm01.modified.txt"
    DESIGN = "ibm01"
    OUT_DIR = f"bookshelf/{DESIGN}"

    nets, grid_w, grid_h = parse_ibm01(INPUT_FILE)
    write_bookshelf(nets, grid_w, grid_h, DESIGN, OUT_DIR)