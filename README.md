1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate


2. Install dependencies
pip install "numpy<2"
pip install torch torchvision torchaudio
pip install torch_geometric
pip install matplotlib networkx plotly

Note: Do not install pyg-lib, torch-sparse, torch-cluster, or torch-spline-conv. They are optional compiled extensions that cause architecture errors on some machines and are not needed here.


3. Running
Build the graph + run a GNN forward pass

python plc_to_hetero_graph.py --ibm01 ibm01.modified.txt

ONLY BUILD GRAPH
python build_graph.py --ibm01 ibm01.modified.txt


4. Output
Building graph directly from ibm01.modified.txt ...
HeteroData(
  cell={ x=[4064, 2] },
  net={ x=[11507, 1] },
  (cell, drives, net)={ edge_index=[2, 11507] },
  (net, fans_out_to, cell)={ edge_index=[2, 32759] }
)
Cell nodes : torch.Size([4064, 2])
Net  nodes : torch.Size([11507, 1])
Output shape: torch.Size([4064, 1])


5. Visualize the graph
python visualize_graph.py