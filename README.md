# GNN+LCC

This repository is the codebase of a paper "Graph Neural Network leveraging Higher-order Class Label Connectivity for Heterophilous Graphs", which was accepted to [ECML PKDD 2025](https://ecmlpkdd.org/2025/).

## Installation

```bash
pip install -r requirements.txt
```

## Run the code

### Parameters

- `--dataset`: Name of the dataset.
    - `cornell`, `texas`, `wisconsin`
    - `squirrel`, `chameleon`
    - `directed-roman-empire`, `directed-amazon-ratings`
- `--model`:Name of the model to use.
    - GNNs: `gcn`, `gat`, `h2gcn`, `linkx`, `glognn`
    - LCC (Label Context Classifier): `lcc`
    - GNN+LCC: `h2gcn+lcc`, `linkx+lcc`, `glognn+lcc`
- `--seed`: Random seed
- `--walk_length_forward`: Walk length for the forward walk
- `--walk_length_backward`: Walk length for the backward walk
- `--walk_length_sibling`: Walk length for the sibling walk
- `--walk_length_guardian`: Walk length for the guardian walk
- `--num_walks_forward`: Number of forward walks per node
- `--num_walks_backward`: Number of backrward walks per node
- `--embedding_dim_forward`: Embedding dimension for the forward walk
- `--embedding_dim_backward`: Embedding dimension for the backward walk
- `--embedding_dim_sibling`: Embedding dimension for the sibling walk
- `--embedding_dim_guardian`: Embedding dimension for the guardian walk
- `--temperature`: Temperature parameter for integration

### Example: H2GCN+LCC on the Cornell Dataset

```bash
python main.py --dataset cornell \
  --walk_length_forward 3 \
  --num_walks_forward 5 \
  --embedding_dim_forward 8 \
  --walk_length_backward 1 \
  --num_walks_backward 3 \
  --embedding_dim_backward 8 \
  --walk_length_sibling 1 \
  --embedding_dim_sibling 8 \
  --walk_length_guardian 3 \
  --embedding_dim_guardian 8 \
  --temperature 0.2 \
  --model h2gcn+lcc
```
