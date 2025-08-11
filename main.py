import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import csv
import os
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor

from model import MLP, H2GCN, LINKX, GCN, GAT, GloGNN
from createLCE import LabelContextEmbedding
from datasets.data_loading import get_dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_dir = 'data'
results_dir = 'results'
num_runs = 5

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train models on datasets.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--model', type=str, required=True, choices=['gcn', 'gat', 'lcc', 'h2gcn', 'linkx', 'glognn', 'linkx+lcc', 'h2gcn+lcc', 'glognn+lcc'], help='model')
    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--walk_length_forward', type=int, default=1, help='Length of forward walk')
    parser.add_argument('--walk_length_backward', type=int, default=1, help='Length of backward walk')
    parser.add_argument('--walk_length_sibling', type=int, default=1, help='Length of sibling walk')
    parser.add_argument('--walk_length_guardian', type=int, default=1, help='Length of guardian walk')
    parser.add_argument('--num_walks_forward', type=int, default=3, help='Number of label walks per node for forward walk')
    parser.add_argument('--num_walks_backward', type=int, default=3, help='Number of label walks per node for backward walk')
    parser.add_argument('--embedding_dim_forward', type=int, default=8, help='Dimension of embeddings for forward walk')
    parser.add_argument('--embedding_dim_backward', type=int, default=8, help='Dimension of embeddings for backward walk')
    parser.add_argument('--embedding_dim_sibling', type=int, default=8, help='Dimension of embeddings for sibling walk')
    parser.add_argument('--embedding_dim_guardian', type=int, default=8, help='Dimension of embeddings for guardian walk')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter for integration weighting')
    args = parser.parse_args()
    return args

def train_LCE(data, test_mask, walk_type, embedding_dim, walk_length, num_walks):
    LCE_EPOCHS = 300
    edge_index = data.edge_index.to(device)
    labels = data.y.squeeze().long().to(device)
    LCE_model = LabelContextEmbedding(
        edge_index=edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        num_walks=num_walks,
        num_nodes=data.num_nodes,
        labels=labels,
        test_mask=test_mask.to(device),
        num_negative_samples=2
    ).to(device)

    if walk_type == 'forward':
        loader = LCE_model.forwardloader(batch_size=2048)
    elif walk_type == 'backward':
        loader = LCE_model.backwardloader(batch_size=2048)
    elif walk_type == 'sibling':
        loader = LCE_model.siblingloader(batch_size=2048)
    elif walk_type == 'guardian':
        loader = LCE_model.guardianloader(batch_size=2048)
    else:
        raise ValueError(f"Unknown walk_type: {walk_type}")

    optimizer_emb = torch.optim.Adam(LCE_model.parameters(), lr=0.01)
    for epoch_emb in range(1, LCE_EPOCHS + 1):
        total_loss_emb = 0
        for pos_rw, neg_rw in loader:
            optimizer_emb.zero_grad()
            pos_rw, neg_rw = pos_rw.to(device), neg_rw.to(device)
            loss_emb = LCE_model.loss(pos_rw, neg_rw)
            loss_emb.backward()
            optimizer_emb.step()
            total_loss_emb += loss_emb.item()
        avg_loss_emb = total_loss_emb / len(loader)
        # if epoch_emb % 50 == 0 or epoch_emb == LCE_EPOCHS:
        #     print(f"Epoch {epoch_emb}/{LCE_EPOCHS}, {walk_type}_embeddings loss: {avg_loss_emb:.4f}")
    print(f"{walk_type}_embeddings training completed with final loss {avg_loss_emb:.4f}")

    LCE_model.eval()
    with torch.no_grad():
        return LCE_model()

def train_mlp(x, y, train_mask, val_mask, test_mask, input_dim, output_dim):
    model = MLP(
        in_channels=input_dim,
        hidden_channels=32,
        out_channels=output_dim,
        num_layers=2,
        dropout=0.5
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    data_obj = Data(graph={'node_feat': x.to(device)}, y=y.to(device),
                    train_mask=train_mask.to(device), val_mask=val_mask.to(device), test_mask=test_mask.to(device))
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_model_state = None
    num_epochs = 1000
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data_obj)
        loss = F.cross_entropy(out[data_obj.train_mask], data_obj.y[data_obj.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_out = model(data_obj)
            val_loss = F.cross_entropy(val_out[data_obj.val_mask], data_obj.y[data_obj.val_mask])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f'MLP: Early stopping at epoch {epoch}')
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print(f'MLP: Training completed {num_epochs} epochs or best model not found by patience.')
    model.eval()
    with torch.no_grad():
        out = model(data_obj)
        pred = out[data_obj.test_mask].max(1)[1]
        accuracy = (pred.eq(data_obj.y[data_obj.test_mask]).sum().item() / data_obj.test_mask.sum().item()) * 100
    return accuracy, model

def train_gcn(x, y, edge_index, train_mask, val_mask, test_mask, input_dim, output_dim, num_nodes):
    model = GCN(in_channels=input_dim, hidden_channels=32, out_channels=output_dim, num_layers=2, dropout=0.5, save_mem=False, use_bn=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    data_obj = Data(graph={'num_nodes': num_nodes, 'node_feat': x.to(device), 'edge_index': edge_index.to(device)},
                    y=y.to(device), train_mask=train_mask.to(device), val_mask=val_mask.to(device), test_mask=test_mask.to(device))
    best_val_loss = float('inf')
    patience = 30
    epochs_since_improvement = 0
    num_epochs = 1000
    best_model_state = None
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data_obj)
        loss = F.cross_entropy(out[data_obj.train_mask], data_obj.y[data_obj.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_out = model(data_obj)
            val_loss = F.cross_entropy(val_out[data_obj.val_mask], data_obj.y[data_obj.val_mask])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
            print(f'GCN: Early stopping on epoch {epoch}')
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print(f'GCN: Training completed {num_epochs} epochs or best model not found by patience.')
    model.eval()
    with torch.no_grad():
        out = model(data_obj)
        pred = out.argmax(dim=1)
        correct = pred[data_obj.test_mask] == data_obj.y[data_obj.test_mask]
        accuracy = (int(correct.sum()) / int(data_obj.test_mask.sum())) * 100
    return accuracy, model

def train_gat(x, y, edge_index, train_mask, val_mask, test_mask, input_dim, output_dim, num_nodes):
    model = GAT(in_channels=input_dim, hidden_channels=32, out_channels=output_dim, num_layers=2, dropout=0.5, heads=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    data_obj = Data(graph={'num_nodes': num_nodes, 'node_feat': x.to(device), 'edge_index': edge_index.to(device)},
                    y=y.to(device), train_mask=train_mask.to(device), val_mask=val_mask.to(device), test_mask=test_mask.to(device))
    best_val_loss = float('inf')
    patience = 30
    epochs_since_improvement = 0
    num_epochs = 1000
    best_model_state = None
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data_obj)
        loss = F.cross_entropy(out[data_obj.train_mask], data_obj.y[data_obj.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_out = model(data_obj)
            val_loss = F.cross_entropy(val_out[data_obj.val_mask], data_obj.y[data_obj.val_mask])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
            print(f'GAT: Early stopping on epoch {epoch}')
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print(f'GAT: Training completed {num_epochs} epochs or best model not found by patience.')
    model.eval()
    with torch.no_grad():
        out = model(data_obj)
        pred = out.argmax(dim=1)
        correct = pred[data_obj.test_mask] == data_obj.y[data_obj.test_mask]
        accuracy = (int(correct.sum()) / int(data_obj.test_mask.sum())) * 100
    return accuracy, model

def train_h2gcn(x, y, edge_index, train_mask, val_mask, test_mask, input_dim, output_dim, num_nodes):
    model = H2GCN(in_channels=input_dim, hidden_channels=32, out_channels=output_dim, edge_index=edge_index.to(device), num_layers=2, num_nodes=num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    data_obj = Data(graph={'num_nodes': num_nodes, 'node_feat': x.to(device), 'edge_index': edge_index.to(device)},
                    y=y.to(device), train_mask=train_mask.to(device), val_mask=val_mask.to(device), test_mask=test_mask.to(device))
    best_val_loss = float('inf')
    patience = 30
    epochs_since_improvement = 0
    num_epochs = 1000
    best_model_state = None
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data_obj)
        loss = F.cross_entropy(out[data_obj.train_mask], data_obj.y[data_obj.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_out = model(data_obj)
            val_loss = F.cross_entropy(val_out[data_obj.val_mask], data_obj.y[data_obj.val_mask])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
            print(f'H2GCN: Early stopping on epoch {epoch}')
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print(f'H2GCN: Training completed {num_epochs} epochs or best model not found by patience.')
    model.eval()
    with torch.no_grad():
        out = model(data_obj)
        pred = out.argmax(dim=1)
        correct = pred[data_obj.test_mask] == data_obj.y[data_obj.test_mask]
        accuracy = (int(correct.sum()) / int(data_obj.test_mask.sum())) * 100
    return accuracy, model

def train_linkx(x, y, edge_index, train_mask, val_mask, test_mask, input_dim, output_dim, num_nodes):
    model = LINKX(in_channels=input_dim, hidden_channels=32, out_channels=output_dim, num_nodes=num_nodes, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    data_obj = Data(graph={'num_nodes': num_nodes, 'node_feat': x.to(device), 'edge_index': edge_index.to(device)},
                    y=y.to(device), train_mask=train_mask.to(device), val_mask=val_mask.to(device), test_mask=test_mask.to(device))
    best_val_loss = float('inf')
    patience = 30
    epochs_since_improvement = 0
    num_epochs = 1000
    best_model_state = None
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data_obj)
        loss = F.cross_entropy(out[data_obj.train_mask], data_obj.y[data_obj.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_out = model(data_obj)
            val_loss = F.cross_entropy(val_out[data_obj.val_mask], data_obj.y[data_obj.val_mask])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
            print(f'LINKX: Early stopping on epoch {epoch}')
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print(f'LINKX: Training completed {num_epochs} epochs or best model not found by patience.')
    model.eval()
    with torch.no_grad():
        out = model(data_obj)
        pred = out.argmax(dim=1)
        correct = pred[data_obj.test_mask] == data_obj.y[data_obj.test_mask]
        accuracy = (int(correct.sum()) / int(data_obj.test_mask.sum())) * 100
    return accuracy, model

def train_glognn(x, y, edge_index, train_mask, val_mask, test_mask, input_dim, output_dim, num_nodes):
    x_glognn = x.to(torch.float64) if x.dtype != torch.float64 else x
    edge_index_glognn = edge_index.to(device)
    data_obj = Data(graph={'node_feat': x_glognn, 'edge_index': edge_index_glognn},
                    y=y.to(device), train_mask=train_mask.to(device), val_mask=val_mask.to(device), test_mask=test_mask.to(device))
    model = GloGNN(in_channels=input_dim, hidden_channels=32, out_channels=output_dim, num_nodes=num_nodes, dropout=0.5,
                   alpha=0.5, beta1=0.5, beta2=0.5, gamma=0.5, norm_func_id=1, norm_layers=1, orders=1, orders_func_id=1).to(device).to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val_loss = float('inf')
    patience = 30
    epochs_since_improve = 0
    num_epochs = 1000
    best_model_state = None
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out, _ = model(data_obj)
        loss = F.cross_entropy(out[data_obj.train_mask], data_obj.y[data_obj.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_out, _ = model(data_obj)
            val_loss = F.cross_entropy(val_out[data_obj.val_mask], data_obj.y[data_obj.val_mask])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_since_improve += 1
        if epochs_since_improve >= patience:
            print(f'GloGNN: Early stopping on epoch {epoch}')
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print(f'GloGNN: Training completed {num_epochs} epochs or best model not found by patience.')
    model.eval()
    with torch.no_grad():
        out, _ = model(data_obj)
        pred = out[data_obj.test_mask].max(1)[1]
        accuracy = (pred.eq(data_obj.y[data_obj.test_mask]).sum().item() / data_obj.test_mask.sum().item()) * 100
    return accuracy, model

def integration_models(
    model_lcc, model_gnn,
    x_lcc, x_gnn,
    y, edge_index_gnn,
    train_mask, val_mask, test_mask,
    num_nodes, temperature,
    model_gnn_name: str
):
    for param in model_lcc.parameters():
        param.requires_grad = False
    for param in model_gnn.parameters():
        param.requires_grad = False
    model_lcc.eval()
    model_gnn.eval()
    data_dict_lcc = {'node_feat': x_lcc.to(device)}
    data_obj_lcc = Data(graph=data_dict_lcc, y=y.to(device), val_mask=val_mask.to(device), test_mask=test_mask.to(device))

    if model_gnn_name == "glognn":
        x_gnn_casted = x_gnn.to(torch.float64) if x_gnn.dtype != torch.float64 else x_gnn
        data_dict_gnn = {'node_feat': x_gnn_casted.to(device), 'edge_index': edge_index_gnn.to(device)}
    else:
        data_dict_gnn = {'num_nodes': num_nodes, 'node_feat': x_gnn.to(device), 'edge_index': edge_index_gnn.to(device)}
    data_obj_gnn = Data(graph=data_dict_gnn, y=y.to(device), val_mask=val_mask.to(device), test_mask=test_mask.to(device))

    with torch.no_grad():
        out_lcc_full = model_lcc(data_obj_lcc)
        if model_gnn_name == "glognn":
            out_lcc_full = out_lcc_full.to(torch.float64)
        val_loss_lcc = F.cross_entropy(out_lcc_full[val_mask], y[val_mask]).item()

        raw_out_gnn_full = model_gnn(data_obj_gnn)
        if isinstance(raw_out_gnn_full, tuple):
            out_gnn_full = raw_out_gnn_full[0]
        else:
            out_gnn_full = raw_out_gnn_full
        val_loss_gnn = F.cross_entropy(out_gnn_full[val_mask], y[val_mask]).item()

        term_lcc_val = (1.0 / (val_loss_lcc + 1e-8)) * (1.0 / temperature)
        term_gnn_val = (1.0 / (val_loss_gnn + 1e-8)) * (1.0 / temperature)

        tensor_dtype = out_lcc_full.dtype
        weights_tensor = torch.tensor([term_lcc_val, term_gnn_val], device=device, dtype=tensor_dtype)
        adjusted_weights = torch.softmax(weights_tensor, dim=0)

        if out_gnn_full.dtype != out_lcc_full.dtype:
             out_gnn_full = out_gnn_full.to(out_lcc_full.dtype)

        logits = adjusted_weights[0] * out_lcc_full + adjusted_weights[1] * out_gnn_full
        pred = logits[test_mask].argmax(dim=1)
        correct = pred.eq(y[test_mask]).sum().item()
        accuracy = (correct / test_mask.sum().item()) * 100
    return accuracy, adjusted_weights

def save_results(results_data, results_dir_path, filename, temperature_results=None):
    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)
    results_file = os.path.join(results_dir_path, filename)
    if temperature_results:
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if os.path.getsize(results_file) == 0:
                writer.writerow(['Dataset', 'Model', 'Mean Accuracy ± Std Dev', 'Mean LCC Weight ± Std Dev', 'Mean gnn Model Weight ± Std Dev', 'Temperature'])
            for temp, mean_acc, std_acc, mean_weight_mlp, std_weight_mlp, mean_weight_gnn, std_weight_gnn in temperature_results:
                writer.writerow([results_data[0][0], results_data[0][1], f'{mean_acc:.4g} ± {std_acc:.4g}', f'{mean_weight_mlp:.4g} ± {std_weight_mlp:.4g}', f'{mean_weight_gnn:.4g} ± {std_weight_gnn:.4g}', temp])
        print(f'Temperature results saved to {results_file}')
    else:
        file_exists = os.path.isfile(results_file)
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(results_file) == 0:
                writer.writerow(['Dataset', 'Model', 'Test Accuracy (Mean ± Std Dev for multiple runs)'])
            for row in results_data:
                writer.writerow(row)
        print(f'Results saved to {results_file}')

def main():
    args = parse_args()
    set_seed(args.seed)

    run_results_summary = []
    print(f"\nProcessing dataset: {args.dataset}")
    dataset, _ = get_dataset(args.dataset, data_dir)
    data_graph = dataset[0].to(device)

    test_accuracies_all_runs = []
    per_run_integration_details = []

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs} for {args.model} on {args.dataset}")
        current_data_run = data_graph.clone()
        train_mask = current_data_run.train_mask[:, run].clone().to(device)
        val_mask = current_data_run.val_mask[:, run].clone().to(device)
        test_mask = current_data_run.test_mask[:, run].clone().to(device)

        x = current_data_run.x
        y = current_data_run.y.squeeze().long()
        edge_index = current_data_run.edge_index
        num_nodes = current_data_run.num_nodes

        input_dim = x.size(1)
        output_dim = dataset.num_classes

        accuracy = 0.0
        model_name_for_integration = ""

        if args.model == 'lcc' or 'lcc' in args.model:
            embedding_configs = [
                ('forward', args.embedding_dim_forward, args.walk_length_forward, args.num_walks_forward),
                ('backward', args.embedding_dim_backward, args.walk_length_backward, args.num_walks_backward),
                ('sibling', args.embedding_dim_sibling, args.walk_length_sibling, 1),
                ('guardian', args.embedding_dim_guardian, args.walk_length_guardian, 1), 
            ]
            lce_embeddings_map = {}
            with ThreadPoolExecutor(max_workers=1) as executor:
                future_to_walk_type = {}
                for walk_type, emb_dim, walk_len, walks_pn in embedding_configs:
                    future = executor.submit(train_LCE, current_data_run, test_mask, walk_type, emb_dim, walk_len, walks_pn)
                    future_to_walk_type[future] = walk_type
                for future_item in future_to_walk_type:
                    walk_type = future_to_walk_type[future_item]
                    try:
                        lce_embeddings_map[walk_type] = future_item.result()
                    except Exception as exc:
                        print(f'{walk_type} embedding generation failed: {exc}')
                        lce_embeddings_map[walk_type] = torch.zeros((num_nodes, emb_dim), device=device)

            forward_embeddings = lce_embeddings_map['forward']
            backward_embeddings = lce_embeddings_map['backward']
            sibling_embeddings = lce_embeddings_map['sibling']
            guardian_embeddings = lce_embeddings_map['guardian']
            x_for_mlp = torch.cat((x, forward_embeddings, backward_embeddings, sibling_embeddings, guardian_embeddings), dim=1)
            input_dim_mlp = x_for_mlp.size(1)

        if args.model == 'h2gcn':
            accuracy, _ = train_h2gcn(x, y, to_undirected(edge_index), train_mask, val_mask, test_mask, input_dim, output_dim, num_nodes)
        elif args.model == 'linkx':
            accuracy, _ = train_linkx(x, y, edge_index, train_mask, val_mask, test_mask, input_dim, output_dim, num_nodes)
        elif args.model == 'gcn':
            accuracy, _ = train_gcn(x, y, to_undirected(edge_index), train_mask, val_mask, test_mask, input_dim, output_dim, num_nodes)
        elif args.model == 'gat':
            accuracy, _ = train_gat(x, y, to_undirected(edge_index), train_mask, val_mask, test_mask, input_dim, output_dim, num_nodes)
        elif args.model == 'glognn':
            accuracy, _ = train_glognn(x, y, to_undirected(edge_index), train_mask, val_mask, test_mask, input_dim, output_dim, num_nodes)
        elif args.model == 'lcc':
            accuracy, _ = train_mlp(x_for_mlp, y, train_mask, val_mask, test_mask, input_dim_mlp, output_dim)
        elif args.model in ['linkx+lcc', 'h2gcn+lcc', 'glognn+lcc']:
            model_mlp = None
            accuracy_mlp = 0.0
            model_gnn = None
            accuracy_gnn = 0.0
            edge_index_for_gnn = edge_index

            with ThreadPoolExecutor(max_workers=1) as executor:
                future_mlp = executor.submit(train_mlp,
                                             x_for_mlp, y, train_mask, val_mask, test_mask,
                                             input_dim_mlp, output_dim)
                gnn_train_func = None
                if args.model == 'linkx+lcc':
                    model_name_for_integration = "linkx"
                    gnn_train_func = train_linkx
                elif args.model == 'h2gcn+lcc':
                    model_name_for_integration = "h2gcn"
                    edge_index_for_gnn = to_undirected(edge_index)
                    gnn_train_func = train_h2gcn
                elif args.model == 'glognn+lcc':
                    model_name_for_integration = "glognn"
                    edge_index_for_gnn = to_undirected(edge_index)
                    gnn_train_func = train_glognn
                future_gnn = executor.submit(gnn_train_func,
                                             x, y, edge_index_for_gnn, train_mask, val_mask, test_mask,
                                             input_dim, output_dim, num_nodes)

                accuracy_mlp, model_mlp = future_mlp.result()
                accuracy_gnn, model_gnn = future_gnn.result()
                
            accuracy_integration, weights = integration_models(
                model_lcc=model_mlp, model_gnn=model_gnn, x_lcc=x_for_mlp, x_gnn=x, y=y,
                edge_index_gnn=edge_index_for_gnn, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                num_nodes=num_nodes, temperature=args.temperature, model_gnn_name=model_name_for_integration
            )
            accuracy = accuracy_integration
            per_run_integration_details.append({'acc': accuracy, 'w_lcc': weights[0].item(), 'w_gnn': weights[1].item()})
            print(f'Run {run + 1}, Temperature {args.temperature}, Test Accuracy: {accuracy_integration:.4f}, '
                  f'LCC Weight: {weights[0].item():.4f}, {model_name_for_integration.upper()} Weight: {weights[1].item():.4f}')
        else:
            raise ValueError(f"Unknown model type: {args.model}")

        test_accuracies_all_runs.append(accuracy)
        if not (args.model in ['linkx+lcc', 'h2gcn+lcc', 'glognn+lcc']):
             print(f'Run {run + 1}, Test Accuracy: {accuracy:.4f}')

    mean_acc = np.mean(test_accuracies_all_runs)
    std_acc = np.std(test_accuracies_all_runs)
    formatted_result_str = f'{mean_acc:.4f} ± {std_acc:.4f}'
    run_results_summary.append([args.dataset, args.model, formatted_result_str])

    print(f'\nDataset: {args.dataset}, Model: {args.model}')
    print(f'Average Test Accuracy over {num_runs} runs: {formatted_result_str}')

    results_file_name = f'result_{args.dataset}_{args.model.replace("+", "_")}.csv'

    if args.model in ['linkx+lcc', 'h2gcn+lcc', 'glognn+lcc']:
        all_accuracies = [d['acc'] for d in per_run_integration_details]
        all_weights_lcc = [d['w_lcc'] for d in per_run_integration_details]
        all_weights_gnn = [d['w_gnn'] for d in per_run_integration_details]
        mean_integration_acc = np.mean(all_accuracies)
        std_integration_acc = np.std(all_accuracies)
        mean_weight_lcc = np.mean(all_weights_lcc)
        std_weight_lcc = np.std(all_weights_lcc)
        mean_weight_gnn = np.mean(all_weights_gnn)
        std_weight_gnn = np.std(all_weights_gnn)
        temperature_summary_for_csv = [(
            args.temperature, mean_integration_acc, std_integration_acc,
            mean_weight_lcc, std_weight_lcc, mean_weight_gnn, std_weight_gnn
        )]
        save_results(run_results_summary, results_dir, results_file_name, temperature_summary_for_csv)
    else:
        save_results(run_results_summary, results_dir, results_file_name)

if __name__ == "__main__":
    main()