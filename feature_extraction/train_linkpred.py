"""Train GraphSAGE link prediction using precomputed protein sequence embeddings.

Features source:
  - Sequence-level embeddings: <DATASET>_ragplm_embeddings.npy (N_proteins, D)
  - Protein ID order: <DATASET>_protein_ids.txt (line aligned with embeddings)
Graph edges + labels:
  - processed_data/<DATASET>_ppi.pkl (assumed list/array shape (E,2) of protein IDs or indices)
  - processed_data/<DATASET>_ppi_label.pkl (shape (E,) 0/1 labels)
Splits (indices into edge list):
  - processed_data/<DATASET>_<split>.json with keys train_index/val_index/test_index referencing edge indices

If pkl edge file stores indices already (ints), set --edge-mode index (default). If they are protein ID strings, set --edge-mode id.

Example:
  python train_linkpred.py --dataset SHS27k --split bfs --emb-type ragplm \
      --epochs 50 --lr 1e-3 --batch-size 8192

Outputs:
  runs/<dataset>/<split>/best_model.pt
  runs/<dataset>/<split>/metrics.json
  runs/<dataset>/<split>/predictions_test.npy

Reports micro-F1 on test set at best validation ROC-AUC (or F1 if desired switch later).

Dependencies: torch, torch_geometric (for GraphSAGE), scikit-learn
"""
from __future__ import annotations
import os
import json
import argparse
import pickle
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

try:
    from torch_geometric.nn import SAGEConv
except ImportError as e:
    raise SystemExit("Please install torch-geometric. For CUDA 11.8 e.g.:\n"
                     "pip install torch==2.2.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html\n"
                     "pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html\n"
                     "pip install torch-geometric")

# ---------------- Model ---------------- #
class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        dims = [in_dim] + [hidden]*(num_layers-1)
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i+1]) for i in range(num_layers-1)])
        self.dropout = dropout
        self.proj = nn.Linear(dims[-1], hidden)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.proj(h)

# ------------- Link Predictor (dot / bilinear) ------------- #
class LinkPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.lin1 = nn.Linear(in_dim*2, hidden)
        self.lin2 = nn.Linear(hidden, 1)

    def forward(self, h, edge_pairs):
        # edge_pairs shape (2,E)
        src, dst = edge_pairs
        z = torch.cat([h[src], h[dst]], dim=-1)
        z = F.relu(self.lin1(z))
        z = self.lin2(z).squeeze(-1)
        return z

# ---------------- Data utilities ---------------- #

def load_embeddings(dataset: str, emb_type: str, root: str) -> Tuple[np.ndarray, list]:
    if emb_type == 'ragplm':
        emb_path = os.path.join(root, 'feature_extraction', f'{dataset}_ragplm_embeddings.npy')
        id_path = os.path.join(root, 'feature_extraction', f'{dataset}_protein_ids.txt')
    else:
        raise ValueError('Unsupported emb_type: ' + emb_type)
    if not os.path.isfile(emb_path):
        raise FileNotFoundError(f'Missing embeddings: {emb_path}')
    embs = np.load(emb_path)
    with open(id_path, 'r') as f:
        ids = [l.strip() for l in f if l.strip()]
    if len(ids) != embs.shape[0]:
        raise ValueError('ID count mismatch embeddings')
    return embs, ids


def load_edges(dataset: str, root: str, mode: str, ids: list):
    ppi_pkl = os.path.join(root, 'processed_data', f'{dataset}_ppi.pkl')
    label_pkl = os.path.join(root, 'processed_data', f'{dataset}_ppi_label.pkl')
    with open(ppi_pkl, 'rb') as f:
        edges_raw = pickle.load(f)
    with open(label_pkl, 'rb') as f:
        labels_raw = pickle.load(f)
    edges_raw = np.array(edges_raw)
    labels_raw = np.array(labels_raw)
    if edges_raw.shape[0] != labels_raw.shape[0]:
        raise ValueError('Edges and labels length mismatch')

    if mode == 'index':
        if edges_raw.max() >= len(ids):
            raise ValueError('Edge index larger than number of proteins')
        edge_index_pairs = edges_raw.astype(int)
    else:  # 'id'
        id_to_idx = {pid: i for i, pid in enumerate(ids)}
        try:
            edge_index_pairs = np.vectorize(lambda a: id_to_idx[a])(edges_raw)
        except KeyError as e:
            raise KeyError(f'Edge contains unknown protein id {e}')
    return edge_index_pairs, labels_raw


def load_split(dataset: str, split: str, root: str):
    path = os.path.join(root, 'processed_data', f'{dataset}_{split}.json')
    with open(path, 'r') as f:
        js = json.load(f)
    return js['train_index'], js['val_index'], js['test_index']

# ---------------- Training Loop ---------------- #

def edge_index_from_pairs(num_nodes: int, edge_pairs: np.ndarray) -> torch.Tensor:
    # Undirected: add reverse
    rev = edge_pairs[:, [1,0]]
    full = np.concatenate([edge_pairs, rev], axis=0)
    return torch.as_tensor(full.T, dtype=torch.long)


def minibatch_iterator(edge_idx: np.ndarray, labels: np.ndarray, batch_size: int, shuffle=True):
    order = np.arange(len(edge_idx))
    if shuffle:
        np.random.shuffle(order)
    for start in range(0, len(order), batch_size):
        sel = order[start:start+batch_size]
        yield edge_idx[sel], labels[sel]


def evaluate(model_g, model_lp, x, edge_pairs, labels, device) -> Tuple[float,float,float]:
    model_g.eval(); model_lp.eval()
    with torch.no_grad():
        h = model_g(x, edge_index_global)
        logits = model_lp(h, torch.as_tensor(edge_pairs.T, dtype=torch.long, device=device))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        f1 = f1_score(labels, preds, average='micro')
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = float('nan')
    acc = (preds == labels).mean()
    return f1, auc, acc

# ---------------- Main ---------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['SHS27k','SHS148k','STRING'])
    parser.add_argument('--split', required=True, choices=['bfs','dfs','random'])
    parser.add_argument('--emb-type', default='ragplm')
    parser.add_argument('--edge-mode', default='index', choices=['index','id'])
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--sage-layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', default='runs')
    args = parser.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Load embeddings
    embs, ids = load_embeddings(args.dataset, args.emb_type, root)
    num_nodes, feat_dim = embs.shape
    x = torch.as_tensor(embs, dtype=torch.float32, device=args.device)

    # Load edges
    edge_pairs, edge_labels = load_edges(args.dataset, root, args.edge_mode, ids)

    # Load split indices
    train_idx, val_idx, test_idx = load_split(args.dataset, args.split, root)
    train_edges = edge_pairs[train_idx]
    train_labels = edge_labels[train_idx]
    val_edges = edge_pairs[val_idx]
    val_labels = edge_labels[val_idx]
    test_edges = edge_pairs[test_idx]
    test_labels = edge_labels[test_idx]

    # Build global graph structure from ALL edges (could also restrict to train only)
    global edge_index_global
    edge_index_global = edge_index_from_pairs(num_nodes, train_edges)  # using only train edges for inductive
    edge_index_global = edge_index_global.to(args.device)

    model_g = GraphSAGE(in_dim=feat_dim, hidden=args.hidden, num_layers=args.sage_layers, dropout=args.dropout).to(args.device)
    model_lp = LinkPredictor(in_dim=args.hidden).to(args.device)

    opt = torch.optim.Adam(list(model_g.parameters()) + list(model_lp.parameters()), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs+1):
        model_g.train(); model_lp.train()
        h = model_g(x, edge_index_global)
        losses = []
        for batch_edges, batch_y in minibatch_iterator(train_edges, train_labels, args.batch_size, shuffle=True):
            batch_edges_t = torch.as_tensor(batch_edges.T, dtype=torch.long, device=args.device)
            batch_y_t = torch.as_tensor(batch_y, dtype=torch.float32, device=args.device)
            logits = model_lp(h, batch_edges_t)
            loss = bce(logits, batch_y_t)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        avg_loss = float(np.mean(losses)) if losses else 0.0

        val_f1, val_auc, val_acc = evaluate(model_g, model_lp, x, val_edges, val_labels, args.device)
        print(f'Epoch {epoch:03d} | loss {avg_loss:.4f} | val_auc {val_auc:.4f} | val_f1 {val_f1:.4f} | val_acc {val_acc:.4f}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                'model_g': model_g.state_dict(),
                'model_lp': model_lp.state_dict(),
                'epoch': epoch,
                'val_auc': val_auc
            }
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print('Early stopping.')
                break

    # Load best
    if best_state is not None:
        model_g.load_state_dict(best_state['model_g'])
        model_lp.load_state_dict(best_state['model_lp'])

    test_f1, test_auc, test_acc = evaluate(model_g, model_lp, x, test_edges, test_labels, args.device)
    print(f'Test micro-F1: {test_f1:.4f} | Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}')

    # Save artifacts
    out_dir = os.path.join(args.save_dir, args.dataset, args.split)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(out_dir, 'best_model.pt'))
    metrics = {
        'test_micro_f1': test_f1,
        'test_auc': test_auc,
        'test_acc': test_acc,
        'best_val_auc': best_val_auc,
        'best_epoch': best_state['epoch'] if best_state else None
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    # Save raw predictions on test
    with torch.no_grad():
        h = model_g(x, edge_index_global)
        test_logits = model_lp(h, torch.as_tensor(test_edges.T, dtype=torch.long, device=args.device))
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
    np.save(os.path.join(out_dir, 'predictions_test.npy'), test_probs)

if __name__ == '__main__':
    main()
