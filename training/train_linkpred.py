"""Generic training script controlled by YAML config.
Move from feature_extraction/ to project root structure.

Config example: see training/config_example.yaml

You may run:
  python training/train_linkpred.py --config training/config_example.yaml

Will create outputs under runs/<dataset>/<split>/... by default.
"""
from __future__ import annotations
import os
import argparse
import pickle
import json
import yaml
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

try:
    from torch_geometric.nn import SAGEConv
except ImportError as e:
    raise SystemExit("torch-geometric not installed. See config_example.yaml header for install guidance.")

# ---------------- Data Classes ---------------- #
@dataclass
class Config:
    seed: int
    dataset: str
    split: str
    edge_mode: str
    emb_type: str
    processed_dir: str
    emb_dir: str
    output_dir: str
    model: Dict[str, Any]
    optim: Dict[str, Any]
    train: Dict[str, Any]
    logging: Dict[str, Any]

# ---------------- Models ---------------- #
class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_layers: int, dropout: float):
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

class LinkPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim*2, hidden)
        self.lin2 = nn.Linear(hidden, 1)

    def forward(self, h, edge_pairs):
        src, dst = edge_pairs
        z = torch.cat([h[src], h[dst]], dim=-1)
        z = F.relu(self.lin1(z))
        return self.lin2(z).squeeze(-1)

# ---------------- Utilities ---------------- #

def load_yaml(path: str) -> Config:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return Config(**data)


def load_embeddings(cfg: Config, root: str) -> Tuple[np.ndarray, list]:
    if cfg.emb_type == 'ragplm':
        emb_path = os.path.join(root, cfg.emb_dir, f'{cfg.dataset}_ragplm_embeddings.npy')
        id_path = os.path.join(root, cfg.emb_dir, f'{cfg.dataset}_protein_ids.txt')
    else:
        raise ValueError('Unsupported emb_type: ' + cfg.emb_type)
    embs = np.load(emb_path)
    with open(id_path, 'r') as f:
        ids = [l.strip() for l in f if l.strip()]
    if embs.shape[0] != len(ids):
        raise ValueError('Mismatch between embeddings and ids length')
    return embs, ids


def load_edges(cfg: Config, root: str, ids: list):
    ppi_pkl = os.path.join(root, cfg.processed_dir, f'{cfg.dataset}_ppi.pkl')
    label_pkl = os.path.join(root, cfg.processed_dir, f'{cfg.dataset}_ppi_label.pkl')
    with open(ppi_pkl, 'rb') as f:
        edges_raw = pickle.load(f)
    with open(label_pkl, 'rb') as f:
        labels_raw = pickle.load(f)
    edges_raw = np.array(edges_raw)
    labels_raw = np.array(labels_raw)
    if edges_raw.shape[0] != labels_raw.shape[0]:
        raise ValueError('Edges and labels length mismatch')

    if cfg.edge_mode == 'index':
        edge_pairs = edges_raw.astype(int)
    else:
        id_to_idx = {pid: i for i, pid in enumerate(ids)}
        edge_pairs = np.vectorize(lambda a: id_to_idx[a])(edges_raw)
    return edge_pairs, labels_raw


def load_split(cfg: Config, root: str):
    path = os.path.join(root, cfg.processed_dir, f'{cfg.dataset}_{cfg.split}.json')
    with open(path, 'r') as f:
        js = json.load(f)
    return js['train_index'], js['val_index'], js['test_index']


def edge_index_from_pairs(num_nodes: int, edge_pairs: np.ndarray) -> torch.Tensor:
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


def evaluate(model_g, model_lp, x, edge_pairs, labels, device, edge_index_global):
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

# ---------------- Main Train ---------------- #

def train(cfg: Config, root: str):
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    embs, ids = load_embeddings(cfg, root)
    num_nodes, feat_dim = embs.shape
    x = torch.as_tensor(embs, dtype=torch.float32, device=device)

    edge_pairs, edge_labels = load_edges(cfg, root, ids)
    train_idx, val_idx, test_idx = load_split(cfg, root)

    train_edges = edge_pairs[train_idx]; train_labels = edge_labels[train_idx]
    val_edges = edge_pairs[val_idx]; val_labels = edge_labels[val_idx]
    test_edges = edge_pairs[test_idx]; test_labels = edge_labels[test_idx]

    # Build global graph
    if cfg.train.get('inductive', True):
        base_edges = train_edges
        if cfg.train.get('use_val_in_graph', False):
            base_edges = np.concatenate([base_edges, val_edges], axis=0)
    else:
        base_edges = edge_pairs
    edge_index_global = edge_index_from_pairs(num_nodes, base_edges).to(device)

    model_g = GraphSAGE(feat_dim, cfg.model['hidden_dim'], cfg.model['sage_layers'], cfg.model['dropout']).to(device)
    model_lp = LinkPredictor(cfg.model['hidden_dim'], cfg.model['link_hidden']).to(device)

    opt = torch.optim.Adam(list(model_g.parameters()) + list(model_lp.parameters()), lr=cfg.optim['lr'])
    loss_fn = nn.BCEWithLogitsLoss()

    best_metric = -1.0
    best_state = None
    patience_left = cfg.optim['patience']
    early_metric_name = cfg.train.get('metric','auc')

    for epoch in range(1, cfg.optim['epochs']+1):
        model_g.train(); model_lp.train()
        h_full = model_g(x, edge_index_global)  # precompute
        losses = []
        for be, bl in minibatch_iterator(train_edges, train_labels, cfg.optim['batch_size'], shuffle=True):
            be_t = torch.as_tensor(be.T, dtype=torch.long, device=device)
            bl_t = torch.as_tensor(bl, dtype=torch.float32, device=device)
            logits = model_lp(h_full, be_t)
            loss = loss_fn(logits, bl_t)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        avg_loss = float(np.mean(losses)) if losses else 0.0

        val_f1, val_auc, val_acc = evaluate(model_g, model_lp, x, val_edges, val_labels, device, edge_index_global)
        if cfg.logging.get('verbose_every',1) and (epoch % cfg.logging.get('verbose_every',1)==0):
            print(f"Epoch {epoch:03d} | loss {avg_loss:.4f} | val_auc {val_auc:.4f} | val_f1 {val_f1:.4f} | val_acc {val_acc:.4f}")

        current_metric = val_auc if early_metric_name=='auc' else val_f1
        if current_metric > best_metric:
            best_metric = current_metric
            best_state = {
                'model_g': model_g.state_dict(),
                'model_lp': model_lp.state_dict(),
                'epoch': epoch,
                'val_auc': val_auc,
                'val_f1': val_f1
            }
            patience_left = cfg.optim['patience']
        else:
            patience_left -= 1
            if patience_left <= 0:
                print('Early stopping.')
                break

    # Restore best
    if best_state is not None:
        model_g.load_state_dict(best_state['model_g'])
        model_lp.load_state_dict(best_state['model_lp'])

    test_f1, test_auc, test_acc = evaluate(model_g, model_lp, x, test_edges, test_labels, device, edge_index_global)
    print(f"Test micro-F1: {test_f1:.4f} | Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}")

    out_dir = os.path.join(root, cfg.output_dir, cfg.dataset, cfg.split)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(out_dir, 'best_model.pt'))
    metrics = {
        'test_micro_f1': test_f1,
        'test_auc': test_auc,
        'test_acc': test_acc,
        'best_epoch': best_state['epoch'] if best_state else None,
        'best_val_auc': best_state['val_auc'] if best_state else None,
        'best_val_f1': best_state['val_f1'] if best_state else None
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Predictions
    with torch.no_grad():
        h_full = model_g(x, edge_index_global)
        test_logits = model_lp(h_full, torch.as_tensor(test_edges.T, dtype=torch.long, device=device))
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
    np.save(os.path.join(out_dir, 'predictions_test.npy'), test_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cfg = load_yaml(args.config)
    train(cfg, root)

if __name__ == '__main__':
    main()
