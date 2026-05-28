"""
train_gnn.py
============
loso training for the minimal gcn defined in gnn_model.py.

dataset construction:
  - graph per epoch (480 graphs total)
  - nodes:   6 rois
  - node features: per-roi band ersp summary (mean over time, 4 features per node)
  - edges:   complete graph over the 6 rois
  - edge weight: mean wpli across the 4 bands (single scalar per edge)
  - target:  15-class scenario label (label_15class from v6)

training:
  - 9-fold leave-one-subject-out
  - adamw, lr=5e-4, weight_decay=1e-2 (strong regularisation)
  - class-weighted cross-entropy
  - max 40 epochs, patience 8 on validation accuracy

permutation test:
  - 50 label shuffles, refit per fold
  - report mean null accuracy and p-value

outputs (approach_b/03_gnn_classification):
  models/fold_<sid>/best_model.pth
  models/fold_<sid>/predictions.csv
  models/fold_<sid>/metrics.json
  evaluation/loso_summary.json
  evaluation/per_scenario_performance.csv
  evaluation/confusion_matrix.png
  evaluation/permutation_results.json
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, classification_report)
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gnn_model import MinimalGCN, count_parameters

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

#device: prefer mps on m1, otherwise cpu
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built()
    else "cpu"
)

#paths
B_DIR     = Path(__file__).resolve().parents[2]
STAGE1    = B_DIR / "01_connectivity_extraction"
STAGE3    = B_DIR / "03_gnn_classification"
FEAT_DIR  = STAGE1 / "features" / "connectivity_per_epoch"
MODEL_DIR = STAGE3 / "models"
EVAL_DIR  = STAGE3 / "evaluation"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

PROJECT  = B_DIR.parent
V6_FEAT  = PROJECT / "approach_a" / "06_v6_multiclass_characterization" / "features"

SUBJECTS = [14, 15, 16, 17, 18, 20, 21, 22, 23]
N_ROI    = 6
N_BAND   = 4
N_CLASS  = 15

SCENARIO_NAMES = [
    'control_action_matched',
    'broken_image','button_delay','coupon_expired','coupon_min_spend',
    'facet_reset_once','feedback_late','first_click_miss','network_jitter',
    'overlay_blocking','price_change','search_irrelevant','skeleton_prolong',
    'slow_image','sort_reset',
]
SCENARIO_TO_LABEL = {s: i for i, s in enumerate(SCENARIO_NAMES)}


def build_node_features(osc: np.ndarray) -> np.ndarray:
    """convert v6 oscillation array (n_ep, 25, 110) to per-node band features
    (n_ep, n_roi, n_band). order in osc: feat_idx loops roi outer, band inner;
    index 24 is faa (ignored here)."""
    n_ep = osc.shape[0]
    nf = np.zeros((n_ep, N_ROI, N_BAND), dtype=np.float32)
    osc_mean = osc.mean(axis=-1)        # (n_ep, 25)
    for r in range(N_ROI):
        for b in range(N_BAND):
            nf[:, r, b] = osc_mean[:, r * N_BAND + b]
    return nf


def normalize_node_features(nf_train: np.ndarray,
                            nf_test: np.ndarray) -> tuple:
    """z-score node features using the training subset's stats."""
    mu = nf_train.mean(axis=(0, 1), keepdims=True)
    sd = nf_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (nf_train - mu) / sd, (nf_test - mu) / sd


def build_graph(node_feat: np.ndarray, wpli_mat: np.ndarray, label: int) -> Data:
    """build pyg data object for one epoch.
    node_feat: (n_roi, n_band)
    wpli_mat:  (n_roi, n_roi) mean across bands
    """
    x = torch.tensor(node_feat, dtype=torch.float32)
    #fully connected directed edges (skip self loops; gcnconv adds them)
    edges = []
    weights = []
    for i in range(N_ROI):
        for j in range(N_ROI):
            if i == j:
                continue
            edges.append([i, j])
            #clamp negatives (only relevant for aec weight if ever used)
            weights.append(max(float(wpli_mat[i, j]), 0.0))
    edge_index  = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)


def build_dataset(node_feat: np.ndarray,
                  wpli_mean: np.ndarray,
                  labels: np.ndarray) -> list:
    return [build_graph(node_feat[i], wpli_mean[i], int(labels[i]))
            for i in range(len(labels))]


def class_weights(labels: np.ndarray, n_class: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_class).astype(np.float32)
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    weights = weights / weights.sum() * (weights > 0).sum()
    return torch.tensor(weights, dtype=torch.float32)


def train_fold(train_data: list, val_data: list, n_class: int,
               sid_held_out: int, fold_dir: Path,
               max_epochs: int = 40, patience: int = 8,
               permuted: bool = False) -> dict:
    """train one loso fold. returns metrics dict."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=128, shuffle=False)

    train_labels = np.array([int(g.y.item()) for g in train_data])
    cw = class_weights(train_labels, n_class).to(DEVICE)

    model = MinimalGCN(in_dim=N_BAND, hidden=16,
                       n_classes=n_class, dropout=0.5).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    best_acc = -1.0
    best_state = None
    bad = 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index,
                           batch.edge_attr, batch.batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            opt.step()
            running_loss += float(loss.item()) * batch.num_graphs

        #val accuracy
        model.eval()
        preds = []
        truths = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                logits = model(batch.x, batch.edge_index,
                               batch.edge_attr, batch.batch)
                preds.append(logits.argmax(dim=1).cpu().numpy())
                truths.append(batch.y.cpu().numpy())
        preds  = np.concatenate(preds)  if preds  else np.array([])
        truths = np.concatenate(truths) if truths else np.array([])
        val_acc = accuracy_score(truths, preds) if len(preds) else 0.0
        val_f1  = f1_score(truths, preds, average="macro", zero_division=0) \
                    if len(preds) else 0.0
        history.append({"epoch": epoch, "train_loss": running_loss / max(1, len(train_data)),
                        "val_acc": val_acc, "val_f1": val_f1})
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if not permuted:
            torch.save(best_state, str(fold_dir / "best_model.pth"))

    #final predictions
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index,
                           batch.edge_attr, batch.batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            truths.append(batch.y.cpu().numpy())
    preds  = np.concatenate(preds)  if preds  else np.array([])
    truths = np.concatenate(truths) if truths else np.array([])

    acc = accuracy_score(truths, preds) if len(preds) else 0.0
    f1  = f1_score(truths, preds, average="macro", zero_division=0) \
            if len(preds) else 0.0

    metrics = {
        "fold_sid":    int(sid_held_out),
        "n_train":     int(len(train_data)),
        "n_val":       int(len(val_data)),
        "best_val_acc": float(best_acc),
        "final_acc":   float(acc),
        "final_f1_macro": float(f1),
        "history":     history,
        "permuted":    bool(permuted),
    }

    if not permuted:
        pd.DataFrame({"true": truths, "pred": preds}).to_csv(
            str(fold_dir / "predictions.csv"), index=False)
        with open(str(fold_dir / "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics, preds, truths


def run_loso(node_feat: np.ndarray, wpli_mean: np.ndarray,
             labels: np.ndarray, subject_ids: np.ndarray,
             permuted: bool = False) -> tuple:
    fold_metrics = []
    all_true, all_pred = [], []
    for sid in SUBJECTS:
        test_mask  = subject_ids == sid
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        nf_tr, nf_te = normalize_node_features(node_feat[train_mask],
                                               node_feat[test_mask])
        tr_data = build_dataset(nf_tr, wpli_mean[train_mask], labels[train_mask])
        te_data = build_dataset(nf_te, wpli_mean[test_mask],  labels[test_mask])
        fold_dir = MODEL_DIR / f"fold_{sid}"
        m, p, t = train_fold(tr_data, te_data, N_CLASS, sid, fold_dir,
                             permuted=permuted)
        fold_metrics.append(m)
        all_true.extend(t.tolist())
        all_pred.extend(p.tolist())
    return fold_metrics, np.array(all_true), np.array(all_pred)


def per_scenario_performance(true: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for c, name in enumerate(SCENARIO_NAMES):
        mask = true == c
        n = int(mask.sum())
        if n == 0:
            rows.append({"label": c, "scenario": name, "n_epochs": 0,
                         "accuracy": np.nan, "f1": np.nan})
            continue
        acc = float((pred[mask] == c).mean())
        try:
            f1 = float(f1_score(true == c, pred == c, zero_division=0))
        except Exception:
            f1 = np.nan
        rows.append({"label": c, "scenario": name, "n_epochs": n,
                     "accuracy": acc, "f1": f1})
    return pd.DataFrame(rows)


def plot_confusion(true: np.ndarray, pred: np.ndarray, out_path: Path):
    cm = confusion_matrix(true, pred, labels=list(range(N_CLASS)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASS))
    ax.set_yticks(range(N_CLASS))
    ax.set_xticklabels(SCENARIO_NAMES, rotation=70, ha="right", fontsize=7)
    ax.set_yticklabels(SCENARIO_NAMES, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("approach b gnn confusion matrix (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, facecolor="white")
    plt.close(fig)


def permutation_test(node_feat: np.ndarray, wpli_mean: np.ndarray,
                     labels: np.ndarray, subject_ids: np.ndarray,
                     n_perm: int = 50) -> dict:
    """label-permutation test. within each fold, shuffle training labels
    before fitting; record val accuracy on the original (unshuffled)
    test labels. report mean null accuracy + p-value."""
    rng = np.random.default_rng(42)
    null_accs = []
    for p_idx in range(n_perm):
        permuted = labels.copy()
        rng.shuffle(permuted)
        fold_metrics, _, _ = run_loso(node_feat, wpli_mean,
                                      permuted, subject_ids, permuted=True)
        if not fold_metrics:
            continue
        null_accs.append(float(np.mean([m["final_acc"] for m in fold_metrics])))
        print(f"  perm {p_idx+1}/{n_perm}: null acc {null_accs[-1]:.3f}")
    return {
        "n_perm": int(n_perm),
        "null_mean": float(np.mean(null_accs)) if null_accs else float("nan"),
        "null_std":  float(np.std(null_accs))  if null_accs else float("nan"),
        "null_accuracies": null_accs,
    }


def main():
    print(f"device: {DEVICE}")
    print(f"model params: {count_parameters(MinimalGCN(in_dim=N_BAND, hidden=16, n_classes=N_CLASS))}")

    wpli   = np.load(str(FEAT_DIR / "all_wpli_v3.npy"))
    labels_v6 = pd.read_csv(str(FEAT_DIR / "labels_v6.csv"))
    osc    = np.load(str(V6_FEAT / "all_oscillation_v6.npy"))
    print(f"  wpli {wpli.shape}, osc {osc.shape}, labels {len(labels_v6)}")

    #node features from oscillation
    node_feat = build_node_features(osc)            # (480, 6, 4)
    wpli_mean = wpli.mean(axis=1)                   # (480, 6, 6) mean across bands

    labels = labels_v6["label_15class"].values.astype(int)
    subject_ids = labels_v6["subject_id"].values.astype(int)

    print("\nloso training...")
    fold_metrics, all_true, all_pred = run_loso(
        node_feat, wpli_mean, labels, subject_ids, permuted=False)

    fold_accs = [m["final_acc"] for m in fold_metrics]
    fold_f1   = [m["final_f1_macro"] for m in fold_metrics]
    summary = {
        "n_folds": len(fold_metrics),
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy":  float(np.std(fold_accs)),
        "mean_f1_macro": float(np.mean(fold_f1)),
        "std_f1_macro":  float(np.std(fold_f1)),
        "chance":        1.0 / N_CLASS,
        "device":        str(DEVICE),
        "per_fold": [
            {"sid": m["fold_sid"], "acc": m["final_acc"],
             "f1": m["final_f1_macro"], "n_val": m["n_val"]}
            for m in fold_metrics
        ],
    }
    with open(str(EVAL_DIR / "loso_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nloso acc {summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f}, "
          f"f1 {summary['mean_f1_macro']:.3f} ± {summary['std_f1_macro']:.3f}")

    perf = per_scenario_performance(all_true, all_pred)
    perf.to_csv(str(EVAL_DIR / "per_scenario_performance.csv"), index=False)
    print(perf.to_string(index=False))

    plot_confusion(all_true, all_pred, EVAL_DIR / "confusion_matrix.png")
    print(f"confusion -> {EVAL_DIR / 'confusion_matrix.png'}")

    print("\npermutation test (50 shuffles)...")
    perm = permutation_test(node_feat, wpli_mean, labels, subject_ids,
                            n_perm=50)
    obs = summary["mean_accuracy"]
    null = np.array(perm["null_accuracies"])
    p_value = (1.0 + (null >= obs).sum()) / (1.0 + len(null))
    perm["observed_accuracy"] = obs
    perm["p_value"] = float(p_value)
    with open(str(EVAL_DIR / "permutation_results.json"), "w") as f:
        json.dump(perm, f, indent=2)
    print(f"null acc {perm['null_mean']:.3f} ± {perm['null_std']:.3f}, "
          f"p={p_value:.4f}")

    print("\nstage 3 done.")


if __name__ == "__main__":
    main()
