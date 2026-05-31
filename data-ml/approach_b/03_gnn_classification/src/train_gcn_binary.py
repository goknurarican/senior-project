"""
train_gnn_binary.py
===================
Binary LOSO training for Approach B GCN.

Task:
  0 = control_action_matched
  1 = frustration scenario

This replaces the original 15-class scenario classification, which is too sparse
for N=9 and highly imbalanced scenario counts.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gnn_model import MinimalGCN, count_parameters

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built()
    else "cpu"
)

B_DIR = Path(__file__).resolve().parents[2]
STAGE1 = B_DIR / "01_connectivity_extraction"
STAGE3 = B_DIR / "03_gnn_classification"

FEAT_DIR = STAGE1 / "features" / "connectivity_per_epoch"
MODEL_DIR = STAGE3 / "models_binary"
EVAL_DIR = STAGE3 / "evaluation_binary"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

PROJECT = B_DIR.parent
V6_FEAT = PROJECT / "approach_a" / "06_v6_multiclass_characterization" / "features"

SUBJECTS = [14, 15, 16, 17, 18, 20, 21, 22, 23]
N_ROI = 6
N_BAND = 4
N_CLASS = 2


def build_node_features(osc: np.ndarray) -> np.ndarray:
    """
    Converts V6 oscillation array (n_ep, 25, n_time) into
    per-node band features (n_ep, 6 ROI, 4 bands).
    """
    n_ep = osc.shape[0]
    nf = np.zeros((n_ep, N_ROI, N_BAND), dtype=np.float32)
    osc_mean = osc.mean(axis=-1)

    for r in range(N_ROI):
        for b in range(N_BAND):
            nf[:, r, b] = osc_mean[:, r * N_BAND + b]

    return nf


def normalize_node_features(nf_train, nf_test):
    mu = nf_train.mean(axis=(0, 1), keepdims=True)
    sd = nf_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (nf_train - mu) / sd, (nf_test - mu) / sd


def build_graph(node_feat: np.ndarray, wpli_mat: np.ndarray, label: int) -> Data:
    x = torch.tensor(node_feat, dtype=torch.float32)

    edges = []
    weights = []

    for i in range(N_ROI):
        for j in range(N_ROI):
            if i == j:
                continue
            edges.append([i, j])
            weights.append(max(float(wpli_mat[i, j]), 0.0))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)


def build_dataset(node_feat, wpli_mean, labels):
    return [
        build_graph(node_feat[i], wpli_mean[i], int(labels[i]))
        for i in range(len(labels))
    ]


def class_weights_binary(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=2).astype(np.float32)
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    weights = weights / weights.sum() * (weights > 0).sum()
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_positive": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_positive": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_positive": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
    }

    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan

    return out


def train_fold(train_data, val_data, sid_held_out, fold_dir,
               max_epochs=60, patience=10, permuted=False):
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

    train_labels = np.array([int(g.y.item()) for g in train_data])
    cw = class_weights_binary(train_labels).to(DEVICE)

    model = MinimalGCN(in_dim=N_BAND, hidden=16, n_classes=2, dropout=0.4).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    best_bal_acc = -1.0
    best_state = None
    bad = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()

            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = loss_fn(logits, batch.y)

            loss.backward()
            opt.step()

            running_loss += float(loss.item()) * batch.num_graphs

        model.eval()
        probs = []
        truths = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                prob = torch.softmax(logits, dim=1)[:, 1]

                probs.append(prob.cpu().numpy())
                truths.append(batch.y.cpu().numpy())

        probs = np.concatenate(probs) if probs else np.array([])
        truths = np.concatenate(truths) if truths else np.array([])

        m = compute_metrics(truths, probs, threshold=0.5)
        val_bal_acc = m["balanced_accuracy"]

        history.append({
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_data)),
            "val_balanced_accuracy": val_bal_acc,
            "val_f1_positive": m["f1_positive"],
            "val_pr_auc": m["pr_auc"],
            "val_roc_auc": m["roc_auc"],
        })

        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if not permuted:
            torch.save(best_state, str(fold_dir / "best_model_binary.pth"))

    model.eval()
    probs = []
    truths = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            prob = torch.softmax(logits, dim=1)[:, 1]

            probs.append(prob.cpu().numpy())
            truths.append(batch.y.cpu().numpy())

    probs = np.concatenate(probs) if probs else np.array([])
    truths = np.concatenate(truths) if truths else np.array([])

    metrics = compute_metrics(truths, probs, threshold=0.5)
    metrics.update({
        "fold_sid": int(sid_held_out),
        "n_train": int(len(train_data)),
        "n_val": int(len(val_data)),
        "best_val_balanced_accuracy": float(best_bal_acc),
        "history": history,
        "permuted": bool(permuted),
    })

    if not permuted:
        pd.DataFrame({
            "true": truths,
            "prob_frustration": probs,
            "pred_05": (probs >= 0.5).astype(int),
        }).to_csv(str(fold_dir / "predictions_binary.csv"), index=False)

        with open(str(fold_dir / "metrics_binary.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics, probs, truths


def run_loso(node_feat, wpli_mean, labels, subject_ids, permuted=False):
    fold_metrics = []
    all_true = []
    all_prob = []

    for sid in SUBJECTS:
        test_mask = subject_ids == sid
        train_mask = ~test_mask

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        nf_tr, nf_te = normalize_node_features(
            node_feat[train_mask],
            node_feat[test_mask],
        )

        tr_data = build_dataset(nf_tr, wpli_mean[train_mask], labels[train_mask])
        te_data = build_dataset(nf_te, wpli_mean[test_mask], labels[test_mask])

        fold_dir = MODEL_DIR / f"fold_{sid}"
        m, p, t = train_fold(
            tr_data,
            te_data,
            sid_held_out=sid,
            fold_dir=fold_dir,
            permuted=permuted,
        )

        fold_metrics.append(m)
        all_prob.extend(p.tolist())
        all_true.extend(t.tolist())

    return fold_metrics, np.array(all_true), np.array(all_prob)


def plot_confusion_binary(y_true, y_prob, out_path):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="gray_r")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["control", "frustration"])
    ax.set_yticklabels(["control", "frustration"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Binary GCN confusion matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, facecolor="white")
    plt.close(fig)


def threshold_sweep(y_true, y_prob):
    rows = []
    for th in np.linspace(0.0, 1.0, 101):
        m = compute_metrics(y_true, y_prob, threshold=th)
        m["threshold"] = float(th)
        rows.append(m)

    return pd.DataFrame(rows)


def permutation_test(node_feat, wpli_mean, labels, subject_ids, n_perm=50):
    rng = np.random.default_rng(42)
    null_bal_accs = []

    for p_idx in range(n_perm):
        permuted = labels.copy()
        rng.shuffle(permuted)

        fold_metrics, y_true, y_prob = run_loso(
            node_feat,
            wpli_mean,
            permuted,
            subject_ids,
            permuted=True,
        )

        m = compute_metrics(y_true, y_prob, threshold=0.5)
        null_bal_accs.append(m["balanced_accuracy"])

        print(f"  perm {p_idx + 1}/{n_perm}: null balanced acc {null_bal_accs[-1]:.3f}")

    return {
        "n_perm": int(n_perm),
        "null_mean_balanced_accuracy": float(np.mean(null_bal_accs)),
        "null_std_balanced_accuracy": float(np.std(null_bal_accs)),
        "null_balanced_accuracies": null_bal_accs,
    }


def main():
    print(f"device: {DEVICE}")
    print(f"model params: {count_parameters(MinimalGCN(in_dim=N_BAND, hidden=16, n_classes=2))}")

    wpli = np.load(str(FEAT_DIR / "all_wpli_v3.npy"))
    osc = np.load(str(V6_FEAT / "all_oscillation_v6.npy"))
    meta = pd.read_csv(str(FEAT_DIR / "connectivity_metadata.csv"))

    print(f"wpli {wpli.shape}, osc {osc.shape}, meta {len(meta)}")

    if len(meta) != len(wpli):
        raise ValueError("metadata and wPLI length mismatch")

    if "label_binary" not in meta.columns:
        raise ValueError("connectivity_metadata.csv must contain label_binary")

    if "subject_id" not in meta.columns:
        raise ValueError("connectivity_metadata.csv must contain subject_id")

    node_feat = build_node_features(osc)
    wpli_mean = wpli.mean(axis=1)

    labels = meta["label_binary"].values.astype(int)
    subject_ids = meta["subject_id"].values.astype(int)

    print("\nClass distribution:")
    print(pd.Series(labels).value_counts().sort_index())

    print("\nLOSO binary training...")
    fold_metrics, all_true, all_prob = run_loso(
        node_feat,
        wpli_mean,
        labels,
        subject_ids,
        permuted=False,
    )

    global_metrics_05 = compute_metrics(all_true, all_prob, threshold=0.5)
    sweep = threshold_sweep(all_true, all_prob)

    best_f1 = sweep.sort_values(
        ["f1_positive", "balanced_accuracy"],
        ascending=[False, False],
    ).iloc[0].to_dict()

    best_bal = sweep.sort_values(
        ["balanced_accuracy", "f1_positive"],
        ascending=[False, False],
    ).iloc[0].to_dict()

    summary = {
        "n_folds": len(fold_metrics),
        "device": str(DEVICE),
        "n_samples": int(len(labels)),
        "n_control": int((labels == 0).sum()),
        "n_frustration": int((labels == 1).sum()),
        "global_metrics_threshold_05": global_metrics_05,
        "best_threshold_by_f1": best_f1,
        "best_threshold_by_balanced_accuracy": best_bal,
        "per_fold": [
            {
                "sid": m["fold_sid"],
                "n_val": m["n_val"],
                "balanced_accuracy": m["balanced_accuracy"],
                "f1_positive": m["f1_positive"],
                "precision_positive": m["precision_positive"],
                "recall_positive": m["recall_positive"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tn": m["tn"],
            }
            for m in fold_metrics
        ],
    }

    with open(str(EVAL_DIR / "binary_loso_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame({
        "true": all_true,
        "prob_frustration": all_prob,
        "pred_05": (all_prob >= 0.5).astype(int),
    }).to_csv(str(EVAL_DIR / "binary_loso_predictions.csv"), index=False)

    sweep.to_csv(str(EVAL_DIR / "binary_threshold_sweep.csv"), index=False)

    plot_confusion_binary(
        all_true,
        all_prob,
        EVAL_DIR / "binary_confusion_matrix.png",
    )

    print("\nGlobal metrics at threshold 0.5:")
    print(json.dumps(global_metrics_05, indent=2))

    print("\nBest threshold by F1:")
    print(best_f1)

    print("\nBest threshold by balanced accuracy:")
    print(best_bal)

    print("\nPermutation test...")
    perm = permutation_test(
        node_feat,
        wpli_mean,
        labels,
        subject_ids,
        n_perm=50,
    )

    obs = global_metrics_05["balanced_accuracy"]
    null = np.array(perm["null_balanced_accuracies"])
    p_value = (1.0 + (null >= obs).sum()) / (1.0 + len(null))

    perm["observed_balanced_accuracy"] = float(obs)
    perm["p_value"] = float(p_value)

    with open(str(EVAL_DIR / "binary_permutation_results.json"), "w") as f:
        json.dump(perm, f, indent=2)

    print("\nPermutation:")
    print(json.dumps(perm, indent=2))

    print("\nStage 3 binary GCN done.")
    print(f"Outputs -> {EVAL_DIR}")


if __name__ == "__main__":
    main()