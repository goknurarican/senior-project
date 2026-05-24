"""
LOSO (Leave-One-Subject-Out) training skeleton for Approach A.

Usage:
    python approach_a/training/train_loso.py [--epochs 50] [--lr 1e-4] [--device cpu]

Reads pre-extracted features from approach_a/features/:
  all_eeg_embeddings.npy   (N, 200)
  all_eye_timeseries.npy   (N, 6, 110)
  all_mouse_timeseries.npy (N, 7, 210)
  epoch_metadata.csv       (N rows, includes subject_id + label_frustration)

Runs 9 LOSO folds (one per subject) and aggregates results.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "approach_a" / "src"))

import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from husformer import HusformerBITIRMEEG

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FEAT_DIR  = ROOT / "approach_a" / "features"
EVAL_DIR  = ROOT / "approach_a" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [14, 15, 16, 17, 18, 20, 21, 22, 23]


def load_features():
    eeg   = np.load(str(FEAT_DIR / "all_eeg_embeddings.npy"))
    eye   = np.load(str(FEAT_DIR / "all_eye_timeseries.npy"))
    mouse = np.load(str(FEAT_DIR / "all_mouse_timeseries.npy"))
    meta  = pd.read_csv(str(FEAT_DIR / "epoch_metadata.csv"))
    return eeg, eye, mouse, meta


def z_score(arr: np.ndarray) -> np.ndarray:
    """Z-score normalize across the N dimension, per feature channel."""
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - mean) / std


def train_one_fold(
    eeg_tr, eye_tr, mouse_tr, y_tr,
    eeg_te, eye_te, mouse_te, y_te,
    device, epochs=50, lr=1e-4, batch_size=16,
):
    model = HusformerBITIRMEEG().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Tensors
    def to_t(arr): return torch.tensor(arr, dtype=torch.float32).to(device)
    def to_y(arr): return torch.tensor(arr, dtype=torch.long).to(device)

    ds = TensorDataset(to_t(eeg_tr), to_t(eye_tr), to_t(mouse_tr), to_y(y_tr))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for eeg_b, eye_b, mou_b, y_b in dl:
            optimizer.zero_grad()
            logits = model(eeg_b, eye_b, mou_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (ep + 1) % 10 == 0:
            log.info(f"    epoch {ep+1}/{epochs}  loss={total_loss/len(dl):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits_te = model(to_t(eeg_te), to_t(eye_te), to_t(mouse_te))
        preds = logits_te.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits_te, dim=1)[:, 1].cpu().numpy()

    acc  = accuracy_score(y_te, preds)
    f1   = f1_score(y_te, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_te, probs)
    except ValueError:
        auc = float("nan")

    return {"accuracy": acc, "f1": f1, "auc": auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--label_col",   type=str,   default="label_frustration",
                        help="Metadata column to use as binary label")
    parser.add_argument("--device",      type=str,   default="cpu",
                        choices=["cpu", "mps", "cuda"])
    parser.add_argument("--sanity",      action="store_true",
                        help="Quick sanity check: test on Feyiz (sub-18), train on 5 epochs only")
    args = parser.parse_args()

    device = torch.device(args.device)
    log.info(f"Device: {device}")

    eeg, eye, mouse, meta = load_features()
    # Normalize eye and mouse per-channel across all N
    eye   = z_score(eye)
    mouse = z_score(mouse)

    labels = meta[args.label_col].values.astype(int)

    results = []

    test_subjects = [18] if args.sanity else SUBJECTS
    train_epochs  = 5   if args.sanity else args.epochs

    for test_sid in test_subjects:
        log.info(f"\nFold: test=sub-{test_sid}")
        te_mask = (meta["subject_id"] == test_sid).values
        tr_mask = ~te_mask

        eeg_tr, eye_tr, mouse_tr, y_tr = eeg[tr_mask], eye[tr_mask], mouse[tr_mask], labels[tr_mask]
        eeg_te, eye_te, mouse_te, y_te = eeg[te_mask], eye[te_mask], mouse[te_mask], labels[te_mask]

        log.info(f"  train N={tr_mask.sum()}  test N={te_mask.sum()}  "
                 f"label dist train={np.bincount(y_tr).tolist()} "
                 f"test={np.bincount(y_te).tolist() if len(y_te) > 0 else '[]'}")

        fold_res = train_one_fold(
            eeg_tr, eye_tr, mouse_tr, y_tr,
            eeg_te, eye_te, mouse_te, y_te,
            device=device, epochs=train_epochs, lr=args.lr, batch_size=args.batch_size,
        )
        fold_res["test_subject"] = test_sid
        results.append(fold_res)
        log.info(f"  → acc={fold_res['accuracy']:.3f}  f1={fold_res['f1']:.3f}  auc={fold_res['auc']:.3f}")

    res_df = pd.DataFrame(results)
    log.info("\n── Aggregate ──────────────────────────────────────────")
    log.info(f"  Mean acc={res_df['accuracy'].mean():.3f}  f1={res_df['f1'].mean():.3f}  auc={res_df['auc'].mean():.3f}")

    out_name = "sanity_check_results.csv" if args.sanity else "loso_results.csv"
    res_df.to_csv(str(EVAL_DIR / out_name), index=False)
    log.info(f"Saved to approach_a/evaluation/{out_name}")

    return res_df


if __name__ == "__main__":
    main()
