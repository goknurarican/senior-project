"""
LOSO Training Pipeline for HusformerBITIRMEEG (v2 dataset).

Pipeline per fold (9 folds, one test subject each):
  1. Per-subject z-score normalization of EEG embeddings
  2. Stratified 85/15 train/val split from training subjects
  3. FrustrationDataset with Gaussian jitter augmentation (eye + mouse)
  4. Train HusformerBITIRMEEG with AdamW, per-fold class weights, early stopping
  5. Evaluate on held-out test subject (attention weights collected)
  6. Save: model_best.pth, training_history.json, predictions.csv,
           attention_weights.npy, metrics.json

Aggregate: approach_a/training/loso_results/loso_summary.json
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(BASE, "features")
OUT_DIR  = os.path.join(BASE, "training", "loso_results")
SRC_DIR  = os.path.join(BASE, "src")
sys.path.insert(0, SRC_DIR)

from husformer import HusformerBITIRMEEG

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 32
MAX_EPOCHS   = 50
PATIENCE     = 5
VAL_RATIO    = 0.15
JITTER_STD   = 0.05
SEED         = 42

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_v2_data():
    eeg   = np.load(os.path.join(FEAT_DIR, "all_eeg_embeddings_v2.npy"))   # (1452, 200)
    eye   = np.load(os.path.join(FEAT_DIR, "all_eye_timeseries_v2.npy"))   # (1452, 6, 110)
    mouse = np.load(os.path.join(FEAT_DIR, "all_mouse_timeseries_v2.npy")) # (1452, 7, 210)
    meta  = pd.read_csv(os.path.join(FEAT_DIR, "all_eeg_embeddings_v2_metadata.csv"))
    labels_df = pd.read_csv(os.path.join(FEAT_DIR, "labels_v2.csv"))
    labels = labels_df["label"].values  # 0=control, 1=variant
    assert len(eeg) == len(meta) == len(labels), "Length mismatch"
    return eeg, eye, mouse, labels, meta


# ---------------------------------------------------------------------------
# Subject normalization
# ---------------------------------------------------------------------------
def subject_normalize(eeg: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Per-subject z-score of EEG embeddings (mean/std from that subject)."""
    eeg_norm = eeg.copy().astype(np.float32)
    for sid in meta["subject_id"].unique():
        mask = (meta["subject_id"] == sid).values
        mu  = eeg_norm[mask].mean(axis=0)
        sig = eeg_norm[mask].std(axis=0) + 1e-8
        eeg_norm[mask] = (eeg_norm[mask] - mu) / sig
    return eeg_norm


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FrustrationDataset(Dataset):
    def __init__(self, eeg, eye, mouse, labels, augment=False):
        self.eeg    = torch.from_numpy(eeg).float()
        self.eye    = torch.from_numpy(eye).float()
        self.mouse  = torch.from_numpy(mouse).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg   = self.eeg[idx]
        eye   = self.eye[idx]
        mouse = self.mouse[idx]
        label = self.labels[idx]
        if self.augment:
            eye   = eye   + torch.randn_like(eye)   * JITTER_STD
            mouse = mouse + torch.randn_like(mouse)  * JITTER_STD
        return eeg, eye, mouse, label


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Per-fold class weights: w_c = n_total / (2 * n_c)."""
    n = len(y)
    classes = np.unique(y)
    weights = np.zeros(len(classes), dtype=np.float32)
    for c in classes:
        weights[int(c)] = n / (2 * (y == c).sum())
    return torch.tensor(weights, device=DEVICE)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_fold(model, train_loader, val_loader, class_weights):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=LR * 0.01)

    best_val_loss = float("inf")
    patience_cnt  = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for eeg, eye, mou, lbl in train_loader:
            eeg, eye, mou, lbl = (eeg.to(DEVICE), eye.to(DEVICE),
                                   mou.to(DEVICE), lbl.to(DEVICE))
            optimizer.zero_grad()
            logits = model(eeg, eye, mou)
            loss = criterion(logits, lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(lbl)
        train_loss /= len(train_loader.dataset)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for eeg, eye, mou, lbl in val_loader:
                eeg, eye, mou, lbl = (eeg.to(DEVICE), eye.to(DEVICE),
                                       mou.to(DEVICE), lbl.to(DEVICE))
                logits = model(eeg, eye, mou)
                val_loss += criterion(logits, lbl).item() * len(lbl)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_true.extend(lbl.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_true, val_preds)

        scheduler.step()
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_state    = {k: v.cpu().clone() for k, v in
                              model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                history["stopped_epoch"] = epoch
                break

    history.setdefault("stopped_epoch", MAX_EPOCHS)
    return best_state, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_fold(model, test_loader, meta_test):
    model.eval()
    all_probs, all_preds, all_true = [], [], []
    attn_eeg, attn_eye, attn_mou   = [], [], []

    with torch.no_grad():
        for eeg, eye, mou, lbl in test_loader:
            eeg, eye, mou = eeg.to(DEVICE), eye.to(DEVICE), mou.to(DEVICE)
            logits, attn = model(eeg, eye, mou, return_attn=True)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_true.extend(lbl.numpy().tolist())
            attn_eeg.append(attn["eeg"].numpy())
            attn_eye.append(attn["eye"].numpy())
            attn_mou.append(attn["mouse"].numpy())

    y_true = np.array(all_true)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_variant":        float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc":               float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "confusion_matrix":  confusion_matrix(y_true, y_pred).tolist(),
        "n_test":            int(len(y_true)),
        "n_variant":         int((y_true == 1).sum()),
        "n_control":         int((y_true == 0).sum()),
    }

    pred_df = meta_test.copy().reset_index(drop=True)
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["prob_variant"] = y_prob

    attn_arr = {
        "eeg":   np.concatenate(attn_eeg,  axis=0),  # (N, 2): [:, 0]=→Eye, [:, 1]=→Mouse
        "eye":   np.concatenate(attn_eye,  axis=0),  # (N, 2): [:, 0]=→EEG, [:, 1]=→Mouse
        "mouse": np.concatenate(attn_mou,  axis=0),  # (N, 2): [:, 0]=→EEG, [:, 1]=→Eye
    }

    return metrics, pred_df, attn_arr


# ---------------------------------------------------------------------------
# Main LOSO loop
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.makedirs(OUT_DIR, exist_ok=True)

    eeg, eye, mouse, labels, meta = load_v2_data()
    print(f"Loaded v2: EEG={eeg.shape} Eye={eye.shape} Mouse={mouse.shape}")
    print(f"Label distribution: control={int((labels==0).sum())} variant={int((labels==1).sum())}")

    # Per-subject EEG normalization (applied globally, no leakage since per-subject)
    eeg_norm = subject_normalize(eeg, meta)
    np.save(os.path.join(FEAT_DIR, "all_eeg_embeddings_v2_normalized.npy"), eeg_norm)
    print("Saved normalized EEG embeddings.")

    subjects = sorted(meta["subject_id"].unique())
    fold_summaries = []

    for fold_idx, test_sid in enumerate(subjects):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{len(subjects)}  -  Test subject: sub-{test_sid}")
        print(f"{'='*60}")
        t0 = time.time()

        fold_dir = os.path.join(OUT_DIR, f"fold_{test_sid}")
        os.makedirs(fold_dir, exist_ok=True)

        # Index masks
        test_mask  = (meta["subject_id"] == test_sid).values
        train_mask = ~test_mask

        # --- Split arrays ---
        eeg_tr_all = eeg_norm[train_mask]
        eye_tr_all = eye[train_mask]
        mou_tr_all = mouse[train_mask]
        lbl_tr_all = labels[train_mask]
        meta_tr    = meta[train_mask].reset_index(drop=True)

        eeg_te  = eeg_norm[test_mask]
        eye_te  = eye[test_mask]
        mou_te  = mouse[test_mask]
        lbl_te  = labels[test_mask]
        meta_te = meta[test_mask].reset_index(drop=True)

        # --- Stratified train / val split ---
        sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO,
                                      random_state=SEED)
        tr_idx, val_idx = next(sss.split(eeg_tr_all, lbl_tr_all))

        eeg_tr,  eye_tr,  mou_tr,  lbl_tr  = (eeg_tr_all[tr_idx],  eye_tr_all[tr_idx],
                                                mou_tr_all[tr_idx],  lbl_tr_all[tr_idx])
        eeg_val, eye_val, mou_val, lbl_val = (eeg_tr_all[val_idx], eye_tr_all[val_idx],
                                               mou_tr_all[val_idx], lbl_tr_all[val_idx])

        print(f"  train={len(lbl_tr)} val={len(lbl_val)} test={len(lbl_te)}")
        print(f"  train variant={int((lbl_tr==1).sum())} control={int((lbl_tr==0).sum())}")

        # --- Datasets & Loaders ---
        train_ds = FrustrationDataset(eeg_tr, eye_tr, mou_tr, lbl_tr, augment=True)
        val_ds   = FrustrationDataset(eeg_val, eye_val, mou_val, lbl_val, augment=False)
        test_ds  = FrustrationDataset(eeg_te, eye_te, mou_te, lbl_te, augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=0, drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # --- Class weights (per fold, from training set) ---
        class_weights = compute_class_weights(lbl_tr)
        print(f"  class_weights: control={class_weights[0]:.3f} variant={class_weights[1]:.3f}")

        # --- Fresh model ---
        model = HusformerBITIRMEEG().to(DEVICE)

        # --- Train ---
        best_state, history = train_fold(model, train_loader, val_loader, class_weights)
        stopped = history["stopped_epoch"]
        print(f"  Stopped at epoch {stopped}  |  best_val_loss={min(history['val_loss']):.4f}")

        # --- Load best weights ---
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

        # --- Evaluate ---
        metrics, pred_df, attn_arr = evaluate_fold(model, test_loader, meta_te)
        elapsed = time.time() - t0
        metrics["fold_time_sec"] = round(elapsed, 1)
        metrics["test_subject"]  = int(test_sid)
        metrics["stopped_epoch"] = stopped

        print(f"  ACC={metrics['accuracy']:.3f}  BAL={metrics['balanced_accuracy']:.3f}  "
              f"F1={metrics['f1_macro']:.3f}  AUC={metrics['auc']:.3f}  ({elapsed:.0f}s)")

        # --- Save ---
        torch.save(best_state, os.path.join(fold_dir, "model_best.pth"))

        with open(os.path.join(fold_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        pred_df.to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)

        np.savez(os.path.join(fold_dir, "attention_weights.npz"),
                 eeg=attn_arr["eeg"],
                 eye=attn_arr["eye"],
                 mouse=attn_arr["mouse"])

        fold_summaries.append(metrics)

    # ---------------------------------------------------------------------------
    # Aggregate summary
    # ---------------------------------------------------------------------------
    accs  = [f["accuracy"]          for f in fold_summaries]
    bals  = [f["balanced_accuracy"] for f in fold_summaries]
    f1s   = [f["f1_macro"]          for f in fold_summaries]
    aucs  = [f["auc"]               for f in fold_summaries]

    summary = {
        "n_folds": len(fold_summaries),
        "subjects": [int(s) for s in subjects],
        "accuracy":          {"mean": float(np.mean(accs)),  "std": float(np.std(accs))},
        "balanced_accuracy": {"mean": float(np.mean(bals)),  "std": float(np.std(bals))},
        "f1_macro":          {"mean": float(np.mean(f1s)),   "std": float(np.std(f1s))},
        "auc":               {"mean": float(np.mean(aucs)),  "std": float(np.std(aucs))},
        "per_fold": fold_summaries,
    }

    summary_path = os.path.join(OUT_DIR, "loso_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("LOSO COMPLETE")
    print(f"{'='*60}")
    print(f"  ACC  {summary['accuracy']['mean']:.3f} ± {summary['accuracy']['std']:.3f}")
    print(f"  BAL  {summary['balanced_accuracy']['mean']:.3f} ± {summary['balanced_accuracy']['std']:.3f}")
    print(f"  F1   {summary['f1_macro']['mean']:.3f} ± {summary['f1_macro']['std']:.3f}")
    print(f"  AUC  {summary['auc']['mean']:.3f} ± {summary['auc']['std']:.3f}")
    print(f"  Summary → {summary_path}")


if __name__ == "__main__":
    main()
