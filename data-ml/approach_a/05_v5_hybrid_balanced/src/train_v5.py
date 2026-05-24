"""
V5 Training Pipeline: 9-fold LOSO with modality balance.

Innovations vs V2/V3:
  - Modality dropout (30% prob): zeroes a random modality per batch
  - Auxiliary modality losses: each branch must classify independently
  - Anchor alignment loss: adapter output aligns with oscillation temporal mean
  - Leakage-free per-fold normalization: fit scaler only on training subjects
  - LaBraM embeddings (subject z-score) + oscillation TS (per-fold scaler) + eye/mouse

Loss:
  total = main_loss + 0.3*(aux_eeg + aux_eye + aux_mouse) + 0.2*anchor_loss

References:
  - Modality dropout: Neverova et al. (2016), TPAMI
  - Auxiliary losses: multi-task learning (Ruder 2017)
  - Knowledge distillation: Hinton et al. (2015)
"""
import json, os, random, sys, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")

BASE    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
V5_DIR  = os.path.join(BASE, "approach_a", "v5_hybrid")
FEAT    = os.path.join(V5_DIR, "features")
OUT_DIR = os.path.join(V5_DIR, "models")
SRC_DIR = os.path.join(V5_DIR, "src")
sys.path.insert(0, SRC_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

from hybrid_model import HybridV5Model

# ── Config ───────────────────────────────────────────────────────────────────
SEED                  = 42
BATCH_SIZE            = 32
LR                    = 5e-4
WEIGHT_DECAY          = 1e-4
MAX_EPOCHS            = 80
PATIENCE              = 10
VAL_RATIO             = 0.15
MODALITY_DROP_PROB    = 0.3
AUX_WEIGHT            = 0.3
ANCHOR_WEIGHT         = 0.2
JITTER_STD            = 0.05

DEVICE = ("mps"  if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available()         else "cpu")
print("Device:", DEVICE)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_v5():
    labram = np.load(os.path.join(FEAT, "all_labram_embeddings_v5.npy")).astype(np.float32)
    osc    = np.load(os.path.join(FEAT, "all_oscillation_v5.npy")).astype(np.float32)
    eye    = np.load(os.path.join(FEAT, "all_eye_v5.npy")).astype(np.float32)
    mouse  = np.load(os.path.join(FEAT, "all_mouse_v5.npy")).astype(np.float32)
    lab_df = pd.read_csv(os.path.join(FEAT, "labels_v5.csv"))
    meta   = pd.read_csv(os.path.join(BASE, "approach_a", "features",
                                      "all_eeg_embeddings_v3_metadata.csv"))
    labels = lab_df["label"].values.astype(np.int64)
    return labram, osc, eye, mouse, labels, meta


# ── Normalization (leakage-free) ──────────────────────────────────────────────
def subject_normalize_labram(labram, meta):
    """Per-subject z-score (same as V2/V3, no fold info needed)."""
    out = labram.copy()
    for sid in meta["subject_id"].unique():
        idx = (meta["subject_id"] == sid).values
        m   = out[idx].mean(0, keepdims=True)
        s   = out[idx].std(0,  keepdims=True) + 1e-8
        out[idx] = (out[idx] - m) / s
    return out


def fold_normalize(train_arr, test_arr):
    """Fit mean/std on train, apply to both. Shape: (N, ...) or (N, C, T)."""
    flat_train = train_arr.reshape(len(train_arr), -1)
    mean = flat_train.mean(0)
    std  = flat_train.std(0) + 1e-8
    def normalize(arr):
        f = arr.reshape(len(arr), -1)
        return ((f - mean) / std).reshape(arr.shape).astype(np.float32)
    return normalize(train_arr), normalize(test_arr)


# ── Dataset ────────────────────────────────────────────────────────────────────
class V5Dataset(Dataset):
    def __init__(self, labram, osc, eye, mouse, labels, augment=False):
        self.labram  = torch.tensor(labram,  dtype=torch.float32)
        self.osc     = torch.tensor(osc,     dtype=torch.float32)
        self.eye     = torch.tensor(eye,     dtype=torch.float32)
        self.mouse   = torch.tensor(mouse,   dtype=torch.float32)
        self.labels  = torch.tensor(labels,  dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        lab = self.labram[idx]
        osc = self.osc[idx]
        eye = self.eye[idx]
        mou = self.mouse[idx]
        lbl = self.labels[idx]
        if self.augment:
            eye = eye + torch.randn_like(eye) * JITTER_STD
            mou = mou + torch.randn_like(mou) * JITTER_STD
        return lab, osc, eye, mou, lbl


# ── Class weights ──────────────────────────────────────────────────────────────
def class_weights(labels):
    n = len(labels)
    counts = np.bincount(labels)
    w = n / (len(counts) * counts.astype(float))
    return torch.tensor(w, dtype=torch.float32).to(DEVICE)


# ── Training ───────────────────────────────────────────────────────────────────
def train_one_fold(model, train_loader, val_loader, cw):
    criterion = nn.CrossEntropyLoss(weight=cw)
    opt  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None
    history       = {"train_loss": [], "val_loss": [], "stopped_epoch": MAX_EPOCHS}

    rng = random.Random(SEED)

    for epoch in range(MAX_EPOCHS):
        # ── Train ──
        model.train()
        ep_loss = 0.0
        for lab_b, osc_b, eye_b, mou_b, lbl_b in train_loader:
            lab_b, osc_b, eye_b, mou_b, lbl_b = (
                lab_b.to(DEVICE), osc_b.to(DEVICE), eye_b.to(DEVICE),
                mou_b.to(DEVICE), lbl_b.to(DEVICE)
            )

            # Modality dropout: drop one entire modality with MODALITY_DROP_PROB
            mask = {'eeg': True, 'eye': True, 'mouse': True}
            if rng.random() < MODALITY_DROP_PROB:
                mask[rng.choice(['eeg', 'eye', 'mouse'])] = False

            opt.zero_grad()
            logits, aux, attn = model(lab_b, osc_b, eye_b, mou_b, modality_mask=mask)

            # Main loss
            main_loss = criterion(logits, lbl_b)

            # Auxiliary losses
            aux_loss = (criterion(aux['eeg'],   lbl_b) +
                        criterion(aux['eye'],   lbl_b) +
                        criterion(aux['mouse'], lbl_b))

            # Anchor alignment: adapter output vs oscillation temporal mean
            osc_mean  = osc_b.mean(dim=-1)                     # (B, 6)
            osc_mean_n = F.normalize(osc_mean,  dim=-1)
            adap_n     = F.normalize(attn['adapter_output'], dim=-1)
            anchor_loss = F.mse_loss(adap_n, osc_mean_n)

            loss = main_loss + AUX_WEIGHT * aux_loss + ANCHOR_WEIGHT * anchor_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()

        sched.step()
        avg_train = ep_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lab_b, osc_b, eye_b, mou_b, lbl_b in val_loader:
                lab_b, osc_b, eye_b, mou_b, lbl_b = (
                    lab_b.to(DEVICE), osc_b.to(DEVICE), eye_b.to(DEVICE),
                    mou_b.to(DEVICE), lbl_b.to(DEVICE)
                )
                logits, _, _ = model(lab_b, osc_b, eye_b, mou_b)
                val_loss += criterion(logits, lbl_b).item()
        avg_val = val_loss / len(val_loader)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_ctr  = 0
            best_state    = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                history["stopped_epoch"] = epoch + 1
                break

    return best_state, history


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_fold(model, test_loader, meta_test):
    model.eval()
    probs, preds, trues = [], [], []
    band_attns, mod_attns, adapter_outs = [], [], []

    with torch.no_grad():
        for lab_b, osc_b, eye_b, mou_b, lbl_b in test_loader:
            lab_b, osc_b, eye_b, mou_b = (
                lab_b.to(DEVICE), osc_b.to(DEVICE),
                eye_b.to(DEVICE), mou_b.to(DEVICE)
            )
            logits, _, attn = model(lab_b, osc_b, eye_b, mou_b)
            p = torch.softmax(logits, -1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            trues.extend(lbl_b.numpy().tolist())
            band_attns.append(attn['band_attention'].cpu().numpy())
            adapter_outs.append(attn['adapter_output'].cpu().numpy())
            mod_attns.append({k: v.cpu().numpy()
                              for k, v in attn['modality_attention'].items()})

    y_true = np.array(trues); y_pred = np.array(preds); y_prob = np.array(probs)
    metrics = {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc":               float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "n_test": int(len(y_true)), "n_variant": int((y_true==1).sum()), "n_control": int((y_true==0).sum()),
    }
    pred_df = meta_test.copy().reset_index(drop=True)
    pred_df["y_true"] = y_true; pred_df["y_pred"] = y_pred; pred_df["prob_variant"] = y_prob

    attn_arrays = {
        "band_attn":    np.concatenate(band_attns),
        "adapter_out":  np.concatenate(adapter_outs),
        "mod_attn_eeg":   np.concatenate([d["eeg"]   for d in mod_attns]),
        "mod_attn_eye":   np.concatenate([d["eye"]   for d in mod_attns]),
        "mod_attn_mouse": np.concatenate([d["mouse"] for d in mod_attns]),
    }
    return metrics, pred_df, attn_arrays


# ── Main LOSO loop ─────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    labram, osc, eye, mouse, labels, meta = load_v5()
    print(f"V5: LaBraM={labram.shape}  osc={osc.shape}  "
          f"eye={eye.shape}  mouse={mouse.shape}")
    print(f"    labels={dict(zip(*np.unique(labels, return_counts=True)))}")

    # Subject z-score for LaBraM (independent of LOSO fold)
    labram_norm = subject_normalize_labram(labram, meta)
    subjects    = sorted(meta["subject_id"].unique())
    fold_summaries = []

    for fold_idx, test_sid in enumerate(subjects):
        t0 = time.time()
        print(f"\nFold {fold_idx+1}/{len(subjects)} - sub-{test_sid}")
        fold_dir = os.path.join(OUT_DIR, f"fold_{test_sid}")
        os.makedirs(fold_dir, exist_ok=True)

        test_mask  = (meta["subject_id"] == test_sid).values
        train_mask = ~test_mask

        # Split raw arrays
        lab_tr = labram_norm[train_mask]; lab_te = labram_norm[test_mask]
        osc_tr = osc[train_mask];         osc_te = osc[test_mask]
        eye_tr = eye[train_mask];         eye_te = eye[test_mask]
        mou_tr = mouse[train_mask];       mou_te = mouse[test_mask]
        lbl_tr = labels[train_mask];      lbl_te = labels[test_mask]
        meta_te = meta[test_mask].reset_index(drop=True)

        # Leakage-free per-fold normalization for osc, eye, mouse
        osc_tr, osc_te = fold_normalize(osc_tr, osc_te)
        eye_tr, eye_te = fold_normalize(eye_tr, eye_te)
        mou_tr, mou_te = fold_normalize(mou_tr, mou_te)

        # Val split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
        try:
            tr_idx, val_idx = next(sss.split(lab_tr, lbl_tr))
        except Exception:
            tr_idx  = np.arange(len(lbl_tr))
            val_idx = tr_idx[:max(2, int(len(lbl_tr) * 0.1))]

        train_ds = V5Dataset(lab_tr[tr_idx],  osc_tr[tr_idx],  eye_tr[tr_idx],
                             mou_tr[tr_idx],  lbl_tr[tr_idx],  augment=True)
        val_ds   = V5Dataset(lab_tr[val_idx], osc_tr[val_idx], eye_tr[val_idx],
                             mou_tr[val_idx], lbl_tr[val_idx], augment=False)
        test_ds  = V5Dataset(lab_te, osc_te, eye_te, mou_te, lbl_te, augment=False)

        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
        cw    = class_weights(lbl_tr[tr_idx])
        model = HybridV5Model().to(DEVICE)
        best_state, history = train_one_fold(model, train_loader, val_loader, cw)

        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        metrics, pred_df, attn_arr = evaluate_fold(model, test_loader, meta_te)

        elapsed = time.time() - t0
        metrics.update({"fold_time_sec": round(elapsed, 1),
                        "test_subject":  int(test_sid),
                        "stopped_epoch": history["stopped_epoch"]})
        print(f"  ACC={metrics['accuracy']:.3f}  BAL={metrics['balanced_accuracy']:.3f}  "
              f"F1={metrics['f1_macro']:.3f}  AUC={metrics['auc']:.3f}  ({elapsed:.0f}s)")

        # Save fold outputs
        torch.save(best_state, os.path.join(fold_dir, "best_model.pth"))
        with open(os.path.join(fold_dir, "metrics.json"),          "w") as f: json.dump(metrics,  f, indent=2)
        with open(os.path.join(fold_dir, "training_history.json"), "w") as f: json.dump(history,  f, indent=2)
        pred_df.to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)
        np.savez(os.path.join(fold_dir, "attention_weights.npz"),
                 band_attn    = attn_arr["band_attn"],
                 adapter_out  = attn_arr["adapter_out"],
                 mod_attn_eeg = attn_arr["mod_attn_eeg"],
                 mod_attn_eye = attn_arr["mod_attn_eye"],
                 mod_attn_mouse = attn_arr["mod_attn_mouse"])
        fold_summaries.append(metrics)

    accs = [f["accuracy"]          for f in fold_summaries]
    bals = [f["balanced_accuracy"] for f in fold_summaries]
    f1s  = [f["f1_macro"]          for f in fold_summaries]
    aucs = [f["auc"]               for f in fold_summaries]
    summary = {
        "n_folds":  len(fold_summaries),
        "subjects": [int(s) for s in subjects],
        "accuracy":          {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
        "balanced_accuracy": {"mean": float(np.mean(bals)), "std": float(np.std(bals))},
        "f1_macro":          {"mean": float(np.mean(f1s)),  "std": float(np.std(f1s))},
        "auc":               {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))},
        "per_fold": fold_summaries,
    }
    with open(os.path.join(OUT_DIR, "loso_summary_v5.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"V5 LOSO COMPLETE")
    print(f"  ACC  {summary['accuracy']['mean']:.3f} ± {summary['accuracy']['std']:.3f}")
    print(f"  BAL  {summary['balanced_accuracy']['mean']:.3f} ± {summary['balanced_accuracy']['std']:.3f}")
    print(f"  F1   {summary['f1_macro']['mean']:.3f} ± {summary['f1_macro']['std']:.3f}")
    print(f"  AUC  {summary['auc']['mean']:.3f} ± {summary['auc']['std']:.3f}")


if __name__ == "__main__":
    main()
