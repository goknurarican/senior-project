"""
V6 Multi-Class LOSO Training
============================
15-class (control + 14 frustration scenarios), 9-fold LOSO.
Leakage-free normalization: fit on training subjects, apply to test.

Hyperparameters:
  lr=3e-4, weight_decay=1e-3, batch=32, max_epochs=60, patience=10
  Modality dropout p=0.3, auxiliary loss weight=0.3
  Time-series jitter aug sigma=0.05
"""

import json, os, random, sys, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
warnings.filterwarnings("ignore")

V6_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR   = os.path.join(V6_DIR, "features")
MODEL_DIR  = os.path.join(V6_DIR, "models")
EVAL_DIR   = os.path.join(V6_DIR, "evaluation")

sys.path.insert(0, os.path.join(V6_DIR, "src"))
from multiclass_model import V6MultiClassModel

SUBJECTS    = [14, 15, 16, 17, 18, 20, 21, 22, 23]
N_CLASSES   = 15
SEED        = 42
BATCH_SIZE  = 32
LR          = 3e-4
WEIGHT_DECAY = 1e-3
MAX_EPOCHS  = 60
PATIENCE    = 10
AUX_WEIGHT  = 0.3
MOD_DROP_P  = 0.3
JITTER_STD  = 0.05
VAL_RATIO   = 0.15

DEVICE = ("mps"  if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available()         else "cpu")
print(f"Device: {DEVICE}")


#── Data loading ──────────────────────────────────────────────────────────
def load_v6():
    osc   = np.load(os.path.join(FEAT_DIR, "all_oscillation_v6.npy")).astype(np.float32)
    eye   = np.load(os.path.join(FEAT_DIR, "all_eye_v6.npy")).astype(np.float32)
    mouse = np.load(os.path.join(FEAT_DIR, "all_mouse_v6.npy")).astype(np.float32)
    lab   = pd.read_csv(os.path.join(FEAT_DIR, "labels_v6.csv"))
    labels   = lab["label_15class"].values.astype(np.int64)
    subjects = lab["subject_id"].values
    return osc, eye, mouse, labels, subjects, lab


def fold_normalize(train_arr, test_arr):
    """Fit mean/std on train, apply to both (over epochs axis)."""
    mu  = train_arr.mean(axis=0, keepdims=True)
    sig = train_arr.std(axis=0, keepdims=True) + 1e-8
    return (train_arr - mu) / sig, (test_arr - mu) / sig


#── Dataset ───────────────────────────────────────────────────────────────
class V6Dataset(Dataset):
    def __init__(self, osc, eye, mouse, labels, augment=False):
        self.osc, self.eye, self.mouse = osc, eye, mouse
        self.labels  = labels
        self.augment = augment

    def __len__(self):  return len(self.labels)

    def __getitem__(self, i):
        o = self.osc[i].copy()
        e = self.eye[i].copy()
        m = self.mouse[i].copy()
        if self.augment:
            o += np.random.randn(*o.shape).astype(np.float32) * JITTER_STD
            e += np.random.randn(*e.shape).astype(np.float32) * JITTER_STD
            m += np.random.randn(*m.shape).astype(np.float32) * JITTER_STD
        return (torch.tensor(o), torch.tensor(e), torch.tensor(m),
                torch.tensor(self.labels[i], dtype=torch.long))


#── Training ──────────────────────────────────────────────────────────────
def train_fold(osc_tr, eye_tr, mouse_tr, lbl_tr,
               osc_te, eye_te, mouse_te, lbl_te,
               fold_dir):
    os.makedirs(fold_dir, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    #Class weights for imbalanced classes
    classes = np.arange(N_CLASSES)
    present  = np.unique(lbl_tr)
    cw = compute_class_weight('balanced', classes=present, y=lbl_tr)
    weight_vec = torch.ones(N_CLASSES).to(DEVICE)
    for cls, w in zip(present, cw):
        weight_vec[cls] = w

    #Val split
    try:
        sss = StratifiedShuffleSplit(1, test_size=VAL_RATIO, random_state=SEED)
        tr_i, val_i = next(sss.split(osc_tr, lbl_tr))
    except Exception:
        tr_i = np.arange(len(lbl_tr))
        val_i = tr_i[-max(1, len(tr_i)//10):]

    tr_ds  = V6Dataset(osc_tr[tr_i],  eye_tr[tr_i],  mouse_tr[tr_i],  lbl_tr[tr_i],  augment=True)
    val_ds = V6Dataset(osc_tr[val_i], eye_tr[val_i], mouse_tr[val_i], lbl_tr[val_i], augment=False)
    te_ds  = V6Dataset(osc_te,        eye_te,        mouse_te,        lbl_te,        augment=False)

    tr_ld  = DataLoader(tr_ds,  BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=len(tr_ds)>BATCH_SIZE)
    val_ld = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    te_ld  = DataLoader(te_ds,  BATCH_SIZE, shuffle=False, num_workers=0)

    model = V6MultiClassModel(n_classes=N_CLASSES).to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS)
    crit  = nn.CrossEntropyLoss(weight=weight_vec)

    best_val_loss = float('inf'); best_state = None; patience_cnt = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        #── Train ──
        model.train()
        tr_loss = 0; n_tr = 0
        for osc_b, eye_b, mou_b, lbl_b in tr_ld:
            osc_b, eye_b, mou_b, lbl_b = (osc_b.to(DEVICE), eye_b.to(DEVICE),
                                           mou_b.to(DEVICE), lbl_b.to(DEVICE))
            #Modality dropout mask
            mask = {}
            r = random.random()
            if r < MOD_DROP_P / 3:
                mask = {'eeg': False, 'eye': True, 'mouse': True}
            elif r < 2 * MOD_DROP_P / 3:
                mask = {'eeg': True, 'eye': False, 'mouse': True}
            elif r < MOD_DROP_P:
                mask = {'eeg': True, 'eye': True, 'mouse': False}

            logits, aux, _ = model(osc_b, eye_b, mou_b, modality_mask=mask)
            main_loss = crit(logits, lbl_b)
            aux_loss  = (crit(aux['eeg'],   lbl_b) +
                         crit(aux['eye'],   lbl_b) +
                         crit(aux['mouse'], lbl_b)) / 3
            loss = main_loss + AUX_WEIGHT * aux_loss

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(lbl_b); n_tr += len(lbl_b)

        sched.step()

        #── Validate ──
        model.eval(); val_loss = 0; n_val = 0
        with torch.no_grad():
            for osc_b, eye_b, mou_b, lbl_b in val_ld:
                osc_b, eye_b, mou_b, lbl_b = (osc_b.to(DEVICE), eye_b.to(DEVICE),
                                               mou_b.to(DEVICE), lbl_b.to(DEVICE))
                logits, aux, _ = model(osc_b, eye_b, mou_b)
                vl = (crit(logits, lbl_b) +
                      AUX_WEIGHT * (crit(aux['eeg'], lbl_b) +
                                    crit(aux['eye'], lbl_b) +
                                    crit(aux['mouse'], lbl_b)) / 3)
                val_loss += vl.item() * len(lbl_b); n_val += len(lbl_b)

        val_loss /= max(n_val, 1)
        history.append({"epoch": epoch, "train_loss": tr_loss / max(n_tr, 1),
                         "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

    #── Evaluate on test ──
    model.load_state_dict(best_state)
    model.eval()

    all_probs, all_preds, all_true = [], [], []
    all_feat_attn, all_modal_attn  = [], []

    with torch.no_grad():
        for osc_b, eye_b, mou_b, lbl_b in te_ld:
            osc_b, eye_b, mou_b = osc_b.to(DEVICE), eye_b.to(DEVICE), mou_b.to(DEVICE)
            logits, _, attn = model(osc_b, eye_b, mou_b)
            probs = torch.softmax(logits, -1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
            all_true.extend(lbl_b.numpy().tolist())
            all_feat_attn.append(attn['feature_attention'].cpu().numpy())
            ma = attn['modality_attention']
            all_modal_attn.append(np.stack([
                ma['eeg'].cpu().numpy(),
                ma['eye'].cpu().numpy(),
                ma['mouse'].cpu().numpy()
            ], axis=1))

    y_true = np.array(all_true); y_pred = np.array(all_preds)
    probs_arr = np.concatenate(all_probs, axis=0)

    metrics = {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "f1_macro":    float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        "n_test":      int(len(y_true)),
        "classes_in_test": sorted(int(c) for c in np.unique(y_true)),
    }

    torch.save(best_state, os.path.join(fold_dir, "best_model.pth"))
    with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(fold_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    for c in range(N_CLASSES):
        pred_df[f"prob_{c}"] = probs_arr[:, c]
    pred_df.to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)

    feat_attn_arr  = np.concatenate(all_feat_attn, axis=0)
    modal_attn_arr = np.concatenate(all_modal_attn, axis=0)
    np.savez(os.path.join(fold_dir, "attention_weights.npz"),
             feature_attention=feat_attn_arr,
             modality_attention=modal_attn_arr)

    return metrics, y_true, y_pred, probs_arr


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    osc, eye, mouse, labels, subjects, meta = load_v6()
    print(f"V6 data: osc={osc.shape}, eye={eye.shape}, mouse={mouse.shape}")
    print(f"Labels: {np.bincount(labels)} (15 classes)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR,  exist_ok=True)

    fold_summaries = []
    all_y_true, all_y_pred = [], []

    for sid in SUBJECTS:
        t0 = time.time()
        print(f"\n{'='*50}\nFold: sub-{sid} as test")
        te_mask = subjects == sid
        tr_mask = ~te_mask

        #Leakage-free normalization
        osc_tr, osc_te     = fold_normalize(osc[tr_mask], osc[te_mask])
        eye_tr, eye_te     = fold_normalize(eye[tr_mask], eye[te_mask])
        mouse_tr, mouse_te = fold_normalize(mouse[tr_mask], mouse[te_mask])

        lbl_tr = labels[tr_mask]
        lbl_te = labels[te_mask]

        print(f"  Train: {tr_mask.sum()} | Test: {te_mask.sum()}")
        print(f"  Test classes: {sorted(np.unique(lbl_te).tolist())}")

        fold_dir = os.path.join(MODEL_DIR, f"fold_{sid}")
        eval_fold_dir = os.path.join(EVAL_DIR, f"fold_{sid}")
        os.makedirs(eval_fold_dir, exist_ok=True)

        metrics, yt, yp, _ = train_fold(
            osc_tr, eye_tr, mouse_tr, lbl_tr,
            osc_te, eye_te, mouse_te, lbl_te,
            fold_dir
        )
        #Copy predictions to eval dir
        import shutil
        for fn in ["predictions.csv", "attention_weights.npz", "metrics.json"]:
            src = os.path.join(fold_dir, fn)
            dst = os.path.join(eval_fold_dir, fn)
            if os.path.exists(src): shutil.copy2(src, dst)

        elapsed = time.time() - t0
        print(f"  Acc={metrics['accuracy']:.3f}  F1={metrics['f1_macro']:.3f}  ({elapsed:.1f}s)")
        fold_summaries.append({"fold": sid, **metrics})
        all_y_true.extend(yt.tolist()); all_y_pred.extend(yp.tolist())

    #── Summary ──
    accs  = [s["accuracy"] for s in fold_summaries]
    f1s   = [s["f1_macro"] for s in fold_summaries]
    summary = {
        "mean_accuracy":    float(np.mean(accs)),
        "std_accuracy":     float(np.std(accs)),
        "mean_f1_macro":    float(np.mean(f1s)),
        "std_f1_macro":     float(np.std(f1s)),
        "chance_baseline":  round(1 / N_CLASSES, 4),
        "folds":            fold_summaries,
    }
    out_path = os.path.join(EVAL_DIR, "loso_summary_v6.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"V6 LOSO SUMMARY")
    print(f"  Accuracy:  {summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f}")
    print(f"  F1 macro:  {summary['mean_f1_macro']:.3f} ± {summary['std_f1_macro']:.3f}")
    print(f"  Chance:    {summary['chance_baseline']:.3f} (1/15)")

    #Overall confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=list(range(N_CLASSES)))
    np.save(os.path.join(EVAL_DIR, "confusion_matrix.npy"), cm)
    np.save(os.path.join(EVAL_DIR, "all_y_true.npy"), np.array(all_y_true))
    np.save(os.path.join(EVAL_DIR, "all_y_pred.npy"), np.array(all_y_pred))
    print(f"\nResults saved to {EVAL_DIR}")

    #── STOP CHECK ──
    if summary['mean_accuracy'] < summary['chance_baseline'] * 1.5:
        print("\nSTOP: Accuracy near/below chance - see evaluation for details.")


if __name__ == "__main__":
    main()
