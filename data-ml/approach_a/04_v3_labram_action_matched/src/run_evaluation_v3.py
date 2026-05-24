"""
V3 Evaluation Pipeline: Ablation study + Permutation test using action-matched dataset.
Results → approach_a/evaluation/v3/
"""
import json, os, sys, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR  = os.path.join(BASE, "features")
V3_RES    = os.path.join(BASE, "training", "loso_results_v3")
EVAL_V3   = os.path.join(BASE, "evaluation", "v3")
SRC_DIR   = os.path.join(BASE, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(BASE, "training"))

from husformer import HusformerBITIRMEEG
from train_loso_v2 import (FrustrationDataset, subject_normalize,
                             compute_class_weights, BATCH_SIZE, SEED, DEVICE)
from train_loso_v3 import load_v3

os.makedirs(EVAL_V3, exist_ok=True)


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------
def run_ablation():
    print("=== V3 Ablation Study ===")
    eeg, eye, mouse, labels, meta = load_v3()
    eeg_norm = subject_normalize(eeg, meta)
    subjects = sorted(meta["subject_id"].unique())

    conditions = {
        "full":       (False, False, False),
        "no_eeg":     (True,  False, False),
        "no_eye":     (False, True,  False),
        "no_mouse":   (False, False, True),
        "eeg_only":   (False, True,  True),
        "eye_only":   (True,  False, True),
        "mouse_only": (True,  True,  False),
    }

    rows = []
    for cond_name, (zero_eeg, zero_eye, zero_mou) in conditions.items():
        fold_aucs, fold_accs, fold_bals = [], [], []
        for test_sid in subjects:
            ckpt = os.path.join(V3_RES, f"fold_{test_sid}", "model_best.pth")
            if not os.path.exists(ckpt):
                print(f"  WARNING: no checkpoint for sub-{test_sid}"); continue

            model = HusformerBITIRMEEG().to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            model.eval()

            mask   = (meta["subject_id"] == test_sid).values
            eeg_te = eeg_norm[mask].copy()
            eye_te = eye[mask].copy()
            mou_te = mouse[mask].copy()
            lbl_te = labels[mask]

            if zero_eeg: eeg_te[:] = 0.0
            if zero_eye: eye_te[:] = 0.0
            if zero_mou: mou_te[:] = 0.0

            ds     = FrustrationDataset(eeg_te, eye_te, mou_te, lbl_te, augment=False)
            loader = DataLoader(ds, BATCH_SIZE, shuffle=False, num_workers=0)

            probs, trues = [], []
            with torch.no_grad():
                for eb, ib, mb, lb in loader:
                    eb, ib, mb = eb.to(DEVICE), ib.to(DEVICE), mb.to(DEVICE)
                    p = torch.softmax(model(eb, ib, mb), -1)[:, 1]
                    probs.extend(p.cpu().numpy()); trues.extend(lb.numpy())

            y_true = np.array(trues); y_prob = np.array(probs)
            y_pred = (y_prob >= 0.5).astype(int)
            fold_accs.append(accuracy_score(y_true, y_pred))
            fold_bals.append(balanced_accuracy_score(y_true, y_pred))
            if len(np.unique(y_true)) > 1:
                fold_aucs.append(roc_auc_score(y_true, y_prob))

        rows.append({
            "condition":    cond_name,
            "acc_mean":     float(np.mean(fold_accs)),
            "acc_std":      float(np.std(fold_accs)),
            "bal_acc_mean": float(np.mean(fold_bals)),
            "bal_acc_std":  float(np.std(fold_bals)),
            "auc_mean":     float(np.mean(fold_aucs)) if fold_aucs else 0.0,
            "auc_std":      float(np.std(fold_aucs))  if fold_aucs else 0.0,
        })
        print(f"  {cond_name:12s}: AUC={rows[-1]['auc_mean']:.3f}±{rows[-1]['auc_std']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EVAL_V3, "ablation_study_v3.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["green" if r["condition"] == "full" else "steelblue" for _, r in df.iterrows()]
    ax.barh(df["condition"], df["auc_mean"], xerr=df["auc_std"],
            color=colors, alpha=0.85, capsize=4)
    ax.axvline(0.5, color="red", lw=1, ls="--", label="Chance")
    ax.set_xlabel("Mean AUC (LOSO)"); ax.set_title("V3 Modality Ablation Study (Action-Matched)")
    ax.set_xlim(0, 1.05); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(EVAL_V3, "ablation_study_v3.png"), dpi=150)
    plt.close(fig)
    print(f"  Ablation saved → {EVAL_V3}")
    return df


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------
def run_permutation(n_perms=20, perm_epochs=3):
    import torch.optim as optim
    print(f"=== V3 Permutation Test ({n_perms} perms × {perm_epochs} epochs) ===")
    torch.manual_seed(SEED); np.random.seed(SEED)

    eeg, eye, mouse, labels, meta = load_v3()
    eeg_norm = subject_normalize(eeg, meta)
    subjects = sorted(meta["subject_id"].unique())

    with open(os.path.join(V3_RES, "loso_summary_v3.json")) as f:
        summary = json.load(f)
    true_auc = summary["auc"]["mean"]

    null_aucs = []
    for perm in range(n_perms):
        rng = np.random.default_rng(SEED + perm)
        labels_perm = rng.permutation(labels)
        fold_aucs = []

        for test_sid in subjects:
            mask_te = (meta["subject_id"] == test_sid).values
            mask_tr = ~mask_te

            eeg_tr = eeg_norm[mask_tr]; eye_tr = eye[mask_tr]
            mou_tr = mouse[mask_tr];    lbl_tr = labels_perm[mask_tr]
            eeg_te = eeg_norm[mask_te]; eye_te = eye[mask_te]
            mou_te = mouse[mask_te];    lbl_te = labels_perm[mask_te]

            if len(np.unique(lbl_te)) < 2:
                continue

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
            try:
                tr_idx, val_idx = next(sss.split(eeg_tr, lbl_tr))
            except ValueError:
                continue

            train_ds = FrustrationDataset(eeg_tr[tr_idx], eye_tr[tr_idx],
                                           mou_tr[tr_idx], lbl_tr[tr_idx], augment=False)
            val_ds   = FrustrationDataset(eeg_tr[val_idx], eye_tr[val_idx],
                                           mou_tr[val_idx], lbl_tr[val_idx], augment=False)
            test_ds  = FrustrationDataset(eeg_te, eye_te, mou_te, lbl_te, augment=False)

            train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
            test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0)

            cw = compute_class_weights(lbl_tr[tr_idx])
            model = HusformerBITIRMEEG().to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=cw)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

            for _ in range(perm_epochs):
                model.train()
                for eb, ib, mb, lb in train_loader:
                    eb, ib, mb, lb = eb.to(DEVICE), ib.to(DEVICE), mb.to(DEVICE), lb.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(eb, ib, mb), lb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            model.eval()
            probs, trues = [], []
            with torch.no_grad():
                for eb, ib, mb, lb in test_loader:
                    eb, ib, mb = eb.to(DEVICE), ib.to(DEVICE), mb.to(DEVICE)
                    p = torch.softmax(model(eb, ib, mb), -1)[:, 1]
                    probs.extend(p.cpu().numpy()); trues.extend(lb.cpu().numpy())
            if len(np.unique(trues)) > 1:
                fold_aucs.append(roc_auc_score(trues, probs))

        if fold_aucs:
            null_aucs.append(float(np.mean(fold_aucs)))
            print(f"  perm {perm+1}/{n_perms}  null_auc={null_aucs[-1]:.3f}")
        else:
            print(f"  perm {perm+1}/{n_perms}  skipped")

    null_arr = np.array(null_aucs)
    p_val = float((null_arr >= true_auc).sum() / len(null_arr)) if len(null_arr) else 1.0
    result = {
        "true_auc": true_auc, "n_perms": int(len(null_arr)),
        "null_mean": float(null_arr.mean()), "null_std": float(null_arr.std()),
        "p_value": p_val, "significant": bool(p_val < 0.05),
        "null_distribution": null_arr.tolist(),
    }
    with open(os.path.join(EVAL_V3, "permutation_test_v3.json"), "w") as f:
        json.dump(result, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_arr, bins=10, color="steelblue", alpha=0.7, label="Null distribution")
    ax.axvline(true_auc, color="red", lw=2, label=f"True AUC={true_auc:.3f}")
    ax.set_xlabel("Mean AUC"); ax.set_ylabel("Count")
    ax.set_title(f"V3 Permutation Test  p={p_val:.4f}")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(EVAL_V3, "permutation_null_dist_v3.png"), dpi=150)
    plt.close(fig)

    print(f"  true_AUC={true_auc:.3f}  p={p_val:.4f}  "
          f"({'SIGNIFICANT' if p_val < 0.05 else 'not significant'})")
    print(f"  Permutation test saved → {EVAL_V3}")
    return result


if __name__ == "__main__":
    ablation_df = run_ablation()
    print()
    perm_result = run_permutation(n_perms=20, perm_epochs=3)
    print("\n=== V3 EVALUATION COMPLETE ===")
    print(f"Ablation saved: {EVAL_V3}/ablation_study_v3.csv")
    print(f"Permutation test saved: {EVAL_V3}/permutation_test_v3.json")
