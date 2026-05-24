"""
V5 Evaluation: Ablation, Permutation Test, Band Attention, Adapter Analysis.
Outputs → approach_a/v5_hybrid/evaluation/
"""
import json, os, random, sys, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")

SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
V5_DIR    = os.path.dirname(SRC_DIR)
APPROACH_A = os.path.dirname(V5_DIR)
sys.path.insert(0, SRC_DIR)

from hybrid_model import HybridV5Model
from train_v5 import (load_v5, subject_normalize_labram, fold_normalize,
                       V5Dataset, class_weights, BATCH_SIZE, SEED, DEVICE,
                       MAX_EPOCHS, PATIENCE, VAL_RATIO, MODALITY_DROP_PROB,
                       AUX_WEIGHT, ANCHOR_WEIGHT, JITTER_STD, LR, WEIGHT_DECAY)

MODEL_DIR = os.path.join(V5_DIR, "models")
EVAL_DIR  = os.path.join(V5_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

BAND_NAMES = ['frontal_theta', 'frontal_alpha', 'parietal_alpha',
              'central_beta', 'faa_dynamic', 'engagement_index']


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_fold_checkpoint(sid):
    ckpt = os.path.join(MODEL_DIR, f"fold_{sid}", "best_model.pth")
    model = HybridV5Model().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


def zero_inference(model, lab, osc, eye, mou):
    ds = V5Dataset(lab, osc, eye, mou, np.zeros(len(lab), dtype=np.int64), augment=False)
    loader = DataLoader(ds, BATCH_SIZE, shuffle=False, num_workers=0)
    probs, trues = [], []
    with torch.no_grad():
        for lb, ob, eb, mb, _ in loader:
            lb, ob, eb, mb = lb.to(DEVICE), ob.to(DEVICE), eb.to(DEVICE), mb.to(DEVICE)
            logits, _, _ = model(lb, ob, eb, mb)
            probs.extend(torch.softmax(logits,-1)[:,1].cpu().numpy())
    return np.array(probs)


# ── Step 1: Ablation ─────────────────────────────────────────────────────────
def run_ablation():
    print("=== V5 Ablation Study ===")
    labram, osc, eye, mouse, labels, meta = load_v5()
    labram_norm = subject_normalize_labram(labram, meta)
    subjects = sorted(meta["subject_id"].unique())

    conditions = {
        "full":       (False,False,False),
        "no_eeg":     (True, False,False),
        "no_eye":     (False,True, False),
        "no_mouse":   (False,False,True),
        "eeg_only":   (False,True, True),
        "eye_only":   (True, False,True),
        "mouse_only": (True, True, False),
    }

    rows = []
    for cond, (z_eeg, z_eye, z_mou) in conditions.items():
        fold_aucs, fold_accs = [], []
        for sid in subjects:
            model = load_fold_checkpoint(sid)
            mask   = (meta["subject_id"] == sid).values
            tr_mask = ~mask

            lab_te = labram_norm[mask].copy()
            osc_te = osc[mask].copy()
            eye_te = eye[mask].copy()
            mou_te = mouse[mask].copy()
            lbl_te = labels[mask]

            # Same leakage-free normalization as training
            _, osc_te_n = fold_normalize(osc[tr_mask], osc_te)
            _, eye_te_n = fold_normalize(eye[tr_mask], eye_te)
            _, mou_te_n = fold_normalize(mouse[tr_mask], mou_te)

            if z_eeg: lab_te[:] = 0.0
            if z_eye: eye_te_n[:] = 0.0
            if z_mou: mou_te_n[:] = 0.0

            ds = V5Dataset(lab_te, osc_te_n, eye_te_n, mou_te_n, lbl_te, augment=False)
            loader = DataLoader(ds, BATCH_SIZE, shuffle=False, num_workers=0)

            probs, trues = [], []
            with torch.no_grad():
                for lb,ob,eb,mb,lbl_b in loader:
                    lb,ob,eb,mb = lb.to(DEVICE),ob.to(DEVICE),eb.to(DEVICE),mb.to(DEVICE)
                    logits,_,_ = model(lb,ob,eb,mb)
                    probs.extend(torch.softmax(logits,-1)[:,1].cpu().numpy())
                    trues.extend(lbl_b.numpy())

            y_true=np.array(trues); y_prob=np.array(probs)
            fold_accs.append(accuracy_score(y_true,(y_prob>=0.5).astype(int)))
            if len(np.unique(y_true))>1:
                fold_aucs.append(roc_auc_score(y_true,y_prob))

        rows.append({
            "condition":  cond,
            "acc_mean":   float(np.mean(fold_accs)),  "acc_std":  float(np.std(fold_accs)),
            "auc_mean":   float(np.mean(fold_aucs)) if fold_aucs else 0.0,
            "auc_std":    float(np.std(fold_aucs))  if fold_aucs else 0.0,
        })
        print(f"  {cond:12s}: AUC={rows[-1]['auc_mean']:.3f}±{rows[-1]['auc_std']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EVAL_DIR, "ablation_v5.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["green" if r["condition"]=="full" else "steelblue" for _,r in df.iterrows()]
    ax.barh(df["condition"], df["auc_mean"], xerr=df["auc_std"], color=colors, alpha=0.85, capsize=4)
    ax.axvline(0.5, color="red", lw=1, ls="--", label="Chance")
    ax.set_xlabel("Mean AUC (LOSO)"); ax.set_title("V5 Modality Ablation (Hybrid Model)")
    ax.set_xlim(0,1.05); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "ablation_v5.png"), dpi=150); plt.close(fig)
    print(f"  Ablation saved → {EVAL_DIR}")
    return df


# ── Step 2: Permutation Test ─────────────────────────────────────────────────
def run_permutation(n_perms=20, perm_epochs=3):
    import random as rlib
    print(f"=== V5 Permutation Test ({n_perms} perms × {perm_epochs} epochs) ===")
    torch.manual_seed(SEED); np.random.seed(SEED); rlib.seed(SEED)

    labram, osc, eye, mouse, labels, meta = load_v5()
    labram_norm = subject_normalize_labram(labram, meta)
    subjects = sorted(meta["subject_id"].unique())

    with open(os.path.join(MODEL_DIR, "loso_summary_v5.json")) as f:
        summ = json.load(f)
    true_auc = summ["auc"]["mean"]

    null_aucs = []
    for perm in range(n_perms):
        rng = np.random.default_rng(SEED + perm)
        lbl_p = rng.permutation(labels)
        fold_aucs = []

        for sid in subjects:
            mask_te = (meta["subject_id"]==sid).values
            mask_tr = ~mask_te

            lab_tr=labram_norm[mask_tr]; lab_te=labram_norm[mask_te]
            osc_tr=osc[mask_tr];        osc_te=osc[mask_te]
            eye_tr=eye[mask_tr];        eye_te=eye[mask_te]
            mou_tr=mouse[mask_tr];      mou_te=mouse[mask_te]
            lbl_tr=lbl_p[mask_tr];      lbl_te=lbl_p[mask_te]

            if len(np.unique(lbl_te))<2: continue
            osc_tr_n,osc_te_n = fold_normalize(osc_tr,osc_te)
            eye_tr_n,eye_te_n = fold_normalize(eye_tr,eye_te)
            mou_tr_n,mou_te_n = fold_normalize(mou_tr,mou_te)

            sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
            try: tr_i,val_i = next(sss.split(lab_tr,lbl_tr))
            except: continue

            tr_ds = V5Dataset(lab_tr[tr_i], osc_tr_n[tr_i], eye_tr_n[tr_i],
                              mou_tr_n[tr_i], lbl_tr[tr_i], augment=False)
            te_ds = V5Dataset(lab_te, osc_te_n, eye_te_n, mou_te_n, lbl_te, augment=False)
            tr_loader = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
            te_loader = DataLoader(te_ds, BATCH_SIZE, shuffle=False, num_workers=0)

            cw    = class_weights(lbl_tr[tr_i])
            model = HybridV5Model().to(DEVICE)
            crit  = nn.CrossEntropyLoss(weight=cw)
            opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            for _ in range(perm_epochs):
                model.train()
                for lb,ob,eb,mb,lbl_b in tr_loader:
                    lb,ob,eb,mb,lbl_b = (lb.to(DEVICE),ob.to(DEVICE),eb.to(DEVICE),
                                          mb.to(DEVICE),lbl_b.to(DEVICE))
                    opt.zero_grad()
                    logits,aux,attn = model(lb,ob,eb,mb)
                    osc_mean = ob.mean(-1)
                    osc_n = F.normalize(osc_mean,-1)
                    adap_n = F.normalize(attn['adapter_output'],-1)
                    loss = (crit(logits,lbl_b)
                            + AUX_WEIGHT*(crit(aux['eeg'],lbl_b)+crit(aux['eye'],lbl_b)+crit(aux['mouse'],lbl_b))
                            + ANCHOR_WEIGHT*F.mse_loss(adap_n,osc_n))
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    opt.step()

            model.eval()
            probs,trues = [],[]
            with torch.no_grad():
                for lb,ob,eb,mb,lbl_b in te_loader:
                    lb,ob,eb,mb = lb.to(DEVICE),ob.to(DEVICE),eb.to(DEVICE),mb.to(DEVICE)
                    logits,_,_ = model(lb,ob,eb,mb)
                    probs.extend(torch.softmax(logits,-1)[:,1].cpu().numpy())
                    trues.extend(lbl_b.cpu().numpy())
            if len(np.unique(trues))>1:
                fold_aucs.append(roc_auc_score(trues,probs))

        if fold_aucs:
            null_aucs.append(float(np.mean(fold_aucs)))
            print(f"  perm {perm+1}/{n_perms}  null_auc={null_aucs[-1]:.3f}")

    null_arr = np.array(null_aucs)
    p_val = float((null_arr>=true_auc).sum()/len(null_arr)) if len(null_arr) else 1.0
    result = {"true_auc": true_auc, "n_perms": int(len(null_arr)),
              "null_mean": float(null_arr.mean()), "null_std": float(null_arr.std()),
              "p_value": p_val, "significant": bool(p_val<0.05),
              "null_distribution": null_arr.tolist()}
    with open(os.path.join(EVAL_DIR,"permutation_test_v5.json"),"w") as f:
        json.dump(result,f,indent=2)

    fig,ax = plt.subplots(figsize=(7,4))
    ax.hist(null_arr, bins=10, color="steelblue", alpha=0.7, label="Null distribution")
    ax.axvline(true_auc, color="red", lw=2, label=f"True AUC={true_auc:.3f}")
    ax.set_xlabel("Mean AUC"); ax.set_ylabel("Count")
    ax.set_title(f"V5 Permutation Test  p={p_val:.4f}")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR,"permutation_null_dist_v5.png"),dpi=150); plt.close(fig)
    print(f"  true_AUC={true_auc:.3f}  p={p_val:.4f}  {'SIGNIFICANT' if p_val<0.05 else 'not sig'}")
    return result


# ── Step 3: Band Attention Analysis ─────────────────────────────────────────
def run_band_attention_analysis():
    print("=== Band Attention Analysis ===")
    labram, osc, eye, mouse, labels, meta = load_v5()
    labram_norm = subject_normalize_labram(labram, meta)
    subjects = sorted(meta["subject_id"].unique())

    all_band_attn   = []
    all_adapter_out = []
    all_labels      = []
    all_subjects    = []

    for sid in subjects:
        fold_dir = os.path.join(MODEL_DIR, f"fold_{sid}")
        npz = np.load(os.path.join(fold_dir, "attention_weights.npz"))
        preds = pd.read_csv(os.path.join(fold_dir, "predictions.csv"))

        all_band_attn.append(npz["band_attn"])       # (n_test, 6)
        all_adapter_out.append(npz["adapter_out"])    # (n_test, 6)
        all_labels.extend(preds["y_true"].tolist())
        all_subjects.extend([sid]*len(preds))

    band_attn_arr   = np.concatenate(all_band_attn)   # (480, 6)
    adapter_out_arr = np.concatenate(all_adapter_out) # (480, 6)
    labels_arr      = np.array(all_labels)

    # Per-class mean band attention
    v_attn  = band_attn_arr[labels_arr==1].mean(0)
    c_attn  = band_attn_arr[labels_arr==0].mean(0)
    diff    = v_attn - c_attn

    print("  Band attention - Variant vs Control:")
    for i, bn in enumerate(BAND_NAMES):
        print(f"    {bn:25s}: variant={v_attn[i]:.4f}  control={c_attn[i]:.4f}  diff={diff[i]:+.4f}")

    # Save table
    attn_df = pd.DataFrame({
        "band":           BAND_NAMES,
        "variant_mean":   v_attn.tolist(),
        "control_mean":   c_attn.tolist(),
        "diff":           diff.tolist(),
    })
    attn_df.to_csv(os.path.join(EVAL_DIR,"band_attention_summary.csv"), index=False)

    # Adapter output correlation with oscillation temporal mean (anchor alignment check)
    osc_arr = np.load(os.path.join(V5_DIR,"features","all_oscillation_v5.npy"))  # (480,6,110)
    osc_mean = osc_arr.mean(-1)  # (480, 6)
    corrs = []
    for i in range(6):
        c = np.corrcoef(adapter_out_arr[:,i], osc_mean[:,i])[0,1]
        corrs.append(c)
        print(f"  Adapter-Osc corr [{BAND_NAMES[i]:25s}]: r={c:.4f}")

    corr_df = pd.DataFrame({"band": BAND_NAMES, "pearson_r": corrs})
    corr_df.to_csv(os.path.join(EVAL_DIR,"adapter_alignment_correlation.csv"), index=False)

    # --- Figures ---
    # 1. Band attention bar: variant vs control
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(BAND_NAMES)); w=0.35
    ax.bar(x-w/2, v_attn, w, label="Variant",  color="tomato",    alpha=0.8)
    ax.bar(x+w/2, c_attn, w, label="Control",  color="steelblue", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(BAND_NAMES, rotation=20, ha="right")
    ax.set_ylabel("Mean Attention Weight"); ax.set_title("Band Attention: Variant vs Control")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR,"band_attention_bars.png"),dpi=150); plt.close(fig)

    # 2. Adapter alignment: scatter per band
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, (bn, ax) in enumerate(zip(BAND_NAMES, axes.flat)):
        ax.scatter(osc_mean[:,i], adapter_out_arr[:,i],
                   c=['tomato' if l==1 else 'steelblue' for l in labels_arr],
                   alpha=0.3, s=10)
        ax.set_title(f"{bn}\nr={corrs[i]:.3f}")
        ax.set_xlabel("Osc temporal mean"); ax.set_ylabel("Adapter output")
    fig.suptitle("Adapter Output vs Oscillation Temporal Mean (anchor alignment)")
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR,"adapter_output_distributions.png"),dpi=150); plt.close(fig)

    print(f"  Band attention + adapter analysis saved → {EVAL_DIR}")
    return attn_df, corr_df


# ── Step 4: Per-subject table ────────────────────────────────────────────────
def per_subject_table():
    with open(os.path.join(MODEL_DIR,"loso_summary_v5.json")) as f:
        summ = json.load(f)
    rows = []
    for fold in summ["per_fold"]:
        rows.append({
            "subject": fold["test_subject"],
            "accuracy": fold["accuracy"], "balanced_accuracy": fold["balanced_accuracy"],
            "f1_macro": fold["f1_macro"], "auc": fold["auc"],
            "n_test": fold["n_test"], "stopped_epoch": fold["stopped_epoch"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EVAL_DIR,"per_subject_v5.csv"), index=False)
    print("  Per-subject table saved.")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    per_subject_table()
    ablation_df = run_ablation()
    print()
    perm_result = run_permutation(n_perms=20, perm_epochs=3)
    print()
    band_attn_df, corr_df = run_band_attention_analysis()
    print("\n=== V5 EVALUATION COMPLETE ===")
