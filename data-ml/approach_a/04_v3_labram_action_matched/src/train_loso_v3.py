"""LOSO Training Pipeline for HusformerBITIRMEEG - v3 (action-matched) dataset."""
import json, os, sys, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(BASE, "features")
OUT_DIR  = os.path.join(BASE, "training", "loso_results_v3")
SRC_DIR  = os.path.join(BASE, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(BASE, "training"))

from husformer import HusformerBITIRMEEG
from train_loso_v2 import (FrustrationDataset, subject_normalize,
                             compute_class_weights, train_fold,
                             BATCH_SIZE, SEED, DEVICE, MAX_EPOCHS, PATIENCE, VAL_RATIO)

os.makedirs(OUT_DIR, exist_ok=True)

def load_v3():
    eeg   = np.load(os.path.join(FEAT_DIR, "all_eeg_embeddings_v3.npy"))
    eye   = np.load(os.path.join(FEAT_DIR, "all_eye_timeseries_v3.npy"))
    mouse = np.load(os.path.join(FEAT_DIR, "all_mouse_timeseries_v3.npy"))
    lab   = pd.read_csv(os.path.join(FEAT_DIR, "labels_v3.csv"))
    meta  = pd.read_csv(os.path.join(FEAT_DIR, "all_eeg_embeddings_v3_metadata.csv"))
    labels = lab["label"].values
    return eeg, eye, mouse, labels, meta

def evaluate_fold(model, test_loader, meta_test):
    model.eval()
    probs, preds, trues = [], [], []
    attn_eeg, attn_eye, attn_mou = [], [], []
    with torch.no_grad():
        for eeg, eye, mou, lbl in test_loader:
            eeg, eye, mou = eeg.to(DEVICE), eye.to(DEVICE), mou.to(DEVICE)
            logits, attn = model(eeg, eye, mou, return_attn=True)
            p = torch.softmax(logits, -1)[:,1].cpu().numpy()
            probs.extend(p.tolist()); preds.extend(logits.argmax(1).cpu().numpy().tolist())
            trues.extend(lbl.numpy().tolist())
            attn_eeg.append(attn["eeg"].numpy()); attn_eye.append(attn["eye"].numpy())
            attn_mou.append(attn["mouse"].numpy())
    y_true, y_pred, y_prob = np.array(trues), np.array(preds), np.array(probs)
    metrics = {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc":               float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true))>1 else 0.0,
        "n_test": int(len(y_true)), "n_variant": int((y_true==1).sum()), "n_control": int((y_true==0).sum()),
    }
    pred_df = meta_test.copy().reset_index(drop=True)
    pred_df["y_true"] = y_true; pred_df["y_pred"] = y_pred; pred_df["prob_variant"] = y_prob
    attn_arr = {
        "eeg":   np.concatenate(attn_eeg), "eye": np.concatenate(attn_eye),
        "mouse": np.concatenate(attn_mou),
    }
    return metrics, pred_df, attn_arr

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    eeg, eye, mouse, labels, meta = load_v3()
    print(f"v3: EEG={eeg.shape} labels={dict(zip(*np.unique(labels, return_counts=True)))}")
    eeg_norm = subject_normalize(eeg, meta)
    subjects = sorted(meta["subject_id"].unique())
    fold_summaries = []

    for fold_idx, test_sid in enumerate(subjects):
        t0 = time.time()
        print(f"\nFold {fold_idx+1}/{len(subjects)} - sub-{test_sid}")
        fold_dir = os.path.join(OUT_DIR, f"fold_{test_sid}")
        os.makedirs(fold_dir, exist_ok=True)
        test_mask  = (meta["subject_id"] == test_sid).values
        train_mask = ~test_mask
        eeg_tr, eye_tr, mou_tr, lbl_tr = eeg_norm[train_mask], eye[train_mask], mouse[train_mask], labels[train_mask]
        eeg_te, eye_te, mou_te, lbl_te = eeg_norm[test_mask],  eye[test_mask],  mouse[test_mask],  labels[test_mask]
        meta_te = meta[test_mask].reset_index(drop=True)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
        try: tr_idx, val_idx = next(sss.split(eeg_tr, lbl_tr))
        except: tr_idx = np.arange(len(lbl_tr)); val_idx = tr_idx[:max(2,int(len(lbl_tr)*0.1))]
        eeg_tv, eye_tv, mou_tv, lbl_tv = eeg_tr[val_idx], eye_tr[val_idx], mou_tr[val_idx], lbl_tr[val_idx]
        eeg_tr, eye_tr, mou_tr, lbl_tr = eeg_tr[tr_idx],  eye_tr[tr_idx],  mou_tr[tr_idx],  lbl_tr[tr_idx]
        print(f"  train={len(lbl_tr)} val={len(lbl_tv)} test={len(lbl_te)}")
        if len(lbl_te) < 4:
            print(f"  WARNING: sub-{test_sid} test set very small ({len(lbl_te)} epochs)")
        train_ds = FrustrationDataset(eeg_tr, eye_tr, mou_tr, lbl_tr, augment=True)
        val_ds   = FrustrationDataset(eeg_tv, eye_tv, mou_tv, lbl_tv, augment=False)
        test_ds  = FrustrationDataset(eeg_te, eye_te, mou_te, lbl_te, augment=False)
        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0)
        cw = compute_class_weights(lbl_tr)
        model = HusformerBITIRMEEG().to(DEVICE)
        best_state, history = train_fold(model, train_loader, val_loader, cw)
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        metrics, pred_df, attn_arr = evaluate_fold(model, test_loader, meta_te)
        elapsed = time.time() - t0
        metrics.update({"fold_time_sec": round(elapsed,1), "test_subject": int(test_sid),
                        "stopped_epoch": history["stopped_epoch"]})
        print(f"  ACC={metrics['accuracy']:.3f} BAL={metrics['balanced_accuracy']:.3f} "
              f"F1={metrics['f1_macro']:.3f} AUC={metrics['auc']:.3f} ({elapsed:.0f}s)")
        torch.save(best_state, os.path.join(fold_dir, "model_best.pth"))
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f: json.dump(metrics, f, indent=2)
        with open(os.path.join(fold_dir, "training_history.json"), "w") as f: json.dump(history, f, indent=2)
        pred_df.to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)
        np.savez(os.path.join(fold_dir, "attention_weights.npz"),
                 eeg=attn_arr["eeg"], eye=attn_arr["eye"], mouse=attn_arr["mouse"])
        fold_summaries.append(metrics)

    accs  = [f["accuracy"]          for f in fold_summaries]
    bals  = [f["balanced_accuracy"] for f in fold_summaries]
    f1s   = [f["f1_macro"]          for f in fold_summaries]
    aucs  = [f["auc"]               for f in fold_summaries]
    summary = {
        "n_folds": len(fold_summaries), "subjects": [int(s) for s in subjects],
        "accuracy":          {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
        "balanced_accuracy": {"mean": float(np.mean(bals)), "std": float(np.std(bals))},
        "f1_macro":          {"mean": float(np.mean(f1s)),  "std": float(np.std(f1s))},
        "auc":               {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))},
        "per_fold": fold_summaries,
    }
    with open(os.path.join(OUT_DIR, "loso_summary_v3.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*50}")
    print(f"v3 LOSO COMPLETE")
    print(f"  ACC  {summary['accuracy']['mean']:.3f} ± {summary['accuracy']['std']:.3f}")
    print(f"  BAL  {summary['balanced_accuracy']['mean']:.3f} ± {summary['balanced_accuracy']['std']:.3f}")
    print(f"  F1   {summary['f1_macro']['mean']:.3f} ± {summary['f1_macro']['std']:.3f}")
    print(f"  AUC  {summary['auc']['mean']:.3f} ± {summary['auc']['std']:.3f}")

if __name__ == "__main__":
    main()
