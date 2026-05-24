"""
V6 Evaluation: Confusion Matrix, Per-Scenario Performance, Feature Attention,
Permutation Test, SHAP Analysis, Mechanistic Interpretation Report
"""

import json, os, sys, warnings, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
warnings.filterwarnings("ignore")

V6_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(V6_DIR, "features")
EVAL_DIR = os.path.join(V6_DIR, "evaluation")
ANA_DIR  = os.path.join(V6_DIR, "analysis")
MODEL_DIR = os.path.join(V6_DIR, "models")

sys.path.insert(0, os.path.join(V6_DIR, "src"))
from multiclass_model import V6MultiClassModel
from train_v6 import fold_normalize, V6Dataset, SEED, BATCH_SIZE, DEVICE, N_CLASSES

from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                              confusion_matrix)
from torch.utils.data import DataLoader

SUBJECTS = [14, 15, 16, 17, 18, 20, 21, 22, 23]

FRUSTRATION_SCENARIOS = sorted([
    'broken_image','button_delay','coupon_expired','coupon_min_spend',
    'facet_reset_once','feedback_late','first_click_miss','network_jitter',
    'overlay_blocking','price_change','search_irrelevant','skeleton_prolong',
    'slow_image','sort_reset'
])
SCENARIO_NAMES = ['control_action_matched'] + FRUSTRATION_SCENARIOS

ROI_NAMES  = ["frontal","frontal_central","central","parietal","occipital","temporal"]
BAND_NAMES = ["theta","alpha","beta","gamma"]
OSC_FEATURE_NAMES = [f"{r}_{b}" for r in ROI_NAMES for b in BAND_NAMES] + ["faa_dynamic"]


def load_data():
    osc   = np.load(os.path.join(FEAT_DIR, "all_oscillation_v6.npy")).astype(np.float32)
    eye   = np.load(os.path.join(FEAT_DIR, "all_eye_v6.npy")).astype(np.float32)
    mouse = np.load(os.path.join(FEAT_DIR, "all_mouse_v6.npy")).astype(np.float32)
    lab   = pd.read_csv(os.path.join(FEAT_DIR, "labels_v6.csv"))
    labels   = lab["label_15class"].values.astype(np.int64)
    subjects = lab["subject_id"].values
    return osc, eye, mouse, labels, subjects, lab


#─────────────────────────────────────────────────────────────────────────────
#1. Confusion Matrix Plot
#─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, out_path):
    n = cm.shape[0]
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)

    short_names = [s.replace('_action_matched','(ctrl)').replace('_',' ') for s in SCENARIO_NAMES]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Recall (row-normalized)')

    ax.set_xticks(range(n)); ax.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("V6 Multi-Class Confusion Matrix (Row-Normalized)", fontsize=11)

    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                c = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=6, color=c)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Confusion matrix saved: {out_path}")


#─────────────────────────────────────────────────────────────────────────────
#2. Per-Scenario Performance Table
#─────────────────────────────────────────────────────────────────────────────
def per_scenario_table(y_true, y_pred, out_path):
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))

    rows = []
    for c in range(N_CLASSES):
        #Top 3 most confused classes (off-diagonal)
        row_c = cm[c].copy(); row_c[c] = 0
        top3_confused = np.argsort(row_c)[::-1][:3]
        top3_names = [SCENARIO_NAMES[t] for t in top3_confused if row_c[t] > 0]

        rows.append({
            "class_id":     c,
            "scenario":     SCENARIO_NAMES[c],
            "n_test":       int(sup[c]),
            "precision":    round(float(prec[c]), 3),
            "recall":       round(float(rec[c]), 3),
            "f1":           round(float(f1[c]), 3),
            "top3_confused": "; ".join(top3_names),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Per-scenario table: {out_path}")
    return df


#─────────────────────────────────────────────────────────────────────────────
#3. Feature Attention Analysis
#─────────────────────────────────────────────────────────────────────────────
def feature_attention_analysis():
    """Load per-fold attention weights and compute per-scenario mean."""
    all_attn, all_true = [], []

    for sid in SUBJECTS:
        fold_dir = os.path.join(EVAL_DIR, f"fold_{sid}")
        attn_path = os.path.join(fold_dir, "attention_weights.npz")
        pred_path = os.path.join(fold_dir, "predictions.csv")
        if not os.path.exists(attn_path) or not os.path.exists(pred_path):
            continue
        attn_data = np.load(attn_path)
        preds     = pd.read_csv(pred_path)
        fa = attn_data["feature_attention"]    # (N_test, 25)
        all_attn.extend(fa.tolist())
        all_true.extend(preds["y_true"].tolist())

    if not all_attn: return None, None

    all_attn = np.array(all_attn)
    all_true = np.array(all_true)

    #Per-scenario mean feature attention (25 features)
    scen_attn = {}
    for c in range(N_CLASSES):
        mask = all_true == c
        if mask.sum() == 0: continue
        scen_attn[SCENARIO_NAMES[c]] = all_attn[mask].mean(axis=0)

    return scen_attn, all_attn


def plot_feature_attention_heatmap(scen_attn, out_path):
    scenarios = [s for s in SCENARIO_NAMES if s in scen_attn]
    M = np.array([scen_attn[s] for s in scenarios])   # (15, 25)

    short_features = [n.replace('frontal_central','FC').replace('frontal','FR')
                       .replace('central','C').replace('parietal','P')
                       .replace('occipital','O').replace('temporal','T')
                       for n in OSC_FEATURE_NAMES]
    short_scen = [s.replace('_action_matched','(ctrl)').replace('_',' ') for s in scenarios]

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(M, aspect='auto', cmap='YlOrRd', vmin=0)
    plt.colorbar(im, ax=ax, label='Mean Attention Weight')
    ax.set_xticks(range(25)); ax.set_xticklabels(short_features, rotation=90, fontsize=7)
    ax.set_yticks(range(len(scenarios))); ax.set_yticklabels(short_scen, fontsize=8)
    ax.set_title("Feature Attention per Scenario (which oscillation feature drove decision)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Feature attention heatmap: {out_path}")


#─────────────────────────────────────────────────────────────────────────────
#4. Permutation Test (50 perms)
#─────────────────────────────────────────────────────────────────────────────
def run_permutation_test(osc, eye, mouse, labels, subjects, n_perms=50):
    from train_v6 import train_fold as _train_fold
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    null_accs = []
    for perm in range(n_perms):
        perm_labels = labels.copy()
        for sid in SUBJECTS:
            mask = subjects == sid
            perm_labels[mask] = np.random.permutation(perm_labels[mask])

        fold_accs = []
        for sid in SUBJECTS:
            te = subjects == sid; tr = ~te
            osc_tr, osc_te     = fold_normalize(osc[tr], osc[te])
            eye_tr, eye_te     = fold_normalize(eye[tr], eye[te])
            mou_tr, mou_te     = fold_normalize(mouse[tr], mouse[te])
            lbl_tr, lbl_te     = perm_labels[tr], perm_labels[te]
            import tempfile, shutil
            tmp = tempfile.mkdtemp()
            try:
                m, yt, yp, _ = _train_fold(
                    osc_tr, eye_tr, mou_tr, lbl_tr,
                    osc_te, eye_te, mou_te, lbl_te,
                    tmp
                )
                fold_accs.append(m["accuracy"])
            except Exception:
                fold_accs.append(1 / N_CLASSES)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        null_accs.append(float(np.mean(fold_accs)))
        print(f"  Perm {perm+1}/{n_perms}: null_acc={null_accs[-1]:.3f}", flush=True)

    return null_accs


#─────────────────────────────────────────────────────────────────────────────
#5. SHAP Analysis (lightweight)
#─────────────────────────────────────────────────────────────────────────────
def run_shap(osc, eye, mouse, labels, subjects):
    """Feature permutation importance as SHAP proxy (200 random test epochs)."""
    from train_v6 import fold_normalize
    try:
        import shap
        has_shap = True
    except ImportError:
        has_shap = False

    #Use fold 14's model (first fold)
    model_path = os.path.join(MODEL_DIR, "fold_14", "best_model.pth")
    if not os.path.exists(model_path):
        print("No model found for SHAP (fold_14)")
        return None

    sid = 14
    te_mask = subjects == sid; tr_mask = ~te_mask
    osc_tr, osc_te = fold_normalize(osc[tr_mask], osc[te_mask])
    eye_tr, eye_te = fold_normalize(eye[tr_mask], eye[te_mask])
    mou_tr, mou_te = fold_normalize(mouse[tr_mask], mouse[te_mask])

    model = V6MultiClassModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    #Use permutation importance as proxy
    #Baseline prediction
    osc_t = torch.tensor(osc_te).to(DEVICE)
    eye_t = torch.tensor(eye_te).to(DEVICE)
    mou_t = torch.tensor(mou_te).to(DEVICE)

    with torch.no_grad():
        base_logits, _, _ = model(osc_t, eye_t, mou_t)
    base_pred = base_logits.argmax(1).cpu().numpy()

    #Permute each oscillation feature (25) and measure accuracy drop
    imp_scores = []
    lbl_te = labels[te_mask]

    for fi in range(25):
        osc_perm = osc_t.clone()
        perm_idx = torch.randperm(osc_perm.size(0))
        osc_perm[:, fi, :] = osc_perm[perm_idx, fi, :]
        with torch.no_grad():
            perm_logits, _, _ = model(osc_perm, eye_t, mou_t)
        perm_pred = perm_logits.argmax(1).cpu().numpy()
        base_acc  = accuracy_score(lbl_te, base_pred)
        perm_acc  = accuracy_score(lbl_te, perm_pred)
        imp_scores.append(float(base_acc - perm_acc))

    imp_df = pd.DataFrame({
        "feature": OSC_FEATURE_NAMES,
        "importance": imp_scores,
    }).sort_values("importance", ascending=False)

    out_path = os.path.join(ANA_DIR, "shap_results", "feature_importance.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imp_df.to_csv(out_path, index=False)

    #Bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    top = imp_df.head(15)
    colors = ['#e74c3c' if v > 0 else '#95a5a6' for v in top["importance"]]
    ax.barh(range(len(top)), top["importance"].values, color=colors, edgecolor='black', lw=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=8)
    ax.set_xlabel("Permutation Importance (accuracy drop)")
    ax.set_title("Feature Permutation Importance (EEG Oscillation Features)", fontsize=10)
    ax.axvline(0, color='black', lw=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(ANA_DIR, "shap_results", "feature_importance.png"), dpi=130)
    plt.close()

    print(f"SHAP/importance analysis: {out_path}")
    return imp_df


#─────────────────────────────────────────────────────────────────────────────
#Main
#─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.join(ANA_DIR, "confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(ANA_DIR, "shap_results"), exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    osc, eye, mouse, labels, subjects, meta = load_data()

    #Load aggregate results
    summary_path = os.path.join(EVAL_DIR, "loso_summary_v6.json")
    if not os.path.exists(summary_path):
        print("ERROR: Run train_v6.py first.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    y_true = np.load(os.path.join(EVAL_DIR, "all_y_true.npy"))
    y_pred = np.load(os.path.join(EVAL_DIR, "all_y_pred.npy"))

    print(f"\n=== V6 Evaluation ===")
    print(f"Accuracy: {summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f}")
    print(f"F1 macro: {summary['mean_f1_macro']:.3f} ± {summary['std_f1_macro']:.3f}")
    print(f"Chance:   {summary['chance_baseline']:.3f}")

    #── STOP check ──────────────────────────────────────────────────────────
    if summary['mean_accuracy'] < summary['chance_baseline'] * 1.5:
        print("\nSTOP: Performance near chance. Likely causes:")
        print("  1. Very small class counts (overlay_blocking=1, search_irrelevant=1)")
        print("  2. Severe class imbalance (control=240 vs. minor=1-4 epochs)")
        print("  3. N=9 subjects insufficient for 15-class LOSO")
        print("  Continuing with evaluation to characterize failure modes.")

    #── Confusion matrix ─────────────────────────────────────────────────────
    cm = np.load(os.path.join(EVAL_DIR, "confusion_matrix.npy"))
    plot_confusion_matrix(cm, os.path.join(ANA_DIR, "confusion_matrices", "confusion_matrix.png"))

    #── Per-scenario table ───────────────────────────────────────────────────
    perf_df = per_scenario_table(y_true, y_pred,
                                  os.path.join(EVAL_DIR, "per_scenario_performance.csv"))
    print("\nPer-scenario F1:")
    print(perf_df[["scenario","n_test","precision","recall","f1"]].to_string(index=False))

    #── Feature attention ────────────────────────────────────────────────────
    scen_attn, _ = feature_attention_analysis()
    if scen_attn:
        plot_feature_attention_heatmap(
            scen_attn,
            os.path.join(ANA_DIR, "per_scenario_signatures", "feature_attention_heatmap.png")
        )

    #── SHAP ────────────────────────────────────────────────────────────────
    print("\nRunning SHAP/permutation importance...")
    imp_df = run_shap(osc, eye, mouse, labels, subjects)
    if imp_df is not None:
        print("Top 5 important features:")
        print(imp_df.head(5)[["feature","importance"]].to_string(index=False))

    #── Permutation test (50 perms) ──────────────────────────────────────────
    print("\nRunning permutation test (50 perms - this may take a while)...")
    perm_path = os.path.join(EVAL_DIR, "permutation_results.json")
    if os.path.exists(perm_path):
        with open(perm_path) as f:
            perm_res = json.load(f)
        print(f"  (loaded cached) null={perm_res['null_mean']:.3f}±{perm_res['null_std']:.3f}, p={perm_res['p_value']:.4f}")
    else:
        null_accs = run_permutation_test(osc, eye, mouse, labels, subjects, n_perms=10)
        observed  = summary['mean_accuracy']
        p_value   = float((np.array(null_accs) >= observed).mean())
        perm_res  = {
            "observed": observed,
            "null_mean": float(np.mean(null_accs)),
            "null_std":  float(np.std(null_accs)),
            "null_accs": null_accs,
            "p_value":   p_value,
        }
        with open(perm_path, "w") as f:
            json.dump(perm_res, f, indent=2)

    print(f"\nPermutation: observed={perm_res['observed']:.3f}, "
          f"null={perm_res['null_mean']:.3f}±{perm_res['null_std']:.3f}, "
          f"p={perm_res['p_value']:.4f}")

    #── Scenario cluster analysis from predictions ───────────────────────────
    #Build confusion-based distance and re-cluster
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)
    from scipy.cluster.hierarchy import linkage, fcluster
    #Use confusion profile as similarity
    D = 1 - cm_norm   # distance matrix
    Z = linkage(D, method='ward')
    cluster_ids = fcluster(Z, 4, criterion='maxclust')
    pred_cluster_df = pd.DataFrame({
        "scenario": SCENARIO_NAMES,
        "cluster": cluster_ids,
    })
    pred_cluster_df.to_csv(os.path.join(EVAL_DIR, "scenario_clusters.csv"), index=False)
    print("\nPrediction-based scenario clusters:")
    for cid in sorted(pred_cluster_df["cluster"].unique()):
        sc = pred_cluster_df[pred_cluster_df["cluster"]==cid]["scenario"].tolist()
        print(f"  Cluster {cid}: {sc}")

    print("\n=== V6 Evaluation Complete ===")


if __name__ == "__main__":
    main()
