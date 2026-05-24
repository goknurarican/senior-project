"""
EEG Pipeline Diagnostic Tests - 4 parallel hypotheses for AUC=1.000

Test 1: Subject normalization leakage (per-subject vs global-from-train)
Test 2: Subject identity classification from LaBraM embeddings
Test 3: Random label shuffle (5 seeds) - structural leakage check
Test 4: Classical EEG features + RandomForest LOSO baseline

All tests use V3 dataset (480 epochs, 9 subjects, balanced).
Quick training: 10 epochs, no early stopping.
"""
import json, os, random, sys, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")

DIAG_DIR   = os.path.dirname(os.path.abspath(__file__))
APPROACH_A = os.path.dirname(DIAG_DIR)
PROJECT    = os.path.dirname(APPROACH_A)
FEAT_V3    = os.path.join(APPROACH_A, "features")
FEAT_V5    = os.path.join(APPROACH_A, "v5_hybrid", "features")

SEED    = 42
EPOCHS  = 10
BS      = 32
LR      = 1e-3
DEVICE  = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Shared data loading ───────────────────────────────────────────────────────
def load_v3():
    lab    = np.load(os.path.join(FEAT_V3, "all_eeg_embeddings_v3.npy")).astype(np.float32)
    labels = pd.read_csv(os.path.join(FEAT_V3, "labels_v3.csv"))["label"].values.astype(np.int64)
    meta   = pd.read_csv(os.path.join(FEAT_V3, "all_eeg_embeddings_v3_metadata.csv"))
    return lab, labels, meta


# ── Simple MLP classifier for EEG-only tests ─────────────────────────────────
class SimpleMLP(nn.Module):
    def __init__(self, in_dim=200, hidden=64, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 32), nn.GELU(),
            nn.Linear(32, n_classes)
        )
    def forward(self, x):
        return self.net(x)


def train_eval_quick(eeg_tr, lbl_tr, eeg_te, lbl_te, n_classes=2):
    """10-epoch MLP, returns AUC and ACC on test set."""
    model = SimpleMLP(200, 64, n_classes).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)
    crit  = nn.CrossEntropyLoss()

    # Tiny val split for loss monitoring (not used for stopping)
    try:
        sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=SEED)
        tr_i, _ = next(sss.split(eeg_tr, lbl_tr))
    except Exception:
        tr_i = np.arange(len(lbl_tr))

    tr_ds  = TensorDataset(torch.tensor(eeg_tr[tr_i]),  torch.tensor(lbl_tr[tr_i]))
    te_ds  = TensorDataset(torch.tensor(eeg_te),        torch.tensor(lbl_te))
    tr_ld  = DataLoader(tr_ds, BS, shuffle=True, num_workers=0)
    te_ld  = DataLoader(te_ds, BS, shuffle=False, num_workers=0)

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            p = torch.softmax(model(xb.to(DEVICE)), -1).cpu().numpy()
            probs.append(p); trues.extend(yb.numpy())
    probs_arr = np.concatenate(probs); trues_arr = np.array(trues)
    preds_arr = probs_arr.argmax(1)

    acc = float(accuracy_score(trues_arr, preds_arr))
    if n_classes == 2 and len(np.unique(trues_arr)) > 1:
        auc = float(roc_auc_score(trues_arr, probs_arr[:, 1]))
    else:
        auc = 0.0
    return acc, auc


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1 - Normalization Leakage
# ═══════════════════════════════════════════════════════════════════════════════
def norm_version_A(labram, meta):
    """Current: per-subject z-score (uses all subject epochs, pre-LOSO)."""
    out = labram.copy()
    for sid in meta["subject_id"].unique():
        idx = (meta["subject_id"] == sid).values
        m = out[idx].mean(0, keepdims=True)
        s = out[idx].std(0, keepdims=True) + 1e-8
        out[idx] = (out[idx] - m) / s
    return out


def test1_normalization_leakage():
    print("\n" + "="*60)
    print("TEST 1: Normalization Leakage")
    print("="*60)
    torch.manual_seed(SEED); np.random.seed(SEED)

    labram, labels, meta = load_v3()
    subjects = sorted(meta["subject_id"].unique())

    # Version A: pre-LOSO per-subject normalization (current approach)
    labram_A = norm_version_A(labram, meta)

    results = {}
    for version_name, use_leaky in [("version_A_leaky_normalization", True),
                                     ("version_B_leakage_free",        False)]:
        fold_accs, fold_aucs = [], []
        for sid in subjects:
            te_mask = (meta["subject_id"] == sid).values
            tr_mask = ~te_mask

            if use_leaky:
                # Version A: use pre-normalized (includes test subject's own stats)
                eeg_tr = labram_A[tr_mask]
                eeg_te = labram_A[te_mask]
            else:
                # Version B: compute global mean/std from TRAINING subjects only
                train_data = labram[tr_mask]
                g_mean = train_data.mean(0, keepdims=True)
                g_std  = train_data.std(0, keepdims=True) + 1e-8
                eeg_tr = (labram[tr_mask] - g_mean) / g_std
                eeg_te = (labram[te_mask] - g_mean) / g_std

            lbl_tr = labels[tr_mask]; lbl_te = labels[te_mask]
            acc, auc = train_eval_quick(eeg_tr, lbl_tr, eeg_te, lbl_te)
            fold_accs.append(acc); fold_aucs.append(auc)

        results[version_name] = {
            "loso_accuracy": round(float(np.mean(fold_accs)), 4),
            "loso_accuracy_std": round(float(np.std(fold_accs)), 4),
            "loso_auc": round(float(np.mean(fold_aucs)), 4),
            "loso_auc_std": round(float(np.std(fold_aucs)), 4),
            "per_fold_auc": [round(a, 4) for a in fold_aucs],
        }
        print(f"  {version_name}: ACC={results[version_name]['loso_accuracy']:.3f}  "
              f"AUC={results[version_name]['loso_auc']:.3f}")

    delta_auc = results["version_A_leaky_normalization"]["loso_auc"] - \
                results["version_B_leakage_free"]["loso_auc"]
    auc_B = results["version_B_leakage_free"]["loso_auc"]

    if delta_auc > 0.10:
        interp = "leakage detected: per-subject normalization inflates AUC"
    elif delta_auc > 0.05:
        interp = "partial leakage: normalization may contribute modestly"
    else:
        interp = "no normalization leakage: both versions similar"

    if auc_B >= 0.95:
        interp += " | AUC remains high even without per-subject norm → signal is elsewhere"

    results["delta_auc"] = round(float(delta_auc), 4)
    results["interpretation"] = interp

    out_dir = os.path.join(DIAG_DIR, "test_1_normalization_leakage")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  → delta_AUC={delta_auc:+.3f}  {interp}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2 - Subject Identity in LaBraM Embeddings
# ═══════════════════════════════════════════════════════════════════════════════
def test2_subject_identity():
    print("\n" + "="*60)
    print("TEST 2: Subject Identity Classification")
    print("="*60)
    torch.manual_seed(SEED); np.random.seed(SEED)

    labram, _, meta = load_v3()
    subjects = sorted(meta["subject_id"].unique())
    n_classes = len(subjects)
    chance = 1.0 / n_classes

    # Map subject_id → 0-based integer label
    sid_map = {sid: i for i, sid in enumerate(subjects)}
    sub_labels = meta["subject_id"].map(sid_map).values.astype(np.int64)

    # 80/20 stratified split (no LOSO needed - we want to test if identity is decodable)
    X_tr, X_te, y_tr, y_te = train_test_split(
        labram, sub_labels, test_size=0.20, random_state=SEED, stratify=sub_labels
    )
    acc, _ = train_eval_quick(X_tr, y_tr, X_te, y_te, n_classes=n_classes)

    # Compute F1 separately
    model = SimpleMLP(200, 64, n_classes).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)
    crit  = nn.CrossEntropyLoss()
    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in DataLoader(tr_ds, BS, shuffle=True, num_workers=0):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), BS):
            xb = torch.tensor(X_te[i:i+BS]).to(DEVICE)
            preds.extend(model(xb).argmax(1).cpu().numpy())
    f1 = float(f1_score(y_te, preds, average="macro", zero_division=0))
    acc_final = float(accuracy_score(y_te, preds))

    if acc_final > 0.70:
        interp = "HIGH subject identity in embeddings - model may exploit subject-specific patterns"
    elif acc_final > 0.35:
        interp = "MEDIUM subject identity - some subject-specific information present"
    else:
        interp = "LOW subject identity - embeddings are subject-agnostic"

    result = {
        "task": "subject_id_classification_from_labram",
        "n_classes": n_classes,
        "chance_accuracy": round(chance, 3),
        "achieved_accuracy": round(acc_final, 4),
        "achieved_f1_macro": round(f1, 4),
        "interpretation": interp,
    }
    out_dir = os.path.join(DIAG_DIR, "test_2_subject_identity")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"  9-class ACC={acc_final:.3f} (chance={chance:.3f})  F1={f1:.3f}")
    print(f"  → {interp}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3 - Random Label Shuffle (Structural Leakage Check)
# ═══════════════════════════════════════════════════════════════════════════════
def test3_random_labels(n_shuffles=5):
    print("\n" + "="*60)
    print("TEST 3: Random Label Shuffle (Structural Leakage)")
    print("="*60)

    labram, labels, meta = load_v3()
    # Pre-compute subject normalization (Version A, current)
    labram_norm = norm_version_A(labram, meta)
    subjects = sorted(meta["subject_id"].unique())

    all_aucs = []
    for seed in range(n_shuffles):
        rng = np.random.default_rng(SEED + seed)
        labels_shuf = rng.permutation(labels)   # epoch-level shuffle
        fold_aucs = []
        torch.manual_seed(SEED + seed); np.random.seed(SEED + seed)

        for sid in subjects:
            te_mask = (meta["subject_id"] == sid).values
            tr_mask = ~te_mask
            eeg_tr = labram_norm[tr_mask]; eeg_te = labram_norm[te_mask]
            lbl_tr = labels_shuf[tr_mask]; lbl_te = labels_shuf[te_mask]
            if len(np.unique(lbl_te)) < 2:
                continue
            _, auc = train_eval_quick(eeg_tr, lbl_tr, eeg_te, lbl_te)
            fold_aucs.append(auc)

        mean_auc = float(np.mean(fold_aucs))
        all_aucs.append(mean_auc)
        print(f"  seed={seed+1}  null_auc={mean_auc:.3f}")

    mean_null = float(np.mean(all_aucs))
    std_null  = float(np.std(all_aucs))

    if mean_null > 0.70:
        interp = "STRUCTURAL LEAKAGE DETECTED - model learns from random labels"
    elif mean_null > 0.60:
        interp = "PARTIAL LEAKAGE - model partially overfits to random labels"
    else:
        interp = "NO STRUCTURAL LEAKAGE - model needs real signal (genuine classification)"

    result = {
        "shuffle_strategy": "epoch-level random permutation",
        "n_shuffles": n_shuffles,
        "per_seed_auc": [round(a, 4) for a in all_aucs],
        "mean_loso_auc": round(mean_null, 4),
        "std_loso_auc": round(std_null, 4),
        "expected_chance": 0.5,
        "interpretation": interp,
    }
    out_dir = os.path.join(DIAG_DIR, "test_3_random_labels")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → mean_AUC={mean_null:.3f} ± {std_null:.3f}  {interp}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4 - Classical EEG Features + RandomForest
# ═══════════════════════════════════════════════════════════════════════════════
def test4_classical_features():
    print("\n" + "="*60)
    print("TEST 4: Classical EEG Features (no LaBraM)")
    print("="*60)

    # Load V5 oscillation time series (480, 6, 110)
    osc = np.load(os.path.join(FEAT_V5, "all_oscillation_v5.npy")).astype(np.float32)
    _, labels, meta = load_v3()
    subjects = sorted(meta["subject_id"].unique())

    # Compute temporal mean → (480, 6) scalar features
    osc_mean = osc.mean(axis=-1)   # (480, 6)

    # 7 features: the 6 oscillation bands + theta/beta ratio
    # frontal_theta idx=0, central_beta idx=3
    eps = 1e-10
    theta_beta_ratio = osc_mean[:, 0] / (np.abs(osc_mean[:, 3]) + eps)
    features = np.column_stack([osc_mean, theta_beta_ratio])   # (480, 7)
    feat_names = ['frontal_theta', 'frontal_alpha', 'parietal_alpha',
                  'central_beta', 'faa_dynamic', 'engagement_index', 'theta_beta_ratio']

    fold_accs, fold_aucs = [], []
    all_importances = []

    for sid in subjects:
        te_mask = (meta["subject_id"] == sid).values
        tr_mask = ~te_mask
        X_tr = features[tr_mask]; X_te = features[te_mask]
        y_tr = labels[tr_mask];   y_te = labels[te_mask]

        rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        probs = rf.predict_proba(X_te)[:, 1]
        preds = rf.predict(X_te)

        fold_accs.append(float(accuracy_score(y_te, preds)))
        if len(np.unique(y_te)) > 1:
            fold_aucs.append(float(roc_auc_score(y_te, probs)))
        all_importances.append(rf.feature_importances_)

    mean_acc = float(np.mean(fold_accs))
    mean_auc = float(np.mean(fold_aucs))
    mean_imp = np.mean(all_importances, axis=0)

    if mean_auc > 0.90:
        interp = "classical features sufficient - LaBraM adds marginal value (or shares the same strong signal)"
    elif mean_auc > 0.75:
        interp = "genuine signal in classical features - LaBraM adds incremental value"
    else:
        interp = "classical features weak - LaBraM adds significant value (or exploits a different signal)"

    result = {
        "n_features": 7,
        "feature_names": feat_names,
        "model": "RandomForest (n_estimators=200)",
        "loso_accuracy": round(mean_acc, 4),
        "loso_auc": round(mean_auc, 4),
        "loso_auc_std": round(float(np.std(fold_aucs)), 4),
        "per_fold_auc": [round(a, 4) for a in fold_aucs],
        "feature_importance": {fn: round(float(v), 4)
                                for fn, v in zip(feat_names, mean_imp)},
        "interpretation": interp,
    }
    out_dir = os.path.join(DIAG_DIR, "test_4_classical_features")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"  RF LOSO: ACC={mean_acc:.3f}  AUC={mean_auc:.3f} ± {result['loso_auc_std']:.3f}")
    print("  Feature importances:")
    for fn, v in sorted(zip(feat_names, mean_imp), key=lambda x: -x[1]):
        print(f"    {fn:25s}: {v:.4f}")
    print(f"  → {interp}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic Report
# ═══════════════════════════════════════════════════════════════════════════════
def generate_report(t1, t2, t3, t4):
    print("\n" + "="*60)
    print("Generating Diagnostic Report")
    print("="*60)

    a_auc = t1["version_A_leaky_normalization"]["loso_auc"]
    b_auc = t1["version_B_leakage_free"]["loso_auc"]
    sub_acc = t2["achieved_accuracy"]
    null_auc = t3["mean_loso_auc"]
    rf_auc = t4["loso_auc"]

    # Determine overall diagnosis
    leakage_from_norm = t1["delta_auc"] > 0.10
    subject_identity_high = sub_acc > 0.70
    structural_leakage = null_auc > 0.70
    classical_strong = rf_auc > 0.80

    lines = ["# EEG Pipeline Diagnostic Report\n",
             "## Hypothesis Under Test\n",
             "Why does EEG-only (LaBraM) AUC reach 1.000 in V2, V3, V5?\n",
             "\n## Test Results Summary\n",
             "| Test | Result | Interpretation |",
             "|------|--------|----------------|",
             f"| 1. Normalization leakage | Version A={a_auc:.3f}, B={b_auc:.3f}, delta={t1['delta_auc']:+.3f} | {t1['interpretation']} |",
             f"| 2. Subject identity | accuracy={sub_acc:.3f} (chance=0.111) | {t2['interpretation']} |",
             f"| 3. Random label shuffle | AUC={null_auc:.3f} ± {t3['std_loso_auc']:.3f} (chance=0.5) | {t3['interpretation']} |",
             f"| 4. Classical features RF | AUC={rf_auc:.3f} ± {t4['loso_auc_std']:.3f} | {t4['interpretation']} |",
             "",
             "## Diagnosis\n",
    ]

    # Core diagnosis
    diag = []
    if structural_leakage:
        diag.append("⚠️  **STRUCTURAL LEAKAGE** confirmed (Test 3). "
                    "Model achieves high AUC even with random labels - there is a data pipeline bug.")
    else:
        diag.append("✓ **No structural leakage** (Test 3): model fails on random labels → "
                    f"it needs genuine signal (null AUC={null_auc:.3f} ≈ chance).")

    if leakage_from_norm:
        diag.append(f"⚠️  **Normalization leakage** (Test 1): per-subject norm inflates AUC by "
                    f"{t1['delta_auc']:+.3f} ({a_auc:.3f}→{b_auc:.3f}).")
    else:
        diag.append(f"✓ **No normalization leakage** (Test 1): delta={t1['delta_auc']:+.3f} is negligible.")

    if subject_identity_high:
        diag.append(f"⚠️  **Subject identity encoded** (Test 2): LaBraM embeddings classify subjects "
                    f"at {sub_acc:.1%} accuracy (chance 11.1%). Model may exploit subject-specific features.")
    else:
        diag.append(f"✓ **Weak subject identity** (Test 2): embeddings not strongly subject-specific "
                    f"(accuracy {sub_acc:.1%} vs chance 11.1%).")

    if classical_strong:
        diag.append(f"✓ **Classical features sufficient** (Test 4): RF on 7 scalar EEG features "
                    f"achieves AUC={rf_auc:.3f}. A strong frustration-relevant signal exists in raw oscillations.")
    else:
        diag.append(f"• Classical features moderate (Test 4): RF AUC={rf_auc:.3f} - "
                    f"LaBraM adds substantial value beyond classical features.")

    lines.extend(diag)
    lines.append("")
    lines.append("### Overall Conclusion\n")

    if not structural_leakage and not leakage_from_norm and classical_strong:
        conclusion = (
            "The high AUC (1.000) appears to reflect a **genuinely strong EEG signal** "
            "that is detectable even with classical features (RF AUC=%.3f). "
            "LaBraM amplifies this via richer temporal-spectral encoding, not data leakage. "
            "The action-matched control design (V3/V5) confirms the signal is not purely task-vs-rest." % rf_auc
        )
    elif structural_leakage:
        conclusion = (
            "CRITICAL: Structural leakage detected. The 1.000 AUC is likely an artifact. "
            "The data pipeline must be audited for train/test contamination."
        )
    elif subject_identity_high and not classical_strong:
        conclusion = (
            "The model may be exploiting subject identity rather than frustration signal. "
            "High subject identity in LaBraM embeddings (%.1f%% accuracy) combined with "
            "weak classical features (RF AUC=%.3f) suggests the model identifies subjects, "
            "not mental states." % (sub_acc*100, rf_auc)
        )
    else:
        auc_leakage_free = b_auc
        conclusion = (
            "Partial evidence: normalization leakage accounts for some AUC inflation "
            f"(leakage-free AUC = {auc_leakage_free:.3f}). The remaining performance may be "
            "genuine signal or subject-specific LaBraM features."
        )

    lines.append(conclusion)
    lines.append("")
    lines.append("## Recommended Strategy Update\n")

    recs = []
    if structural_leakage:
        recs.append("1. **URGENT**: Audit full data pipeline for any train/test bleed (feature files, normalization, epoch indexing).")
    if leakage_from_norm:
        recs.append("1. **Replace normalization**: use global mean/std from training subjects only (leakage-free version B).")
    if subject_identity_high:
        recs.append("- **Add subject-ID adversarial loss** (domain adaptation) to prevent the model from using subject identity.")
        recs.append("- **Consider subject-matched baseline**: within-subject LOSO or leave-one-session-out.")
    if not structural_leakage and classical_strong:
        recs.append("- **Investigate classical features**: the signal detectable by RF is worth characterizing in terms of which band explains variance.")
        recs.append("- **Increase N**: with 9 subjects and strong per-subject EEG patterns, a larger cohort would confirm generalizability.")
    if not any([structural_leakage, leakage_from_norm, subject_identity_high]):
        recs.append("- **Primary concern**: sample size (N=9). The signal is likely real but may not generalize. Recruit more subjects.")
        recs.append("- **Publish negative finding too**: if classical RF achieves 0.80+, the fancy model adds little practical value.")

    if not recs:
        recs = ["- All tests passed: pipeline appears sound. Main limitation is small N."]
    lines.extend(recs)

    report = "\n".join(lines)
    out_path = os.path.join(DIAG_DIR, "diagnostic_report.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport saved → {out_path}")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t1 = test1_normalization_leakage()
    t2 = test2_subject_identity()
    t3 = test3_random_labels(n_shuffles=5)
    t4 = test4_classical_features()
    generate_report(t1, t2, t3, t4)
    print("\n=== ALL DIAGNOSTICS COMPLETE ===")
