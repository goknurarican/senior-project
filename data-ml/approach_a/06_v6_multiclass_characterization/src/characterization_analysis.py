"""
V6 Per-Scenario Characterization Analysis
==========================================
For each of 14 frustration scenarios vs control:
  - Mean ERSP change per feature
  - Cohen's d (within-subject: mean per-subject then test)
  - Wilcoxon signed-rank test on per-subject means
  - FDR correction (Benjamini-Hochberg)
  - Forest plots (top 5 changes per scenario)
  - Heatmap (scenario × feature, color = effect size)
  - Hierarchical clustering of scenarios

Outputs:
  analysis/per_scenario_signatures/scenario_signatures.csv
  analysis/per_scenario_signatures/all_scenarios_heatmap.png
  analysis/per_scenario_signatures/scenario_clustering.png
  analysis/per_scenario_signatures/forest_plots/scenario_X.png
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
warnings.filterwarnings("ignore")

V6_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(V6_DIR, "features")
ANA_DIR  = os.path.join(V6_DIR, "analysis", "per_scenario_signatures")
FOREST_DIR = os.path.join(ANA_DIR, "forest_plots")
os.makedirs(FOREST_DIR, exist_ok=True)

SUBJECTS = [14, 15, 16, 17, 18, 20, 21, 22, 23]

#Feature names: 6 ROI × 4 bands + FAA
ROI_NAMES  = ["frontal","frontal_central","central","parietal","occipital","temporal"]
BAND_NAMES = ["theta","alpha","beta","gamma"]
OSC_FEATURE_NAMES = [f"{r}_{b}" for r in ROI_NAMES for b in BAND_NAMES] + ["faa_dynamic"]

#These are temporal-mean scalars for per-scenario comparison
#25 osc + 6 eye + 7 mouse = 38 total feature scalars
EYE_FEATURE_NAMES   = [f"eye_{i}" for i in range(6)]
MOUSE_FEATURE_NAMES = [f"mouse_{i}" for i in range(7)]
ALL_FEATURE_NAMES   = OSC_FEATURE_NAMES + EYE_FEATURE_NAMES + MOUSE_FEATURE_NAMES

FRUSTRATION_SCENARIOS = sorted([
    'broken_image','button_delay','coupon_expired','coupon_min_spend',
    'facet_reset_once','feedback_late','first_click_miss','network_jitter',
    'overlay_blocking','price_change','search_irrelevant','skeleton_prolong',
    'slow_image','sort_reset'
])


def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvals)
    pvals = np.array(pvals)
    rank  = np.argsort(pvals) + 1
    adj   = pvals * n / rank
    adj   = np.minimum(1.0, adj)
    #Enforce monotonicity (cumulative min from tail)
    out = np.empty_like(adj)
    out_sorted = adj[np.argsort(pvals)[::-1]]
    cummin = np.minimum.accumulate(out_sorted)
    out[np.argsort(pvals)[::-1]] = cummin
    return out


def cohens_d_paired(x, y):
    """Cohen's d for paired samples (within-subject design)."""
    diff = np.array(x) - np.array(y)
    if len(diff) < 2 or diff.std() == 0:
        return 0.0
    return float(diff.mean() / (diff.std(ddof=1) + 1e-10))


def load_data():
    osc   = np.load(os.path.join(FEAT_DIR, "all_oscillation_v6.npy")).astype(np.float32)
    eye   = np.load(os.path.join(FEAT_DIR, "all_eye_v6.npy")).astype(np.float32)
    mouse = np.load(os.path.join(FEAT_DIR, "all_mouse_v6.npy")).astype(np.float32)
    lab   = pd.read_csv(os.path.join(FEAT_DIR, "labels_v6.csv"))

    #Temporal means per epoch per feature
    osc_mean   = osc.mean(axis=-1)           # (480, 25)
    eye_mean   = eye.mean(axis=-1)           # (480, 6)
    mouse_mean = mouse.mean(axis=-1)         # (480, 7)

    X = np.concatenate([osc_mean, eye_mean, mouse_mean], axis=1)   # (480, 38)
    return X, lab


def analyze_scenario(scenario_name, X, lab):
    """
    Compare scenario epochs vs control epochs using within-subject design.
    Returns dict with per-feature statistics.
    """
    ctrl_mask = lab["scenario_name"] == "control_action_matched"
    scen_mask = lab["scenario_name"] == scenario_name
    sids      = lab["subject_id"].values

    rows = []   # per-feature results

    for fi, fname in enumerate(ALL_FEATURE_NAMES):
        #Per-subject means for paired test
        ctrl_means, scen_means = [], []
        for sid in SUBJECTS:
            sm = (sids == sid) & ctrl_mask.values
            ss = (sids == sid) & scen_mask.values
            if sm.sum() == 0 or ss.sum() == 0:
                continue
            ctrl_means.append(float(X[sm, fi].mean()))
            scen_means.append(float(X[ss, fi].mean()))

        n_paired = len(ctrl_means)
        if n_paired < 2:
            rows.append({
                "feature": fname, "n_subjects_paired": n_paired,
                "ctrl_mean": float(X[ctrl_mask, fi].mean()) if ctrl_mask.any() else np.nan,
                "scen_mean": float(X[scen_mask, fi].mean()) if scen_mask.any() else np.nan,
                "pct_change": np.nan, "cohens_d": np.nan,
                "p_wilcoxon": np.nan, "p_fdr": np.nan, "significant": False,
            })
            continue

        ctrl_arr = np.array(ctrl_means)
        scen_arr = np.array(scen_means)
        ctrl_m   = float(ctrl_arr.mean())
        scen_m   = float(scen_arr.mean())
        pct_change = (scen_m - ctrl_m) / (abs(ctrl_m) + 1e-10) * 100

        d = cohens_d_paired(scen_arr, ctrl_arr)

        try:
            _, p = stats.wilcoxon(scen_arr, ctrl_arr, alternative='two-sided')
        except Exception:
            p = 1.0

        rows.append({
            "feature": fname, "n_subjects_paired": n_paired,
            "ctrl_mean": ctrl_m, "scen_mean": scen_m,
            "pct_change": float(pct_change), "cohens_d": float(d),
            "p_wilcoxon": float(p), "p_fdr": np.nan, "significant": False,
        })

    df = pd.DataFrame(rows)
    #FDR correction
    valid = ~df["p_wilcoxon"].isna()
    if valid.sum() > 0:
        adj = fdr_bh(df.loc[valid, "p_wilcoxon"].values)
        df.loc[valid, "p_fdr"] = adj
        df.loc[valid, "significant"] = adj < 0.05
    return df


def forest_plot(scen_name, sig_df, out_path):
    """Forest plot showing top 5 features by |effect size| for a scenario."""
    df = sig_df.copy().dropna(subset=["cohens_d"])
    if len(df) == 0: return
    top = df.reindex(df["cohens_d"].abs().sort_values(ascending=False).index).head(8)

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = range(len(top))
    colors = ['#d32f2f' if d > 0 else '#1976D2' for d in top["cohens_d"]]
    ax.barh(list(y_pos), top["cohens_d"].values, color=colors, alpha=0.8, edgecolor='black', lw=0.5)
    ax.axvline(0, color='black', lw=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top["feature"].values, fontsize=8)
    ax.set_xlabel("Cohen's d (positive = higher in scenario)", fontsize=9)
    ax.set_title(f"Top features: {scen_name.replace('_',' ')}\n"
                 f"(N={top['n_subjects_paired'].iloc[0] if len(top)>0 else 0} subjects, "
                 f"red=↑ blue=↓)", fontsize=9)
    #Mark significant
    for i, (_, row) in enumerate(top.iterrows()):
        if row.get("significant", False):
            ax.text(row["cohens_d"] + (0.02 if row["cohens_d"] >= 0 else -0.02),
                    i, "*", ha='center', va='center', fontsize=10, color='black')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def all_heatmap(signatures, out_path):
    """Heatmap: scenarios × features, value = Cohen's d."""
    scenarios = list(signatures.keys())
    features  = ALL_FEATURE_NAMES

    M = np.zeros((len(scenarios), len(features)))
    for i, scen in enumerate(scenarios):
        df = signatures[scen].set_index("feature")
        for j, feat in enumerate(features):
            if feat in df.index:
                M[i, j] = float(df.loc[feat, "cohens_d"]) if not pd.isna(df.loc[feat, "cohens_d"]) else 0

    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(M, aspect='auto', cmap='RdBu_r', vmin=-1.5, vmax=1.5)
    plt.colorbar(im, ax=ax, label="Cohen's d")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([s.replace('_', ' ') for s in scenarios], fontsize=8)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=7)
    ax.set_title("Scenario × Feature Signature Heatmap (Cohen's d vs control)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Heatmap saved: {out_path}")


def clustering_plot(signatures, out_path):
    """Hierarchical clustering of scenarios based on signature vectors."""
    scenarios = list(signatures.keys())
    M = []
    for scen in scenarios:
        df  = signatures[scen].set_index("feature")
        vec = [float(df.loc[f, "cohens_d"]) if f in df.index and not pd.isna(df.loc[f, "cohens_d"]) else 0
               for f in ALL_FEATURE_NAMES]
        M.append(vec)
    M = np.array(M)

    if len(scenarios) < 3:
        print("Not enough scenarios for clustering.")
        return {}

    Z = linkage(M, method='ward', metric='euclidean')
    #Assign clusters
    n_clusters = min(4, len(scenarios) - 1)
    cluster_ids = fcluster(Z, n_clusters, criterion='maxclust')

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=[s.replace('_', ' ') for s in scenarios],
               leaf_rotation=45, leaf_font_size=9, ax=ax,
               color_threshold=Z[-n_clusters+1, 2] if len(Z) >= n_clusters else None)
    ax.set_title("Hierarchical Clustering of Frustration Scenarios\n(Ward linkage, Euclidean distance on signature vectors)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Clustering plot saved: {out_path}")

    cluster_df = pd.DataFrame({"scenario": scenarios, "cluster": cluster_ids})
    return cluster_df


def main():
    X, lab = load_data()
    print(f"Features: {X.shape}, Labels: {len(lab)}")

    signatures = {}
    all_rows   = []

    for scen in FRUSTRATION_SCENARIOS:
        n_scen = (lab["scenario_name"] == scen).sum()
        print(f"  {scen}: N={n_scen}", end="")
        df = analyze_scenario(scen, X, lab)
        signatures[scen] = df

        forest_path = os.path.join(FOREST_DIR, f"forest_{scen}.png")
        forest_plot(scen, df, forest_path)

        #Add to combined table
        df.insert(0, "scenario", scen)
        df.insert(1, "n_epochs", n_scen)
        all_rows.append(df)

        n_sig = df["significant"].sum()
        top_d = df.dropna(subset=["cohens_d"]).reindex(
            df["cohens_d"].abs().sort_values(ascending=False).index
        )["feature"].head(3).tolist()
        print(f"  sig={n_sig}  top3={top_d}")

    #Combined signature table
    sig_table = pd.concat(all_rows, ignore_index=True)
    sig_table.to_csv(os.path.join(ANA_DIR, "scenario_signatures.csv"), index=False)
    print(f"\nScenario signatures saved.")

    #Heatmap
    all_heatmap(signatures, os.path.join(ANA_DIR, "all_scenarios_heatmap.png"))

    #Clustering
    cluster_df = clustering_plot(signatures, os.path.join(ANA_DIR, "scenario_clustering.png"))
    if isinstance(cluster_df, pd.DataFrame) and len(cluster_df) > 0:
        cluster_df.to_csv(os.path.join(ANA_DIR, "scenario_clusters.csv"), index=False)
        print("\nClusters:")
        for cid in sorted(cluster_df["cluster"].unique()):
            sc = cluster_df[cluster_df["cluster"]==cid]["scenario"].tolist()
            print(f"  Cluster {cid}: {sc}")

    #Top-5 feature summary per scenario
    print("\n=== Per-Scenario Top-5 Features (|Cohen's d|) ===")
    for scen in FRUSTRATION_SCENARIOS:
        df = signatures[scen].dropna(subset=["cohens_d"])
        if len(df) == 0: continue
        top = df.reindex(df["cohens_d"].abs().sort_values(ascending=False).index).head(5)
        print(f"\n{scen} (N={top['n_subjects_paired'].iloc[0]} subjects):")
        for _, r in top.iterrows():
            sig_mark = "*" if r.get("significant", False) else " "
            print(f"  {sig_mark}{r['feature']:30s} d={r['cohens_d']:+.3f}  "
                  f"Δ={r['pct_change']:+.1f}%  p_fdr={r['p_fdr']:.3f}" if not pd.isna(r.get("p_fdr")) else
                  f"  {r['feature']:30s} d={r['cohens_d']:+.3f}  Δ={r['pct_change']:+.1f}%  p_fdr=N/A")


if __name__ == "__main__":
    main()
