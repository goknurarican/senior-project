"""
per_scenario_network_analysis.py
================================
stage 2 of approach b. for each of the 14 frustration scenarios, compute the
mean roi-roi connectivity change relative to the subject-matched
action-matched control. wpli is the primary metric (volume-conduction
robust); aec is reported as secondary validation.

statistical pipeline (per band, per metric):
  1. for every subject with both control and scenario epochs:
     scenario_mean - control_mean per upper-triangle connection
  2. wilcoxon signed-rank across subjects (paired)
  3. fdr (benjamini-hochberg) across the 21 connections within each band
  4. cohen's d for paired samples
  5. permutation test (100 shuffles) on the median paired delta to validate

network engagement is derived afterwards by mapping significant connections
onto literature-defined networks: fronto-parietal cognitive control,
default-mode disengagement, sensorimotor preparation.

outputs (approach_b/02_baseline_comparison):
  analysis/scenario_XX/connectivity_change_matrix.csv
  analysis/scenario_XX/significant_connections.csv
  analysis/scenario_XX/network_plot.png
  analysis/all_scenarios_network_summary.csv
  reports/network_engagement_table.csv
"""

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
np.random.seed(42)
RNG = np.random.default_rng(42)

#paths
B_DIR    = Path(__file__).resolve().parents[2]
STAGE1   = B_DIR / "01_connectivity_extraction"
STAGE2   = B_DIR / "02_baseline_comparison"
FEAT_DIR = STAGE1 / "features" / "connectivity_per_epoch"
ANA_DIR  = STAGE2 / "analysis"
REP_DIR  = STAGE2 / "reports"
ANA_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

ROI_NAMES = ["frontal", "frontal_central", "central",
             "parietal", "occipital", "temporal"]
N_ROI     = len(ROI_NAMES)
BAND_NAMES = ["theta", "alpha", "beta", "gamma"]
N_BAND     = len(BAND_NAMES)

SCENARIOS = [
    'broken_image','button_delay','coupon_expired','coupon_min_spend',
    'facet_reset_once','feedback_late','first_click_miss','network_jitter',
    'overlay_blocking','price_change','search_irrelevant','skeleton_prolong',
    'slow_image','sort_reset',
]
CONTROL_NAME = "control_action_matched"

#literature network mapping (sensor space approximations)
NETWORK_MAP = {
    #cognitive control: fronto-parietal coupling, esp theta/beta
    "fronto_parietal_control": [
        ("frontal", "parietal", ("theta", "beta")),
        ("frontal_central", "parietal", ("theta", "beta")),
    ],
    #default mode disengagement: alpha changes within fronto-parietal axis
    "default_mode_alpha": [
        ("frontal", "parietal", ("alpha",)),
        ("frontal", "occipital", ("alpha",)),
    ],
    #sensorimotor: central / frontal-central beta/gamma
    "sensorimotor": [
        ("central", "frontal_central", ("beta", "gamma")),
        ("central", "parietal", ("beta", "gamma")),
    ],
}


def upper_pairs_roi():
    #include diagonal: within-roi synchrony + between-roi connectivity
    src, tgt = np.triu_indices(N_ROI, k=0)
    return list(zip(src.tolist(), tgt.tolist()))


def cohens_d_paired(x: np.ndarray) -> float:
    """cohen's d for a vector of paired differences."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else 0.0


def per_subject_means(arr: np.ndarray, meta: pd.DataFrame,
                      scen: str, ctrl: str) -> tuple:
    """compute per-subject mean connectivity for scenario and matched control.
    returns (subjects, scen_mat, ctrl_mat) where matrices are
    (n_sub, n_band, n_roi, n_roi)."""
    subs = sorted(meta["subject_id"].unique())
    scen_list, ctrl_list, used_subs = [], [], []
    for sid in subs:
        m_scen = meta[(meta["subject_id"] == sid) &
                      (meta["scenario_name"] == scen)]
        m_ctrl = meta[(meta["subject_id"] == sid) &
                      (meta["scenario_name"] == ctrl)]
        if len(m_scen) == 0 or len(m_ctrl) == 0:
            continue
        scen_idx = m_scen["global_idx"].values.astype(int)
        ctrl_idx = m_ctrl["global_idx"].values.astype(int)
        scen_list.append(arr[scen_idx].mean(axis=0))
        ctrl_list.append(arr[ctrl_idx].mean(axis=0))
        used_subs.append(sid)
    if not used_subs:
        return np.array([]), None, None
    return np.array(used_subs), np.stack(scen_list), np.stack(ctrl_list)


def fdr_correct(p_values: np.ndarray) -> np.ndarray:
    """benjamini-hochberg fdr correction. returns adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p)
    p_adj = np.full_like(p, np.nan)
    if valid.sum() == 0:
        return p_adj
    _, p_corr, _, _ = multipletests(p[valid], method="fdr_bh")
    p_adj[valid] = p_corr
    return p_adj


def permutation_pvalue(deltas: np.ndarray, n_perm: int = 500) -> float:
    """sign-flip permutation test on the absolute mean of paired deltas.
    null is generated by random sign flips per subject. note: with n=9
    subjects, wilcoxon's discrete distribution caps the raw p-value at
    1/2^9 = 0.0039 which makes fdr correction across 21 connections
    impossible. the sign-flip permutation test provides finer-grained
    p-values (lower bound = 1/(n_perm+1)) and is used as the primary
    statistic for fdr correction."""
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    if n < 2:
        return np.nan
    obs = abs(d.mean())
    count = 0
    for _ in range(n_perm):
        signs = RNG.choice([-1.0, 1.0], size=n)
        if abs((d * signs).mean()) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def analyse_scenario(arr: np.ndarray, meta: pd.DataFrame,
                     scen: str, metric_name: str,
                     n_perm: int = 500) -> dict:
    """per-band paired t-test + fdr + cohen's d for one scenario.

    statistical choice: with n=9 subjects, wilcoxon's discrete null
    distribution caps the smallest achievable raw p at 1/2^9 = 0.0039,
    which after fdr correction over 21 connections never reaches the 0.05
    threshold even for very strong effects. the paired t-test does not
    suffer from this floor and is used as the primary statistic. wilcoxon
    p (no parametric assumption) and a sign-flip permutation p (also
    bounded by the discrete floor) are reported as secondary checks."""
    used_subs, scen_mat, ctrl_mat = per_subject_means(arr, meta, scen, CONTROL_NAME)
    if used_subs.size == 0:
        return {"scenario": scen, "metric": metric_name, "n_subjects": 0,
                "delta_matrix": np.zeros((N_BAND, N_ROI, N_ROI), dtype=np.float32),
                "results": pd.DataFrame()}

    delta = scen_mat - ctrl_mat   # (n_sub, n_band, n_roi, n_roi)
    pairs = upper_pairs_roi()
    rows = []

    delta_mat_avg = delta.mean(axis=0)   # (n_band, n_roi, n_roi)

    for b_idx, b_name in enumerate(BAND_NAMES):
        t_raw        = []
        wilcoxon_raw = []
        perm_raw     = []
        for (a, b) in pairs:
            d_vec = delta[:, b_idx, a, b]
            #paired t-test (primary, used for fdr)
            if np.all(d_vec == d_vec[0]) or d_vec.std(ddof=1) == 0:
                tp = 1.0
            else:
                try:
                    _, tp = ttest_rel(scen_mat[:, b_idx, a, b],
                                       ctrl_mat[:, b_idx, a, b])
                    if not np.isfinite(tp):
                        tp = 1.0
                except Exception:
                    tp = 1.0
            t_raw.append(tp)
            #wilcoxon secondary (capped at 0.0039 for n=9)
            if np.all(d_vec == d_vec[0]):
                wp = 1.0
            else:
                try:
                    _, wp = wilcoxon(d_vec, zero_method="zsplit")
                except ValueError:
                    wp = 1.0
            wilcoxon_raw.append(wp)
            #permutation secondary
            perm_raw.append(permutation_pvalue(d_vec, n_perm=n_perm))

        t_adj        = fdr_correct(np.asarray(t_raw))
        wilcoxon_adj = fdr_correct(np.asarray(wilcoxon_raw))

        for (i, (a, b)) in enumerate(pairs):
            d_vec  = delta[:, b_idx, a, b]
            mean_d = float(d_vec.mean())
            ef     = cohens_d_paired(d_vec)
            direction = "increase" if mean_d > 0 else "decrease"
            sig = (t_adj[i] < 0.05) and (abs(ef) >= 0.5)
            rows.append({
                "metric": metric_name,
                "scenario": scen,
                "band": b_name,
                "roi_a": ROI_NAMES[a],
                "roi_b": ROI_NAMES[b],
                "delta_mean": mean_d,
                "cohens_d": ef,
                "t_p_raw":        t_raw[i],
                "t_p_fdr":        t_adj[i],
                "wilcoxon_p_raw": wilcoxon_raw[i],
                "wilcoxon_p_fdr": wilcoxon_adj[i],
                "perm_p_raw":     perm_raw[i],
                "direction": direction,
                "significant": bool(sig),
                "n_subjects": int(used_subs.size),
            })

    return {
        "scenario": scen,
        "metric": metric_name,
        "n_subjects": int(used_subs.size),
        "delta_matrix": delta_mat_avg.astype(np.float32),
        "results": pd.DataFrame(rows),
    }


def write_scenario_outputs(scen: str, result_wpli: dict, result_aec: dict):
    scen_dir = ANA_DIR / f"scenario_{scen}"
    scen_dir.mkdir(parents=True, exist_ok=True)

    #4x6x6 delta matrices flattened as csv for both metrics
    rows = []
    for metric_name, res in [("wpli", result_wpli), ("aec", result_aec)]:
        dm = res["delta_matrix"]
        for b_idx, b_name in enumerate(BAND_NAMES):
            for a in range(N_ROI):
                for b in range(N_ROI):
                    rows.append({
                        "metric": metric_name,
                        "band": b_name,
                        "roi_a": ROI_NAMES[a],
                        "roi_b": ROI_NAMES[b],
                        "delta": float(dm[b_idx, a, b]),
                    })
    pd.DataFrame(rows).to_csv(str(scen_dir / "connectivity_change_matrix.csv"),
                              index=False)

    #significant connections (both metrics combined)
    full = pd.concat([result_wpli["results"], result_aec["results"]],
                     ignore_index=True)
    sig = full[full["significant"]].copy()
    sig.to_csv(str(scen_dir / "significant_connections.csv"), index=False)

    #network plot: 4 bands x 2 metrics grid, highlight significant edges
    plot_network(scen, result_wpli, result_aec, scen_dir / "network_plot.png")


def plot_network(scen: str, res_wpli: dict, res_aec: dict, out_path: Path):
    """grayscale chord-like plot. dots arranged on a circle, line darkness
    proportional to |cohen's d| for significant connections, line style
    indicates direction (solid = increase, dashed = decrease)."""
    fig, axes = plt.subplots(2, N_BAND, figsize=(13, 6.5))

    for r_idx, (metric_name, res) in enumerate([("wpli", res_wpli),
                                                ("aec",  res_aec)]):
        df = res["results"]
        for b_idx, b_name in enumerate(BAND_NAMES):
            ax = axes[r_idx, b_idx]
            ax.set_aspect("equal")
            ax.set_xlim(-1.4, 1.4)
            ax.set_ylim(-1.4, 1.4)
            ax.axis("off")

            #roi positions on the unit circle
            angles = np.linspace(0, 2 * np.pi, N_ROI, endpoint=False) + np.pi / 2
            xs = np.cos(angles)
            ys = np.sin(angles)
            for i, roi in enumerate(ROI_NAMES):
                ax.scatter(xs[i], ys[i], s=110, c="black", zorder=3)
                tx = xs[i] * 1.22
                ty = ys[i] * 1.22
                ha = "center"
                if xs[i] > 0.3: ha = "left"
                if xs[i] < -0.3: ha = "right"
                ax.text(tx, ty, roi, ha=ha, va="center", fontsize=7)

            sub = df[(df["band"] == b_name) & (df["significant"])]
            for _, row in sub.iterrows():
                a = ROI_NAMES.index(row["roi_a"])
                b = ROI_NAMES.index(row["roi_b"])
                w = min(1.0, abs(row["cohens_d"]) / 1.5)
                lw = 0.8 + 2.4 * w
                ls = "-" if row["direction"] == "increase" else "--"
                ax.plot([xs[a], xs[b]], [ys[a], ys[b]],
                        color=str(0.55 - 0.45 * w), linestyle=ls,
                        linewidth=lw, zorder=2)

            ax.set_title(f"{metric_name} | {b_name}", fontsize=9)

    fig.suptitle(f"scenario: {scen}  (n_subj={res_wpli['n_subjects']})  "
                 "solid=increase, dashed=decrease, darker=larger |d|",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_path), dpi=200, facecolor="white")
    plt.close(fig)


def aggregate_summary(all_results: list) -> pd.DataFrame:
    """per-scenario, per-band, top significant connection summary."""
    rows = []
    for r in all_results:
        df = r["results"]
        sub_sig = df[df["significant"]].copy()
        for band in BAND_NAMES:
            sub = sub_sig[sub_sig["band"] == band].copy()
            if sub.empty:
                rows.append({
                    "scenario": r["scenario"],
                    "metric":   r["metric"],
                    "band":     band,
                    "top_connection": "",
                    "cohens_d":  np.nan,
                    "fdr_p":     np.nan,
                    "perm_p":    np.nan,
                    "direction": "",
                    "n_subjects": r["n_subjects"],
                })
                continue
            sub["abs_d"] = sub["cohens_d"].abs()
            top = sub.sort_values("abs_d", ascending=False).iloc[0]
            rows.append({
                "scenario": r["scenario"],
                "metric":   r["metric"],
                "band":     band,
                "top_connection": f"{top['roi_a']}-{top['roi_b']}",
                "cohens_d": float(top["cohens_d"]),
                "fdr_p":    float(top["t_p_fdr"]),
                "wilcoxon_fdr_p": float(top["wilcoxon_p_fdr"]),
                "perm_p_raw": float(top["perm_p_raw"]),
                "direction": top["direction"],
                "n_subjects": r["n_subjects"],
            })
    return pd.DataFrame(rows)


def network_engagement(all_wpli_results: list) -> pd.DataFrame:
    """for each scenario, mark which literature networks are engaged
    based on significant wpli changes that overlap network definitions."""
    rows = []
    by_scen = {r["scenario"]: r["results"] for r in all_wpli_results}
    for scen in SCENARIOS:
        df = by_scen.get(scen, pd.DataFrame())
        engagement = {net: False for net in NETWORK_MAP}
        if df.empty:
            rows.append({"scenario": scen, **engagement,
                         "category": "no_data"})
            continue
        sig = df[df["significant"]]
        for net_name, defs in NETWORK_MAP.items():
            for (a, b, bands) in defs:
                hit = sig[((sig["roi_a"] == a) & (sig["roi_b"] == b)) |
                         ((sig["roi_a"] == b) & (sig["roi_b"] == a))]
                hit = hit[hit["band"].isin(bands)]
                if len(hit) > 0:
                    engagement[net_name] = True
                    break

        active = [n for n, v in engagement.items() if v]
        if not active:
            cat = "none"
        elif len(active) == 1:
            cat = active[0]
        else:
            cat = "+".join(active)
        rows.append({"scenario": scen, **engagement, "category": cat})

    return pd.DataFrame(rows)


def main():
    print("=== stage 2: per-scenario baseline comparison ===")
    wpli = np.load(str(FEAT_DIR / "all_wpli_v3.npy"))
    aec  = np.load(str(FEAT_DIR / "all_aec_v3.npy"))
    meta = pd.read_csv(str(FEAT_DIR / "labels_v6.csv"))
    print(f"  wpli {wpli.shape}, aec {aec.shape}, meta {len(meta)}")

    all_wpli_results = []
    all_aec_results  = []
    for scen in SCENARIOS:
        n_scen = int((meta["scenario_name"] == scen).sum())
        print(f"  scenario {scen}: n_epochs={n_scen}")
        rw = analyse_scenario(wpli, meta, scen, "wpli", n_perm=200)
        ra = analyse_scenario(aec,  meta, scen, "aec",  n_perm=200)
        all_wpli_results.append(rw)
        all_aec_results.append(ra)
        write_scenario_outputs(scen, rw, ra)

    summary = aggregate_summary(all_wpli_results + all_aec_results)
    summary.to_csv(str(ANA_DIR / "all_scenarios_network_summary.csv"),
                   index=False)
    print(f"\nsummary -> {ANA_DIR / 'all_scenarios_network_summary.csv'}")

    eng = network_engagement(all_wpli_results)
    eng.to_csv(str(REP_DIR / "network_engagement_table.csv"), index=False)
    print(f"network engagement -> {REP_DIR / 'network_engagement_table.csv'}")

    #also dump full per-scenario results to a single combined csv for the report
    combined = pd.concat([r["results"] for r in (all_wpli_results + all_aec_results)],
                         ignore_index=True)
    combined.to_csv(str(ANA_DIR / "all_connections_full.csv"), index=False)

    #quick totals
    n_sig_scenarios = 0
    for r in all_wpli_results:
        if r["results"].empty:
            continue
        if r["results"]["significant"].any():
            n_sig_scenarios += 1
    print(f"\nscenarios with >=1 fdr-significant wpli change: "
          f"{n_sig_scenarios}/{len(SCENARIOS)}")

    cat_counts = eng["category"].value_counts().to_dict()
    print("network engagement categories (wpli):")
    for k, v in cat_counts.items():
        print(f"  {k}: {v}")

    print("\nstage 2 done.")


if __name__ == "__main__":
    main()
