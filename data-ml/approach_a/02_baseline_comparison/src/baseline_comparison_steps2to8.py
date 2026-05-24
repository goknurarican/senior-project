"""
Baseline Comparison - Steps 2-8 only.
Loads the already-computed (and phase-fixed) per_epoch_features.csv
and runs aggregates, statistics, visualizations, and report.
"""

import sys, warnings, logging
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "approach_a" / "src"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SUBJECTS = [14, 15, 16, 17, 18, 20, 21, 22, 23]
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

GATE_MIN_FEATURES = 5
GATE_MIN_EFFECT   = 0.3

OUT = ROOT / "approach_a" / "analysis" / "baseline_comparison"
SUBJ_COLORS = plt.cm.tab10.colors[:9]

# ── Feature column lists (must match per_epoch_features.csv) ──────────────────
BANDS = {"delta": (1,4), "theta": (4,8), "alpha": (8,13), "beta": (13,30), "gamma": (30,40)}
EEG_FEAT_COLS = (
    [f"{b}_power" for b in BANDS]
    + [f"{g}_{b}" for g in ("frontal","parietal","occipital","central") for b in BANDS]
    + ["FAA", "theta_beta_ratio", "engagement_index", "spectral_entropy"]
)
EYE_FEAT_COLS   = ["fixation_count","fixation_mean_dur_ms","saccade_count",
                   "saccade_mean_amp_deg","blink_count","gaze_dispersion",
                   "nan_ratio","n_samples","blink_rate_per_sec"]
MOUSE_FEAT_COLS = ["velocity_mean","velocity_max","acceleration_mean",
                   "path_length_px","auc_deviation","x_flips","y_flips",
                   "idle_ratio","click_count","rage_click_flag"]
ALL_FEAT_COLS = EEG_FEAT_COLS + EYE_FEAT_COLS + MOUSE_FEAT_COLS


def _friendly(f):
    return (f.replace("_power","").replace("frontal_","F-").replace("parietal_","P-")
             .replace("occipital_","O-").replace("central_","C-")
             .replace("_"," ").upper())


# ── Step 2: aggregates ────────────────────────────────────────────────────────
def build_aggregates(pef):
    rows = []
    for sid in SUBJECTS:
        for phase in ["variant", "control"]:
            sub = pef[(pef["subject_id"]==sid) & (pef["phase"]==phase)]
            n   = len(sub)
            for feat in ALL_FEAT_COLS:
                if feat not in sub.columns:
                    continue
                vals = pd.to_numeric(sub[feat], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                rows.append(dict(subject_id=sid, phase=phase, feature_name=feat,
                                 mean=vals.mean(), std=vals.std(),
                                 median=vals.median(), n_epochs=n))
    df = pd.DataFrame(rows)
    df.to_csv(str(OUT/"features"/"per_subject_aggregates.csv"), index=False)
    log.info(f"per_subject_aggregates: {df.shape}")
    return df


# ── Statistics helpers ────────────────────────────────────────────────────────
def rank_biserial(diffs):
    diffs = np.asarray(diffs, float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return np.nan
    n = len(diffs)
    ranks = stats.rankdata(np.abs(diffs))
    T_plus  = float(ranks[diffs > 0].sum()) if any(diffs > 0) else 0.0
    T_minus = float(ranks[diffs < 0].sum()) if any(diffs < 0) else 0.0
    return (T_plus - T_minus) / (n * (n + 1) / 2)


def bootstrap_ci(diffs, n_boot=1000):
    diffs = np.asarray(diffs, float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(RANDOM_STATE)
    bs = [rank_biserial(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cohens_d(diffs):
    diffs = np.asarray(diffs, float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 2:
        return np.nan
    return float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-12))


def power_calc(d, n=9):
    if np.isnan(d):
        return np.nan
    try:
        return TTestPower().power(effect_size=abs(d), nobs=n, alpha=0.05, alternative="two-sided")
    except Exception:
        return np.nan


# ── Level 1 ───────────────────────────────────────────────────────────────────
def level1(agg):
    results = []
    for feat in ALL_FEAT_COLS:
        sub = agg[agg["feature_name"] == feat]
        v = sub[sub["phase"]=="variant"].set_index("subject_id")["mean"]
        c = sub[sub["phase"]=="control"].set_index("subject_id")["mean"]
        sids = sorted(set(v.index) & set(c.index))
        if len(sids) < 3:
            continue
        v_arr = v.loc[sids].values.astype(float)
        c_arr = c.loc[sids].values.astype(float)
        mask  = ~(np.isnan(v_arr) | np.isnan(c_arr))
        if mask.sum() < 3:
            continue
        diffs = (v_arr - c_arr)[mask]
        v_arr, c_arr = v_arr[mask], c_arr[mask]

        try:
            stat, pval = wilcoxon(diffs, alternative="two-sided")
        except Exception:
            stat, pval = np.nan, np.nan

        rb      = rank_biserial(diffs)
        lo, hi  = bootstrap_ci(diffs)
        cd      = cohens_d(diffs)
        pwr     = power_calc(cd, n=len(diffs))
        dirn    = "variant>" if np.nanmean(diffs) > 0 else "control>"

        results.append(dict(
            feature=feat,
            variant_median=float(np.nanmedian(v_arr)),
            control_median=float(np.nanmedian(c_arr)),
            n_subjects=int(mask.sum()),
            wilcoxon_statistic=stat,
            p_value=pval,
            rank_biserial=rb,
            rb_ci_low=lo, rb_ci_high=hi,
            cohens_d=cd,
            achieved_power=pwr,
            direction=dirn,
        ))

    df = pd.DataFrame(results)
    if df.empty:
        log.warning("Level1: empty! Check feature names.")
        return df

    p_vals = df["p_value"].values
    valid  = ~np.isnan(p_vals)
    p_fdr  = np.full(len(p_vals), np.nan)
    if valid.sum() > 0:
        _, p_corr, _, _ = multipletests(p_vals[valid], method="fdr_bh", alpha=0.05)
        p_fdr[valid]    = p_corr
    df["p_value_fdr"]   = p_fdr
    df["significant_fdr"] = df["p_value_fdr"] < 0.05
    df = df.sort_values("p_value_fdr")

    df.to_csv(str(OUT/"statistics"/"overall_comparison.csv"), index=False)
    log.info(f"Level1: {int(df['significant_fdr'].sum())} significant / {len(df)} tested")
    return df


# ── Level 2 ───────────────────────────────────────────────────────────────────
def level2(pef):
    results = []
    for sid in SUBJECTS:
        for feat in ALL_FEAT_COLS:
            if feat not in pef.columns:
                continue
            v = pd.to_numeric(pef[(pef["subject_id"]==sid)&(pef["phase"]=="variant")][feat], errors="coerce").dropna()
            c = pd.to_numeric(pef[(pef["subject_id"]==sid)&(pef["phase"]=="control")][feat], errors="coerce").dropna()
            if len(v) < 3 or len(c) < 3:
                continue
            try:
                u, p = mannwhitneyu(v, c, alternative="two-sided")
                rb   = 1 - 2*u/(len(v)*len(c))
                dirn = "variant>" if v.mean() > c.mean() else "control>"
            except Exception:
                u, p, rb, dirn = np.nan, np.nan, np.nan, "n/a"
            results.append(dict(subject_id=sid, feature=feat,
                                variant_mean=float(v.mean()), control_mean=float(c.mean()),
                                u_statistic=u, p_value=p, rank_biserial=rb, direction=dirn,
                                n_variant=len(v), n_control=len(c)))

    df = pd.DataFrame(results)
    df.to_csv(str(OUT/"statistics"/"per_subject_comparison.csv"), index=False)
    log.info(f"Level2: {len(df)} rows ({len(SUBJECTS)} subjects × {len(ALL_FEAT_COLS)} features)")
    return df


# ── Level 3 ───────────────────────────────────────────────────────────────────
def level3(pef):
    ctrl      = pef[pef["phase"]=="control"]
    scenarios = pef[pef["phase"]=="variant"]["scenario_name"].unique()
    results   = []
    for scen in scenarios:
        scen_df = pef[pef["scenario_name"]==scen]
        n_scen  = len(scen_df)
        for feat in ALL_FEAT_COLS:
            if feat not in pef.columns:
                continue
            sv = pd.to_numeric(scen_df[feat], errors="coerce").dropna()
            cv = pd.to_numeric(ctrl[feat],    errors="coerce").dropna()
            if len(sv) < 3 or len(cv) < 3:
                continue
            try:
                u, p = mannwhitneyu(sv, cv, alternative="two-sided")
                rb   = 1 - 2*u/(len(sv)*len(cv))
                dirn = "scenario>" if sv.mean() > cv.mean() else "control>"
            except Exception:
                u, p, rb, dirn = np.nan, np.nan, np.nan, "n/a"
            results.append(dict(scenario_name=scen, n_epochs=n_scen, feature=feat,
                                scenario_mean=float(sv.mean()), control_mean=float(cv.mean()),
                                u_statistic=u, p_value=p, effect_size=rb, direction=dirn))

    df = pd.DataFrame(results)
    df.to_csv(str(OUT/"statistics"/"per_scenario_comparison.csv"), index=False)
    log.info(f"Level3: {len(scenarios)} scenarios × {len(ALL_FEAT_COLS)} features")
    return df


# ── Visualizations ────────────────────────────────────────────────────────────
def plot_forest(overall):
    df = overall[~overall["rank_biserial"].isna()].copy()
    df = df.sort_values("rank_biserial", key=abs, ascending=False).head(30)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df)*0.38)))
    y = np.arange(len(df))
    colors = ["#2ca02c" if s else "#aec7e8" for s in df["significant_fdr"]]
    ax.barh(y, df["rank_biserial"], color=colors, alpha=0.75, height=0.6)
    for i, (_, r) in enumerate(df.iterrows()):
        ax.plot([r["rb_ci_low"], r["rb_ci_high"]], [i, i], "k-", lw=1.5, solid_capstyle="round")
        ax.plot([r["rb_ci_low"]], [i], "k|", ms=6)
        ax.plot([r["rb_ci_high"]], [i], "k|", ms=6)
    ax.axvline(0,    color="black", lw=0.8, ls="--")
    ax.axvline(0.3,  color="gray",  lw=0.5, ls=":")
    ax.axvline(-0.3, color="gray",  lw=0.5, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([_friendly(f) for f in df["feature"]], fontsize=8)
    ax.set_xlabel("Rank-biserial r  (← control higher  |  variant higher →)")
    ax.set_title("Effect Sizes - Variant vs Control Baseline\n(95% bootstrap CI, green = FDR q<0.05)")
    sig_p = mpatches.Patch(color="#2ca02c", alpha=0.75, label="Significant (q<0.05)")
    ns_p  = mpatches.Patch(color="#aec7e8", alpha=0.75, label="Not significant")
    ax.legend(handles=[sig_p, ns_p], fontsize=8)
    plt.tight_layout()
    plt.savefig(str(OUT/"visualizations"/"effect_sizes_forest_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Forest plot saved")


def plot_violins(pef, overall):
    sig = overall[overall["significant_fdr"]].sort_values("rank_biserial", key=abs, ascending=False)
    top = [f for f in sig["feature"].head(10) if f in pef.columns]
    if not top:
        top = [f for f in ALL_FEAT_COLS if f in pef.columns][:8]

    ncols = 2
    nrows = (len(top)+1)//2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows*3.2))
    axes = np.array(axes).flatten()

    for ai, feat in enumerate(top):
        ax = axes[ai]
        data_v = pd.to_numeric(pef[pef["phase"]=="variant"][feat], errors="coerce").dropna().values
        data_c = pd.to_numeric(pef[pef["phase"]=="control"][feat], errors="coerce").dropna().values
        parts  = ax.violinplot([data_v, data_c], positions=[1,2], showmedians=True, widths=0.6)
        parts["bodies"][0].set(facecolor="#E84C4C", alpha=0.4)
        parts["bodies"][1].set(facecolor="#4C9BE8", alpha=0.4)
        for pc in ["cbars","cmins","cmaxes","cmedians"]:
            if pc in parts:
                parts[pc].set(color="black", linewidth=0.8)

        for si, sid in enumerate(SUBJECTS):
            v_m = pd.to_numeric(pef[(pef["subject_id"]==sid)&(pef["phase"]=="variant")][feat], errors="coerce").mean()
            c_m = pd.to_numeric(pef[(pef["subject_id"]==sid)&(pef["phase"]=="control")][feat], errors="coerce").mean()
            if np.isnan(v_m) or np.isnan(c_m):
                continue
            jit = np.random.uniform(-0.07, 0.07)
            ax.plot([1+jit, 2+jit], [v_m, c_m], color=SUBJ_COLORS[si], alpha=0.7, lw=1.2)
            ax.scatter([1+jit, 2+jit], [v_m, c_m], color=SUBJ_COLORS[si], s=28, zorder=5)

        row = overall[overall["feature"]==feat]
        pstar = ""
        if not row.empty:
            p_fdr = row["p_value_fdr"].values[0]
            rb    = row["rank_biserial"].values[0]
            pstar = f"  r={rb:+.2f}, {'*' if p_fdr<0.05 else 'ns'}"
        ax.set_xticks([1,2]); ax.set_xticklabels(["Variant","Control"])
        ax.set_title(f"{_friendly(feat)}{pstar}", fontsize=9)
        ax.set_ylabel("Value", fontsize=8)

    for ai in range(len(top), len(axes)):
        axes[ai].set_visible(False)

    plt.suptitle("Feature Distributions: Variant vs Control  (per-subject dots+lines)", y=1.01, fontsize=11)
    plt.tight_layout()
    plt.savefig(str(OUT/"visualizations"/"feature_distributions_violin.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Violin plots saved")


def plot_heatmap(agg, overall):
    top = (overall.dropna(subset=["rank_biserial"])
                   .sort_values("rank_biserial", key=abs, ascending=False)
                   .head(20))
    feat_cols = [f for f in top["feature"] if f in agg["feature_name"].unique()]

    mat = np.full((len(SUBJECTS), len(feat_cols)), np.nan)
    for fi, feat in enumerate(feat_cols):
        sub = agg[agg["feature_name"]==feat]
        for si, sid in enumerate(SUBJECTS):
            vr = sub[(sub["subject_id"]==sid)&(sub["phase"]=="variant")]
            cr = sub[(sub["subject_id"]==sid)&(sub["phase"]=="control")]
            if vr.empty or cr.empty:
                continue
            sv = float(vr["std"].values[0]); sc = float(cr["std"].values[0])
            pool = np.sqrt((sv**2+sc**2)/2+1e-12)
            mat[si, fi] = (vr["mean"].values[0] - cr["mean"].values[0]) / pool

    fig, ax = plt.subplots(figsize=(max(10, len(feat_cols)*0.65), 5))
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    plt.colorbar(im, ax=ax, label="(variant − control) / pooled SD")
    ax.set_xticks(range(len(feat_cols)))
    ax.set_xticklabels([_friendly(f) for f in feat_cols], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(SUBJECTS)))
    ax.set_yticklabels([f"sub-{s}" for s in SUBJECTS], fontsize=9)
    ax.set_title("Per-Subject Standardized Difference (Variant − Control)\nBlue=control higher, Red=variant higher")
    plt.tight_layout()
    plt.savefig(str(OUT/"visualizations"/"per_subject_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Heatmap saved")


def plot_radar(pef, overall):
    sig = overall[overall["significant_fdr"]].sort_values("rank_biserial", key=abs, ascending=False)
    radar_feats = [f for f in sig["feature"].head(8) if f in pef.columns]
    if len(radar_feats) < 3:
        radar_feats = [f for f in ALL_FEAT_COLS if f in pef.columns][:6]

    scen_counts = pef[pef["phase"]=="variant"].groupby("scenario_name").size()
    scenarios   = scen_counts[scen_counts >= 5].index.tolist()

    # Per-feature normalization across all scenarios + control
    all_vals = {f: [] for f in radar_feats}
    scen_list = scenarios + ["_control"]
    for scen in scen_list:
        sub = pef[pef["phase"]=="control"] if scen == "_control" else pef[pef["scenario_name"]==scen]
        for f in radar_feats:
            all_vals[f].append(pd.to_numeric(sub[f], errors="coerce").mean())

    feat_min = {f: np.nanmin(all_vals[f]) for f in radar_feats}
    feat_max = {f: np.nanmax(all_vals[f]) for f in radar_feats}

    def norm(val, f):
        r = feat_max[f] - feat_min[f]
        return (val - feat_min[f]) / r if r > 1e-9 else 0.5

    n_ax   = len(radar_feats)
    angles = np.linspace(0, 2*np.pi, n_ax, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    cmap = plt.cm.tab20.colors

    ctrl_means = {f: pd.to_numeric(pef[pef["phase"]=="control"][f], errors="coerce").mean() for f in radar_feats}
    ctrl_norm  = [norm(ctrl_means[f], f) for f in radar_feats] + [norm(ctrl_means[radar_feats[0]], radar_feats[0])]
    ax.fill(angles, ctrl_norm, alpha=0.12, color="gray")
    ax.plot(angles, ctrl_norm, color="gray", lw=2, ls="--", label="Control baseline")

    for si, scen in enumerate(scenarios[:12]):
        sub  = pef[pef["scenario_name"]==scen]
        vals = [norm(pd.to_numeric(sub[f], errors="coerce").mean(), f) for f in radar_feats]
        vals += [vals[0]]
        ax.plot(angles, vals, color=cmap[si%20], lw=1.5, alpha=0.85, label=scen.replace("_"," "))
        ax.scatter(angles[:-1], vals[:-1], color=cmap[si%20], s=22, alpha=0.9)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_friendly(f) for f in radar_feats], size=8)
    ax.set_ylim(0, 1); ax.set_title("Per-Scenario Feature Profile vs Control Baseline", pad=20, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=7)
    plt.tight_layout()
    plt.savefig(str(OUT/"visualizations"/"per_scenario_radar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Radar chart saved")


# ── Sanity gate ───────────────────────────────────────────────────────────────
def sanity_gate(overall):
    sig  = overall[(overall["significant_fdr"]==True) & (overall["rank_biserial"].abs()>=GATE_MIN_EFFECT)]
    n    = len(sig)
    dec  = "GREEN" if n >= GATE_MIN_FEATURES else ("YELLOW" if n >= 3 else "RED")
    return dec, n, sig


# ── Report ────────────────────────────────────────────────────────────────────
def build_report(overall, l2, l3, gate, pef, agg):
    decision, n_pass, sig_feats = gate
    icons = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}

    sig_top  = overall[overall["significant_fdr"]].sort_values("rank_biserial", key=abs, ascending=False).head(12)
    ns_list  = overall[~overall["significant_fdr"]]["feature"].tolist()[:8]
    n_sig    = int(overall["significant_fdr"].sum())
    n_tested = len(overall)

    # Per-subject consistency
    consist = []
    for feat in sig_feats["feature"].tolist():
        exp = sig_feats[sig_feats["feature"]==feat]["direction"].values[0]
        n_c = 0
        for sid in SUBJECTS:
            v = pd.to_numeric(pef[(pef["subject_id"]==sid)&(pef["phase"]=="variant")][feat], errors="coerce").mean()
            c = pd.to_numeric(pef[(pef["subject_id"]==sid)&(pef["phase"]=="control")][feat], errors="coerce").mean()
            if np.isnan(v) or np.isnan(c):
                continue
            if (exp=="variant>" and v>c) or (exp=="control>" and c>v):
                n_c += 1
        consist.append((feat, n_c))

    # Power
    pwr_df = overall[overall["significant_fdr"]].dropna(subset=["achieved_power"])
    n_well = int((pwr_df["achieved_power"] >= 0.8).sum())
    n_under= int((pwr_df["achieved_power"] <  0.8).sum())

    # Per-scenario
    if not l3.empty:
        ss = l3.groupby("scenario_name")["effect_size"].apply(lambda x: x.abs().mean()).sort_values(ascending=False)
        strongest = ss.head(3).index.tolist()
        weakest   = ss.tail(3).index.tolist()
    else:
        strongest, weakest = [], []

    # Tables
    top_table = ("| Feature | V.Median | C.Median | r | p_fdr | d | Direction |\n"
                 "|---------|----------|----------|---|-------|---|-----------|\n")
    for _, r in sig_top.iterrows():
        arrow = "↑ variant" if r["direction"]=="variant>" else "↓ control"
        top_table += (f"| `{r['feature']}` | {r['variant_median']:.3f} | {r['control_median']:.3f} "
                      f"| {r['rank_biserial']:.3f} | {r['p_value_fdr']:.4f} | {r['cohens_d']:.3f} | {arrow} |\n")

    pwr_table = "| Feature | d | Power | Status |\n|---------|---|-------|--------|\n"
    for _, r in pwr_df.iterrows():
        st = "✓ ≥0.80" if r["achieved_power"]>=0.8 else "⚠ <0.80"
        pwr_table += f"| `{r['feature']}` | {r['cohens_d']:.3f} | {r['achieved_power']:.2f} | {st} |\n"

    c_table = f"| Feature | Consistent subjects (/{len(SUBJECTS)}) |\n|---------|--------------------------------|\n"
    for feat, nc in consist:
        c_table += f"| `{feat}` | {nc}/{len(SUBJECTS)} |\n"

    n_var = int((pef["phase"]=="variant").sum())
    n_ctl = int((pef["phase"]=="control").sum())

    report = f"""# Baseline Comparison Analysis Report

Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}

## Overview

| | |
|--|--|
| Dataset | v2 master arrays - 1452 epochs |
| Variant (frustration) | {n_var} epochs |
| Control (baseline) | {n_ctl} epochs |
| Subjects | 9 |
| Features tested | {n_tested} |
| Statistics | Wilcoxon signed-rank (N=9 pairs, L1), Mann-Whitney U (L2/L3) |
| Correction | Benjamini-Hochberg FDR q<0.05 |

---

## {icons.get(decision,'')} Sanity Gate: **{decision}**

**Criterion:** ≥{GATE_MIN_FEATURES} features with p_fdr<0.05 AND |r_biserial|≥{GATE_MIN_EFFECT}

**Result:** {n_pass} features pass both thresholds → **{decision}**

{"✅ Proceed to modeling." if decision=="GREEN" else
 "⚠️ Proceed with caution - marginal signal present." if decision=="YELLOW" else
 "🛑 STOP - re-evaluate pseudo-marker strategy."}

---

## Level 1 - Overall Group Comparison

### Significant features (FDR) : {n_sig} / {n_tested}

{top_table}

### Non-significant features (sample)
{", ".join(f"`{f}`" for f in ns_list)}

---

## Level 2 - Per-Subject Consistency

{c_table}

---

## Level 3 - Per-Scenario Differentiation

**Strongest frustration scenarios** (highest mean |effect|):
{chr(10).join(f"  {i+1}. {s}" for i,s in enumerate(strongest))}

**Weakest scenarios**:
{chr(10).join(f"  - {s}" for s in weakest)}

---

## Power Analysis (N=9, paired t approx.)

{pwr_table}

- Well-powered (≥0.80): **{n_well}**
- Underpowered (<0.80): **{n_under}**

*Note: Wilcoxon efficiency ≈ 0.955 × t-test. True power slightly higher than shown.*

---

## Implications for Modeling

**Features likely to drive classification:**
{chr(10).join(f"  - `{f}`" for f in sig_feats["feature"].tolist())}

**Features unlikely to contribute** (variant ≈ control):
{chr(10).join(f"  - `{f}`" for f in ns_list[:6])}

---

## Visualizations

| File | Content |
|------|---------|
| `effect_sizes_forest_plot.png` | **Main finding** - all features sorted by effect size |
| `feature_distributions_violin.png` | Top significant features with per-subject lines |
| `per_subject_heatmap.png` | Cross-subject consistency (standardized differences) |
| `per_scenario_radar.png` | Scenario-level feature profiles vs baseline |

---

## Ready for Modeling?

**{decision}** - {"YES, proceed to HusformerBITIRMEEG training." if decision=="GREEN"
                   else "YES with caution." if decision=="YELLOW"
                   else "NO - review design first."}
"""
    (OUT/"reports"/"baseline_comparison_report.md").write_text(report)
    log.info("  Report saved")
    return decision


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pef = pd.read_csv(str(OUT/"features"/"per_epoch_features.csv"))
    log.info(f"Loaded per_epoch_features: {pef.shape}, phases={pef['phase'].value_counts().to_dict()}")

    feats_in_pef = [f for f in ALL_FEAT_COLS if f in pef.columns]
    log.info(f"Matched feature columns: {len(feats_in_pef)}/{len(ALL_FEAT_COLS)}")

    log.info("\n=== Step 2: Aggregates ===")
    agg = build_aggregates(pef)

    log.info("\n=== Step 3: Level 1 Overall ===")
    overall = level1(agg)

    log.info("\n=== Step 4: Level 2 Per-Subject ===")
    l2 = level2(pef)

    log.info("\n=== Step 5: Level 3 Per-Scenario ===")
    l3 = level3(pef)

    log.info("\n=== Step 6: Visualizations ===")
    plot_forest(overall)
    plot_violins(pef, overall)
    plot_heatmap(agg, overall)
    plot_radar(pef, overall)

    log.info("\n=== Step 7: Sanity Gate + Report ===")
    gate     = sanity_gate(overall)
    decision = build_report(overall, l2, l3, gate, pef, agg)

    # ── Terminal summary ──────────────────────────────────────────────────────
    n_sig = int(overall["significant_fdr"].sum())
    log.info("\n" + "="*65)
    log.info(f"SANITY GATE : {gate[0]}  ({gate[1]} features pass effect+significance)")
    log.info(f"Significant : {n_sig} / {len(overall)} features (FDR q<0.05)")
    log.info("\nTop 5 by |effect size|:")
    for _, r in overall[overall["significant_fdr"]].head(5).iterrows():
        log.info(f"  {r['feature']:35s}  r={r['rank_biserial']:+.3f}  d={r['cohens_d']:+.3f}  p_fdr={r['p_value_fdr']:.4f}  pwr={r['achieved_power']:.2f}")
    if gate[0] == "RED":
        log.info("\n⚠️  RED gate - stop modeling, revisit pseudo-marker strategy")
    log.info("="*65)


if __name__ == "__main__":
    main()
