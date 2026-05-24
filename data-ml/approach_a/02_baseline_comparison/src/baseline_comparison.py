"""
Baseline Comparison Analysis - Adım 2-8
Variant (frustration) vs Control (baseline) within-subject paired statistics.

Output: approach_a/analysis/baseline_comparison/
"""

import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "approach_a" / "src"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal, stats
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
SUBJECTS = [14, 15, 16, 17, 18, 20, 21, 22, 23]
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SFREQ      = 500.0
ERP_WINDOW = (0.0, 2.0)   # post-stimulus window for EEG features (seconds)

# 32-ch layout (from inspection)
ALL_CH = ['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5','CP1',
          'Pz','P3','P7','O1','Oz','O2','P4','P8','TP10','CP6','CP2','Cz',
          'C4','T8','FT10','FC6','FC2','F4','F8','Fp2']

FRONTAL_CH   = [c for c in ALL_CH if c in {'Fp1','Fp2','F3','F4','Fz','F7','F8'}]
PARIETAL_CH  = [c for c in ALL_CH if c in {'P3','P4','Pz','P7','P8'}]
OCCIPITAL_CH = [c for c in ALL_CH if c in {'O1','O2','Oz','PO9','PO10'}]
CENTRAL_CH   = [c for c in ALL_CH if c in {'Cz','C3','C4','FC1','FC2','CP1','CP2'}]

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 40),
}

FAA_LOW_CONF_SIDS = {14, 17}   # F4 interpolated by AutoReject

EYE_WIN_S  = 2.2   # ERP window total duration: -200ms to +2000ms
GATE_MIN_FEATURES   = 5
GATE_MIN_EFFECT     = 0.3   # |rank_biserial| threshold
GATE_MIN_COHENS_D   = 0.5

OUT = ROOT / "approach_a" / "analysis" / "baseline_comparison"
FEAT_DIR  = ROOT / "approach_a" / "features"
PROC_DIR  = ROOT / "data" / "processed"

for d in [OUT/"features", OUT/"statistics", OUT/"visualizations", OUT/"reports"]:
    d.mkdir(parents=True, exist_ok=True)


# ── EEG Feature Extraction ─────────────────────────────────────────────────────
def _band_power(data_ch, sfreq, band):
    """Band power via Welch, log-transformed. data_ch: (n_times,)"""
    f, psd = signal.welch(data_ch, sfreq, nperseg=int(sfreq), noverlap=int(sfreq//2))
    mask = (f >= band[0]) & (f <= band[1])
    power = np.trapz(psd[mask], f[mask])
    return np.log(power + 1e-12)


def _spectral_entropy(data_ch, sfreq):
    """Shannon entropy of normalized broadband PSD."""
    f, psd = signal.welch(data_ch, sfreq, nperseg=int(sfreq), noverlap=int(sfreq//2))
    mask = (f >= 1) & (f <= 40)
    p = psd[mask]
    p = p / (p.sum() + 1e-12)
    return float(stats.entropy(p + 1e-12))


def compute_eeg_features_batch(epoch_data: np.ndarray,
                                ch_names: list,
                                sid: int) -> pd.DataFrame:
    """
    Compute EEG features for a batch of epochs.
    epoch_data: (n_epochs, n_channels, n_times) - already post-stimulus window
    Returns DataFrame with one row per epoch.
    """
    ch_idx = {ch: i for i, ch in enumerate(ch_names)}

    def ch_ids(names):
        return [ch_idx[c] for c in names if c in ch_idx]

    front_idx   = ch_ids(FRONTAL_CH)
    pariet_idx  = ch_ids(PARIETAL_CH)
    occip_idx   = ch_ids(OCCIPITAL_CH)
    central_idx = ch_ids(CENTRAL_CH)

    f3_idx = ch_idx.get("F3")
    f4_idx = ch_idx.get("F4")

    rows = []
    n_ep, n_ch, n_t = epoch_data.shape

    for ei in range(n_ep):
        x = epoch_data[ei]  # (32, n_times)

        # Per-electrode band powers → mean across all channels (global)
        feats = {}
        for band_name, band_range in BANDS.items():
            powers = [_band_power(x[c], SFREQ, band_range) for c in range(n_ch)]
            feats[f"{band_name}_power"] = float(np.mean(powers))

        # Spatial aggregates
        for grp_name, idx in [("frontal", front_idx), ("parietal", pariet_idx),
                               ("occipital", occip_idx), ("central", central_idx)]:
            for band_name, band_range in BANDS.items():
                if not idx:
                    feats[f"{grp_name}_{band_name}"] = np.nan
                    continue
                pws = [_band_power(x[c], SFREQ, band_range) for c in idx]
                feats[f"{grp_name}_{band_name}"] = float(np.mean(pws))

        # Key derived features
        ft = feats.get("frontal_theta", np.nan)
        fa = feats.get("frontal_alpha", np.nan)
        fb = feats.get("frontal_beta",  np.nan)
        ta = feats.get("theta_power",   np.nan)
        aa = feats.get("alpha_power",   np.nan)
        ba = feats.get("beta_power",    np.nan)

        # FAA = log(F4_alpha) - log(F3_alpha)
        if f3_idx is not None and f4_idx is not None:
            f3_alpha = _band_power(x[f3_idx], SFREQ, BANDS["alpha"])
            f4_alpha = _band_power(x[f4_idx], SFREQ, BANDS["alpha"])
            feats["FAA"] = float(f4_alpha - f3_alpha)
        else:
            feats["FAA"] = np.nan
        feats["FAA_low_confidence"] = int(sid in FAA_LOW_CONF_SIDS)

        # Theta/beta ratio (frontal cognitive load)
        feats["theta_beta_ratio"] = float(ft - fb) if not (np.isnan(ft) or np.isnan(fb)) else np.nan

        # Engagement index = beta / (alpha + theta) - global averages (log space → exp first)
        try:
            beta_lin  = np.exp(ba)
            alpha_lin = np.exp(aa)
            theta_lin = np.exp(ta)
            feats["engagement_index"] = float(beta_lin / (alpha_lin + theta_lin + 1e-12))
        except Exception:
            feats["engagement_index"] = np.nan

        # Spectral entropy (mean across all channels)
        feats["spectral_entropy"] = float(np.mean([_spectral_entropy(x[c], SFREQ) for c in range(n_ch)]))

        rows.append(feats)

    return pd.DataFrame(rows)


def load_eeg_features(sid: int, meta_rows: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Load epoch file, extract post-stimulus window, compute features."""
    if phase == "variant":
        epo_path = PROC_DIR / f"subject_{sid}" / "epochs_erp-epo.fif"
    else:
        epo_path = PROC_DIR / f"subject_{sid}" / "epochs_erp_control-epo.fif"

    epochs = mne.read_epochs(str(epo_path), preload=True, verbose=False)
    ch_names = epochs.ch_names
    sfreq    = epochs.info["sfreq"]
    times    = epochs.times

    # Post-stimulus window indices
    t_mask = (times >= ERP_WINDOW[0]) & (times <= ERP_WINDOW[1])

    indices = meta_rows["eeg_index"].values.astype(int)
    data_all = epochs.get_data()   # (n_all, 32, n_times)

    # Clip indices to valid range
    valid = indices < len(data_all)
    if not valid.all():
        log.warning(f"  sub-{sid} {phase}: {(~valid).sum()} eeg_index out of range, clipping")
    indices = np.clip(indices, 0, len(data_all) - 1)

    data_sel = data_all[indices][:, :, t_mask]   # (n_epochs, 32, n_post_samples)

    log.info(f"  sub-{sid} {phase}: EEG features for {len(indices)} epochs "
             f"({data_sel.shape[2]} post-stim samples)")
    feat_df = compute_eeg_features_batch(data_sel, ch_names, sid)
    feat_df.index = meta_rows.index
    return feat_df


# ── Eye / Mouse Feature Loading ────────────────────────────────────────────────
EYE_COLS   = ["fixation_count", "fixation_mean_dur_ms", "saccade_count",
              "saccade_mean_amp_deg", "blink_count", "gaze_dispersion",
              "nan_ratio", "n_samples"]
MOUSE_COLS = ["velocity_mean", "velocity_max", "acceleration_mean",
              "path_length_px", "auc_deviation", "x_flips", "y_flips",
              "idle_ratio", "click_count", "rage_click_flag"]


def _join_nearest(meta_wt: np.ndarray, feat_df: pd.DataFrame,
                  cols: list, tol_ms: float = 500.0) -> pd.DataFrame:
    """Join feature CSV to epoch list by nearest wall_time_ms within tolerance."""
    out = np.full((len(meta_wt), len(cols)), np.nan)
    feat_wt = feat_df["wall_time_ms"].values
    for i, t in enumerate(meta_wt):
        diffs = np.abs(feat_wt - t)
        best  = diffs.argmin()
        if diffs[best] <= tol_ms:
            out[i] = feat_df.iloc[best][cols].values.astype(float)
    return pd.DataFrame(out, columns=cols)


def load_eye_features(sid: int, meta_rows: pd.DataFrame, phase: str) -> pd.DataFrame:
    suffix = "" if phase == "variant" else "_control"
    path   = PROC_DIR / f"subject_{sid}" / f"eye_epoch_features{suffix}_erp.csv"
    if not path.exists():
        return pd.DataFrame(np.nan, index=meta_rows.index, columns=EYE_COLS)
    feat = pd.read_csv(str(path))
    result = _join_nearest(meta_rows["wall_time_ms"].values, feat, EYE_COLS)
    result.index = meta_rows.index
    # Add blink_rate_per_sec
    result["blink_rate_per_sec"] = result["blink_count"] / EYE_WIN_S
    return result


def load_mouse_features(sid: int, meta_rows: pd.DataFrame, phase: str) -> pd.DataFrame:
    suffix = "" if phase == "variant" else "_control"
    path   = PROC_DIR / f"subject_{sid}" / f"mouse_epoch_features{suffix}_erp.csv"
    if not path.exists():
        return pd.DataFrame(np.nan, index=meta_rows.index, columns=MOUSE_COLS)
    feat = pd.read_csv(str(path))
    result = _join_nearest(meta_rows["wall_time_ms"].values, feat, MOUSE_COLS)
    result.index = meta_rows.index
    return result


# ── Feature column lists ───────────────────────────────────────────────────────
EEG_FEAT_COLS = (
    [f"{b}_power" for b in BANDS]
    + [f"{g}_{b}" for g in ("frontal","parietal","occipital","central") for b in BANDS]
    + ["FAA", "theta_beta_ratio", "engagement_index", "spectral_entropy"]
)
EYE_FEAT_COLS   = EYE_COLS + ["blink_rate_per_sec"]
MOUSE_FEAT_COLS = MOUSE_COLS
ALL_FEAT_COLS   = EEG_FEAT_COLS + EYE_FEAT_COLS + MOUSE_FEAT_COLS


# ── Step 1: per_epoch_features.csv ────────────────────────────────────────────
def build_per_epoch_features(meta_df: pd.DataFrame) -> pd.DataFrame:
    all_rows = []

    for sid in SUBJECTS:
        for phase, lclass in [("variant", 1), ("control", 0)]:
            rows = meta_df[
                (meta_df["subject_id"] == sid) & (meta_df["label_class"] == lclass)
            ].copy()
            if rows.empty:
                continue

            log.info(f"sub-{sid} {phase}: {len(rows)} epochs")

            eeg   = load_eeg_features(sid, rows, phase)
            eye   = load_eye_features(sid, rows, phase)
            mouse = load_mouse_features(sid, rows, phase)

            base = rows[["global_idx","subject_id","epoch_id","phase",
                          "scenario_name","wall_time_ms","label_class"]].copy()
            base = base.rename(columns={"label_class": "label"})
            # Normalize phase: "variant_a/b/c" → "variant"
            base["phase"] = phase

            combined = pd.concat([base.reset_index(drop=True),
                                   eeg.reset_index(drop=True),
                                   eye.reset_index(drop=True),
                                   mouse.reset_index(drop=True)], axis=1)
            all_rows.append(combined)

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(str(OUT / "features" / "per_epoch_features.csv"), index=False)
    log.info(f"per_epoch_features.csv: {df.shape}")
    return df


# ── Step 2: per_subject_aggregates.csv ────────────────────────────────────────
def build_aggregates(pef: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid in SUBJECTS:
        for phase in ["variant", "control"]:
            sub = pef[(pef["subject_id"] == sid) & (pef["phase"] == phase)]
            n   = len(sub)
            for feat in ALL_FEAT_COLS:
                if feat not in sub.columns:
                    continue
                vals = pd.to_numeric(sub[feat], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                rows.append(dict(subject_id=sid, phase=phase, feature_name=feat,
                                 mean=vals.mean(), std=vals.std(), median=vals.median(),
                                 n_epochs=n))
    df = pd.DataFrame(rows)
    df.to_csv(str(OUT / "features" / "per_subject_aggregates.csv"), index=False)
    log.info(f"per_subject_aggregates.csv: {df.shape}")
    return df


# ── Statistics helpers ─────────────────────────────────────────────────────────
def rank_biserial(diffs: np.ndarray) -> float:
    """Rank-biserial correlation from paired differences."""
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return np.nan
    n = len(diffs)
    abs_d = np.abs(diffs)
    ranks = stats.rankdata(abs_d)
    T_plus  = ranks[diffs > 0].sum() if any(diffs > 0) else 0.0
    T_minus = ranks[diffs < 0].sum() if any(diffs < 0) else 0.0
    return float((T_plus - T_minus) / (n * (n + 1) / 2))


def bootstrap_ci_rb(diffs: np.ndarray, n_boot: int = 1000,
                     ci: float = 95) -> tuple:
    """Bootstrap 95% CI for rank-biserial correlation."""
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(RANDOM_STATE)
    bs = [rank_biserial(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
    lo = (100 - ci) / 2
    return float(np.percentile(bs, lo)), float(np.percentile(bs, 100 - lo))


def cohens_d_paired(diffs: np.ndarray) -> float:
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 2:
        return np.nan
    return float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-12))


def achieved_power(cohens_d: float, n: int = 9, alpha: float = 0.05) -> float:
    if np.isnan(cohens_d):
        return np.nan
    try:
        return TTestPower().power(effect_size=abs(cohens_d), nobs=n, alpha=alpha, alternative="two-sided")
    except Exception:
        return np.nan


# ── Level 1: Overall Comparison ────────────────────────────────────────────────
def level1_overall(agg: pd.DataFrame) -> pd.DataFrame:
    results = []
    for feat in ALL_FEAT_COLS:
        sub = agg[agg["feature_name"] == feat]
        v = sub[sub["phase"] == "variant"].set_index("subject_id")["mean"]
        c = sub[sub["phase"] == "control"].set_index("subject_id")["mean"]
        sids = sorted(set(v.index) & set(c.index))
        if len(sids) < 3:
            continue
        v_arr = v.loc[sids].values.astype(float)
        c_arr = c.loc[sids].values.astype(float)
        diffs = v_arr - c_arr

        # Drop NaN pairs
        mask = ~(np.isnan(v_arr) | np.isnan(c_arr))
        if mask.sum() < 3:
            continue
        v_arr, c_arr, diffs = v_arr[mask], c_arr[mask], diffs[mask]

        try:
            stat, pval = wilcoxon(diffs, alternative="two-sided")
        except Exception:
            stat, pval = np.nan, np.nan

        rb   = rank_biserial(diffs)
        rb_lo, rb_hi = bootstrap_ci_rb(diffs)
        cd   = cohens_d_paired(diffs)
        pwr  = achieved_power(cd, n=len(diffs))
        dirn = "variant>" if np.nanmean(diffs) > 0 else "control>"

        results.append(dict(
            feature=feat,
            variant_median=float(np.nanmedian(v_arr)),
            control_median=float(np.nanmedian(c_arr)),
            n_subjects=int(mask.sum()),
            wilcoxon_statistic=stat,
            p_value=pval,
            rank_biserial=rb,
            rb_ci_low=rb_lo,
            rb_ci_high=rb_hi,
            cohens_d=cd,
            achieved_power=pwr,
            direction=dirn,
        ))

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # BH FDR correction
    p_vals = df["p_value"].values
    valid  = ~np.isnan(p_vals)
    p_fdr  = np.full(len(p_vals), np.nan)
    if valid.sum() > 0:
        _, p_corr, _, _ = multipletests(p_vals[valid], method="fdr_bh", alpha=0.05)
        p_fdr[valid] = p_corr
    df["p_value_fdr"] = p_fdr
    df["significant_fdr"] = df["p_value_fdr"] < 0.05
    df = df.sort_values("p_value_fdr")

    df.to_csv(str(OUT / "statistics" / "overall_comparison.csv"), index=False)
    log.info(f"Level1: {df['significant_fdr'].sum()} significant features (FDR)")
    return df


# ── Level 2: Per-Subject Comparison ───────────────────────────────────────────
def level2_per_subject(pef: pd.DataFrame) -> pd.DataFrame:
    results = []
    for sid in SUBJECTS:
        for feat in ALL_FEAT_COLS:
            if feat not in pef.columns:
                continue
            v = pd.to_numeric(pef[(pef["subject_id"]==sid) & (pef["phase"]=="variant")][feat], errors="coerce").dropna()
            c = pd.to_numeric(pef[(pef["subject_id"]==sid) & (pef["phase"]=="control")][feat], errors="coerce").dropna()
            if len(v) < 3 or len(c) < 3:
                continue
            try:
                u, p = mannwhitneyu(v, c, alternative="two-sided")
                rb   = 1 - 2*u/(len(v)*len(c))
                dirn = "variant>" if v.mean() > c.mean() else "control>"
            except Exception:
                u, p, rb, dirn = np.nan, np.nan, np.nan, "n/a"
            results.append(dict(
                subject_id=sid, feature=feat,
                variant_mean=float(v.mean()), control_mean=float(c.mean()),
                u_statistic=u, p_value=p, rank_biserial=rb, direction=dirn,
                n_variant=len(v), n_control=len(c),
            ))

    df = pd.DataFrame(results)
    df.to_csv(str(OUT / "statistics" / "per_subject_comparison.csv"), index=False)
    log.info(f"Level2 done: {len(df)} rows")
    return df


# ── Level 3: Per-Scenario Comparison ──────────────────────────────────────────
def level3_per_scenario(pef: pd.DataFrame) -> pd.DataFrame:
    control = pef[pef["phase"] == "control"]
    scenarios = pef[pef["phase"] == "variant"]["scenario_name"].unique()
    results = []
    for scen in scenarios:
        scen_df = pef[pef["scenario_name"] == scen]
        n_scen  = len(scen_df)
        for feat in ALL_FEAT_COLS:
            if feat not in pef.columns:
                continue
            sv = pd.to_numeric(scen_df[feat], errors="coerce").dropna()
            cv = pd.to_numeric(control[feat],  errors="coerce").dropna()
            if len(sv) < 3 or len(cv) < 3:
                continue
            try:
                u, p = mannwhitneyu(sv, cv, alternative="two-sided")
                rb   = 1 - 2*u/(len(sv)*len(cv))
                dirn = "scenario>" if sv.mean() > cv.mean() else "control>"
            except Exception:
                u, p, rb, dirn = np.nan, np.nan, np.nan, "n/a"
            results.append(dict(
                scenario_name=scen, n_epochs=n_scen, feature=feat,
                scenario_mean=float(sv.mean()), control_mean=float(cv.mean()),
                u_statistic=u, p_value=p, effect_size=rb, direction=dirn,
            ))

    df = pd.DataFrame(results)
    df.to_csv(str(OUT / "statistics" / "per_scenario_comparison.csv"), index=False)
    log.info(f"Level3 done: {len(scenarios)} scenarios")
    return df


# ── Visualizations ─────────────────────────────────────────────────────────────
SUBJ_COLORS = plt.cm.tab10.colors[:9]

def _friendly_name(f):
    return (f.replace("_power","").replace("frontal_","F-").replace("parietal_","P-")
             .replace("occipital_","O-").replace("central_","C-")
             .replace("_"," ").upper())


def plot_forest(overall: pd.DataFrame):
    df = overall[~overall["rank_biserial"].isna()].copy()
    df = df.sort_values("rank_biserial", key=abs, ascending=False).head(30)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df)*0.38)))
    y = np.arange(len(df))
    colors = ["#2ca02c" if s else "#aec7e8" for s in df["significant_fdr"]]

    ax.barh(y, df["rank_biserial"], color=colors, alpha=0.75, height=0.6)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot([row["rb_ci_low"], row["rb_ci_high"]], [i, i],
                color="black", linewidth=1.5, solid_capstyle="round")
        ax.plot([row["rb_ci_low"], row["rb_ci_high"]], [i, i], "k|", markersize=6)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0.3,  color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(-0.3, color="gray", linewidth=0.5, linestyle=":")
    ax.set_yticks(y)
    ax.set_yticklabels([_friendly_name(f) for f in df["feature"]], fontsize=9)
    ax.set_xlabel("Rank-biserial correlation (variant vs control)\n← control > variant | variant > control →")
    ax.set_title("Effect Sizes - Variant vs Control Baseline\n(95% bootstrap CI, green = FDR q<0.05)")
    sig_patch = mpatches.Patch(color="#2ca02c", alpha=0.75, label="Significant (FDR q<0.05)")
    ns_patch  = mpatches.Patch(color="#aec7e8", alpha=0.75, label="Not significant")
    ax.legend(handles=[sig_patch, ns_patch], fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(str(OUT / "visualizations" / "effect_sizes_forest_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Forest plot saved")


def plot_violins(pef: pd.DataFrame, overall: pd.DataFrame):
    sig = overall[overall["significant_fdr"]].sort_values("rank_biserial", key=abs, ascending=False)
    top_feats = [f for f in sig["feature"].head(10).tolist() if f in pef.columns]
    if not top_feats:
        top_feats = [f for f in ALL_FEAT_COLS if f in pef.columns][:8]

    ncols = 2
    nrows = (len(top_feats) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = np.array(axes).flatten()

    for ax_i, feat in enumerate(top_feats):
        ax = axes[ax_i]
        data = [
            pd.to_numeric(pef[pef["phase"] == "variant"][feat], errors="coerce").dropna().values,
            pd.to_numeric(pef[pef["phase"] == "control"][feat], errors="coerce").dropna().values,
        ]
        parts = ax.violinplot(data, positions=[1, 2], showmedians=True, widths=0.6)
        parts["bodies"][0].set_facecolor("#E84C4C"); parts["bodies"][0].set_alpha(0.4)
        parts["bodies"][1].set_facecolor("#4C9BE8"); parts["bodies"][1].set_alpha(0.4)
        for pc in ["cbars","cmins","cmaxes","cmedians"]:
            if pc in parts:
                parts[pc].set_color("black"); parts[pc].set_linewidth(0.8)

        # Per-subject dots + lines
        for si, sid in enumerate(SUBJECTS):
            v_val = pd.to_numeric(pef[(pef["subject_id"]==sid)&(pef["phase"]=="variant")][feat], errors="coerce").mean()
            c_val = pd.to_numeric(pef[(pef["subject_id"]==sid)&(pef["phase"]=="control")][feat], errors="coerce").mean()
            if np.isnan(v_val) or np.isnan(c_val):
                continue
            jitter = np.random.uniform(-0.08, 0.08)
            ax.plot([1+jitter, 2+jitter], [v_val, c_val], color=SUBJ_COLORS[si],
                    alpha=0.7, linewidth=1.0)
            ax.scatter([1+jitter, 2+jitter], [v_val, c_val], color=SUBJ_COLORS[si],
                       s=30, zorder=5, alpha=0.9)

        row = overall[overall["feature"] == feat]
        pstar = ""
        if not row.empty:
            p_fdr = row["p_value_fdr"].values[0]
            rb    = row["rank_biserial"].values[0]
            pstar = f" (r={rb:.2f}, {'*' if p_fdr<0.05 else 'ns'})"

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Variant", "Control"])
        ax.set_title(f"{_friendly_name(feat)}{pstar}", fontsize=9)
        ax.set_ylabel("Value")

    for ax_i in range(len(top_feats), len(axes)):
        axes[ax_i].set_visible(False)

    plt.suptitle("Feature Distributions: Variant vs Control (per-subject dots + lines)", y=1.01, fontsize=11)
    plt.tight_layout()
    plt.savefig(str(OUT / "visualizations" / "feature_distributions_violin.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Violin plots saved")


def plot_heatmap(agg: pd.DataFrame, overall: pd.DataFrame):
    # Select top 20 features by |rank_biserial|
    top = overall.dropna(subset=["rank_biserial"]).sort_values("rank_biserial", key=abs, ascending=False).head(20)
    feat_cols = [f for f in top["feature"].tolist() if f in
                 agg["feature_name"].unique()]

    mat  = np.full((len(SUBJECTS), len(feat_cols)), np.nan)
    for fi, feat in enumerate(feat_cols):
        for si, sid in enumerate(SUBJECTS):
            sub = agg[agg["feature_name"] == feat]
            v   = sub[(sub["subject_id"]==sid) & (sub["phase"]=="variant")]["mean"].values
            c   = sub[(sub["subject_id"]==sid) & (sub["phase"]=="control")]["mean"].values
            if len(v) and len(c):
                # Pooled std
                sv = sub[(sub["subject_id"]==sid) & (sub["phase"]=="variant")]["std"].values[0]
                sc = sub[(sub["subject_id"]==sid) & (sub["phase"]=="control")]["std"].values[0]
                pool = np.sqrt((sv**2 + sc**2) / 2 + 1e-12)
                mat[si, fi] = (v[0] - c[0]) / pool

    fig, ax = plt.subplots(figsize=(max(10, len(feat_cols)*0.65), 5))
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    plt.colorbar(im, ax=ax, label="(variant − control) / pooled SD")
    ax.set_xticks(range(len(feat_cols)))
    ax.set_xticklabels([_friendly_name(f) for f in feat_cols], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(SUBJECTS)))
    ax.set_yticklabels([f"sub-{s}" for s in SUBJECTS], fontsize=9)
    ax.set_title("Per-Subject Standardized Difference (Variant − Control)\nBlue=control higher, Red=variant higher")
    plt.tight_layout()
    plt.savefig(str(OUT / "visualizations" / "per_subject_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Heatmap saved")


def plot_radar(pef: pd.DataFrame, overall: pd.DataFrame):
    # Select top 8 significant features for radar axes
    sig = overall[overall["significant_fdr"]].sort_values("rank_biserial", key=abs, ascending=False)
    radar_feats = [f for f in sig["feature"].head(8).tolist() if f in pef.columns]
    if len(radar_feats) < 3:
        radar_feats = [f for f in ALL_FEAT_COLS if f in pef.columns][:6]

    # Scenarios with >= 10 epochs
    scen_counts = pef[pef["phase"]=="variant"].groupby("scenario_name").size()
    scenarios   = scen_counts[scen_counts >= 10].index.tolist()
    if not scenarios:
        scenarios = scen_counts.index.tolist()

    # Normalize each feature across scenarios + control for [0,1] scale
    ctrl_means = {f: pd.to_numeric(pef[pef["phase"]=="control"][f], errors="coerce").mean()
                  for f in radar_feats}
    all_vals   = {f: [] for f in radar_feats}
    for scen in scenarios + ["control_baseline"]:
        sub = pef[pef["scenario_name"]==scen] if scen != "control_baseline" else pef[pef["phase"]=="control"]
        for f in radar_feats:
            all_vals[f].append(pd.to_numeric(sub[f], errors="coerce").mean())
    feat_min = {f: np.nanmin(all_vals[f]) for f in radar_feats}
    feat_max = {f: np.nanmax(all_vals[f]) for f in radar_feats}

    def norm(val, f):
        r = feat_max[f] - feat_min[f]
        return (val - feat_min[f]) / r if r > 1e-9 else 0.5

    n_ax   = len(radar_feats)
    angles = np.linspace(0, 2*np.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    cmap = plt.cm.tab20.colors

    # Draw control baseline first (filled)
    ctrl_norm = [norm(ctrl_means[f], f) for f in radar_feats] + [norm(ctrl_means[radar_feats[0]], radar_feats[0])]
    ax.fill(angles, ctrl_norm, alpha=0.15, color="gray")
    ax.plot(angles, ctrl_norm, color="gray", linewidth=2, linestyle="--", label="Control baseline")

    for si, scen in enumerate(scenarios[:12]):
        sub = pef[pef["scenario_name"]==scen]
        vals = [norm(pd.to_numeric(sub[f], errors="coerce").mean(), f) for f in radar_feats]
        vals += vals[:1]
        ax.plot(angles, vals, color=cmap[si % len(cmap)], linewidth=1.5,
                alpha=0.8, label=scen.replace("_", " "))
        ax.scatter(angles[:-1], vals[:-1], color=cmap[si % len(cmap)], s=20, alpha=0.8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_friendly_name(f) for f in radar_feats], size=8)
    ax.set_ylim(0, 1)
    ax.set_title("Per-Scenario Feature Profile vs Control Baseline", pad=20, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=7)
    plt.tight_layout()
    plt.savefig(str(OUT / "visualizations" / "per_scenario_radar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Radar chart saved")


# ── Sanity Gate ────────────────────────────────────────────────────────────────
def sanity_gate(overall: pd.DataFrame) -> tuple:
    """Returns (decision, n_passing, passing_feats)"""
    sig = overall[
        (overall["significant_fdr"] == True) &
        (overall["rank_biserial"].abs() >= GATE_MIN_EFFECT)
    ]
    n = len(sig)
    decision = "GREEN" if n >= GATE_MIN_FEATURES else ("YELLOW" if n >= 3 else "RED")
    return decision, n, sig


# ── Report ─────────────────────────────────────────────────────────────────────
def build_report(overall: pd.DataFrame, l2: pd.DataFrame, l3: pd.DataFrame,
                 gate: tuple, pef: pd.DataFrame):
    decision, n_pass, sig_feats = gate

    gate_color = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(decision, "")
    sig_top = overall[overall["significant_fdr"]].sort_values("rank_biserial", key=abs, ascending=False).head(10)
    ns_feats = overall[~overall["significant_fdr"]]["feature"].tolist()[:10]

    # Per-subject consistency: count subjects where each feature goes in expected direction
    consist_rows = []
    for feat in sig_feats["feature"].tolist():
        expected_dir = sig_feats[sig_feats["feature"]==feat]["direction"].values[0]
        n_consistent = 0
        for sid in SUBJECTS:
            v = pd.to_numeric(pef[(pef["subject_id"]==sid)&(pef["phase"]=="variant")][feat], errors="coerce").mean()
            c = pd.to_numeric(pef[(pef["subject_id"]==sid)&(pef["phase"]=="control")][feat], errors="coerce").mean()
            if np.isnan(v) or np.isnan(c):
                continue
            if (expected_dir == "variant>" and v > c) or (expected_dir == "control>" and c > v):
                n_consistent += 1
        consist_rows.append((feat, n_consistent))

    # Power analysis rows
    power_rows = overall[overall["significant_fdr"]].dropna(subset=["achieved_power"])
    n_well_powered = int((power_rows["achieved_power"] >= 0.8).sum())
    n_underpowered = int((power_rows["achieved_power"] < 0.8).sum())

    # Per-scenario: strongest / weakest (by mean |effect_size| across features)
    if not l3.empty:
        scen_strength = l3.groupby("scenario_name")["effect_size"].apply(
            lambda x: x.abs().mean()).sort_values(ascending=False)
        strongest = scen_strength.head(3).index.tolist()
        weakest   = scen_strength.tail(3).index.tolist()
    else:
        strongest, weakest = [], []

    top_table = "| Feature | Variant Median | Control Median | r_biserial | p_fdr | Cohen's d | Direction |\n"
    top_table += "|---------|---------------|----------------|------------|-------|-----------|----------|\n"
    for _, row in sig_top.iterrows():
        dirn_arrow = "↑ variant" if row["direction"] == "variant>" else "↓ control"
        top_table += (f"| {row['feature']} | {row['variant_median']:.3f} | {row['control_median']:.3f} "
                      f"| {row['rank_biserial']:.3f} | {row['p_value_fdr']:.4f} "
                      f"| {row['cohens_d']:.3f} | {dirn_arrow} |\n")

    power_table = "| Feature | Cohen's d | Achieved Power | Status |\n|---------|-----------|---------------|--------|\n"
    for _, row in power_rows.iterrows():
        status = "✓ well-powered" if row["achieved_power"] >= 0.8 else "⚠ underpowered"
        power_table += f"| {row['feature']} | {row['cohens_d']:.3f} | {row['achieved_power']:.2f} | {status} |\n"

    consist_table = "| Feature | N subjects consistent (/{}) |\n|---------|----------------------------|\n".format(len(SUBJECTS))
    for feat, n_c in consist_rows:
        consist_table += f"| {feat} | {n_c}/{len(SUBJECTS)} |\n"

    n_var_total = int((pef["phase"]=="variant").sum())
    n_ctl_total = int((pef["phase"]=="control").sum())
    n_sig_total = int(overall["significant_fdr"].sum())
    n_tested    = len(overall)

    report = f"""# Baseline Comparison Analysis Report

Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}

## Overview

- Dataset: v2 master arrays (1452 epochs)
  - Variant (frustration): {n_var_total} epochs
  - Control (baseline):    {n_ctl_total} epochs
- 9 subjects, within-subject paired statistics
- Features tested: {n_tested} (EEG + Eye + Mouse)
- Multiple comparison correction: Benjamini-Hochberg FDR (q<0.05)
- Wilcoxon signed-rank (N=9 subject pairs), Mann-Whitney U (per-subject/per-scenario)

---

## {gate_color} Sanity Gate Decision: **{decision}**

**Criterion:** ≥{GATE_MIN_FEATURES} features with p_fdr<0.05 AND |r_biserial|≥{GATE_MIN_EFFECT}

**Result:** {n_pass} features pass both criteria → **{decision}**

{"✅ Ready to proceed to modeling." if decision == "GREEN" else
 "⚠️ Proceed to modeling with caution - signal is present but marginal." if decision == "YELLOW" else
 "🛑 STOP. Re-evaluate pseudo-marker strategy before modeling."}

---

## Top Findings - Level 1 (Group Overall)

### Significant features after FDR correction ({n_sig_total}/{n_tested})

{top_table}

### Non-significant features (sample)
{", ".join(ns_feats)}

---

## Per-Subject Consistency - Level 2

{consist_table}

---

## Per-Scenario Differentiation - Level 3

**Strongest frustration-inducing scenarios** (highest mean |effect size|):
{chr(10).join(f"  {i+1}. {s}" for i, s in enumerate(strongest))}

**Weakest scenarios** (lowest mean |effect size|):
{chr(10).join(f"  - {s}" for s in weakest)}

---

## Power Analysis (N=9, α=0.05)

{power_table}

- Well-powered (≥0.80): {n_well_powered}
- Underpowered (<0.80): {n_underpowered}

Note: N=9 is small. Effect sizes are reliable indicators; power values are approximate
(paired t-test approximation for Wilcoxon efficiency ~0.955).

---

## Implications for Modeling

**Features that should drive classification** (significant, large effect):
{chr(10).join(f"  - {f}" for f in sig_feats["feature"].tolist())}

**Features that may not contribute** (similar in variant and control):
{chr(10).join(f"  - {f}" for f in ns_feats[:8])}

---

## Visualizations

- `visualizations/effect_sizes_forest_plot.png` - Main finding summary
- `visualizations/feature_distributions_violin.png` - Distributions with per-subject lines
- `visualizations/per_subject_heatmap.png` - Consistency across subjects
- `visualizations/per_scenario_radar.png` - Scenario differentiation

---

## Ready for Modeling?

**{decision}** - {"YES, proceed." if decision == "GREEN" else "YES but note marginal signal." if decision == "YELLOW" else "NO - revisit design."}
"""
    (OUT / "reports" / "baseline_comparison_report.md").write_text(report)
    log.info("  Report saved")
    return decision


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    meta = pd.read_csv(str(FEAT_DIR / "all_eeg_embeddings_v2_metadata.csv"))

    log.info("=== Step 1: per_epoch_features.csv ===")
    pef = build_per_epoch_features(meta)

    log.info("\n=== Step 2: per_subject_aggregates.csv ===")
    agg = build_aggregates(pef)

    log.info("\n=== Step 3: Level 1 - Overall comparison ===")
    overall = level1_overall(agg)

    log.info("\n=== Step 4: Level 2 - Per-subject ===")
    l2 = level2_per_subject(pef)

    log.info("\n=== Step 5: Level 3 - Per-scenario ===")
    l3 = level3_per_scenario(pef)

    log.info("\n=== Step 6: Visualizations ===")
    plot_forest(overall)
    plot_violins(pef, overall)
    plot_heatmap(agg, overall)
    plot_radar(pef, overall)

    log.info("\n=== Step 7: Sanity gate + Report ===")
    gate = sanity_gate(overall)
    decision = build_report(overall, l2, l3, gate, pef)

    # Print summary
    n_sig = int(overall["significant_fdr"].sum())
    log.info("\n" + "="*60)
    log.info(f"SANITY GATE: {gate[0]}  ({gate[1]} features pass both criteria)")
    log.info(f"Significant features (FDR): {n_sig}/{len(overall)}")
    top5 = overall[overall["significant_fdr"]].head(5)
    if not top5.empty:
        log.info("Top 5 features:")
        for _, r in top5.iterrows():
            log.info(f"  {r['feature']:35s}  r={r['rank_biserial']:+.3f}  d={r['cohens_d']:+.3f}  p_fdr={r['p_value_fdr']:.4f}")
    log.info(f"\nOutput: {OUT}")
    log.info("="*60)


if __name__ == "__main__":
    main()
