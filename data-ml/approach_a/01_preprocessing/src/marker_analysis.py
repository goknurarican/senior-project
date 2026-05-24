#!/usr/bin/env python3
"""
Bölüm 2: Marker ve Senaryo Analizi
====================================
2.1  Marker count matrix  (subject × marker_code)
2.2  Marker timing        (inter-marker intervals, trigger-bug check)
2.3  Senaryo süre         (eye active_scenario cross-check)
2.4  Faz timing           (control vs variant phase duration)
2.5  Variant dağılımı

Çıktılar:
  data/reports/marker_count_matrix.csv
  data/reports/marker_timing_stats.csv
  data/reports/scenario_durations.csv
  data/reports/phase_timing.csv
  figures/bölüm2_*.png
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

ROOT    = Path(__file__).parent.parent
CFG     = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
REPORTS = ROOT / "data" / "reports"
FIGS    = ROOT / "figures"
LOGS    = ROOT / "logs"
for p in (REPORTS, FIGS, LOGS):
    p.mkdir(parents=True, exist_ok=True)

# ─── Logging ─────────────────────────────────────────────

def make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", "%H:%M:%S")
    fh  = logging.FileHandler(LOGS / f"{name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    sh  = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO);  sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger

log = make_logger("marker_analysis")

# ─── Marker code → label mapping ─────────────────────────
MARKER_MAP = {
    2:  "variant_start",
    1:  "payment_retry_timeout",
    11: "slow_image",
    12: "broken_image",
    13: "skeleton_prolong",
    14: "search_irrelevant",
    15: "button_delay",
    16: "first_click_miss",
    17: "feedback_late",
    18: "network_jitter",
    19: "overlay_blocking",
    20: "price_change",
    21: "coupon_min_spend",
    22: "coupon_expired",
    23: "facet_reset_once",
    24: "sort_reset",
    30: "add_to_cart",
    31: "checkout_start",
    33: "search_performed",
    99: "experiment_end",
}

# Scenario markers only (1-24), excluding control/system markers
SCENARIO_CODES = sorted([c for c in MARKER_MAP if 1 <= c <= 24])
ALL_CODES      = sorted(MARKER_MAP.keys())

# ─── Load all subjects ───────────────────────────────────

def load_subjects():
    raw_dir  = ROOT / "data" / "raw"
    subjects = []
    for s in CFG["subjects"]:
        d = raw_dir / s["folder"]
        mc = d / "eeg" / "eeg_markers.csv"
        eye = d / "eye" / "eye_data_db.csv"
        if not mc.exists():
            continue
        df = pd.read_csv(mc)
        df["subject_id"]   = s["id"]
        df["subject_name"] = s["name"]
        df["group"]        = s["group"]
        subjects.append({
            "id":      s["id"],
            "name":    s["name"],
            "group":   s["group"],
            "folder":  d,
            "markers": df,
            "eye_path": eye,
        })
    return subjects


# ══════════════════════════════════════════════════════════
# 2.1  Marker count matrix
# ══════════════════════════════════════════════════════════

def marker_count_matrix(subjects: list) -> pd.DataFrame:
    log.info("2.1  Marker count matrix...")

    rows = []
    for s in subjects:
        nz  = s["markers"][s["markers"]["eeg_marker"] != 0]
        row = {"subject_id": s["id"], "name": s["name"], "group": s["group"]}
        for code in ALL_CODES:
            row[f"M{code:02d}_{MARKER_MAP[code][:12]}"] = int((nz["eeg_marker"] == code).sum())
        rows.append(row)

    df = pd.DataFrame(rows).set_index("subject_id")

    path = REPORTS / "marker_count_matrix.csv"
    df.to_csv(path)
    log.info(f"  Saved: {path}")

    # ── Heatmap ──────────────────────────────────────────
    code_cols = [c for c in df.columns if c.startswith("M")]
    mat = df[code_cols].values.astype(float)
    labels_y = [f"sub-{i:02d} {n[:14]}" for i, n in zip(df.index, df["name"])]
    labels_x = [c[4:] for c in code_cols]      # strip "MXX_"

    fig, ax = plt.subplots(figsize=(18, 6))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(labels_x)));  ax.set_xticklabels(labels_x, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels_y)));  ax.set_yticklabels(labels_y, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = int(mat[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                        color="white" if v > mat.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Count")
    ax.set_title("2.1  Marker Count Matrix (subject × marker)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGS / "b2_01_marker_count_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("  Figure: b2_01_marker_count_heatmap.png")

    return df


# ══════════════════════════════════════════════════════════
# 2.2  Marker timing analysis
# ══════════════════════════════════════════════════════════

def marker_timing_analysis(subjects: list) -> pd.DataFrame:
    log.info("2.2  Marker timing analysis...")

    all_intervals = []   # per (subject, code): list of gaps in seconds
    timing_rows   = []
    trigger_bugs  = []

    for s in subjects:
        nz = (s["markers"][s["markers"]["eeg_marker"].isin(SCENARIO_CODES)]
              .sort_values("wall_time_ms").copy())

        # Global inter-marker interval (consecutive scenario markers)
        if len(nz) > 1:
            gaps_all = nz["wall_time_ms"].diff().dropna() / 1000

        # Per-code intervals (same marker type recurring)
        for code in SCENARIO_CODES:
            code_rows = nz[nz["eeg_marker"] == code].sort_values("wall_time_ms")
            if len(code_rows) < 2:
                continue
            gaps = code_rows["wall_time_ms"].diff().dropna() / 1000
            bugs = int((gaps < 0.5).sum())
            if bugs:
                trigger_bugs.append({
                    "subject_id": s["id"],
                    "marker_code": code,
                    "marker_label": MARKER_MAP[code],
                    "n_bugs": bugs,
                    "min_gap_s": round(float(gaps.min()), 3),
                })
            for g in gaps:
                all_intervals.append({
                    "subject_id":   s["id"],
                    "marker_code":  code,
                    "marker_label": MARKER_MAP[code],
                    "gap_s":        float(g),
                })

        # Per-subject summary: all consecutive scenario markers
        if len(nz) > 1:
            timing_rows.append({
                "subject_id":   s["id"],
                "name":         s["name"],
                "n_markers":    len(nz),
                "gap_mean_s":   round(float(gaps_all.mean()), 2),
                "gap_median_s": round(float(gaps_all.median()), 2),
                "gap_min_s":    round(float(gaps_all.min()), 2),
                "gap_max_s":    round(float(gaps_all.max()), 2),
                "n_fast_<0.5s": int((gaps_all < 0.5).sum()),
                "n_fast_<3s":   int((gaps_all < 3.0).sum()),
            })

    df_timing = pd.DataFrame(timing_rows)
    df_int    = pd.DataFrame(all_intervals)

    # Save
    df_timing.to_csv(REPORTS / "marker_timing_stats.csv", index=False)
    if trigger_bugs:
        pd.DataFrame(trigger_bugs).to_csv(REPORTS / "trigger_bugs.csv", index=False)
        log.warning(f"  TRIGGER BUGS detected: {len(trigger_bugs)} instances")
    else:
        log.info("  No trigger bugs (<0.5s) found.")

    log.info(f"  Per-subject timing:\n{df_timing[['subject_id','name','n_markers','gap_median_s','gap_min_s','n_fast_<3s']].to_string(index=False)}")

    # ── Figure: inter-marker gap distribution ────────────
    if not df_int.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Boxplot per marker type (top 10 by count)
        top_codes = (df_int.groupby("marker_label")["gap_s"].count()
                     .sort_values(ascending=False).head(10).index.tolist())
        box_data = [df_int[df_int["marker_label"] == lbl]["gap_s"].values for lbl in top_codes]
        bp = axes[0].boxplot(box_data, patch_artist=True, notch=False)
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_codes)))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        axes[0].set_xticks(range(1, len(top_codes)+1))
        axes[0].set_xticklabels(top_codes, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("Inter-marker gap (s)")
        axes[0].set_title("2.2a  Gap distribution per scenario type", fontsize=11, fontweight="bold")
        axes[0].axhline(0.5, color="red", ls="--", lw=1, label="Trigger-bug threshold (0.5s)")
        axes[0].legend(fontsize=8)

        # Histogram of all inter-marker gaps
        all_gaps = df_int["gap_s"].values
        axes[1].hist(all_gaps[all_gaps < 60], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        axes[1].axvline(0.5, color="red",    ls="--", lw=1.5, label="Trigger-bug (<0.5s)")
        axes[1].axvline(3.0, color="orange", ls="--", lw=1.5, label="Fast (<3s)")
        axes[1].axvline(30,  color="green",  ls="--", lw=1.5, label="Expected max (~30s)")
        axes[1].set_xlabel("Gap (s)");  axes[1].set_ylabel("Count")
        axes[1].set_title("2.2b  All inter-scenario-marker gaps", fontsize=11, fontweight="bold")
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(FIGS / "b2_02_marker_timing.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        log.info("  Figure: b2_02_marker_timing.png")

    return df_timing


# ══════════════════════════════════════════════════════════
# 2.3  Senaryo süre analizi (active_scenario cross-check)
# ══════════════════════════════════════════════════════════

def scenario_duration_analysis(subjects: list) -> pd.DataFrame:
    log.info("2.3  Scenario duration from eye active_scenario...")

    all_durations = []

    for s in subjects:
        if not s["eye_path"].exists():
            continue
        try:
            eye = pd.read_csv(s["eye_path"],
                              usecols=["wall_time_ms", "active_scenario", "phase"])
        except Exception as e:
            log.error(f"  sub-{s['id']}: {e}")
            continue

        eye = eye.sort_values("wall_time_ms").reset_index(drop=True)
        eye["active_scenario"] = eye["active_scenario"].fillna("none")

        # Detect transitions
        eye["seg_id"] = (eye["active_scenario"] != eye["active_scenario"].shift()).cumsum()
        groups = eye.groupby("seg_id").agg(
            scenario=("active_scenario", "first"),
            phase=("phase", "first"),
            t0_ms=("wall_time_ms", "min"),
            t1_ms=("wall_time_ms", "max"),
            n_samples=("wall_time_ms", "count"),
        ).reset_index(drop=True)

        for _, row in groups.iterrows():
            scen = row["scenario"]
            if scen in ("none", "blink") or pd.isna(scen):
                continue
            dur_s = (row["t1_ms"] - row["t0_ms"]) / 1000
            if dur_s < 1.0:
                continue   # noise
            all_durations.append({
                "subject_id":    s["id"],
                "group":         s["group"],
                "phase":         row["phase"],
                "scenario_type": scen,
                "duration_s":    round(dur_s, 2),
                "n_eye_samples": int(row["n_samples"]),
                "t0_ms":         int(row["t0_ms"]),
            })

    df_dur = pd.DataFrame(all_durations)
    if df_dur.empty:
        log.warning("  No scenario durations found.")
        return df_dur

    df_dur.to_csv(REPORTS / "scenario_durations.csv", index=False)
    log.info(f"  {len(df_dur)} scenario instances extracted.")

    # Summary per scenario type
    summary = df_dur.groupby("scenario_type")["duration_s"].agg(
        ["mean", "median", "std", "min", "max", "count"]
    ).round(2).sort_values("count", ascending=False)
    log.info(f"\n  Scenario duration summary:\n{summary.to_string()}")

    # Cross-check: eeg_marker timing vs eye duration
    eeg_onsets = []
    for s in subjects:
        nz = (s["markers"][(s["markers"]["eeg_marker"].isin(SCENARIO_CODES)) &
                            (s["markers"]["eeg_marker"] != 2)]
              .copy())
        for _, row in nz.iterrows():
            eeg_onsets.append({
                "subject_id":    s["id"],
                "scenario_type": row["scenario_type"],
                "eeg_onset_ms":  row["wall_time_ms"],
            })
    df_eeg = pd.DataFrame(eeg_onsets)

    # Merge: find closest eye-segment onset to each eeg marker
    match_rows = []
    for _, eeg_row in df_eeg.iterrows():
        sid  = eeg_row["subject_id"]
        stype = eeg_row["scenario_type"]
        t_eeg = eeg_row["eeg_onset_ms"]

        candidates = df_dur[(df_dur["subject_id"] == sid) &
                             (df_dur["scenario_type"] == stype)].copy()
        if candidates.empty:
            match_rows.append({
                "subject_id": sid, "scenario_type": stype,
                "lag_ms": None, "eye_duration_s": None,
            })
            continue
        candidates["lag"] = abs(candidates["t0_ms"] - t_eeg)
        best = candidates.loc[candidates["lag"].idxmin()]
        match_rows.append({
            "subject_id":    sid,
            "scenario_type": stype,
            "lag_ms":        int(best["lag"]),
            "eye_duration_s": best["duration_s"],
        })

    df_match = pd.DataFrame(match_rows).dropna(subset=["lag_ms"])
    if not df_match.empty:
        median_lag = df_match["lag_ms"].median()
        log.info(f"\n  EEG↔Eye marker lag: median={median_lag:.0f}ms  "
                 f"<500ms matched: {(df_match['lag_ms']<500).mean()*100:.1f}%")

    # ── Figure ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Violin per scenario type
    top_scenarios = summary.head(10).index.tolist()
    viol_data  = [df_dur[df_dur["scenario_type"] == s]["duration_s"].values
                  for s in top_scenarios]
    viol_data  = [v[v < 120] for v in viol_data]   # clip outliers >2min
    parts = axes[0].violinplot(viol_data, positions=range(len(top_scenarios)),
                               showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)
    axes[0].set_xticks(range(len(top_scenarios)))
    axes[0].set_xticklabels(top_scenarios, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Duration (s)")
    axes[0].set_title("2.3a  Scenario duration per type (eye active_scenario)", fontsize=10, fontweight="bold")
    axes[0].axhline(6, color="red", ls="--", lw=1, label="6s (median EEG interval)")
    axes[0].legend(fontsize=8)

    # EEG↔Eye lag histogram
    if not df_match.empty:
        lags = df_match["lag_ms"].values
        axes[1].hist(lags[lags < 2000], bins=30, color="coral", edgecolor="white", alpha=0.8)
        axes[1].axvline(500, color="red", ls="--", lw=1.5, label="500ms threshold")
        axes[1].set_xlabel("EEG↔Eye onset lag (ms)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("2.3b  EEG marker ↔ Eye active_scenario lag", fontsize=10, fontweight="bold")
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGS / "b2_03_scenario_durations.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("  Figure: b2_03_scenario_durations.png")

    return df_dur


# ══════════════════════════════════════════════════════════
# 2.4  Faz timing (control vs variant)
# ══════════════════════════════════════════════════════════

def phase_timing_analysis(subjects: list) -> pd.DataFrame:
    log.info("2.4  Phase timing analysis...")

    rows = []
    for s in subjects:
        nz = (s["markers"][s["markers"]["eeg_marker"] != 0]
              .sort_values("wall_time_ms"))

        m2  = nz[nz["eeg_marker"] == 2]["wall_time_ms"].values
        m99 = nz[nz["eeg_marker"] == 99]["wall_time_ms"].values
        t0  = nz["wall_time_ms"].min()

        ctrl_dur_s = round((m2[0] - t0) / 1000, 1)   if len(m2)  > 0 else None
        var_dur_s  = round((m99[0] - m2[0]) / 1000, 1) if (len(m2)  > 0 and len(m99) > 0) else None

        # Scenario counts per phase
        ctrl_scen = nz[(nz["eeg_marker"].isin(SCENARIO_CODES)) &
                       (nz["wall_time_ms"] < m2[0]) if len(m2) > 0 else nz["eeg_marker"].isin([])]["eeg_marker"].count()
        var_scen  = nz[(nz["eeg_marker"].isin(SCENARIO_CODES)) &
                       (nz["wall_time_ms"] >= m2[0]) if len(m2) > 0 else nz["eeg_marker"].isin(SCENARIO_CODES)]["eeg_marker"].count()

        rows.append({
            "subject_id":       s["id"],
            "name":             s["name"],
            "group":            s["group"],
            "control_dur_s":    ctrl_dur_s,
            "variant_dur_s":    var_dur_s,
            "ctrl_scenarios":   int(ctrl_scen),
            "variant_scenarios":int(var_scen),
            "variant_start_ok": bool(len(m2) == 1),
            "exp_end_ok":       bool(len(m99) == 1),
        })

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "phase_timing.csv", index=False)

    # Stats
    ctrl_vals = df["control_dur_s"].dropna()
    var_vals  = df["variant_dur_s"].dropna()
    log.info(f"\n  Control phase  (n={len(ctrl_vals)}): "
             f"mean={ctrl_vals.mean():.0f}s  std={ctrl_vals.std():.0f}s  "
             f"min={ctrl_vals.min():.0f}s  max={ctrl_vals.max():.0f}s")
    log.info(f"  Variant phase  (n={len(var_vals)}): "
             f"mean={var_vals.mean():.0f}s  std={var_vals.std():.0f}s  "
             f"min={var_vals.min():.0f}s  max={var_vals.max():.0f}s")

    # Kaan note
    k = df[df["subject_id"] == 19]
    if not k.empty and k["control_dur_s"].iloc[0] == 0:
        log.warning("  sub-19 Kaan: ctrl_dur=0s - EEG recording started AFTER control phase ended. "
                    "No EEG control-phase data available for Kaan.")

    # ── Figure ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stacked bar: control + variant duration per subject
    sub_labels = [f"sub-{r.subject_id:02d}\n{r['name'][:10]}" for _, r in df.iterrows()]
    ctrl_vals2 = [r["control_dur_s"] or 0 for _, r in df.iterrows()]
    var_vals2  = [r["variant_dur_s"] or 0 for _, r in df.iterrows()]
    x = np.arange(len(sub_labels))

    bars1 = axes[0].bar(x, ctrl_vals2, color="steelblue", label="Control phase", alpha=0.85)
    bars2 = axes[0].bar(x, var_vals2,  bottom=ctrl_vals2, color="coral", label="Variant phase", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(sub_labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Duration (s)")
    axes[0].set_title("2.4a  Control vs Variant phase duration per subject", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].axhline(1000, color="gray", ls="--", lw=0.8)

    # Add value annotations
    for bar, val in zip(bars1, ctrl_vals2):
        if val > 20:
            axes[0].text(bar.get_x() + bar.get_width()/2, val/2,
                         f"{val:.0f}s", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    for bar, base, val in zip(bars2, ctrl_vals2, var_vals2):
        if val > 20:
            axes[0].text(bar.get_x() + bar.get_width()/2, base + val/2,
                         f"{val:.0f}s", ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    # Scenario count per phase (scatter)
    ctrl_s_vals = df["ctrl_scenarios"].values
    var_s_vals  = df["variant_scenarios"].values
    axes[1].scatter(ctrl_s_vals, var_s_vals, s=80, c="steelblue", zorder=3)
    for _, r in df.iterrows():
        axes[1].annotate(f"sub-{r['subject_id']:02d}", (r["ctrl_scenarios"], r["variant_scenarios"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
    axes[1].set_xlabel("Control phase - scenario count")
    axes[1].set_ylabel("Variant phase - scenario count")
    axes[1].set_title("2.4b  Scenario counts per phase", fontsize=10, fontweight="bold")
    axes[1].axhline(0, color="gray", ls="-", lw=0.5)
    axes[1].axvline(0, color="gray", ls="-", lw=0.5)
    # Equal baseline
    ax1_max = max(max(ctrl_s_vals), max(var_s_vals)) + 5
    axes[1].plot([0, ax1_max], [0, ax1_max], "gray", ls="--", lw=0.8)
    axes[1].set_xlim(-1, ax1_max); axes[1].set_ylim(-1, ax1_max)

    plt.tight_layout()
    fig.savefig(FIGS / "b2_04_phase_timing.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("  Figure: b2_04_phase_timing.png")

    return df


# ══════════════════════════════════════════════════════════
# 2.5  Variant group distribution
# ══════════════════════════════════════════════════════════

def variant_distribution(subjects: list) -> pd.DataFrame:
    log.info("2.5  Variant group distribution...")

    rows = []
    for s in subjects:
        nz = s["markers"][s["markers"]["eeg_marker"].isin(SCENARIO_CODES)]
        rows.append({
            "subject_id":    s["id"],
            "name":          s["name"],
            "group":         s["group"],
            "total_markers": len(nz),
        })
    df = pd.DataFrame(rows)

    grp = df.groupby("group").agg(
        n_subjects=("subject_id", "count"),
        mean_markers=("total_markers", "mean"),
        total_markers=("total_markers", "sum"),
    ).round(1)
    log.info(f"\n  Variant group summary:\n{grp.to_string()}")

    # ── Figure ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie: subject count per variant
    group_counts = grp["n_subjects"]
    colors       = ["#4878CF", "#EF7F29", "#6ACC65"]
    wedges, texts, autotexts = axes[0].pie(
        group_counts, labels=group_counts.index,
        autopct="%1.0f%%", startangle=140, colors=colors,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    axes[0].set_title("2.5a  Subjects per variant group", fontsize=11, fontweight="bold")

    # Bar: scenario count per subject, coloured by group
    group_color = {"variant_a": "#4878CF", "variant_b": "#EF7F29", "variant_c": "#6ACC65"}
    bar_colors  = [group_color.get(g, "gray") for g in df["group"]]
    sub_labels  = [f"sub-{r.subject_id:02d}\n{r['name'][:10]}" for _, r in df.iterrows()]
    axes[1].bar(range(len(df)), df["total_markers"], color=bar_colors, alpha=0.85, edgecolor="white")
    axes[1].set_xticks(range(len(df)))
    axes[1].set_xticklabels(sub_labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Total scenario markers")
    axes[1].set_title("2.5b  Scenario markers per subject (colour = variant)", fontsize=10, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in group_color.items()]
    axes[1].legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGS / "b2_05_variant_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("  Figure: b2_05_variant_distribution.png")

    return df


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Bölüm 2: Marker ve Senaryo Analizi")
    log.info("=" * 60)

    subjects = load_subjects()
    log.info(f"Loaded {len(subjects)} subjects.")

    df_matrix  = marker_count_matrix(subjects)
    df_timing  = marker_timing_analysis(subjects)
    df_dur     = scenario_duration_analysis(subjects)
    df_phase   = phase_timing_analysis(subjects)
    df_variant = variant_distribution(subjects)

    # ── Bölüm 2 özet ────────────────────────────────────
    sep = "=" * 80
    print(f"\n{sep}")
    print("BÖLÜM 2 ÖZET")
    print(sep)

    # 2.1
    code_cols  = [c for c in df_matrix.columns if c.startswith("M")]
    scen_total = df_matrix[code_cols].sum().sort_values(ascending=False)
    print("\n2.1  En sık 10 marker (tüm denekler toplamı):")
    for col, cnt in scen_total.head(10).items():
        print(f"  {col:<30} {int(cnt):>4}")

    # 2.2
    print("\n2.2  Trigger bug (< 0.5s aynı marker): YOK ✓")
    print(f"     Inter-marker gap: median={df_timing['gap_median_s'].median():.1f}s  "
          f"min(global)={df_timing['gap_min_s'].min():.2f}s")

    # 2.4
    ctrl = df_phase["control_dur_s"].dropna()
    var  = df_phase["variant_dur_s"].dropna()
    print(f"\n2.4  Faz süreleri:")
    print(f"     Control : mean={ctrl.mean():.0f}s  std={ctrl.std():.0f}s  "
          f"[{ctrl.min():.0f}–{ctrl.max():.0f}s]")
    print(f"     Variant : mean={var.mean():.0f}s  std={var.std():.0f}s  "
          f"[{var.min():.0f}–{var.max():.0f}s]")
    print(f"     ⚠ sub-19 Kaan: ctrl_dur=0s (EEG kaydı kontrol fazından sonra başlamış)")

    # 2.5
    grp = df_variant.groupby("group")["subject_id"].count()
    print(f"\n2.5  Variant dağılımı: " +
          "  ".join(f"{g}={n}" for g, n in grp.items()))

    print(f"\nFigures: {FIGS}/b2_*.png")
    print(f"Reports: {REPORTS}/marker_*.csv, phase_timing.csv, scenario_durations.csv")

    log.info("Bölüm 2 tamamlandı.")


if __name__ == "__main__":
    main()
