from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


def cohen_d_onesample(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    if sd < 1e-12:
        return np.nan
    return float(x.mean() / sd)


def safe_ttest_against_zero(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    if x.std(ddof=1) < 1e-12:
        return np.nan
    return float(ttest_1samp(x, popmean=0.0).pvalue)


def safe_wilcoxon_against_zero(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    if np.allclose(x, 0):
        return np.nan
    try:
        return float(wilcoxon(x).pvalue)
    except Exception:
        return np.nan


def feature_modality(feature):
    if feature.startswith("eye_"):
        return "eye"
    if feature.startswith("mouse_"):
        return "mouse"
    return "other"


def main():
    # file path layout:
    # data-ml/approach_b/05_behavioral_comparison/src/behavioral_scenario_analysis.py
    b_dir = Path(__file__).resolve().parents[2]       # approach_b
    data_ml_dir = b_dir.parent                        # data-ml
    repo_root = data_ml_dir.parent

    rf_feature_path = data_ml_dir / "approach_rf_baseline" / "outputs" / "feature_table_clean.csv"
    eeg_summary_path = b_dir / "02_baseline_comparison" / "analysis" / "all_scenarios_network_summary.csv"

    out_dir = b_dir / "05_behavioral_comparison" / "outputs"
    rep_dir = b_dir / "05_behavioral_comparison" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    if not rf_feature_path.exists():
        raise FileNotFoundError(
            f"Missing RF feature table: {rf_feature_path}\n"
            "Run data-ml/approach_rf_baseline/src/01_build_feature_table.py first."
        )

    df = pd.read_csv(rf_feature_path)

    required_cols = {"subject_id", "scenario_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"feature_table_clean.csv missing required columns: {missing}")

    eye_features = [
        "eye_valid_ratio",
        "eye_mean",
        "eye_std",
        "eye_range",
        "eye_entropy",
    ]

    mouse_features = [
        "mouse_velocity_mean",
        "mouse_velocity_max",
        "mouse_velocity_std",
        "mouse_acc_abs_mean",
        "mouse_acc_abs_max",
        "mouse_click_count",
        "mouse_click_rate",
        "mouse_idle_ratio",
        "mouse_path_proxy",
    ]

    features = [c for c in eye_features + mouse_features if c in df.columns]

    if not features:
        raise ValueError("No eye/mouse features found in feature_table_clean.csv")

    scenarios = sorted(df["scenario_name"].dropna().unique().tolist())
    subjects = sorted(df["subject_id"].dropna().unique().tolist())

    print("=" * 90)
    print("APPROACH B STAGE 5: BEHAVIORAL SCENARIO ANALYSIS")
    print("=" * 90)
    print(f"Loaded feature table: {rf_feature_path}")
    print(f"Rows: {len(df)}")
    print(f"Subjects: {subjects}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Features: {len(features)}")

    rows = []

    # Scenario-vs-other-scenarios, computed within subject.
    # This is not a control comparison. It asks:
    # "For this subject, is feature X higher/lower during this scenario than during the subject's other scenarios?"
    for scenario in scenarios:
        for feat in features:
            deltas = []
            n_epochs_scenario_total = 0
            n_subjects_used = 0

            for sid in subjects:
                s_df = df[df["subject_id"] == sid]
                scen_vals = s_df[s_df["scenario_name"] == scenario][feat].dropna().values
                other_vals = s_df[s_df["scenario_name"] != scenario][feat].dropna().values

                n_epochs_scenario_total += len(scen_vals)

                if len(scen_vals) < 1 or len(other_vals) < 3:
                    continue

                delta = float(np.nanmean(scen_vals) - np.nanmean(other_vals))
                if np.isfinite(delta):
                    deltas.append(delta)
                    n_subjects_used += 1

            deltas = np.asarray(deltas, dtype=float)
            delta_mean = float(np.nanmean(deltas)) if len(deltas) else np.nan
            delta_median = float(np.nanmedian(deltas)) if len(deltas) else np.nan
            delta_std = float(np.nanstd(deltas, ddof=1)) if len(deltas) > 1 else np.nan

            p_t = safe_ttest_against_zero(deltas)
            p_w = safe_wilcoxon_against_zero(deltas)
            d = cohen_d_onesample(deltas)

            direction = "increase" if np.isfinite(delta_mean) and delta_mean > 0 else "decrease"

            rows.append({
                "scenario": scenario,
                "feature": feat,
                "modality": feature_modality(feat),
                "n_subjects": int(n_subjects_used),
                "n_epochs_scenario": int(n_epochs_scenario_total),
                "delta_mean_vs_other_scenarios": delta_mean,
                "delta_median_vs_other_scenarios": delta_median,
                "delta_std": delta_std,
                "cohens_d": d,
                "t_p_raw": p_t,
                "wilcoxon_p_raw": p_w,
                "direction": direction,
            })

    stats = pd.DataFrame(rows)

    # FDR correction over valid t-test p-values.
    stats["t_p_fdr"] = np.nan
    valid = stats["t_p_raw"].notna()
    if valid.sum() > 0:
        _, p_corr, _, _ = multipletests(stats.loc[valid, "t_p_raw"], method="fdr_bh")
        stats.loc[valid, "t_p_fdr"] = p_corr

    stats["significant"] = (
        (stats["t_p_fdr"] < 0.05) &
        (stats["cohens_d"].abs() >= 0.5) &
        (stats["n_subjects"] >= 5)
    )

    stats_path = out_dir / "behavioral_all_feature_stats.csv"
    stats.to_csv(stats_path, index=False)

    # Top behavioral feature per scenario:
    # prefer significant features; if none, take strongest absolute effect size.
    top_rows = []
    for scenario, g in stats.groupby("scenario"):
        g = g.copy()
        g_valid = g[g["cohens_d"].notna()].copy()

        if len(g_valid) == 0:
            continue

        sig = g_valid[g_valid["significant"] == True]
        if len(sig):
            chosen = sig.sort_values(["t_p_fdr", "cohens_d"], ascending=[True, False]).iloc[0]
        else:
            chosen = g_valid.assign(abs_d=g_valid["cohens_d"].abs()).sort_values(
                ["abs_d", "t_p_raw"], ascending=[False, True]
            ).iloc[0]

        top_rows.append(chosen.drop(labels=["abs_d"], errors="ignore"))

    top_behavior = pd.DataFrame(top_rows)
    top_behavior_path = out_dir / "behavioral_top_feature_per_scenario.csv"
    top_behavior.to_csv(top_behavior_path, index=False)

    # Optional EEG + behavioral combined summary.
    combined_path = out_dir / "eeg_behavior_combined_summary.csv"
    if eeg_summary_path.exists():
        eeg = pd.read_csv(eeg_summary_path)

        # Keep rows that have a reported top EEG connection.
        if "top_connection" in eeg.columns:
            eeg_nonempty = eeg[eeg["top_connection"].notna()].copy()
            eeg_nonempty = eeg_nonempty[eeg_nonempty["top_connection"].astype(str).str.strip() != ""]
        else:
            eeg_nonempty = eeg.copy()

        # Choose best EEG finding per scenario by FDR p when possible.
        if len(eeg_nonempty):
            if "fdr_p" in eeg_nonempty.columns:
                eeg_best = (
                    eeg_nonempty.sort_values(["scenario", "fdr_p"], ascending=[True, True])
                    .groupby("scenario")
                    .head(1)
                    .copy()
                )
            else:
                eeg_best = eeg_nonempty.groupby("scenario").head(1).copy()

            eeg_best = eeg_best.add_prefix("eeg_")
            top_behavior_pref = top_behavior.add_prefix("behavior_")

            combined = eeg_best.merge(
                top_behavior_pref,
                left_on="eeg_scenario",
                right_on="behavior_scenario",
                how="outer",
            )
            combined.to_csv(combined_path, index=False)
        else:
            pd.DataFrame().to_csv(combined_path, index=False)
    else:
        print(f"[WARN] EEG summary not found: {eeg_summary_path}")
        pd.DataFrame().to_csv(combined_path, index=False)

    report_path = rep_dir / "behavioral_scenario_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("APPROACH B STAGE 5: BEHAVIORAL SCENARIO ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input feature table: {rf_feature_path}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Subjects: {subjects}\n")
        f.write(f"Scenarios: {len(scenarios)}\n")
        f.write(f"Features tested: {len(features)}\n\n")
        f.write("Important interpretation note:\n")
        f.write(
            "This analysis compares each scenario against the same subject's other scenario epochs. "
            "It is not a clean control-vs-scenario comparison, because the available RF feature table "
            "contains scenario epochs rather than action-matched behavioral control epochs.\n\n"
        )
        f.write("Top behavioral feature per scenario:\n")
        if len(top_behavior):
            f.write(top_behavior[
                [
                    "scenario",
                    "feature",
                    "modality",
                    "n_subjects",
                    "n_epochs_scenario",
                    "delta_mean_vs_other_scenarios",
                    "cohens_d",
                    "t_p_raw",
                    "t_p_fdr",
                    "direction",
                    "significant",
                ]
            ].to_string(index=False))
        else:
            f.write("No valid behavioral features.\n")

    print("\nDONE")
    print(f"Saved all stats      : {stats_path}")
    print(f"Saved top behavior   : {top_behavior_path}")
    print(f"Saved combined       : {combined_path}")
    print(f"Saved report         : {report_path}")

    print("\nTop behavioral feature per scenario:")
    if len(top_behavior):
        cols = [
            "scenario",
            "feature",
            "modality",
            "n_subjects",
            "n_epochs_scenario",
            "delta_mean_vs_other_scenarios",
            "cohens_d",
            "t_p_raw",
            "t_p_fdr",
            "direction",
            "significant",
        ]
        print(top_behavior[cols].to_string(index=False))
    else:
        print("No valid top behavior rows.")


if __name__ == "__main__":
    main()