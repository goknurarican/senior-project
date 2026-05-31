"""
summarize_connectivity_results.py
=================================

Approach B Stage 6: Robustness summary for EEG connectivity results.

This script strengthens Approach B by summarizing the statistical reliability of
the scenario-wise EEG connectivity analysis from Stage 2.

It does not retrain the GNN. The 15-class GNN performed below chance, so the
main value of Approach B is the scenario-wise connectivity analysis.

Inputs:
  approach_b/02_baseline_comparison/analysis/all_connections_full.csv
  approach_b/02_baseline_comparison/analysis/all_scenarios_network_summary.csv
  optional: approach_b/03_gnn_classification/evaluation/loso_summary.json
  optional: approach_b/03_gnn_classification/evaluation/permutation_results.json
  optional: approach_b/05_behavioral_comparison/outputs/behavioral_top_feature_per_scenario.csv

Outputs:
  approach_b/06_robustness_summary/outputs/significant_connections_clean.csv
  approach_b/06_robustness_summary/outputs/exploratory_trends.csv
  approach_b/06_robustness_summary/outputs/top_effects_by_scenario.csv
  approach_b/06_robustness_summary/outputs/metric_band_summary.csv
  approach_b/06_robustness_summary/outputs/scenario_reliability_flags.csv
  approach_b/06_robustness_summary/outputs/report_ready_findings.csv
  approach_b/06_robustness_summary/reports/approach_b_robustness_report.txt
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


def as_bool(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def classify_network(row):
    band = str(row.get("band", "")).lower()
    a = str(row.get("roi_a", "")).lower()
    b = str(row.get("roi_b", "")).lower()
    pair = {a, b}

    if band in {"theta", "beta"} and (
        pair == {"frontal", "parietal"} or
        pair == {"frontal_central", "parietal"}
    ):
        return "fronto-parietal_control"

    if band == "alpha" and (
        pair == {"frontal", "parietal"} or
        pair == {"frontal", "occipital"}
    ):
        return "default-mode_attention_reorientation"

    if band in {"beta", "gamma"} and (
        pair == {"central", "frontal_central"} or
        pair == {"central", "parietal"}
    ):
        return "sensorimotor"

    if "occipital" in pair:
        return "visual_occipital"

    if "temporal" in pair and "parietal" in pair:
        return "parietal_temporal"

    return "other"


def reliability_flag(n_subjects):
    if pd.isna(n_subjects):
        return "unknown"

    n_subjects = int(n_subjects)

    if n_subjects < 5:
        return "very_low_n"
    if n_subjects < 7:
        return "limited_n"
    return "usable_n"


def main():
    b_dir = Path(__file__).resolve().parents[2]

    analysis_dir = b_dir / "02_baseline_comparison" / "analysis"
    gnn_eval_dir = b_dir / "03_gnn_classification" / "evaluation"
    behavioral_dir = b_dir / "05_behavioral_comparison" / "outputs"

    out_dir = b_dir / "06_robustness_summary" / "outputs"
    report_dir = b_dir / "06_robustness_summary" / "reports"

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    all_connections_path = analysis_dir / "all_connections_full.csv"
    scenario_summary_path = analysis_dir / "all_scenarios_network_summary.csv"

    if not all_connections_path.exists():
        raise FileNotFoundError(f"Missing input: {all_connections_path}")

    df = pd.read_csv(all_connections_path)

    required = {
        "metric",
        "scenario",
        "band",
        "roi_a",
        "roi_b",
        "delta_mean",
        "cohens_d",
        "t_p_raw",
        "t_p_fdr",
        "wilcoxon_p_raw",
        "wilcoxon_p_fdr",
        "perm_p_raw",
        "direction",
        "significant",
        "n_subjects",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"all_connections_full.csv missing columns: {missing}")

    for col in [
        "delta_mean",
        "cohens_d",
        "t_p_raw",
        "t_p_fdr",
        "wilcoxon_p_raw",
        "wilcoxon_p_fdr",
        "perm_p_raw",
    ]:
        df[col] = df[col].apply(safe_float)

    df["significant"] = df["significant"].apply(as_bool)
    df["connection"] = df["roi_a"].astype(str) + "-" + df["roi_b"].astype(str)
    df["abs_d"] = df["cohens_d"].abs()
    df["network_category"] = df.apply(classify_network, axis=1)
    df["reliability_flag"] = df["n_subjects"].apply(reliability_flag)

    # Confirmatory FDR-significant effects.
    significant = df[df["significant"]].copy()
    significant = significant.sort_values(
        ["t_p_fdr", "abs_d"],
        ascending=[True, False],
    )

    # Exploratory trends:
    # Not FDR-significant, but raw p < .05, medium/large effect size, and usable N.
    trends = df[
        (~df["significant"])
        & (df["t_p_raw"] < 0.05)
        & (df["abs_d"] >= 0.8)
        & (df["n_subjects"] >= 5)
    ].copy()

    trends["permutation_support"] = trends["perm_p_raw"] < 0.05
    trends = trends.sort_values(
        ["permutation_support", "abs_d", "t_p_raw"],
        ascending=[False, False, True],
    )

    # Top effect per scenario:
    # Prefer significant effects; otherwise strongest exploratory trend; otherwise strongest absolute effect.
    top_rows = []

    for scenario, g in df.groupby("scenario"):
        g = g.copy()

        sig = g[g["significant"]].copy()
        if len(sig):
            chosen = sig.sort_values(
                ["t_p_fdr", "abs_d"],
                ascending=[True, False],
            ).iloc[0]
            status = "confirmatory_fdr_significant"
        else:
            tr = g[
                (g["t_p_raw"] < 0.05)
                & (g["abs_d"] >= 0.8)
                & (g["n_subjects"] >= 5)
            ].copy()

            if len(tr):
                chosen = tr.sort_values(
                    ["abs_d", "t_p_raw"],
                    ascending=[False, True],
                ).iloc[0]
                status = "exploratory_trend"
            else:
                valid = g[g["cohens_d"].notna()].copy()
                if len(valid) == 0:
                    continue

                chosen = valid.sort_values(
                    "abs_d",
                    ascending=False,
                ).iloc[0]
                status = "weak_or_no_effect"

        row = chosen.to_dict()
        row["interpretation_status"] = status
        top_rows.append(row)

    top_effects = pd.DataFrame(top_rows)
    if len(top_effects):
        top_effects = top_effects.sort_values(
            ["interpretation_status", "abs_d"],
            ascending=[True, False],
        )

    # Metric-band summary.
    metric_band_summary = (
        df.groupby(["metric", "band"])
        .agg(
            n_tests=("scenario", "count"),
            n_significant=("significant", "sum"),
            n_raw_p_lt_05=("t_p_raw", lambda x: int((x < 0.05).sum())),
            median_abs_d=("abs_d", "median"),
            max_abs_d=("abs_d", "max"),
            min_fdr_p=("t_p_fdr", "min"),
            median_n_subjects=("n_subjects", "median"),
        )
        .reset_index()
        .sort_values(
            ["n_significant", "n_raw_p_lt_05", "max_abs_d"],
            ascending=[False, False, False],
        )
    )

    # Scenario-level reliability.
    scenario_reliability = (
        df.groupby("scenario")
        .agg(
            max_n_subjects=("n_subjects", "max"),
            min_n_subjects=("n_subjects", "min"),
            n_tests=("scenario", "count"),
            n_significant=("significant", "sum"),
            n_raw_p_lt_05=("t_p_raw", lambda x: int((x < 0.05).sum())),
            max_abs_d=("abs_d", "max"),
            min_fdr_p=("t_p_fdr", "min"),
        )
        .reset_index()
    )

    scenario_reliability["reliability_flag"] = scenario_reliability[
        "max_n_subjects"
    ].apply(reliability_flag)

    scenario_reliability = scenario_reliability.sort_values(
        ["n_significant", "max_abs_d"],
        ascending=[False, False],
    )

    # Report-ready findings: significant + strongest trends.
    report_ready = pd.concat(
        [
            significant.assign(interpretation_status="confirmatory_fdr_significant"),
            trends.assign(interpretation_status="exploratory_trend"),
        ],
        ignore_index=True,
    )

    if len(report_ready):
        sort_cols = ["interpretation_status"]

        if "t_p_fdr" in report_ready.columns:
            sort_cols.append("t_p_fdr")

        if "abs_d" in report_ready.columns:
            sort_cols.append("abs_d")

        ascending = [True] * len(sort_cols)

        if "abs_d" in sort_cols:
            ascending[sort_cols.index("abs_d")] = False

        report_ready = report_ready.sort_values(
            sort_cols,
            ascending=ascending,
        )

        report_ready = report_ready[
            [
                "interpretation_status",
                "scenario",
                "metric",
                "band",
                "connection",
                "network_category",
                "direction",
                "delta_mean",
                "cohens_d",
                "t_p_raw",
                "t_p_fdr",
                "wilcoxon_p_raw",
                "wilcoxon_p_fdr",
                "perm_p_raw",
                "n_subjects",
                "reliability_flag",
            ]
        ]

    # Optional GNN summary.
    gnn_summary = None
    gnn_summary_path = gnn_eval_dir / "loso_summary.json"

    if gnn_summary_path.exists():
        with open(gnn_summary_path, "r", encoding="utf-8") as f:
            gnn_summary = json.load(f)

    gnn_perm = None
    gnn_perm_path = gnn_eval_dir / "permutation_results.json"

    if gnn_perm_path.exists():
        with open(gnn_perm_path, "r", encoding="utf-8") as f:
            gnn_perm = json.load(f)

    # Optional behavioral summary.
    behavioral_top = None
    behavioral_top_path = behavioral_dir / "behavioral_top_feature_per_scenario.csv"

    if behavioral_top_path.exists():
        behavioral_top = pd.read_csv(behavioral_top_path)

    # Optional scenario summary from Stage 2.
    scenario_summary = None
    if scenario_summary_path.exists():
        scenario_summary = pd.read_csv(scenario_summary_path)

    # Save outputs.
    significant.to_csv(out_dir / "significant_connections_clean.csv", index=False)
    trends.to_csv(out_dir / "exploratory_trends.csv", index=False)
    top_effects.to_csv(out_dir / "top_effects_by_scenario.csv", index=False)
    metric_band_summary.to_csv(out_dir / "metric_band_summary.csv", index=False)
    scenario_reliability.to_csv(out_dir / "scenario_reliability_flags.csv", index=False)
    report_ready.to_csv(out_dir / "report_ready_findings.csv", index=False)

    if scenario_summary is not None:
        scenario_summary.to_csv(out_dir / "stage2_network_summary_copy.csv", index=False)

    # Text report.
    report_path = report_dir / "approach_b_robustness_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("APPROACH B ROBUSTNESS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. Input\n")
        f.write("-" * 80 + "\n")
        f.write(f"Input file: {all_connections_path}\n")
        f.write(f"Total connection-level tests: {len(df)}\n")
        f.write(f"Metrics: {sorted(df['metric'].dropna().unique().tolist())}\n")
        f.write(f"Bands: {sorted(df['band'].dropna().unique().tolist())}\n")
        f.write(f"Scenarios: {sorted(df['scenario'].dropna().unique().tolist())}\n\n")

        f.write("2. Confirmatory FDR-significant connectivity effects\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of FDR-significant effects: {len(significant)}\n\n")

        if len(significant):
            cols = [
                "scenario",
                "metric",
                "band",
                "connection",
                "network_category",
                "direction",
                "cohens_d",
                "t_p_fdr",
                "wilcoxon_p_raw",
                "wilcoxon_p_fdr",
                "perm_p_raw",
                "n_subjects",
                "reliability_flag",
            ]
            f.write(significant[cols].to_string(index=False))
            f.write("\n\n")
        else:
            f.write("No FDR-significant connectivity effects were found.\n\n")

        f.write("3. Exploratory trends\n")
        f.write("-" * 80 + "\n")
        f.write(
            "Definition: not FDR-significant, but raw p < 0.05, |Cohen's d| >= 0.8, "
            "and n_subjects >= 5.\n"
        )
        f.write(f"Number of exploratory trends: {len(trends)}\n\n")

        if len(trends):
            cols = [
                "scenario",
                "metric",
                "band",
                "connection",
                "network_category",
                "direction",
                "cohens_d",
                "t_p_raw",
                "t_p_fdr",
                "wilcoxon_p_raw",
                "wilcoxon_p_fdr",
                "perm_p_raw",
                "permutation_support",
                "n_subjects",
                "reliability_flag",
            ]
            f.write(trends[cols].head(40).to_string(index=False))
            f.write("\n\n")
        else:
            f.write("No exploratory trends matched the criteria.\n\n")

        f.write("4. Metric-band summary\n")
        f.write("-" * 80 + "\n")
        f.write(metric_band_summary.to_string(index=False))
        f.write("\n\n")

        f.write("5. Scenario reliability flags\n")
        f.write("-" * 80 + "\n")
        f.write(scenario_reliability.to_string(index=False))
        f.write("\n\n")

        if gnn_summary is not None:
            f.write("6. GNN validation summary\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean LOSO accuracy: {gnn_summary.get('mean_accuracy')}\n")
            f.write(f"Chance level: {gnn_summary.get('chance')}\n")
            f.write(f"Mean macro F1: {gnn_summary.get('mean_f1_macro')}\n")

            if gnn_perm is not None:
                f.write(f"Permutation p-value: {gnn_perm.get('p_value')}\n")

            f.write(
                "Interpretation: the 15-class GNN did not generalize under LOSO and "
                "should be treated as an auxiliary negative validation rather than the "
                "main result of Approach B.\n\n"
            )

        if behavioral_top is not None:
            f.write("7. Behavioral extension summary\n")
            f.write("-" * 80 + "\n")
            f.write(
                "The eye/mouse behavioral scenario comparison is exploratory and compares "
                "each scenario against the same subject's other scenario epochs. It is not "
                "a clean control-vs-scenario comparison.\n"
            )

            if "significant" in behavioral_top.columns:
                n_beh_sig = int(behavioral_top["significant"].apply(as_bool).sum())
                f.write(f"Behavioral top-feature rows marked significant: {n_beh_sig}\n")

            f.write("\n")

        f.write("8. Recommended interpretation\n")
        f.write("-" * 80 + "\n")
        f.write(
            "Approach B is strongest as a scenario-wise EEG functional connectivity "
            "analysis rather than as a classifier. A small number of scenario-specific "
            "connectivity effects survive FDR correction and show large effect sizes, "
            "but these findings should be interpreted cautiously due to N=9 and sparse "
            "scenario counts. The GNN classifier does not exceed chance-level performance, "
            "so it should not be presented as the main contribution. Eye/mouse behavioral "
            "features provide an exploratory multimodal extension, but no strong confirmatory "
            "behavioral effect should be claimed unless it survives correction.\n"
        )

    print("=" * 90)
    print("APPROACH B ROBUSTNESS SUMMARY DONE")
    print("=" * 90)
    print(f"Total connection-level tests : {len(df)}")
    print(f"FDR-significant effects      : {len(significant)}")
    print(f"Exploratory trends           : {len(trends)}")
    print(f"Saved outputs to             : {out_dir}")
    print(f"Saved report to              : {report_path}")

    if len(significant):
        print("\nFDR-significant effects:")
        cols = [
            "scenario",
            "metric",
            "band",
            "connection",
            "direction",
            "cohens_d",
            "t_p_fdr",
            "n_subjects",
        ]
        print(significant[cols].to_string(index=False))

    print("\nTop effects by scenario:")
    if len(top_effects):
        cols = [
            "scenario",
            "interpretation_status",
            "metric",
            "band",
            "connection",
            "direction",
            "cohens_d",
            "t_p_fdr",
            "n_subjects",
        ]
        print(top_effects[cols].to_string(index=False))
    else:
        print("No valid scenario-level effects found.")


if __name__ == "__main__":
    main()