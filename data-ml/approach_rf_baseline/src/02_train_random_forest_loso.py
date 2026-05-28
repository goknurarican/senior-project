from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


warnings.filterwarnings("ignore")


META_COLS = {
    "global_idx",
    "subject_id",
    "local_epoch_idx",
    "eeg_marker",
    "scenario_name",
    "is_scenario_marker",
    "alignment_row",
    "wall_time_ms",
    "phase",
    "eye_epoch_idx",
    "mouse_epoch_idx",
    "alignment_status",
    "label_rage_click",
}


MODALITY_PREFIXES = {
    "eeg": ["eeg_"],
    "eye": ["eye_"],
    "mouse": ["mouse_"],
}


ABLATIONS = {
    "eeg_only": ["eeg"],
    "eye_only": ["eye"],
    "mouse_only": ["mouse"],
    "eeg_eye": ["eeg", "eye"],
    "eeg_mouse": ["eeg", "mouse"],
    "eye_mouse": ["eye", "mouse"],
    "all_modalities": ["eeg", "eye", "mouse"],
}


def load_config():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_output_dir(cfg):
    out = Path(cfg["output_dir"])
    if out.is_absolute():
        return out

    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / out


def get_feature_cols(df, modalities):
    prefixes = []
    for modality in modalities:
        prefixes.extend(MODALITY_PREFIXES[modality])

    cols = []
    for col in df.columns:
        if col in META_COLS:
            continue
        if any(col.startswith(prefix) for prefix in prefixes):
            cols.append(col)

    return cols


def safe_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {}

    # Ranking metrics
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan

    # Thresholded metrics
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["f1_positive"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision_positive"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall_positive"] = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["tn"] = int(tn)
    out["fp"] = int(fp)
    out["fn"] = int(fn)
    out["tp"] = int(tp)

    out["n"] = int(len(y_true))
    out["positives"] = int(y_true.sum())
    out["positive_rate"] = float(y_true.mean())

    return out


def build_model(cfg):
    rf_cfg = cfg.get("random_forest", {})

    clf = RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 500),
        max_depth=rf_cfg.get("max_depth", None),
        min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
        class_weight=rf_cfg.get("class_weight", "balanced_subsample"),
        random_state=rf_cfg.get("random_state", 42),
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", clf),
        ]
    )

    return model


def run_loso(df, feature_cols, cfg, ablation_name):
    subjects = sorted(df["subject_id"].unique())

    all_pred_rows = []
    fold_rows = []

    for test_sid in subjects:
        train_df = df[df["subject_id"] != test_sid].copy()
        test_df = df[df["subject_id"] == test_sid].copy()

        X_train = train_df[feature_cols].values
        y_train = train_df["label_rage_click"].astype(int).values

        X_test = test_df[feature_cols].values
        y_test = test_df["label_rage_click"].astype(int).values

        if len(np.unique(y_train)) < 2:
            print(f"[SKIP] {ablation_name} subject_{test_sid}: training has one class only")
            continue

        model = build_model(cfg)
        model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:, 1]

        fold_metrics = safe_metrics(y_test, prob, threshold=0.5)
        fold_metrics["ablation"] = ablation_name
        fold_metrics["test_subject"] = int(test_sid)
        fold_rows.append(fold_metrics)

        for i, (_, row) in enumerate(test_df.iterrows()):
            all_pred_rows.append(
                {
                    "ablation": ablation_name,
                    "test_subject": int(test_sid),
                    "global_idx": int(row["global_idx"]),
                    "subject_id": int(row["subject_id"]),
                    "local_epoch_idx": int(row["local_epoch_idx"]),
                    "scenario_name": row["scenario_name"],
                    "y_true": int(y_test[i]),
                    "y_prob": float(prob[i]),
                    "y_pred_05": int(prob[i] >= 0.5),
                }
            )

    pred_df = pd.DataFrame(all_pred_rows)
    fold_df = pd.DataFrame(fold_rows)

    global_metrics = safe_metrics(
        pred_df["y_true"].values,
        pred_df["y_prob"].values,
        threshold=0.5,
    )
    global_metrics["ablation"] = ablation_name
    global_metrics["n_features"] = len(feature_cols)

    return global_metrics, fold_df, pred_df


def train_feature_importance(df, feature_cols, cfg, ablation_name):
    X = df[feature_cols].values
    y = df["label_rage_click"].astype(int).values

    model = build_model(cfg)
    model.fit(X, y)

    rf = model.named_steps["rf"]
    importances = rf.feature_importances_

    imp = pd.DataFrame(
        {
            "ablation": ablation_name,
            "feature": feature_cols,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    return imp


def main():
    cfg = load_config()
    output_dir = resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_table_path = output_dir / "feature_table_clean.csv"
    if not feature_table_path.exists():
        raise FileNotFoundError(
            f"Missing feature table: {feature_table_path}\n"
            "Run 01_build_feature_table.py first."
        )

    df = pd.read_csv(feature_table_path)
    df = df[df["label_rage_click"].notna()].copy()
    df["label_rage_click"] = df["label_rage_click"].astype(int)

    print("=" * 80)
    print("RANDOM FOREST LOSO BASELINE")
    print("=" * 80)
    print(f"Rows: {len(df)}")
    print(f"Subjects: {sorted(df['subject_id'].unique().tolist())}")
    print(f"Positive labels: {int(df['label_rage_click'].sum())}")
    print(f"Positive rate: {df['label_rage_click'].mean() * 100:.2f}%")

    global_results = []
    all_folds = []
    all_preds = []
    all_importances = []

    for ablation_name, modalities in ABLATIONS.items():
        feature_cols = get_feature_cols(df, modalities)

        print("\n" + "-" * 80)
        print(f"Ablation: {ablation_name}")
        print(f"Modalities: {modalities}")
        print(f"Feature count: {len(feature_cols)}")

        if not feature_cols:
            print("[SKIP] No features found.")
            continue

        global_metrics, fold_df, pred_df = run_loso(
            df=df,
            feature_cols=feature_cols,
            cfg=cfg,
            ablation_name=ablation_name,
        )

        importance_df = train_feature_importance(
            df=df,
            feature_cols=feature_cols,
            cfg=cfg,
            ablation_name=ablation_name,
        )

        global_results.append(global_metrics)
        all_folds.append(fold_df)
        all_preds.append(pred_df)
        all_importances.append(importance_df)

        print(
            f"Global PR-AUC={global_metrics['pr_auc']:.4f} | "
            f"ROC-AUC={global_metrics['roc_auc']:.4f} | "
            f"BalAcc={global_metrics['balanced_accuracy']:.4f} | "
            f"F1+={global_metrics['f1_positive']:.4f} | "
            f"Recall+={global_metrics['recall_positive']:.4f}"
        )
        print(
            f"Confusion: TN={global_metrics['tn']} FP={global_metrics['fp']} "
            f"FN={global_metrics['fn']} TP={global_metrics['tp']}"
        )

    results_df = pd.DataFrame(global_results)
    folds_df = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    importances_df = pd.concat(all_importances, ignore_index=True) if all_importances else pd.DataFrame()

    results_path = output_dir / "rf_loso_global_results.csv"
    folds_path = output_dir / "rf_loso_fold_results.csv"
    preds_path = output_dir / "rf_loso_predictions.csv"
    imp_path = output_dir / "rf_feature_importances.csv"
    report_path = output_dir / "rf_loso_report.txt"

    results_df.to_csv(results_path, index=False)
    folds_df.to_csv(folds_path, index=False)
    preds_df.to_csv(preds_path, index=False)
    importances_df.to_csv(imp_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RANDOM FOREST LOSO BASELINE REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Positive labels: {int(df['label_rage_click'].sum())}\n")
        f.write(f"Positive rate: {df['label_rage_click'].mean() * 100:.2f}%\n\n")
        f.write("Global results:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        f.write("Important note:\n")
        f.write(
            "The target label_rage_click is highly imbalanced and represents a behavioral proxy, "
            "not a direct self-report frustration label. Accuracy should not be used as the main metric.\n"
        )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved global results : {results_path}")
    print(f"Saved fold results   : {folds_path}")
    print(f"Saved predictions    : {preds_path}")
    print(f"Saved importances    : {imp_path}")
    print(f"Saved report         : {report_path}")

    print("\nSorted by PR-AUC:")
    if len(results_df):
        cols = [
            "ablation",
            "n_features",
            "pr_auc",
            "roc_auc",
            "balanced_accuracy",
            "f1_positive",
            "recall_positive",
            "tp",
            "fp",
            "fn",
            "tn",
        ]
        print(results_df[cols].sort_values("pr_auc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()