from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


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


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_positive": f1_score(y_true, y_pred, zero_division=0),
        "precision_positive": precision_score(y_true, y_pred, zero_division=0),
        "recall_positive": recall_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    cfg = load_config()
    out_dir = resolve_output_dir(cfg)

    pred_path = out_dir / "rf_loso_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    preds = pd.read_csv(pred_path)

    rows = []

    for ablation, g in preds.groupby("ablation"):
        y_true = g["y_true"].astype(int).values
        y_prob = g["y_prob"].astype(float).values

        base = {
            "ablation": ablation,
            "n": len(g),
            "positives": int(y_true.sum()),
            "positive_rate": float(y_true.mean()),
            "pr_auc": average_precision_score(y_true, y_prob)
            if len(np.unique(y_true)) == 2 else np.nan,
            "roc_auc": roc_auc_score(y_true, y_prob)
            if len(np.unique(y_true)) == 2 else np.nan,
        }

        # Dense thresholds from 0.00 to 0.50.
        # 0.5 already failed, so low thresholds matter here.
        for threshold in np.linspace(0.0, 0.5, 101):
            m = compute_metrics(y_true, y_prob, threshold)
            row = dict(base)
            row.update(m)
            rows.append(row)

    sweep = pd.DataFrame(rows)

    sweep_path = out_dir / "rf_threshold_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    print("=" * 90)
    print("RF THRESHOLD SWEEP")
    print("=" * 90)
    print(f"Saved: {sweep_path}")

    print("\nBest threshold by F1 positive:")
    cols = [
        "ablation",
        "threshold",
        "pr_auc",
        "roc_auc",
        "balanced_accuracy",
        "f1_positive",
        "precision_positive",
        "recall_positive",
        "tp",
        "fp",
        "fn",
        "tn",
    ]

    best_f1 = (
        sweep.sort_values(["ablation", "f1_positive", "balanced_accuracy"], ascending=[True, False, False])
        .groupby("ablation")
        .head(1)
        .sort_values("f1_positive", ascending=False)
    )

    print(best_f1[cols].to_string(index=False))

    print("\nBest threshold by balanced accuracy:")
    best_bal = (
        sweep.sort_values(["ablation", "balanced_accuracy", "f1_positive"], ascending=[True, False, False])
        .groupby("ablation")
        .head(1)
        .sort_values("balanced_accuracy", ascending=False)
    )

    print(best_bal[cols].to_string(index=False))


if __name__ == "__main__":
    main()