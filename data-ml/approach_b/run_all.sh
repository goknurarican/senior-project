#!/usr/bin/env bash
#run the full approach b pipeline in sequence.
#assumes stage 1 outputs already exist in
#approach_b/01_connectivity_extraction/features/connectivity_per_epoch/.
set -e
cd "$(dirname "$0")/.."

echo "== stage 2: per-scenario baseline comparison =="
python3 approach_b/02_baseline_comparison/src/per_scenario_network_analysis.py

echo "== stage 3: gnn classification =="
python3 approach_b/03_gnn_classification/src/train_gnn.py

echo "== stage 4: report generation =="
python3 approach_b/04_final_outputs/generate_report.py

echo "approach b pipeline complete."
