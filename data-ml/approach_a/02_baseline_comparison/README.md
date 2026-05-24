# 02 Baseline Comparison

Statistical analysis of raw EEG band power, eye, and mouse features
before any deep learning.

## Contents

src/            baseline_comparison.py and pipeline scripts
statistics/     per-feature group comparison tables, effect sizes
reports/        forest plots, violin plots, report markdown

## Main result

Classical band power features at scalar level are not sufficient for
frustration detection (consistent with V3 diagnostic: AUC ~0.54).
