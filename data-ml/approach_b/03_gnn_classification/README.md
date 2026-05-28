#Stage 3: GNN Classification

This stage is the optional validation step of Approach B. It asks whether
the connectivity matrices computed in Stage 1 are by themselves sufficient
to discriminate the 15 classes (action-matched control plus 14 frustration
scenarios).

##Architecture

`gnn_model.MinimalGCN`:

- 2 GCNConv layers with hidden dimension 16
- Global mean pooling, dropout 0.5
- Final linear classifier to 15 logits
- Total trainable parameters: under one thousand

The model is deliberately small. With N=9 subjects and the connectivity-only
input, an expressive architecture would overfit immediately. The goal is to
report what is and is not learnable from network structure alone, not to
maximise accuracy.

##Graph construction (per epoch)

- Nodes: 6 ROIs
- Node features: per-ROI band power summaries from Approach A V6 oscillation
  features (mean over time), giving 4 features per node
- Edges: complete graph over the 6 ROIs
- Edge weight: mean wPLI across the 4 bands

##Training

- 9-fold leave-one-subject-out
- AdamW, lr=5e-4, weight_decay=1e-2
- Class-weighted cross-entropy
- Max 40 epochs, patience 8 on validation accuracy
- Random seed 42, MPS or CPU device

##Permutation test

50 label shuffles. For each shuffle the full LOSO is rerun on permuted
labels; the mean accuracy across folds becomes one sample of the null
distribution. The p-value is the proportion of null samples that meet or
exceed the observed mean accuracy.

##Outputs

- `models/fold_<sid>/best_model.pth, predictions.csv, metrics.json`
- `evaluation/loso_summary.json`
- `evaluation/per_scenario_performance.csv`
- `evaluation/confusion_matrix.png`
- `evaluation/permutation_results.json`

##Run

```
python approach_b/03_gnn_classification/src/train_gnn.py
```

Runtime: 20 to 45 minutes depending on device.
