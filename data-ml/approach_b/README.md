#Approach B: Functional Connectivity Analysis

Complementary analysis to Approach A. The aim is to characterise inter-region
neural communication during frustration scenarios and test whether the resulting
network signatures are sufficient to discriminate between scenarios.

##Pipeline stages

1. `01_connectivity_extraction/`, per-epoch wPLI and AEC over 6 ROIs and
   4 frequency bands. Operates on the V3 action-matched dataset (480 epochs,
   9 subjects).
2. `02_baseline_comparison/`, primary mechanistic analysis. Per-scenario
   scenario-minus-control change matrices, paired t-test with FDR
   correction, Cohen's d, permutation validation, network categorisation.
3. `03_gnn_classification/`, optional validation. Minimal 2-layer GCN
   (under 1K parameters) trained with 9-fold LOSO on connectivity matrices.
4. `04_final_outputs/`, final DOCX and PDF report, figures and tables.

##Sample-size discipline

N=9 subjects is small for connectivity estimation. The pipeline is designed
defensively: only established metrics (wPLI primary, AEC secondary), FDR
correction across connections, permutation validation, effect sizes reported
alongside p-values, and explicit uncertainty statements throughout.

##Reproducibility

Seed 42 everywhere. M1 Mac MPS for GNN training, CPU fallback elsewhere.
Total compute estimate: 1-2 hours for the full pipeline.
