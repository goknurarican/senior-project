#Stage 4: Final Outputs

Generates the final Approach B report and supporting artefacts.

##Run

```
python approach_b/04_final_outputs/generate_report.py
```

##Outputs

- `Approach_B_Report.docx` and `Approach_B_Report.pdf` (via Word.app)
- `figures/`:
  - `fig1_roi_topography.png` - ROI definition on the 32-channel layout
  - `fig2_change_matrices.png` - wPLI scenario-minus-control matrices for
    the six largest-N scenarios across four bands
  - `fig3_network_engagement.png` - count of scenarios engaging each
    literature-defined network
  - `fig4_gnn_confusion.png` - GNN confusion matrix from Stage 3
  - `fig5_v6_vs_gnn_f1.png` - per-scenario F1 comparison, Approach A V6 vs
    Approach B GNN
- `tables/`:
  - `table1_top_connections.csv` - top wPLI connection per scenario
  - `table2_network_engagement.csv` - network category per scenario
  - `table3_gnn_folds.csv` - per-fold GNN metrics
  - `table4_a_vs_b.csv` - feature and model contrast between approaches

##Style

The report follows the conventions agreed for Approach B:

- Times New Roman 11pt body, 12pt headings
- Black and white only
- Numbered headings, prose paragraphs (no bullet lists in the body)
- Tables: grid borders, no shading
- Figures: grayscale, 300 DPI
- In-text references in (Author Year) form
