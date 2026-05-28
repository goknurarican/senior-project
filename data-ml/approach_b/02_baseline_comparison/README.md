#Stage 2: Per-Scenario Baseline Comparison

This stage is the primary mechanistic analysis of Approach B. For each of
the 14 frustration scenarios it asks: which inter-region connections change
relative to subject-matched action-matched control epochs, in which bands,
and how large are those changes?

##Pipeline (per scenario, per band, per metric)

1. Build per-subject mean connectivity matrix from scenario epochs.
2. Build per-subject mean connectivity matrix from action-matched control
   epochs.
3. Take the paired difference (scenario minus control) per subject.
4. Paired t-test across subjects for each connection. The t-test is
   primary because at n=9 the Wilcoxon and sign-flip permutation tests
   share a discrete null distribution that caps the smallest possible
   raw p at 1/2^9 = 0.0039, which never survives FDR correction over 21
   connections.
5. False discovery rate correction (Benjamini and Hochberg) across the
   21 connections within each band, applied to the t-test p-values.
6. Cohen's d for paired samples per connection.
7. Wilcoxon p and sign-flip permutation p (500 shuffles) reported as
   secondary checks in the per-connection output file.
8. A connection is flagged significant only when the FDR-corrected t-test
   p is below 0.05 and the absolute Cohen's d is at least 0.5.

##Network categorisation

Significant connections are mapped onto literature-defined networks:

- Fronto-parietal control: frontal to parietal and frontal-central to
  parietal coupling in theta or beta. Indicates conflict monitoring and
  cognitive control (Cavanagh and Frank 2014, Sauseng et al. 2008).
- Default-mode alpha: frontal to parietal and frontal to occipital alpha.
  Indicates task disengagement and attention reorientation.
- Sensorimotor: central to frontal-central and central to parietal in beta
  or gamma. Indicates motor preparation.

##Outputs

`analysis/scenario_<scenario>/`:

- `connectivity_change_matrix.csv` - 4 by 6 by 6 delta values per metric
- `significant_connections.csv` - rows that pass the joint threshold
- `network_plot.png` - circular diagram of significant edges, both metrics

`analysis/all_scenarios_network_summary.csv` - top significant connection per
scenario, per band, per metric. Used by Stage 4 to build Table 1.

`analysis/all_connections_full.csv` - per-connection statistics for all
scenarios in long format.

`reports/network_engagement_table.csv` - per-scenario network category
based on the wPLI significant edges. Used by Stage 4 to build Table 2.

##Run

```
python approach_b/02_baseline_comparison/src/per_scenario_network_analysis.py
```

Runtime: under five minutes (most of the cost is the permutation tests).
