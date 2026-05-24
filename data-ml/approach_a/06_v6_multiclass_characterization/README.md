# V6 - Multi-Class Scenario Characterization

## Purpose
15-class classification of 14 frustration scenarios plus control.
Goal is mechanistic characterization, not just detection.
Uses Morlet wavelet ERSP instead of LaBraM for interpretability.

## Key findings
- Macro accuracy: 39.5% (chance 6.7%, p<0.001, 500 permutations)
- Four scenario clusters: temporal/waiting, interface/filter, visual/navigation, singleton
- network_jitter: FAA +1.18 (right frontal activation)
- skeleton_prolong: temporal_gamma +1.05, frontal_central_alpha +0.93

## Notes
mouse_5 and mouse_6 features are zero in all variant epochs (extraction artifact).
overlay_blocking (N=1) and search_irrelevant (N=1) are not learnable in LOSO.
N=9 limits per-scenario statistical power significantly.
