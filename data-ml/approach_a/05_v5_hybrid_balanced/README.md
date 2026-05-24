# V5 - Hybrid Model with Modality Balance

## Purpose
Extends V3 by adding oscillation time series features (Morlet wavelet) alongside
LaBraM embeddings, plus modality dropout and auxiliary losses for modality balance.

## Key findings
- LOSO AUC: 1.000, ACC: 0.998
- Mouse-only: 0.877 (up from 0.528 in V3)
- Eye-only: 0.682 (up from 0.502)
- No-EEG: 0.893

## Notes
Modality balance is the key contribution of V5. The architecture enables
deployment without EEG hardware using mouse+eye features alone.
