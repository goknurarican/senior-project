# BITIRMEEG - Approach A

Multimodal frustration detection pipeline using EEG, eye tracking, and mouse dynamics.
9 subjects, 3 session variants, 14 injected UX frustration scenarios.

## Structure

01_preprocessing/              raw data -> cleaned epochs
02_baseline_comparison/        statistical analysis of raw features
03_v2_labram_pseudo_control/   LaBraM embeddings, pseudo-marker control (historical)
04_v3_labram_action_matched/   LaBraM embeddings, action-matched control (main)
05_v5_hybrid_balanced/         LaBraM + oscillation features, modality balance
06_v6_multiclass_characterization/  15-class scenario characterization
07_diagnostics/                leakage tests and sanity checks
99_final_thesis_outputs/       consolidated thesis-ready outputs

## Main findings

V3: LOSO AUC 1.000, action-matched control eliminates task-vs-rest confound
V5: Mouse-only AUC 0.877 after modality balance training
V6: 39.5% macro accuracy on 15-class problem (chance 6.7%), p<0.001
Four neurobiologically meaningful frustration sub-type clusters identified

## Dataset

Custom Next.js e-commerce platform, 14 UX frustration events injected.
EEG: 32-channel BrainVision, 500Hz. Eye: Tobii. Mouse: platform-logged.
