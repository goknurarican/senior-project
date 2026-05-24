# V2 - LaBraM with Pseudo-Marker Control

## Purpose
First attempt at frustration classification. Control epochs defined by pseudo-markers
near variant events, leading to a task-vs-rest confound.

## Key findings
- LOSO AUC: 0.999, ACC: 0.998
- Confound identified: control = free browsing, variant = active task
- EEG ablation: 1.000 (signal is EEG-dominated)
- Mouse-only: 0.489 (behavioral signal weak under confound)

## Notes
Results are inflated by the task-vs-rest artifact. See V3 for corrected results.
Do not cite V2 numbers as frustration detection performance.
