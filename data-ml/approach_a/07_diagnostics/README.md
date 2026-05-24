# Diagnostics

Leakage tests and sanity checks for the V3 pipeline.

## Tests

Test 1: Normalization leakage - no leakage found (delta_AUC = 0.000)
Test 2: Subject identity in LaBraM - low (21.9% vs 11.1% chance)
Test 3: Random label shuffle - no structural leakage (null AUC 0.494)
Test 4: Classical band power features - AUC 0.539 (chance level)

Window analysis: no single 500ms window is sufficient. Signal requires
full epoch context, consistent with LaBraM's temporal integration.

## Notes
Test 4 is important: scalar band power cannot discriminate frustration.
LaBraM's AUC=1.000 comes from fine-grained temporal dynamics.
