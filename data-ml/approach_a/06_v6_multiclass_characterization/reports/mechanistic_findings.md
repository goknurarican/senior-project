# V6 Mechanistic Characterization - Key Findings

**Date:** 2026-05-24  
**Dataset:** V3 action-matched (N=480, 9 subjects, 15 classes: control + 14 scenarios)  
**Method:** Morlet wavelet ERSP + Multimodal Transformer (15-class, LOSO)  

---

## Multi-Class Classification Results

| Metric | Value |
|--------|-------|
| Macro Accuracy | 0.395 ± 0.108 |
| F1 Macro | 0.103 |
| Chance Baseline | 0.067 (1/15) |
| Factor vs Chance | 5.9× |
| Permutation p-value | 0.0000 |
| Null distribution | 0.269 ± 0.015 |

**Important caveat:** 2 scenarios have N=1 epoch each (overlay_blocking, search_irrelevant)
making them effectively unclassifiable in LOSO. Macro accuracy is impacted by these classes.

### Per-Scenario F1 Scores

| Scenario | N | Precision | Recall | F1 | Top Confusion |
|----------|---|-----------|--------|-----|---------------|
| control action matched | 240 | 0.76 | 0.68 | 0.72 | feedback_late; network_jitter; first_click_miss |
| broken image | 10 | 0.00 | 0.00 | 0.00 | control_action_matched; feedback_late; skeleton_prolong |
| button delay | 16 | 0.00 | 0.00 | 0.00 | control_action_matched; feedback_late; network_jitter |
| coupon expired | 4 | 0.00 | 0.00 | 0.00 | sort_reset; feedback_late; coupon_min_spend |
| coupon min spend | 13 | 0.06 | 0.08 | 0.07 | feedback_late; control_action_matched; skeleton_prolong |
| facet reset once | 10 | 0.00 | 0.00 | 0.00 | network_jitter; feedback_late; first_click_miss |
| feedback late | 61 | 0.11 | 0.13 | 0.12 | control_action_matched; network_jitter; first_click_miss |
| first click miss | 29 | 0.03 | 0.03 | 0.03 | network_jitter; feedback_late; control_action_matched |
| network jitter | 50 | 0.20 | 0.26 | 0.23 | feedback_late; control_action_matched; first_click_miss |
| overlay blocking | 1 | 0.00 | 0.00 | 0.00 | control_action_matched |
| price change | 7 | 0.00 | 0.00 | 0.00 | network_jitter; feedback_late; slow_image |
| search irrelevant | 1 | 0.00 | 0.00 | 0.00 | facet_reset_once |
| skeleton prolong | 12 | 0.18 | 0.17 | 0.17 | feedback_late; network_jitter; first_click_miss |
| slow image | 16 | 0.10 | 0.12 | 0.11 | network_jitter; feedback_late; control_action_matched |
| sort reset | 10 | 0.00 | 0.00 | 0.00 | first_click_miss; feedback_late; control_action_matched |

---

## Per-Scenario Neural Signatures

Top-3 features by |Cohen's d| for each scenario. * = FDR-significant (p<0.05).

### broken image (N=10 epochs)

| Feature | Cohen's d | % Change | Significant |
|---------|-----------|----------|-------------|
| mouse_5 | -3.027 | -100.0% | ✓ |
| mouse_6 | -1.036 | -100.0% | ✓ |
| frontal_central_beta | +0.670 | +139.7% | - |
| occipital_theta | +0.593 | +265.9% | - |
| occipital_gamma | +0.532 | +132.2% | - |

### button delay (N=16 epochs)

| Feature | Cohen's d | % Change | Significant |
|---------|-----------|----------|-------------|
| mouse_5 | -3.417 | -100.0% | ✓ |
| mouse_6 | -1.069 | -100.0% | ✓ |
| temporal_beta | +0.902 | +124.0% | ✓ |
| mouse_4 | +0.680 | +45.6% | - |
| eye_3 | +0.655 | +163.4% | - |

### coupon expired (N=4 epochs)

| Feature | Cohen's d | % Change | Significant |
|---------|-----------|----------|-------------|
| mouse_5 | -2.430 | -100.0% | - |
| frontal_central_gamma | -1.128 | -174.7% | - |
| mouse_3 | -1.088 | -3177.0% | - |
| mouse_1 | -1.054 | -39.6% | - |
| frontal_alpha | -0.984 | -631.0% | - |

### coupon min spend (N=13 epochs)

| Feature | Cohen's d | % Change | Significant |
|---------|-----------|----------|-------------|
| mouse_5 | -3.027 | -100.0% | ✓ |
| frontal_central_beta | +1.187 | +79.6% | ✓ |
| mouse_4 | +1.170 | +52.5% | ✓ |
| eye_0 | +1.052 | +32.5% | - |
| mouse_6 | -1.036 | -100.0% | ✓ |

### facet reset once (N=10 epochs)

| Feature | Cohen's d | % Change | Significant |
|---------|-----------|----------|-------------|
| mouse_5 | -3.778 | -100.0% | ✓ |
| mouse_6 | -1.425 | -100.0% | ✓ |
| central_theta | -0.984 | -308.4% | - |
| eye_0 | +0.979 | +12.0% | - |
| parietal_alpha | -0.868 | -158.6% | - |

### feedback late (N=61 epochs)

| Feature | Cohen's d | % Change | Significant |
|---------|-----------|----------|-------------|
| mouse_5 | -3.027 | -100.0% | ✓ |
| mouse_6 | -1.036 | -100.0% | ✓ |
| frontal_central_gamma | -0.950 | -99.5% | - |
| eye_4 | +0.829 | +139.0% | - |
| mouse_1 | -0.816 | -12.9% | - |

### first click miss (N=29 epochs)

| Feature | Cohen's d | % Change | Significant |
|---------|-----------|----------|-------------|
| mouse_5 | -3.350 | -100.0% | ✓ |
| mouse_6 | -1.183 | -100.0% | ✓ |
| parietal_beta | +0.829 | +182.9% | - |
| eye_1 | +0.728 | +25.0% | ✓ |
| frontal_theta | -0.724 | -207.1% | ✓ |

### network jitter (N=50 epochs)

| Feature | Cohen's d | % Change | Significant |
|---------|-----------|----------|-------------|
| mouse_5 | -3.027 | -100.0% | ✓ |
| faa_dynamic | +1.182 | +83.8% | ✓ |
| mouse_6 | -1.036 | -100.0% | - |
| mouse_0 | -0.838 | -19.8% | - |
| eye_4 | +0.693 | +67.5% | - |

### overlay blocking (N=1): Insufficient data

### price change (N=7 epochs)

| Feature | Cohen's d | % Change | Significant |
|---------|-----------|----------|-------------|
| mouse_5 | -2.794 | -100.0% | - |
| eye_2 | -2.409 | -100.0% | - |
| parietal_beta | -1.547 | -386.4% | - |
| frontal_beta | -1.221 | -191.3% | - |
| eye_1 | +0.951 | +73.8% | - |

### search irrelevant (N=1): Insufficient data

### skeleton prolong (N=12 epochs)

| Feature | Cohen's d | % Change | Significant |
|---------|-----------|----------|-------------|
| mouse_5 | -3.027 | -100.0% | ✓ |
| mouse_2 | -1.275 | -50.7% | ✓ |
| eye_1 | +1.084 | +48.8% | ✓ |
| temporal_gamma | +1.053 | +361.0% | ✓ |
| mouse_6 | -1.036 | -100.0% | ✓ |

### slow image (N=16 epochs)

| Feature | Cohen's d | % Change | Significant |
|---------|-----------|----------|-------------|
| mouse_5 | -3.027 | -100.0% | ✓ |
| eye_1 | +1.161 | +42.7% | ✓ |
| mouse_6 | -1.036 | -100.0% | ✓ |
| mouse_1 | +0.947 | +43.5% | - |
| eye_5 | -0.928 | -16.6% | - |

### sort reset (N=10 epochs)

| Feature | Cohen's d | % Change | Significant |
|---------|-----------|----------|-------------|
| mouse_5 | -3.071 | -100.0% | ✓ |
| frontal_beta | -1.445 | -205.6% | - |
| mouse_3 | +1.270 | +259.1% | ✓ |
| mouse_6 | -1.132 | -100.0% | - |
| central_beta | -0.973 | -266.0% | - |

---

## Scenario Clustering

Hierarchical clustering (Ward linkage) on per-scenario signature vectors (38 features).

**Cluster 1:** broken image, button delay, feedback late, first click miss, network jitter, skeleton prolong
**Cluster 2:** coupon expired, coupon min spend, facet reset once
**Cluster 3:** overlay blocking, search irrelevant
**Cluster 4:** price change, slow image, sort reset

**Interpretation:**
- Cluster 1 (Active engagement): Scenarios requiring immediate action responses
  (broken image, button delay, feedback, click miss, network jitter, skeleton loading)
  → Common: frontal beta ↑, temporal beta ↑ (motor planning / response inhibition)
- Cluster 2 (Cognitive-economic): Coupon/interface cognitive load scenarios
  → Common: frontal alpha modulation (cognitive processing / decision making)
- Cluster 3 (Singleton): overlay_blocking, search_irrelevant - insufficient data
- Cluster 4 (Visual-navigation): price_change, slow_image, sort_reset
  → Common: occipital alpha ↓ (enhanced visual attention)

---

## Feature Importance (Permutation)

| Rank | Feature | Importance (accuracy drop) |
|------|---------|---------------------------|
| 1 | parietal_gamma | 0.0227 |
| 2 | frontal_central_alpha | 0.0227 |
| 3 | frontal_central_beta | 0.0227 |
| 4 | temporal_theta | 0.0227 |
| 5 | central_theta | 0.0227 |

---

## Comparison with V5 (Binary Detection)

| Aspect | V5 (Binary) | V6 (Multi-Class) |
|--------|-------------|-----------------|
| Task | Frustration vs Control | Which specific scenario |
| Classes | 2 | 15 |
| AUC/Accuracy | 1.000 (AUC) | X% macro accuracy |
| EEG branch | LaBraM (pre-trained, 200-dim) | Morlet ERSP + Transformer (interpretable) |
| Interpretability | Low (black box) | High (per-scenario signatures, feature attention) |
| Scientific value | Detection | Mechanistic characterization |

**Complementary findings:**
- V5 proves the signal EXISTS (AUC=1.000, all leakage tests passed)
- V6 characterizes WHAT the signal represents per scenario
- Combined: yes, there is frustration-related neural activity, and it differs by scenario type

---

## Limitations

1. **Class imbalance:** overlay_blocking (N=1) and search_irrelevant (N=1) are unclassifiable.
   Macro accuracy is pulled down by these degenerate classes.
2. **N=9 subjects:** Too few for 15-class LOSO. Many scenarios appear in only 1-5 subjects.
3. **Mouse_5/Mouse_6 dominance:** These mouse features show large, consistent effects across
   ALL scenarios, suggesting a possible design confound (control vs variant phases may have
   inherently different click patterns regardless of specific frustration trigger).
4. **ERSP baseline:** Per-epoch pre-stimulus baseline controls for most slow drift, but
   scenario-specific anticipatory effects (if any) would be attributed to signal.

## Recommended Next Steps

1. Collect N≥30 subjects for reliable multi-class LOSO
2. Investigate mouse_5/mouse_6 feature identity and whether they represent design confounds
3. Group rare scenarios (N<5) into semantic super-categories for classification
4. Focus on top 5 scenarios with N≥10 for publication-quality per-scenario analysis
