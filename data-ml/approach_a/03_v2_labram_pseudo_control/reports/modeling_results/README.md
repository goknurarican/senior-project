# Approach A Modeling Results

## Bu Klasörde Ne Var?

Yaklaşım A LOSO eğitim sonuçlarının resmi raporu - **V2 ve V3 pipeline karşılaştırması dahil**.

## Dosyalar

### Final Raporlar (V2 + V3 Karşılaştırmalı)

- **BITIRMEEG_Modeling_Results_FINAL_TR.docx**: Türkçe final rapor - V2 & V3 yan yana
- **BITIRMEEG_Modeling_Results_FINAL_EN.docx**: İngilizce final rapor - V2 & V3 yan yana
- **BITIRMEEG_Modeling_Results_FINAL_EN.pdf**: İngilizce PDF versiyonu
- **BITIRMEEG_Modeling_Results_FINAL_TR.pdf**: Türkçe PDF versiyonu

### Önceki Raporlar (V2 Only)

- **BITIRMEEG_Modeling_Results_TR.docx**: V2-only Türkçe rapor
- **BITIRMEEG_Modeling_Results_EN.docx**: V2-only İngilizce rapor
- **BITIRMEEG_Modeling_Results_EN.pdf**: V2-only PDF versiyonu

## Ana Bulgular (Özet)

### V2 Pipeline (Pseudo-marker Control)
1. **LOSO: ACC=0.998±0.003, AUC=0.999±0.002** - 1.452 epoch, 9 özne
2. **Permutation test p<0.0001** - sinyal gerçek (null=0.500±0.017)
3. **EEG dominant** (eeg_only AUC=1.000); eye=0.579, mouse=0.489
4. **Design confound mevcut** - kontrol = serbest gezinme, variant = aktif görev

### V3 Pipeline (Action-Matched Control)
1. **LOSO: ACC=0.992±0.016, AUC=1.000±0.000** - 480 epoch, 1:1 dengeli
2. **Permutation test p<0.0001** - sinyal gerçek (null=0.482±0.033)
3. **EEG dominant** (eeg_only AUC=1.000); no_eeg=0.600, eye_only=0.502
4. **Confound azaltıldı** - kontrol epochları S30/S32 user-action'larına kilitli

### V2 vs V3 En Büyük Farklar
| | V2 | V3 |
|---|---|---|
| no_eeg AUC | 0.507 | **0.600** (+confound gitti) |
| eye_only AUC | 0.579 | **0.502** (şans seviyesi) |
| ACC std | ±0.003 | ±0.016 (küçük n) |
| AUC | 0.999 | **1.000** |

## Figures

| Dosya | İçerik |
|-------|--------|
| figures/per_subject_distribution.png | Özne bazlı LOSO doğruluk dağılımı |
| figures/permutation_null_dist.png | V2 permütasyon testi null dağılımı |
| figures/ablation_study.png | V2 modalite ablasyon barları |
| figures/attention_heatmap.png | Cross-modal attention ısı haritası |
| figures/baseline_comparison.png | RF / SVM / MLP karşılaştırması |
| evaluation/v3/ablation_study_v3.png | V3 modalite ablasyon barları |
| evaluation/v3/permutation_null_dist_v3.png | V3 permütasyon testi null dağılımı |

## Tables

| Dosya | İçerik |
|-------|--------|
| tables/per_subject_performance.csv | V2 özne bazlı ACC, AUC, F1 |
| tables/ablation_summary.csv | V2 koşul bazlı AUC |
| tables/baseline_comparison.csv | RF/SVM/MLP LOSO AUC |
| evaluation/v3/ablation_study_v3.csv | V3 koşul bazlı AUC |
| evaluation/v3/permutation_test_v3.json | V3 permütasyon test sonuçları |

## Kaynak Dosyalar

- V2 değerlendirme: `approach_a/evaluation/`
- V3 değerlendirme: `approach_a/evaluation/v3/`
- V2 model checkpointları: `approach_a/training/loso_results/fold_{14..23}/`
- V3 model checkpointları: `approach_a/training/loso_results_v3/fold_{14..23}/`
- V2 LOSO özet: `approach_a/training/loso_results/loso_summary.json`
- V3 LOSO özet: `approach_a/training/loso_results_v3/loso_summary_v3.json`
- V2 vs V3 karşılaştırma: `approach_a/reports/v2_vs_v3_comparison.md`
