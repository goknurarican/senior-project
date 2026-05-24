# V2 / V3 / V5 Pipeline Karşılaştırması

**Tarih:** 2026-05-23  
**Model:** HusformerBITIRMEEG (V2/V3) → HybridV5Model (V5)

---

## 1. Pipeline Farkları

| Özellik | V2 | V3 | V5 |
|---------|----|----|-----|
| Dataset | 1,452 epoch | 480 epoch | 480 epoch |
| Kontrol strateji | Pseudo-marker (5s) | Action-matched (S30/S32) | Action-matched (aynı V3) |
| Sınıf dengesi | 45/55 | 50/50 | 50/50 |
| Design confound | **Var** | Azaltıldı | Azaltıldı |
| Transfer learning | LaBraM (frozen) | LaBraM (frozen) | LaBraM (frozen) |
| EEG temsil | 200-dim embedding | 200-dim embedding | 200-dim embedding + osc adapter |
| Osilasyon feature | Yok | Yok | 6 band × 110 timepoint |
| Yorumlanabilirlik | Yok | Yok | Adapter + band attention |
| Modality balance | Yok | Yok | Dropout (p=0.3) + aux losses |
| Anchor alignment | Yok | Yok | MSE(adapter, osc_mean) × 0.2 |
| Trainable params | ~140K | ~140K | ~89K |

---

## 2. LOSO Performans Karşılaştırması

| Metrik | V2 | V3 | V5 |
|--------|----|----|-----|
| **Accuracy** | **0.998 ± 0.003** | **0.992 ± 0.016** | **0.998 ± 0.006** |
| Balanced Acc | 0.998 ± 0.003 | 0.992 ± 0.016 | 0.998 ± 0.006 |
| F1 Macro | 0.998 ± 0.003 | 0.992 ± 0.016 | 0.998 ± 0.006 |
| **AUC** | **0.999 ± 0.002** | **1.000 ± 0.000** | **1.000 ± 0.000** |

V5, V3 ile özdeş AUC (1.000) elde ederken, ACC std'si V3'ün (±0.016) üzerinden V2 ile aynı seviyeye (±0.006) indi. Modality balance öğrenimi daha stabil bir model üretiyor.

---

## 3. Permütasyon Testi

| Parametre | V2 | V3 | V5 |
|-----------|----|----|-----|
| True AUC | 0.999 | 1.000 | 1.000 |
| Null ortalama | 0.500 ± 0.017 | 0.482 ± 0.033 | 0.501 ± 0.031 |
| **p-değeri** | **< 0.0001** | **< 0.0001** | **< 0.0001** |
| Anlamlı? | ✓ | ✓ | ✓ |

Her üç versiyonda da sinyal şansla açıklanamaz (p < 0.0001).

---

## 4. Modalite Ablasyon Karşılaştırması - En Kritik Tablo

| Koşul | V2 AUC | V3 AUC | V5 AUC | V5−V3 Δ |
|-------|--------|--------|--------|---------|
| full | 0.999 | 1.000 | **1.000** | 0.000 |
| eeg_only | 1.000 | 1.000 | **1.000** | 0.000 |
| **no_eeg** | 0.507 | 0.600 | **0.893** | **+0.293 ✓** |
| no_eye | 0.999 | 1.000 | **1.000** | 0.000 |
| no_mouse | 1.000 | 1.000 | **1.000** | 0.000 |
| **eye_only** | 0.579 | 0.502 | **0.682** | **+0.180 ✓** |
| **mouse_only** | 0.489 | 0.528 | **0.877** | **+0.349 ✓** |

### Yorum - Modality Balance Başarısı

**Modality dropout + auxiliary losses teknikleri açıkça çalıştı:**
- `mouse_only` V3'te 0.528 (şans sınırında) → V5'te **0.877** (+0.349). Mouse tracking artık başlı başına güçlü bir sınıflandırıcı.
- `eye_only` V3'te 0.502 (şans) → V5'te **0.682** (+0.180). Göz izleme artık frustrasyon sinyali taşıyor.
- `no_eeg` V3'te 0.600 → V5'te **0.893** (+0.293). EEG olmadan göz+mouse kombinasyonu çok güçlü.

**Mekanizma:** V2/V3'te model EEG-dominant öğrenir; eye/mouse branch'ları düşük gradyan alır. V5'te:
1. Modality dropout: %30 olasılıkla bir modality sıfırlanır → kalan modaliteler sinyali öğrenmek ZORUNDA kalır.
2. Aux losses: Her branch kendi başına doğru sınıflandırma yapmalı → backpropagation her branch'a güçlü gradyan verir.

### Neden EEG hâlâ tam?

eeg_only hâlâ 1.000 - LaBraM gömme vektörleri, eylem-eşleşmeli koşullarda bile tek başına yeterli. Bu:
- V3'te confound'un tamamen açıklama olmadığını (gerçek neural frustration signal) doğrular
- veya LaBraM'ın çok güçlü özellik çıkarıcı olduğunu (küçük örneklemde bile mükemmel)
- ve/veya hesaplama kapasitesi farkından (200-dim vs 6-dim × zaman) kaynaklanabilir

---

## 5. Band Attention Analizi

| Band | Variant Attn | Control Attn | Fark |
|------|-------------|--------------|------|
| frontal_theta | 0.1577 | 0.1568 | +0.0008 |
| frontal_alpha | 0.1737 | 0.1733 | +0.0004 |
| parietal_alpha | 0.1650 | 0.1644 | +0.0006 |
| central_beta | 0.1655 | 0.1657 | −0.0002 |
| faa_dynamic | 0.1598 | 0.1602 | −0.0004 |
| engagement_index | 0.1783 | 0.1796 | −0.0013 |

**Band attention neredeyse uniform** (~1/6 = 0.167 her band). Tüm farklar < 0.002.

**Yorum:** Model frustrasyon sinyalini tek bir band'a lokalize etmiyor. Bu:
1. Frustrasyon çok-band bir fenomen olabilir (literatürde frontal theta + FAA birlikte bahsedilir)
2. Modelin discriminative bilgiyi `OscillationTemporalEncoder`'dan değil, `labram_adapter + temporal encoding`'in kombinasyonundan çekiyor olabilir
3. Band attention mekanizması interpretability için yeterince granüler değil

**Literatür uyumu sınırlı:** Cavanagh & Frank (2014) frontal theta dominansını öngörüyor, Davidson (2004) FAA'yı. V5 band attention bu hipotezleri desteklemiyor (farklılaşmış değil).

---

## 6. Adapter-Oscillation Korelasyonu (Anchor Alignment)

| Band | Pearson r |
|------|-----------|
| frontal_theta | 0.045 |
| frontal_alpha | 0.078 |
| parietal_alpha | −0.171 |
| central_beta | −0.193 |
| faa_dynamic | −0.061 |
| engagement_index | 0.129 |

**Tüm korelasyonlar çok düşük** (max |r|=0.193). Anchor alignment loss (ağırlık=0.2) adapter'ı oscillation uzayına yeterince çekmedi.

**Yorum:** LaBraM'ın 200-dim gömmesi, bu 6 oscillation boyutuyla zayıf hizalanıyor. Bu beklenebilir bir sonuç çünkü:
- LaBraM pek çok band power'ın ötesinde bilgi yakalıyor (senkroni, faz, cross-frequency coupling...)
- Anchor loss ağırlığı (0.2) task loss'a (1.0) kıyasla çok düşük kalıyor
- Öğrenilen adapter projeksiyon alanı, "oscillation uzayı" olmak yerine "LaBraM'ın ayırt edici boyutları" oluyor

---

## 7. Özet: V2 / V3 / V5 Kritik Karşılaştırma

| Kriter | V2 | V3 | V5 |
|--------|----|----|-----|
| Dataset boyutu | 1,452 | 480 | 480 |
| Design confound | Var | Azaltıldı | Azaltıldı |
| AUC | 0.999 | 1.000 | 1.000 |
| no_eeg AUC | 0.507 | 0.600 | **0.893** |
| eye_only AUC | 0.579 | 0.502 | **0.682** |
| mouse_only AUC | 0.489 | 0.528 | **0.877** |
| Permütasyon p | <0.0001 | <0.0001 | <0.0001 |
| Band attention (literatür) | N/A | N/A | Uniform (desteklemiyor) |
| Adapter-Osc korelasyon | N/A | N/A | Zayıf (max r=0.19) |
| **Modality balance** | Yok | Yok | **✓ Başarılı** |

---

## 8. Sonuç

V5'in en önemli katkısı **modality balance** teknikleridir:
- Eye-only AUC 0.502 → 0.682, Mouse-only 0.528 → 0.877, No-EEG 0.600 → 0.893
- Bu, aux losses + modality dropout'un tek-modalite sistemlere bile güvenilir sinyal öğrettiğini kanıtlıyor

V5'in sınırlılıkları:
- Band attention uniform → literatür-aligned interpretability elde edilemedi
- Adapter-oscillation korelasyonu zayıf → LaBraM → oscillation distillation başarısız

Pratik öneri: Mouse tracking (V5 0.877) **güçlü bir tek-modalite baseline** olarak kullanılabilir; göz izleme ekipmanı yokken bile yüksek AUC elde edilebilir.
