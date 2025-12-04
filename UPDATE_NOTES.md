# 🚀 E-Commerce Experiment Platform v2.0 - Tam Güncellenmiş!

## ✅ YAPILAN GÜNCELLEMELER

### 1. **Senaryo Enable/Disable Artık Çalışıyor!**
- Admin panelden senaryo kapatıp açtığınızda **anında** user tarafına yansıyor
- SDK her sayfa yüklemesinde enabled durumunu kontrol ediyor
- Database'de değişiklik yapılır yapılmaz aktif oluyor

### 2. **Yeni Senaryo Kataloğu (16+ Senaryo)**
Dokümanda verdiğiniz tüm senaryolar eklendi:

**A) Yükleme/Görsel (3 adet)**
- `slow_image` - Görseller geç yüklenir
- `broken_image` - Kırık görsel placeholder'ı
- `skeleton_prolong` - Uzun süren skeleton loader

**B) Etkileşim/Sürtünme (3 adet)**
- `button_delay` - Buton gecikmesi
- `first_click_miss` - İlk tıklama yutulur
- `feedback_late` - Toast/feedback geç gelir

**C) Arama/Navigasyon (3 adet)**
- `search_irrelevant` - Alakasız arama sonuçları
- `facet_reset_once` - Filtreler sıfırlanır
- `sort_reset` - Sıralama varsayılana döner

**D) Fiyat/Veri (1 adet)**
- `price_change` - Checkout'ta fiyat değişim uyarısı

**E) Sepet/Kupon (2 adet)**
- `coupon_min_spend` - Minimum tutar hatası
- `coupon_expired` - Süresi dolmuş kupon

**F) Ödeme (2 adet)**
- `3ds_soft_fail` - İlk 3DS denemesi başarısız
- `payment_retry_timeout` - Payment timeout sonra başarı

**G) Overlay (1 adet)**
- `overlay_blocking` - Kapatılabilir modal overlay

**H) Ağ (1 adet)**
- `network_jitter` - Network gecikmeleri

### 3. **Olasılık Tier Sistemi**
4 deney grubu ile probability stratejisi:

| Grup | Trafik | Olasılık | Açıklama |
|------|--------|----------|----------|
| **Control** | 25% | 0% | Hiç senaryo yok |
| **Variant A** | 25% | 30% | Low tier (düşük olasılık) |
| **Variant B** | 25% | 60% | Medium tier (orta olasılık) |
| **Variant C** | 25% | 100% | Full tier (yüksek olasılık) |

### 4. **Guardrails & Limitler**
- Max 2 senaryo per session
- 120 saniye cooldown between scenarios
- Ödeme adımında kalıcı blok yok
- Her overlay kapatılabilir (a11y uyumlu)

### 5. **Test Lab Sayfası** 
Yeni `/test-scenarios` sayfası eklendi:
- Hangi grupta olduğunuzu görün
- Tüm senaryoları listeleyin
- Enable/Disable durumlarını kontrol edin
- Manuel olarak senaryo tetikleyin
- Gerçek zamanlı trigger monitoring

## 📋 KURULUM

```bash
# 1. Projeyi açın
cd ecommerce-experiment

# 2. Bağımlılıkları yükleyin
npm install

# 3. Database'i sıfırlayın (önemli!)
npm run reset-db

# 4. Uygulamayı başlatın
npm run dev
```

## 🧪 TEST ADIMLARI

### 1. Admin Panel'den Senaryo Yönetimi
```
1. http://localhost:3000/login → admin@test.com / admin123
2. Admin → Scenarios sekmesine gidin
3. Senaryoları Enable/Disable edin
4. Delete ile silebilirsiniz
5. "+ Add New Scenario" ile yeni ekleyin
```

### 2. Test Lab'da Doğrulama
```
1. http://localhost:3000/test-scenarios açın
2. Experiment grubunuzu kontrol edin (Control/A/B/C)
3. Enabled senaryoları görün
4. 3+ saniye bekleyin (otomatik tetikleme)
5. Veya "Force Execute" ile manuel tetikleyin
```

### 3. Normal Sitede Test
```
1. http://localhost:3000/products gidin
2. Admin'den senaryoları açıp kapatın
3. Sayfayı yenileyin
4. Console'da logları izleyin (F12)
5. Senaryolar probability'ye göre tetiklenir
```

## 🔍 NASIL ÇALIŞIYOR?

### SDK Akışı:
```javascript
1. SDK init → Session bilgisi al
2. Experiment group kontrolü (control ise dur)
3. /api/scenarios/active'den senaryoları çek
4. Her saniye enabled ve probability kontrolü
5. Cooldown ve limit kontrolü (max 2, 120s)
6. Koşullar sağlanırsa executeScenario()
7. Event logging ve tracking
```

### Probability Hesaplama:
```javascript
// Base probability: 0.6 (60%)
Control → 0% (hiç tetiklenmez)
Variant A → 0.6 * 0.5 = 0.3 (30%)
Variant B → 0.6 * 0.8 = 0.48 (48%)
Variant C → 0.6 * 1.5 = 0.9 (90%)
```

## 📊 METRIKLER

Admin → Experiments'ta grup bazlı metrikler:
- Session sayısı
- Ortalama event/session
- Cart add sayısı
- Tetiklenen senaryo sayısı

Admin → Sessions'ta detaylı tracking:
- Session timeline
- Event akışı
- Senaryo trigger zamanları
- Relative timestamp (EEG sync için)

## 🐛 DEBUG İPUÇLARI

### Browser Console'da:
```javascript
// SDK durumu
window.ExperimentSDK.sessionId
window.ExperimentSDK.experimentGroup
window.ExperimentSDK.scenarios
window.ExperimentSDK.triggeredScenarios

// Manuel tetikleme
window.ExperimentSDK.executeScenario({
  id: 1,
  type: 'button_delay',
  params: '{"delay": 2000}'
})
```

### Database Reset:
```bash
# Duplicate senaryoları temizler
npm run reset-db
# Veya manuel
rm experiment.db
npm run dev
```

## ⚠️ BİLİNEN SORUNLAR & ÇÖZÜMLERİ

| Sorun | Çözüm |
|-------|--------|
| Senaryolar tetiklenmiyor | Experiment grubunuzu kontrol edin (Control'de iseniz tetiklenmez) |
| Enable/Disable yansımıyor | Sayfayı yenileyin (F5) |
| Duplicate senaryolar | `npm run reset-db` çalıştırın |
| SDK yüklenmiyor | Console'da hata var mı kontrol edin |

## 🎯 SONUÇ

Artık tam fonksiyonel bir negatif case injection platformunuz var:
- ✅ 16+ farklı senaryo tipi
- ✅ 4 probability tier'ı
- ✅ Enable/Disable anında yansıyor
- ✅ Guardrails ve limitler
- ✅ Test Lab ile kolay doğrulama
- ✅ Admin panel tam kontrol
- ✅ EEG için relative timestamp

Projeniz için harika bir temel! 🚀
