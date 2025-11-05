# E-Commerce Experiment Platform

## 🚀 Hızlı Kurulum

### 1. Bağımlılıkları yükleyin:
```bash
npm install
```

### 2. Uygulamayı başlatın:
```bash
npm run dev
```

Uygulama http://localhost:3000 adresinde çalışacaktır.

## 📁 Proje Yapısı

```
ecommerce-experiment/
├── pages/                 # Next.js sayfaları
│   ├── api/              # API endpoints
│   ├── admin/            # Admin panel sayfaları
│   ├── index.tsx         # Ana sayfa
│   ├── products.tsx      # Ürünler listesi
│   ├── cart.tsx          # Sepet
│   └── checkout.tsx      # Ödeme
├── components/           # React components
├── lib/                  # Veritabanı ve utilities
├── public/               # Static dosyalar
│   └── injection-sdk.js # Frontend injection SDK
├── styles/               # CSS dosyaları
└── experiment.db         # SQLite veritabanı (otomatik oluşur)
```

## 🔬 Deney Sistemi

### Experiment Grupları:
- **control** (50%): Hiçbir senaryo uygulanmaz
- **variant_a** (25%): Senaryolar uygulanır
- **variant_b** (25%): Senaryolar uygulanır

### Senaryolar:

1. **Slow Image Load**: Ürün görsellerinin gecikmeli yüklenmesi
2. **Button Delay**: Sepete ekle butonunda gecikme
3. **Search Irrelevant**: Arama sonuçlarının geçici karışması
4. **3DS Soft Fail**: İlk ödeme denemesinde hata
5. **Price Change Warning**: Checkout'ta fiyat değişikliği uyarısı

## 🛠️ Admin Panel

Admin paneline http://localhost:3000/admin adresinden ulaşabilirsiniz.

### Özellikler:
- Gerçek zamanlı oturum izleme
- Senaryo yönetimi (etkinleştir/devre dışı bırak)
- Event tracking ve raporlama
- Deney gruplarına göre metrikler

## 📊 Event Tracking

SDK otomatik olarak şu eventleri toplar:
- Page views
- Clicks (button, link)
- Scroll events
- Scenario triggers (start/end)
- Cart actions
- Checkout events

## 🔧 Senaryo Parametreleri

Her senaryo için ayarlanabilir parametreler:

```javascript
{
  delay: 2000,        // Gecikme süresi (ms)
  duration: 5000,     // Etki süresi (ms)
  probability: 0.3,   // Tetiklenme olasılığı (0-1)
  target_page: "*",   // Hedef sayfa pattern
  selector: ".class"  // DOM seçici
}
```

## 📝 Veritabanı Tabloları

- **sessions**: Kullanıcı oturumları ve deney grupları
- **events**: Tüm kullanıcı etkileşimleri
- **experiments**: Deney tanımları
- **scenarios**: Senaryo konfigürasyonları
- **scenario_triggers**: Tetiklenen senaryoların logu
- **products**: Ürün verileri
- **cart_items**: Sepet içerikleri

## 🚦 API Endpoints

### Public APIs:
- `GET /api/session/info` - Session bilgisi
- `GET /api/products` - Ürünleri listele
- `GET/POST/DELETE /api/cart` - Sepet işlemleri
- `POST /api/events/batch` - Event gönderimi
- `GET /api/scenarios/active` - Aktif senaryolar

### Admin APIs:
- `GET /api/admin/stats` - Dashboard istatistikleri
- `GET /api/admin/events/recent` - Son eventler
- `GET/POST /api/admin/scenarios` - Senaryo yönetimi
- `PATCH /api/admin/scenarios/[id]` - Senaryo güncelleme

## 💡 Kullanım İpuçları

1. **Test için**: Tarayıcı konsolu açık tutun, SDK loglarını görebilirsiniz
2. **Session Reset**: Cookie'leri temizleyerek yeni session başlatabilirsiniz
3. **Senaryo Test**: Admin panelden senaryoları manuel tetikleyebilirsiniz
4. **Metrik İzleme**: Events tablosunu SQL ile sorgulayarak detaylı analiz yapabilirsiniz

## 🐛 Debug

Konsolda SDK durumunu kontrol edin:
```javascript
window.ExperimentSDK.sessionId        // Session ID
window.ExperimentSDK.experimentGroup  // Deney grubu
window.ExperimentSDK.scenarios        // Yüklü senaryolar
window.ExperimentSDK.triggeredScenarios // Tetiklenen senaryolar
```

## 📈 Metrik Hesaplama

Örnek SQL sorguları:

```sql
-- CTR hesaplama (scenario öncesi/sonrası)
SELECT 
  COUNT(CASE WHEN event_type='click' THEN 1 END) * 100.0 / 
  COUNT(CASE WHEN event_type='page_view' THEN 1 END) as CTR
FROM events
WHERE session_id IN (
  SELECT session_id FROM sessions 
  WHERE experiment_group = 'variant_a'
);

-- Ortalama oturum süresi
SELECT 
  AVG(duration) as avg_session_duration
FROM (
  SELECT 
    session_id,
    (MAX(timestamp) - MIN(timestamp)) / 1000.0 as duration
  FROM events
  GROUP BY session_id
);
```

## ⚡ Performans Optimizasyonları

- Event batching (2 saniyede bir gönderim)
- Lazy loading for images
- Session-based caching
- Deterministik grup atama (consistent experience)
